import numpy as np
import random
import torch
import torch.nn as nn
from util.tool import targetItemSelect
from util.metrics import AttackMetric
from util.algorithm import find_k_largest
import torch.nn.functional as F
import scipy.sparse as sp
from copy import deepcopy
from util.loss import bpr_loss, l2_reg_loss
from sklearn.neighbors import LocalOutlierFactor as LOF
from recommender.GMF import GMF
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings


class New():
    def __init__(self, arg, data):
        """
        :param arg: parameter configuration
        :param data: dataLoder
        """
        self.data = data
        self.interact = data.matrix()
        self.userNum = self.interact.shape[0]
        self.itemNum = self.interact.shape[1]

        self.targetItem = targetItemSelect(data, arg)
        self.targetItem = [data.item[i.strip()] for i in self.targetItem]
        self.Epoch = arg.Epoch
        self.innerEpoch = arg.innerEpoch
        self.outerEpoch = arg.outerEpoch

        # capability prior knowledge
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = True

        # limitation 
        self.maliciousUserSize = arg.maliciousUserSize
        self.maliciousFeedbackSize = arg.maliciousFeedbackSize
        if self.maliciousFeedbackSize == 0:
            self.maliciousFeedbackNum = int(self.interact.sum() / data.user_num)
        elif self.maliciousFeedbackSize >= 1:
            self.maliciousFeedbackNum = self.maliciousFeedbackSize
        else:
            self.maliciousFeedbackNum = int(self.maliciousFeedbackSize * self.item_num)

        if self.maliciousUserSize < 1:
            self.fakeUserNum = int(data.user_num * self.maliciousUserSize)
        else:
            self.fakeUserNum = int(self.maliciousUserSize)

        self.batchSize = 2048

    def posionDataAttack(self, recommender):
        selected_indices = self.pretrain_and_cluster(recommender)
        self.fakeUserInject(recommender, selected_indices)
        uiAdj = recommender.data.matrix()
        optimizer = torch.optim.Adam(recommender.model.parameters(), lr=recommender.args.lRate / 10)
        topk = min(recommender.topN)
        bestTargetHitRate = -1
        for epoch in range(self.Epoch):
            # outer optimization
            tmpRecommender = deepcopy(recommender)
            uiAdj2 = uiAdj[:, :]
            ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                self.userNum + self.fakeUserNum + self.itemNum, self.userNum + self.fakeUserNum + self.itemNum),
                                    dtype=np.float32)
            ui_adj[:self.userNum + self.fakeUserNum, self.userNum + self.fakeUserNum:] = uiAdj2
            tmpRecommender.model._init_uiAdj(ui_adj + ui_adj.T)
            optimizer_attack = torch.optim.Adam(tmpRecommender.model.parameters(), lr=recommender.args.lRate)
            for _ in range(self.outerEpoch):
                # Pu, Pi = tmpRecommender.model()
                # We do not know Pu, so learn Pu
                optimizer = torch.optim.Adam([tmpRecommender.model.embedding_dict["user_emb"]], lr=recommender.args.lRate)
                tmpRecommender.train(Epoch=5, optimizer=optimizer, evalNum=5)
                Pu, Pi = tmpRecommender.model()

                scores = torch.zeros((self.userNum + self.fakeUserNum, self.itemNum))
                for batch in range(0,self.userNum + self.fakeUserNum, self.batchSize):
                    scores[batch:batch + self.batchSize, :] = (Pu[batch:batch + self.batchSize, :] \
                                    @ Pi.T).detach()
                nozeroInd = uiAdj2.indices
                scores[nozeroInd[0],nozeroInd[1]] = -10e8
                _, top_items = torch.topk(scores, topk)
                top_items = [[iid.item() for iid in user_top] for user_top in top_items]
                users, pos_items, neg_items = [], [], []
                users_only = []
                for idx, u_index in enumerate(list(range(self.userNum))):
                    users_only.append(u_index)
                    for item in self.targetItem:
                        users.append(u_index)
                        pos_items.append(item)
                        neg_items.append(top_items[u_index].pop())
                user_emb = Pu[users]
                user_emb_only = Pu[users_only]
                pos_items_emb = Pi[pos_items]
                neg_items_emb = Pi[neg_items]
                pos_score = torch.mul(user_emb, pos_items_emb).sum(dim=1)
                neg_score = torch.mul(user_emb, neg_items_emb).sum(dim=1)
                CWloss = neg_score - pos_score
                CWloss = CWloss.mean()


                def uniformity_opt(H, batch_size=512, temp=1):
                    num_users = H.size(0)
                    total_similarity = 0.0
                    total_pairs = 0

                    with torch.no_grad():
                        for start in range(0, num_users, batch_size):
                            end = min(start + batch_size, num_users)
                            current_batch = H[start:end]  

                            inner_product = torch.matmul(current_batch, H.T)  # (B, M)
                            
                            for i in range(start, end):
                                inner_product[i - start, i] = float('-inf')

                            exp_sim = torch.exp(inner_product / temp)  
                            
                            if start > 0:
                                upper_triangle_sum = exp_sim[:, start:].sum()
                                num_elements = exp_sim[:, start:].numel()
                            else:
                                upper_triangle_sum = exp_sim.sum()
                                num_elements = exp_sim.numel()

                            total_similarity += upper_triangle_sum.item()
                            total_pairs += num_elements

                        torch.cuda.empty_cache() 

                    return total_similarity / total_pairs

                user_emb_only.cuda()
                uniform_loss = uniformity_opt(user_emb_only)
                
                lossall = CWloss - uniform_loss
                optimizer_attack.zero_grad()
                lossall.backward()
                optimizer_attack.step()
            for batch in range(0,len(self.fakeUser),self.batchSize):
                uiAdj2[self.fakeUser[batch:batch + self.batchSize], :] = (Pu[self.fakeUser[batch:batch + self.batchSize], :] @ Pi.T).detach().cpu().numpy()
            uiAdj2[self.fakeUser, :] = self.project(uiAdj2[self.fakeUser, :],
                                                          self.maliciousFeedbackNum)
            for u in self.fakeUser:
                uiAdj2[u,self.targetItem] = 1

            uiAdj = uiAdj2[:, :]

            # inner optimization
            ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                self.userNum + self.fakeUserNum + self.itemNum, self.userNum + self.fakeUserNum + self.itemNum),
                                   dtype=np.float32)
            ui_adj[:self.userNum + self.fakeUserNum, self.userNum + self.fakeUserNum:] = uiAdj

            recommender.model._init_uiAdj(ui_adj + ui_adj.T)
            recommender.train(Epoch=self.innerEpoch, optimizer=optimizer, evalNum=5)

            attackmetrics = AttackMetric(recommender, self.targetItem, [topk])
            targetHitRate = attackmetrics.hitRate()[0]
            print(targetHitRate)
            if targetHitRate > bestTargetHitRate:
                bestAdj = uiAdj[:,:]
                bestTargetHitRate = targetHitRate
            
            uiAdj = bestAdj[:,:]
            
            print("BiLevel epoch {} is over\n".format(epoch + 1))
        self.interact = bestAdj
        return self.interact

    def project(self, mat, n):
        try:
            matrix = torch.tensor(mat[:, :].todense())
            _, indices = torch.topk(matrix, n, dim=1)
            matrix.zero_()
            matrix.scatter_(1, indices, 1)
        except:
            matrix = mat[:,:]
            for i in range(matrix.shape[0]):
                subMatrix = torch.tensor(matrix[i, :].todense())
                topk_values, topk_indices = torch.topk(subMatrix, n)
                subMatrix.zero_()  
                subMatrix[0, topk_indices] = 1
                matrix[i, :] = subMatrix[:, :].flatten()
        return matrix

    def fakeUserInject(self, recommender, selected_indices):
        Pu, Pi = recommender.model()
        recommender.data.user_num += self.fakeUserNum

        for i in range(self.fakeUserNum):
            recommender.data.user[f"fakeuser{i}"] = len(recommender.data.user)
            recommender.data.id2user[len(recommender.data.user) - 1] = f"fakeuser{i}"

        self.fakeUser = list(range(self.userNum, self.userNum + self.fakeUserNum))
        row, col, entries = [], [], []

        for idx, u in enumerate(self.fakeUser):
    
            source_idx = source_idx = selected_indices[idx % len(selected_indices)]  
            
            source_user_history = self.interact[source_idx].indices

            fake_user_history = list(source_user_history)

            for i in range(len(self.targetItem)):
                fake_user_history.append(self.targetItem[i])  
            
            while len(fake_user_history) < self.maliciousFeedbackNum:
                random_item = random.sample(list(set(range(self.itemNum)) - set(fake_user_history)), 1)[0]
                fake_user_history.append(random_item)

            for item in fake_user_history:
                recommender.data.training_data.append((recommender.data.id2user[u], recommender.data.id2item[item]))

        for pair in recommender.data.training_data:
            row.append(recommender.data.user[pair[0]])
            col.append(recommender.data.item[pair[1]])
            entries.append(1.0)

        recommender.data.interaction_mat = sp.csr_matrix(
            (entries, (row, col)), 
            shape=(recommender.data.user_num, recommender.data.item_num), 
            dtype=np.float32
        )

        recommender.__init__(recommender.args, recommender.data)

        with torch.no_grad():
            try:
                recommender.model.embedding_dict['user_emb'][:Pu.shape[0]] = Pu
                recommender.model.embedding_dict['item_emb'][:] = Pi
            except:
                recommender.model.embedding_dict['user_mf_emb'][:Pu.shape[0]] = Pu[:, :Pu.shape[1]//2]
                recommender.model.embedding_dict['user_mlp_emb'][:Pu.shape[0]] = Pu[:, Pu.shape[1]//2:]
                recommender.model.embedding_dict['item_mf_emb'][:] = Pi[:, :Pi.shape[1]//2]
                recommender.model.embedding_dict['item_mlp_emb'][:] = Pi[:, Pi.shape[1]//2:]

        recommender.model = recommender.model.cuda()



    def pretrain_and_cluster(self, recommender, max_clusters=20, alpha=1):
        warnings.filterwarnings("ignore", category=FutureWarning)

        proxy_model = deepcopy(recommender.model)
        proxy_model.train()

        with torch.no_grad():
            Pu, Pi = proxy_model()

        user_emb = Pu.detach().cpu().numpy()

        best_n_clusters = 2
        best_silhouette_score = -1
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(user_emb)
            silhouette_avg = silhouette_score(user_emb, kmeans.labels_)
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_n_clusters = n_clusters

        kmeans = KMeans(n_clusters=best_n_clusters, n_init='auto').fit(user_emb)
        cluster_labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        adjacency_matrix = recommender.data.interaction_mat
        node_degrees = adjacency_matrix.sum(axis=1).A1

        cluster_sizes = np.bincount(cluster_labels)
        fake_users_per_cluster = (cluster_sizes / cluster_sizes.sum() * self.fakeUserNum).astype(int)
        leftover_users = self.fakeUserNum - fake_users_per_cluster.sum()
        if leftover_users > 0:
            adjust_indices = np.argsort(-cluster_sizes)[:leftover_users]
            fake_users_per_cluster[adjust_indices] += 1

        selected_indices = []
        global_indices = np.arange(len(user_emb))
        for cluster_idx, center in enumerate(cluster_centers):
            cluster_mask = cluster_labels == cluster_idx
            cluster_user_emb = user_emb[cluster_mask]
            distances = np.linalg.norm(cluster_user_emb - center, axis=1)
            cluster_degrees = node_degrees[cluster_mask]

            scores = distances + alpha * cluster_degrees
            n_top_nodes = fake_users_per_cluster[cluster_idx]

            sorted_indices = np.argsort(scores)[-n_top_nodes:]
            selected_indices.extend(global_indices[cluster_mask][sorted_indices])

        return selected_indices
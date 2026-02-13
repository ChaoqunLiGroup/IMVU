import time
from conf.attack_parser import attack_parse_args
from conf.recommend_parser import recommend_parse_args
from util.DataLoader import DataLoader
from util.tool import seedSet
from ATK import ATK
import os
import torch
import numpy as np
import random


if __name__ == '__main__':

    recommend_args = recommend_parse_args()
    attack_args = attack_parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = recommend_args.gpu_id
    seed = recommend_args.seed
    seedSet(seed)

    import_str = 'from recommender.' + recommend_args.model_name + ' import ' + recommend_args.model_name
    exec(import_str)
    import_str = 'from attack.' + attack_args.attackCategory + "." + attack_args.attackModelName + ' import ' + attack_args.attackModelName
    exec(import_str)

    data = DataLoader(recommend_args)

    recommend_model = eval(recommend_args.model_name)(recommend_args, data)
    attack_model = eval(attack_args.attackModelName)(attack_args, data)
    atk = ATK(recommend_model, attack_model, recommend_args, attack_args)

    s = time.time()

    atk.RecommendTrain()
    atk.RecommendTest()
    atk.PoisonDataAttack()
    for step in range(atk.times):
        print("attack step:{}".format(step))
        atk.RecommendTrain(attack=step)
        atk.RecommendTest(attack=step)

    atk.ResultAnalysis()

    e = time.time()
    print("Running time: %f s" % (e - s))

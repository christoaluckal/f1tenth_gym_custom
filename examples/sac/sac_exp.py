import multiprocessing as mp
from subprocess import call
import random
import argparse
import numpy as np



parser = argparse.ArgumentParser()

parser.add_argument('--multi', action='store_true', help='multi agent')
parser.add_argument('--rand_kl', action='store_true', help='random kl')
parser.add_argument('--ep',type=int,default=1200)
parser.add_argument('--decay_ep',type=int,default=1200)

args = parser.parse_args()

def run_exp(exp):
    call(exp, shell=True)

def multi(rand_flag=False):
    exp_type = "f110"
    kl_scales = []

    total_configs = 3

    if rand_flag:
        kl_scales = [[0,np.round(np.random.uniform(1,10),1)],[np.round(np.random.uniform(10,50),1),np.round(np.random.uniform(50,100),1)]]
    else:
        kl_scales = [[0,0.5],[5,50]]

    base_exp_str = f" --multi {args.multi} --max_episodes {args.ep} --decay_ep {args.decay_ep} --total_configs {total_configs} --cup_flag True --cuda"

    for _ in range(3):
        

        for idx,kl_scale in enumerate(kl_scales):

            exp_1_1 = f"python3 main.py --own_policy_idx 1 --config 1 --env-name {exp_type}_1_sta_{idx}_{(kl_scale[0])} --kl_scale {kl_scale[0]}"+base_exp_str
            exp_2_1 = f"python3 main.py --own_policy_idx 2 --config 2 --env-name {exp_type}_2_sta_{idx}_{(kl_scale[0])} --kl_scale {kl_scale[0]}"+base_exp_str
            exp_3_1 = f"python3 main.py --own_policy_idx 3 --config 3 --env-name {exp_type}_3_sta_{idx}_{(kl_scale[0])} --kl_scale {kl_scale[0]}"+base_exp_str
            

            exp_1_2 = f"python3 main.py --own_policy_idx 1 --config 1 --env-name {exp_type}_1_sta_{idx}_{(kl_scale[1])} --kl_scale {kl_scale[1]}"+base_exp_str
            exp_2_2 = f"python3 main.py --own_policy_idx 2 --config 2 --env-name {exp_type}_2_sta_{idx}_{(kl_scale[1])} --kl_scale {kl_scale[1]}"+base_exp_str
            exp_3_2 = f"python3 main.py --own_policy_idx 3 --config 3 --env-name {exp_type}_3_sta_{idx}_{(kl_scale[1])} --kl_scale {kl_scale[1]}"+base_exp_str

            processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp_1_1, exp_2_1, exp_3_1, exp_1_2, exp_2_2, exp_3_2]]

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            call(args="rm policy_*.pth",shell=True)

def non_multi(rand_flag=False):
    exp_type = "f110"

    total_configs = 3

    if rand_flag:
        # kl_scales = [[0,np.round(np.random.uniform(0,1),2)],[np.round(np.random.uniform(1,10),2),np.round(np.random.uniform(10,50),2)]]
        kl_scales = [0,np.round(np.random.uniform(0,1),2),np.round(np.random.uniform(1,10),2),np.round(np.random.uniform(10,50),2)]
    else:
        kl_scales = [1,0.5,0.05]

    base_exp_str = f" --max_episodes {args.ep} --decay_ep {args.decay_ep} --total_configs {total_configs} --cuda"

    exp_base_2 = f"python3 main.py --own_policy_idx 2 --config 2 --env-name {exp_type}_2_sta_{0}_{0} --kl_scale {0}"+base_exp_str

  
    processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp_base_2]]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    base_exp_str = f" --max_episodes {args.ep} --decay_ep {args.decay_ep} --total_configs {total_configs} --cup_flag True --cuda"


    for e in range(3):
        for idx,k in enumerate(kl_scales):
            # exp_idx = idx+2
            exp_idx = e*len(kl_scales)+idx+1
            exp_1_1 = f"python3 main.py --own_policy_idx 1 --config 1 --env-name {exp_type}_1_sta_{exp_idx}_{k}  --kl_scale {k}"+base_exp_str
            exp_1_3 = f"python3 main.py --own_policy_idx 3 --config 3 --env-name {exp_type}_3_sta_{exp_idx}_{k} --kl_scale {k}"+base_exp_str

            processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp_1_1, exp_1_3]]

            for p in processes:
                p.start()

            for p in processes:
                p.join()     


if args.multi:
    multi(args.rand_kl)

else:
    non_multi(args.rand_kl)
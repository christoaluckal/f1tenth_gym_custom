import multiprocessing as mp
from subprocess import call
import random

def run_exp(exp):
    call(exp, shell=True)

exp_type = "lunar"
exp_type = "f110"


ep = 500000
total_configs = 3

kl_scales = [0,1,0.5,0.05]

# exp_base_1 = f"python3 main_multi.py --own_policy_idx 1 --config 1 --env-name {exp_type}_1_sta_{0}_{0} --cuda --num_steps {ep} --total_configs {total_configs} --kl_scale {0}"
exp_base_2 = f"python3 main_multi.py --own_policy_idx 2 --config 2 --env-name {exp_type}_2_sta_{0}_{0} --cuda --num_steps {ep} --total_configs {total_configs} --kl_scale {0}"
# exp_base_3 = f"python3 main_multi.py --own_policy_idx 3 --config 3 --env-name {exp_type}_3_sta_{0}_{0} --cuda --num_steps {ep} --total_configs {total_configs} --kl_scale {0}"

processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp_base_2]]

for p in processes:
    p.start()

for p in processes:
    p.join()


for e in range(3):
    for idx,k in enumerate(kl_scales):
        # exp_idx = idx+2
        exp_idx = e*len(kl_scales)+idx+1
        exp_1_1 = f"python3 main.py --own_policy_idx 1 --config 1 --env-name {exp_type}_1_sta_{exp_idx}_{k} --cuda --num_steps {ep} --total_configs {total_configs} --cup_flag True --kl_scale {k}"
        exp_1_3 = f"python3 main.py --own_policy_idx 3 --config 3 --env-name {exp_type}_3_sta_{exp_idx}_{k} --cuda --num_steps {ep} --total_configs {total_configs} --cup_flag True --kl_scale {k}"

        processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp_1_1, exp_1_3]]

        for p in processes:
            p.start()

        for p in processes:
            p.join()     


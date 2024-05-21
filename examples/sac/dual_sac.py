import multiprocessing as mp
from subprocess import call
import random

def run_exp(exp):
    call(exp, shell=True)

exp_type = "lunar"

kl_scales = []

# beta1s = [0,1,10,15,20,30,50]
# beta2s = [0,1,1e-1,5e-2,1e-2,5e-3,1e-3]

beta1s = [random.randint(0, 50) for _ in range(10)]
beta2s = [random.randint(1, 5)*(10**(-random.randint(0, 3))) for _ in range(10)]
beta1s.insert(0, 0)
beta2s.insert(0, 0)
ep = 300000
total_configs = 3

kl_scales = zip(beta1s, beta2s)

# for idx,kl_scale in enumerate(kl_scales):
#     exp_1 = f"python3 main.py --own_policy_idx 1 --config 1 --env-name {exp_type}_1_adp_{idx} --cup_flag True --cuda --num_steps {ep} --beta1 {kl_scale[0]} --beta2 {kl_scale[1]} --adaptive --total_configs {total_configs}"
#     exp_2 = f"python3 main.py --own_policy_idx 2 --config 2 --env-name {exp_type}_2_adp_{idx} --cup_flag True --cuda --num_steps {ep} --beta1 {kl_scale[0]} --beta2 {kl_scale[1]} --adaptive --total_configs {total_configs}"
#     #exp_3 = f"python3 main.py --own_policy_idx 3 --config 3 --env-name {exp_type}_3_adp_{idx} --cup_flag True --cuda --num_steps {ep} --beta1 {kl_scale[0]} --beta2 {kl_scale[1]} --adaptive"

#     if idx==0:
#         kls = 0
#     else:
#         kls = int(random.uniform(1, 3)*(10**(random.randint(0,1))))

#     exp_1_na = f"python3 main.py --own_policy_idx 1 --config 1 --env-name {exp_type}_1_sta_{idx} --cup_flag True --cuda --num_steps {ep} --kl_scale {kls} --total_configs {total_configs}"
#     exp_2_na = f"python3 main.py --own_policy_idx 2 --config 2 --env-name {exp_type}_2_sta_{idx} --cup_flag True --cuda --num_steps {ep} --kl_scale {kls} --total_configs {total_configs}"
#     #exp_3_na = f"python3 main.py --own_policy_idx 3 --config 3 --env-name {exp_type}_3_sta_{idx} --cup_flag True --cuda --num_steps {ep} --kl_scale {kls}"

#     processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp_1, exp_2, exp_1_na, exp_2_na]]

#     for p in processes:
#         p.start()

#     for p in processes:
#         p.join()

#     call(args="rm policy_*.pth",shell=True)

# kl_scales = [0,1,5,10,0,30,50,0,100]

for _ in range(1):
    kl_scales = [0,1,5,10]

    for idx,kl_scale in enumerate(kl_scales):
        exp_1 = f"python3 main.py --own_policy_idx 1 --config 1 --env-name {exp_type}_1_sta_{idx}_{kl_scale} --cup_flag True --cuda --num_steps {ep} --kl_scale {kl_scale} --total_configs {total_configs}"
        exp_2 = f"python3 main.py --own_policy_idx 2 --config 2 --env-name {exp_type}_2_sta_{idx}_{kl_scale} --cup_flag True --cuda --num_steps {ep} --kl_scale {kl_scale} --total_configs {total_configs}"
        exp_3 = f"python3 main.py --own_policy_idx 3 --config 3 --env-name {exp_type}_3_sta_{idx}_{kl_scale} --cup_flag True --cuda --num_steps {ep} --kl_scale {kl_scale} --total_configs {total_configs}"

        processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp_1, exp_2, exp_3]]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        call(args="rm policy_*.pth",shell=True)

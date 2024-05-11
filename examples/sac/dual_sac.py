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

kl_scales = zip(beta1s, beta2s)

for idx,kl_scale in enumerate(kl_scales):
    exp_1 = f"python3 main.py --own_policy_idx 1 --config 1 --env-name {exp_type}_1_{idx} --cup_flag True --cuda --num_steps 500000 --beta1 {kl_scale[0]} --beta2 {kl_scale[1]}"
    exp_2 = f"python3 main.py --own_policy_idx 2 --config 2 --env-name {exp_type}_2_{idx} --cup_flag True --cuda --num_steps 500000 --beta1 {kl_scale[0]} --beta2 {kl_scale[1]}"
    exp_3 = f"python3 main.py --own_policy_idx 3 --config 3 --env-name {exp_type}_3_{idx} --cup_flag True --cuda --num_steps 500000 --beta1 {kl_scale[0]} --beta2 {kl_scale[1]}"
    processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp_1, exp_2, exp_3]]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    call(args="rm policy_*.pth",shell=True)

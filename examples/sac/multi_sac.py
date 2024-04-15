import multiprocessing as mp
from subprocess import call

def run_exp(exp):
    call(exp, shell=True)

steps = 300000
kl_scales = [0,1,2,5,10,30]

for kl_scale in kl_scales:
    exp_str = f" --num_steps {steps} --kl_scale {kl_scale}"

    exp_1 = f"python3 main.py --own_policy_idx 1 --config 1 --env-name config_1_{kl_scale} --cup_flag True"
    exp_2 = f"python3 main.py --own_policy_idx 2 --config 2 --env-name config_2_{kl_scale} --cup_flag True"

    processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp_1, exp_2]]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

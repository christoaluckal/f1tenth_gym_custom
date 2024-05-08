import multiprocessing as mp
from subprocess import call

def run_exp(exp):
    call(exp, shell=True)

kl_scales = [1, 3, 5, 10]
for kl_scale in kl_scales:
    exp_1 = f"python3 main.py --own_policy_idx 1 --config 1 --env-name config_1_{kl_scale*10} --cup_flag True --cuda --num_steps 500000 --kl_scale {kl_scale}"
    exp_2 = f"python3 main.py --own_policy_idx 2 --config 2 --env-name config_2_{kl_scale*10} --cup_flag True --cuda --num_steps 500000 --kl_scale {kl_scale}"
    exp_3 = f"python3 main.py --own_policy_idx 3 --config 3 --env-name config_3_{kl_scale*10} --cup_flag True --cuda --num_steps 500000 --kl_scale {kl_scale}"
    processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp_1, exp_2, exp_3]]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    call(args="rm policy_*.pth",shell=True)

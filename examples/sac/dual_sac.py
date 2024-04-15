import multiprocessing as mp
from subprocess import call

def run_exp(exp):
    call(exp, shell=True)

exp_1_base = "python3 main.py --own_policy_idx 1 --config 1 --env-name config_1_baseline --cup_flag False"
exp_2_base = "python3 main.py --own_policy_idx 2 --config 2 --env-name config_2_baseline --cup_flag False"
exp_1 = "python3 main.py --own_policy_idx 1 --config 1 --env-name config_1 --cup_flag True"
exp_2 = "python3 main.py --own_policy_idx 2 --config 2 --env-name config_2 --cup_flag True"

processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp_1, exp_2]]

for p in processes:
    p.start()

for p in processes:
    p.join()

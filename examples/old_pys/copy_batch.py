import multiprocessing as mp
from subprocess import call

def run_exp(exp):
    call(exp, shell=True)

base_cmd = f"python3 test_policy_copy.py --total_timesteps 1000000"

exp1 = base_cmd + " --config 1 --car_idx 1 --to_save 1 --save_name logs/policy_1"

processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp1]]
try:
    for p in processes:
        p.start()

    for p in processes:
        p.join()
    
except KeyboardInterrupt:
    for p in processes:
        p.terminate()

exp2 = base_cmd + " --config 2 --car_idx 2 --to_save 0 --new_policy logs/policy_1"
exp3 = base_cmd + " --config 2 --car_idx 3 --to_save 1"

processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp2,exp3]]
try:
    for p in processes:
        p.start()

    for p in processes:
        p.join()
    
except KeyboardInterrupt:
    for p in processes:
        p.terminate()


exp2 = base_cmd + " --config 3 --car_idx 4 --to_save 0 --new_policy logs/policy_1"
exp3 = base_cmd + " --config 3 --car_idx 5 --to_save 1"

processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp2,exp3]]
try:
    for p in processes:
        p.start()

    for p in processes:
        p.join()
    
except KeyboardInterrupt:
    for p in processes:
        p.terminate()






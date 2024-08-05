import multiprocessing as mp
from subprocess import call
import os

def run_exp(exp):
    call(exp, shell=True)

if not os.path.exists("logs"):
    os.makedirs("logs",exist_ok=True)

te = int(4e5)
exp_counter = 0

exp_str = " --total_timesteps " + str(te)

exp1 = "python3 test_cascade.py --config 1 --car_idx 1 --exp 1 --verbose 1 --is_baseline 1 --own_policy_name 'policy_1'" + exp_str

exp2 = "python3 test_cascade.py --config 2 --car_idx 2 --exp 2 --verbose 1 --from_easier 1 --easier_name 'policy_1'" + exp_str
exp2_base = "python3 test_cascade.py --config 2 --car_idx 2 --exp 3 --verbose 1 --is_baseline 1 --own_policy_name 'policy_2'" + exp_str

exp3 = "python3 test_cascade.py --config 3 --car_idx 3 --exp 4 --verbose 1 --from_easier 1 --easier_name 'policy_2'" + exp_str
exp3_base = "python3 test_cascade.py --config 3 --car_idx 3 --exp 5 --verbose 1 --is_baseline 1 --own_policy_name 'policy_3'" + exp_str

# run exp1
run_exp(exp1)

exps = [exp2,exp2_base]

processes = [mp.Process(target=run_exp, args=(exp,)) for exp in exps]

try:
    for p in processes:
        p.start()
    
    for p in processes:
        p.join()

except Exception as e:
    print(e)
    for p in processes:
        p.join()


exps = [exp3,exp3_base]

processes = [mp.Process(target=run_exp, args=(exp,)) for exp in exps]

try:
    for p in processes:
        p.start()
    
    for p in processes:
        p.join()

except Exception as e:
    print(e)
    for p in processes:
        p.join()

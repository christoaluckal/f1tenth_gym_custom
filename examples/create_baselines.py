import multiprocessing as mp
from subprocess import call

def run_exp(exp):
    call(exp, shell=True)

sf = 10000
me = 20000
te = int(10000)

exp_str = " --total_timesteps " + str(te)


exp1 = "python3 test_sb.py --config 1 --car_idx 1 --exp 1 --verbose 1" + exp_str
exp3 = "python3 test_sb.py --config 2 --car_idx 3 --exp 3 --verbose 1" + exp_str
exp5 = "python3 test_sb.py --config 3 --car_idx 5 --exp 5 --verbose 1" + exp_str

processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp1,exp3,exp5]]
try:
    for p in processes:
        p.start()

    for p in processes:
        p.join()
    
except KeyboardInterrupt:
    for p in processes:
        p.terminate()


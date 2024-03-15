import multiprocessing as mp
from subprocess import call

def run_exp(exp):
    call(exp, shell=True)


e = 1
sf = 5000
me = 10000
te = int(1e6)

exp_str = " --save_freq " + str(sf) + " --modify_epoch " + str(me) + " --total_timesteps " + str(te)

exp1 = "python3 test_sb.py --config 1 --car_idx 1 --base 0 --retain 0.75" + exp_str  +" --exp 1 --verbose 1" 
exp2 = "python3 test_sb.py --config 1 --car_idx 2 --base 1 --retain 1" + exp_str +" --exp 2 --verbose 0"
exp3 = "python3 test_sb.py --config 3 --car_idx 3 --base 0 --retain 0.75" + exp_str +" --exp 3 --verbose 1"
exp4 = "python3 test_sb.py --config 3 --car_idx 4 --base 1 --retain 1" + exp_str +" --exp 4 --verbose 0"

processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp1, exp2, exp3, exp4]]

for p in processes:
    p.start()

for p in processes:
    p.join()



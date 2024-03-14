import multiprocessing as mp
from subprocess import call

def run_exp(exp):
    call(exp, shell=True)


e = 1
sf = 1000
me = 5000
te = int(1e6)

exp_str = " --exp "+ str(e) +" --save_freq " + str(sf) + " --modify_epoch " + str(me) + " --total_timesteps " + str(te)

exp1 = "python3 test_sb.py --config 1 --car_idx 1 --base 0 --retain 0.75" + exp_str
exp2 = "python3 test_sb.py --config 3 --car_idx 2 --base 0 --retain 0.75" + exp_str

processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp1, exp2]]

for p in processes:
    p.start()

for p in processes:
    p.join()



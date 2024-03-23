import multiprocessing as mp
from subprocess import call

def run_exp(exp):
    call(exp, shell=True)


sf = 100
me = 200
te = int(5e6)

exp_str = " --ws True --ws_count 3 --save_freq " + str(sf) + " --modify_epoch " + str(me) + " --total_timesteps " + str(te)

exp1 = "python3 test_sb.py --config 1 --car_idx 1 --base 0 --retain 1" + exp_str  +" --exp 1 --verbose 1" 
exp3 = "python3 test_sb.py --config 2 --car_idx 3 --base 0 --retain 0.8" + exp_str +" --exp 3 --verbose 1"
exp5 = "python3 test_sb.py --config 3 --car_idx 5 --base 0 --retain 0.6" + exp_str +" --exp 5 --verbose 1"

processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp1,exp3,exp5]]
try:
    for p in processes:
        p.start()

    for p in processes:
        p.join()
    
except KeyboardInterrupt:
    for p in processes:
        p.terminate()




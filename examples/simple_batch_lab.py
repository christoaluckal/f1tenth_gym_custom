import multiprocessing as mp
from subprocess import call

def run_exp(exp):
    call(exp, shell=True)

ratios = [[1.0,1.0,1.0],[1.0,0.9,0.5],[1.0,0.8,0.3],[1.0,0.7,0.1]]

sf = 10000
me = 20000
te = int(5e5)

exp_str = "--is_lab True --ws True --ws_count 3 --save_freq " + str(sf) + " --modify_epoch " + str(me) + " --total_timesteps " + str(te)

for r in ratios:
    exp1 = "python3 test_sb.py --config 1 --car_idx 1 --base 0 --retain " + r[0] + exp_str  +" --exp 1 --verbose 1" 
    exp3 = "python3 test_sb.py --config 2 --car_idx 3 --base 0 --retain 0.8" + r[1] + exp_str +" --exp 3 --verbose 1"
    exp5 = "python3 test_sb.py --config 3 --car_idx 5 --base 0 --retain 0.6" + r[2] + exp_str +" --exp 5 --verbose 1"

    processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp1,exp3,exp5]]
    try:
        for p in processes:
            p.start()

        for p in processes:
            p.join()
        
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()




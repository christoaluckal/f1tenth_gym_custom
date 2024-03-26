import multiprocessing as mp
from subprocess import call
import time

def run_exp(exp):
    call(exp, shell=True)

ratios = [
    [1.0,1.0,1.0,'both'],
    [0.9,0.9,0.9,'both'],
    [0.9,0.9,0.9,'actor'],
    [0.9,0.9,0.9,'critic'],
    [0.8,0.8,0.8,'both'],
    [0.8,0.8,0.8,'actor'],
    [0.8,0.8,0.8,'critic'],
    [0.7,0.7,0.7,'both'],
    [0.7,0.7,0.7,'actor'],
    [0.7,0.7,0.7,'critic'],
    [-1.0,-1.0,-1.0,'both'],
    [-1.0,-1.0,-1.0,'actor'],
    [-1.0,-1.0,-1.0,'critic']
]


sf = 10000
me = 20000
te = int(2.5e5)
exp_counter = 0

exp_str = " --from_pretrained True --ws True --ws_count 3 --save_freq " + str(sf) + " --modify_epoch " + str(me) + " --total_timesteps " + str(te)

for ra in ratios:
    for f in [1]:
        exp_counter += 1
        r = [f*x for x in ra[:-1]]
        r.append(ra[-1])
        exp1 = "python3 test_sb.py --config 1 --car_idx 1 --base 0 --retain " + str(r[0]) + exp_str  +" --exp " + str(exp_counter) +" --verbose 1 --mod_type "+str(r[3]) 
        exp3 = "python3 test_sb.py --config 2 --car_idx 3 --base 0 --retain " + str(r[1]) + exp_str +" --exp " + str(exp_counter) +" --verbose 1 --mod_type "+str(r[3])
        exp5 = "python3 test_sb.py --config 3 --car_idx 5 --base 0 --retain " + str(r[2]) + exp_str +" --exp " + str(exp_counter) +" --verbose 1 --mod_type "+str(r[3])
        processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp1,exp3,exp5]]
        try:
            for p in processes:
                p.start()

            for p in processes:
                p.join()
            
        except KeyboardInterrupt:
            for p in processes:
                p.terminate()
                
            exit(0)
                
        time.sleep(10)


import subprocess
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

exp = "python3 f1gym_rand_track.py --save True"

subprocess.call("rm -rf maps/ centerline/ generated.csv",shell=True)

trs = [0.05,0.06,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
# scales = [1.5,2,3,4,5,6]
scales = [1.5,2]

# for t in tqdm(trs):
#     tr_seed = np.random.randint(0,1000)
#     for s in scales:
#         t_exp = exp+f" --turn_rate {t} --scale {s} --seed {tr_seed}"
#         print(t_exp)
#         subprocess.call(t_exp,shell=True)
        
turn_rate_seeds = {}

for t in trs:
    tr_seed = np.random.randint(0,1000)
    turn_rate_seeds[t] = tr_seed

def run_exp(tr,s):
    tr_seed = turn_rate_seeds[tr]
    t_exp = exp+f" --turn_rate {tr} --scale {s} --seed {tr_seed}"
    print(t_exp)
    subprocess.call(t_exp,shell=True)
    
pool = mp.Pool(processes=4)
for t in trs:
    for s in scales:
        pool.apply_async(run_exp,args=(t,s))
pool.close()
pool.join()

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pprint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type',type=str,default='train')
parser.add_argument('--smooth',type=float,default=0.99)

def smooth(scalars, weight=0.99):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

runs_folder = 'runs'
trains = None
tests = None

trains = [os.path.join('runs',x) for x in os.listdir(runs_folder) if "train" in x]
tests = [os.path.join('runs',x) for x in os.listdir(runs_folder) if "test" in x]

exp_dict = {"train":None,"test":None}

kl_scales = []
configs = []

for t in trains:
    f = t
    t = t[5:]
    t_split = t.split('_')
    kl_scale = t_split[5]
    config = t_split[2]
    if kl_scale not in kl_scales:
        kl_scales.append(kl_scale)
    if config not in configs:
        configs.append(config)

    

for t in trains:
    f = t
    t = t[5:]
    t_split = t.split('_')
    log_type = t_split[0]
    exp_type = t_split[1]
    config = t_split[2]
    exp_idx = t_split[4]
    kl_scale = t_split[5]
    data = pd.read_csv(f,skiprows=1).to_numpy()
    # data = None

    if exp_dict[log_type] is None:
        exp_dict[log_type] = {}
        for k in kl_scales:
            exp_dict[log_type][k] = {}
            for c in configs:
                exp_dict[log_type][k][c] = []

    exp_dict[log_type][kl_scale][config].append(data)

for t in tests:
    f = t
    t = t[5:]
    t_split = t.split('_')
    log_type = t_split[0]
    exp_type = t_split[1]
    config = t_split[2]
    exp_idx = t_split[4]
    kl_scale = t_split[5]
    data = pd.read_csv(f,skiprows=1).to_numpy()
    # data = None

    if exp_dict[log_type] is None:
        exp_dict[log_type] = {}
        for k in kl_scales:
            exp_dict[log_type][k] = {}
            for c in configs:
                exp_dict[log_type][k][c] = []

    exp_dict[log_type][kl_scale][config].append(data)

# import collections

# for k,v in exp_dict.items():
#     for k1,v1 in v.items():
#         v[k1] = collections.OrderedDict(sorted(v1.items()))
    
#     exp_dict[k] = collections.OrderedDict(sorted(v.items()))



plot_type = parser.parse_args().type

if plot_type == "train":
    skips = 50
else:
    skips = 10

for k in kl_scales:
    for c in configs:
        data_row = []
        try:
            if len(exp_dict[plot_type][k][c]) > 0:
                idx_row = exp_dict[plot_type][k][c][0][:,0].tolist()
                for d in exp_dict[plot_type][k][c]:
                    data_row.append(d[:,1].tolist())
                print(f"kl_scale:{k},config:{c} has {len(data_row)} runs")
                data_row = np.array(data_row)
                data_row = np.mean(data_row,axis=0)
                data_row = smooth(data_row,weight=parser.parse_args().smooth)
                data_row = data_row[::skips]
                idx_row = idx_row[::skips]
                
                if k == "0":
                    plt.plot(idx_row,data_row,label=f"kl_scale:{k},config:{c}",marker='D')
                else:
                    plt.plot(idx_row,data_row,label=f"kl_scale:{k},config:{c}",marker='o')

        except:
            pass

# plt.ylim(-600,400)
plt.grid()
plt.title(f"{plot_type} Reward Graph")
plt.legend()
plt.show()
plt.close()




import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

runs_folder = 'runs'
trains = None
tests = None

trains = [os.path.join('runs',x) for x in os.listdir(runs_folder) if "train" in x]
tests = [os.path.join('runs',x) for x in os.listdir(runs_folder) if "test" in x]

exp_dict = {}

for t in trains:
    df = pd.read_csv(t)
    exp = t.split('/')[1].split('_')
    exp_type = exp[1]
    exp_config = exp[2]
    kl_scale = exp[5]
    exp_full_name = f"{exp_type}_{exp_config}_{kl_scale}"
    mask = df.loc[df['episode'] == -1].index.values.tolist()
    exp_dict[exp_full_name] = []
    mask.insert(0,0)
    for i in range(len(mask)):
        if i==0:
            exp_dict[exp_full_name].append(df.iloc[mask[i]:mask[i+1]-1])
        elif i==len(mask)-1:
            exp_dict[exp_full_name].append(df.iloc[mask[i]+1:])
        else:
            exp_dict[exp_full_name].append(df.iloc[mask[i]:mask[i+1]])


for k,v in exp_dict.items():
    print(k)

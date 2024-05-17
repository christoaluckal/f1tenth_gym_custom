import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm
import pickle
import os
import numpy as np
import sys

pkl_file = sys.argv[1]
config_s = sys.argv[2]

tag1 = 'avg_reward/test'
tag2 = 'reward/train'


color_scheme = {
    '0': 'r',
    '1': 'g',
    '5': 'b',
    '10': 'y',
    '30': 'm',
    '50': 'c',
    '100': 'k'
}


def plot_tensorflow_log(log_path, tag):
    event_acc = EventAccumulator(log_path)
    event_acc.Reload()
    scalar_event_acc = event_acc.Scalars(tag)
    steps = [scalar.step for scalar in scalar_event_acc]
    values = [scalar.value for scalar in scalar_event_acc]
    return steps, values

def smooth(scalars, weight=0.9):
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def generate_pkl(folder_loc,pkl_name):
    base_dir = folder_loc
    dirs = os.listdir(base_dir)
    dirs = [os.path.join(base_dir, d) for d in dirs if os.path.isdir(os.path.join(base_dir, d))]
    dirs_files = [os.path.join(d, f) for d in dirs for f in os.listdir(d)]

    res_dict = {}

    for df in tqdm(dirs_files):
        name = df.split('/')[1]
        splits = name.split('_')
        config = splits[3]
        scale = splits[6]
        key = f'{config}_{scale}'
        if scale == '0':
            if key not in res_dict:
                res_dict[key] = []
            
            steps_test, values_test = plot_tensorflow_log(df, tag1)
            steps_train, values_train = plot_tensorflow_log(df, tag2)

            res_dict[key].append({
                'test': {
                    'steps': steps_test,
                    'values': values_test
                },
                'train': {
                    'steps': steps_train,
                    'values': values_train
                }
            })
            

        else:
            if key not in res_dict:
                res_dict[key] = {
                    'train': {},
                    'test': {}
                }
            

            steps, values = plot_tensorflow_log(df, tag1)
            res_dict[key]['test']['steps'] = steps
            res_dict[key]['test']['values'] = values

            steps, values = plot_tensorflow_log(df, tag2)
            res_dict[key]['train']['steps'] = steps
            res_dict[key]['train']['values'] = values

    with open(pkl_name, 'wb') as f:
        pickle.dump(res_dict, f)

    return



def plot(pkl_file,config_s='1'):
    with open(pkl_file, 'rb') as f:
        res_dict = pickle.load(f)

    config = config_s

    for key in res_dict:
        if key[0] == config:
            if int(key[2:]) != 0:
                # plt.plot(res_dict[key]['test']['steps'], res_dict[key]['test']['values'], label=key,alpha=0.5)
                # plt.scatter(res_dict[key]['test']['steps'], res_dict[key]['test']['values'], alpha=0.5)
                steps = res_dict[key]['test']['steps']
                values = res_dict[key]['test']['values']
                smoothed = smooth(values)

                # plt.plot(steps, smoothed, label=key, alpha=0.5)
                # plt.scatter(steps, smoothed, alpha=0.5)

                color = color_scheme[key[2:]]
                plt.plot(steps, smoothed, label=key, color=color,alpha=0.8)
                # plt.scatter(steps, smoothed, color=color)



            else:
                counts = []
                values = []
                res_dict[key] = res_dict[key]
                for i in range(len(res_dict[key])):
                    counts.append(len(res_dict[key][i]['test']['steps']))

                min_count = min(counts)
                for i in range(len(res_dict[key])):
                    values.append(res_dict[key][i]['test']['values'][:min_count])
                
                values = np.array(values)
                avg = np.mean(values, axis=0)
                
                steps = res_dict[key][0]['test']['steps'][:min_count]

                smoothed = smooth(avg)

                plt.plot(steps, smoothed, label=key,color='r',linewidth=2)
                plt.scatter(steps, smoothed, color='r',s=10)
                    

    plt.legend()
    plt.show()

if __name__ == "__main__":
    pass
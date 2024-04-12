import time
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import numpy as np
from argparse import Namespace
from gym.envs.registration import register
import argparse

from numba import njit

from pyglet.gl import GL_POINTS

from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback,CallbackList,EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from stable_baselines3 import PPO,SAC
from torch.utils.tensorboard import SummaryWriter

import yaml

with open("device.yaml") as f:
    device = yaml.load(f, Loader=yaml.FullLoader)
    
is_lab = device['is_lab']



def render_callback(env_renderer):
    # custom extra drawing function

    e = env_renderer

    # update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 700
    e.left = left - 800
    e.right = right + 800
    e.top = top + 800
    e.bottom = bottom - 800



def main():
    import os
    import pickle
    with open("maps.pkl","rb") as f:
        maps = pickle.load(f)
        
    configs = maps
    
    if not os.path.exists("logs"):
        os.makedirs("logs",exist_ok=True)
    
    if is_lab:
        for i in configs:
            i['map'] = i['map'].replace('/home/christo/Developer/thesis/f1tenth_gym_custom/examples','/home/christoa/Developer/spring2024/thesis/f1tenth_gym_custom/examples')
            i['waypoints'] = i['waypoints'].replace('/home/christo/Developer/thesis/f1tenth_gym_custom/examples','/home/christoa/Developer/spring2024/thesis/f1tenth_gym_custom/examples')
    

    testing_config = configs[1:]
    current_config = testing_config[args.config-1]
    
    register('f110_gym:f110-cust-v0', entry_point='f110_gym.envs:F110_Cust_Env', max_episode_steps=10000)

    
    env = gym.make('f110_gym:f110-cust-v0',config=current_config, num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False)
    eval_env = gym.make('f110_gym:f110-cust-v0',config=current_config, num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False)

    eval_cb = EvalCallback(
        eval_env=eval_env,
        eval_freq=10000,
        render=False,
        deterministic=True,
    )

    if args.is_baseline:
        experiment_name = f"config_{args.config}_{args.car_idx}_baseline_{args.exp}"
        model = SAC("MlpPolicy", env, verbose=args.verbose, tensorboard_log=f"logs/{experiment_name}")
        model.learn(total_timesteps=args.total_timesteps, callback=eval_cb)
        policy = model.policy
        torch.save(policy.state_dict(),f"logs/{args.own_policy_name}.pth")
    else:
        experiment_name = f"config_{args.config}_{args.car_idx}_cascade_{args.exp}"
        if args.from_easier:
            policy = torch.load(f"logs/{args.easier_name}.pth")
            model = SAC("MlpPolicy", env, verbose=args.verbose, tensorboard_log=f"logs/{experiment_name}")
            model.policy.load_state_dict(policy)
            model.learn(total_timesteps=args.total_timesteps, callback=eval_cb)




    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the f1tenth gym environment')
    parser.add_argument('--config', type=int, default=1, help='Config to run')
    parser.add_argument('--car_idx', type=int, default=1, help='Car index to use')
    parser.add_argument('--exp',type=int,default=1,help='Experiment number')
    parser.add_argument('--total_timesteps',type=int,default=1e6,help='Total timesteps for training')
    parser.add_argument('--verbose',type=int,default=0,help='Verbosity level')
    parser.add_argument('--is_baseline',type=int,default=0,help='Whether to run baseline or not')
    parser.add_argument('--from_easier',type=int,default=0,help='Whether to run from easier task or not')
    parser.add_argument('--own_policy_name',type=str,default="",help='Name of own policy')
    parser.add_argument('--easier_name',type=str,default="",help='Name of easier task')

    args = parser.parse_args()

    main()

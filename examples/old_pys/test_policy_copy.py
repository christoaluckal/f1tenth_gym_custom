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

from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement,EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from stable_baselines3 import PPO,SAC
from torch.utils.tensorboard import SummaryWriter


def main():
    """
    main entry point
    """

    import os
    map_location = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','gym','f110_gym','unittest')

    # env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    # env.add_render_callback(render_callback)
    map_config_1 = {
        'map_ext': '.png',
        'map': os.path.join(map_location,'maps/map_44_1000_'),
        'waypoints': os.path.join(map_location,'centerline/map_44_1000_.csv'),
        'reset_pose': [0.0,0.0,np.pi/5]
    }

    map_config_2 = {
        'map_ext': '.png',
        'map': os.path.join(map_location,'maps/map_7_100_'),
        'waypoints': os.path.join(map_location,'centerline/map_7_100_.csv'),
        'reset_pose': [0.0,0.0,0.0]
    }

    map_config_3 = {
        'map_ext': '.png',
        'map': os.path.join(map_location,'maps/map_15_100_'),
        'waypoints': os.path.join(map_location,'centerline/map_15_100_.csv'),
        'reset_pose': [0.0,0.0,0.0]
    }

    eval_config = {
        'map_ext': '.png',
        'map': os.path.join(map_location,'maps/map_44_1000_'),
        'waypoints': os.path.join(map_location,'centerline/map_44_1000_.csv'),
        'reset_pose': [0.0,0.0,np.pi/5]
    }

    register('f110_gym:f110-cust-v0', entry_point='f110_gym.envs:F110_Cust_Env', max_episode_steps=10000)

    if args.config == 1:
        env = gym.make('f110_gym:f110-cust-v0',config=map_config_1, num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False,render_mode='human')
    elif args.config == 2:
        env = gym.make('f110_gym:f110-cust-v0',config=map_config_2, num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False)
    elif args.config == 3:
        env = gym.make('f110_gym:f110-cust-v0',config=map_config_3, num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False)
        

    
    experiment_name = f"{args.config}_{args.car_idx}"

    model = SAC('MlpPolicy', env, verbose=args.verbose, tensorboard_log=f"logs/{experiment_name}")

    if args.to_save == 0 and args.new_policy != "":
        model.load(args.new_policy)

    stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=20, verbose=1)
    eval_callback = EvalCallback(env, eval_freq=20000, callback_after_eval=stop_cb, verbose=1)
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    if args.to_save == 1 and args.save_name != "":
        model.save(args.save_name)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the f1tenth gym environment')
    parser.add_argument('--config', type=int, default=1, help='Config to run')
    parser.add_argument('--car_idx', type=int, default=1, help='Car index to use')
    parser.add_argument('--total_timesteps', type=int, default=10000, help='Name of Policy')

    parser.add_argument('--to_save',default=0,type=int)
    parser.add_argument('--save_name',default="",type=str)
    parser.add_argument('--new_policy',default="",type=str)
    parser.add_argument('--verbose',type=int, default=0)

    args = parser.parse_args()

    print(args)

    main()

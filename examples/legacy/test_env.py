import os
import pickle
from gym.envs.registration import register
import gym
from f110_gym.envs.base_classes import Integrator

def registerf1():
    with open("maps.pkl","rb") as f:
        maps = pickle.load(f)
        
    configs = maps

    dir_path = os.path.dirname(os.path.realpath(__file__)).split('/')[:-1]
    dir_path = '/'.join(dir_path)

    for i in configs:
        base_map = i['map']
        base_wpt = i['waypoints']
        i['map']=dir_path+base_map
        i['waypoints']=dir_path+base_wpt
        print(dir_path,base_map)


    testing_config = configs[1:]
    current_config = testing_config[0]

    register('f110_gym:f110-cust-v0', entry_point='f110_gym.envs:F110_Cust_Env', max_episode_steps=10000)


    env = gym.make('f110_gym:f110-cust-v0',config=current_config, num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False)
    eval_env = gym.make('f110_gym:f110-cust-v0',config=configs[0], num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False)

    return env,eval_env

# print(eval_env.get_dummies())

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

_,eval_env = registerf1()

eval_env.add_render_callback(render_callback)

import numpy as np

done = False
obs = eval_env.reset(np.array([[0,0,np.pi/2]]))
ep_reward = 0
while not done:
    action = eval_env.action_space.sample()
    obs, reward, done, _, info = eval_env.step(action)
    ep_reward += reward
    eval_env.render()

print(ep_reward)


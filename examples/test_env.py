import os
import pickle
from gym.envs.registration import register
import gym
from f110_gym.envs.base_classes import Integrator
with open("maps.pkl","rb") as f:
    maps = pickle.load(f)
    
configs = maps

if not os.path.exists("logs"):
    os.makedirs("logs",exist_ok=True)


for i in configs:
    i['map'] = i['map'].replace('/home/christo/Developer/thesis/f1tenth_gym_custom/examples','/home/caluckal/Developer/spring2024/thesis/f1tenth_gym_custom/examples')
    i['waypoints'] = i['waypoints'].replace('/home/christo/Developer/thesis/f1tenth_gym_custom/examples','/home/caluckal/Developer/spring2024/thesis/f1tenth_gym_custom/examples')


testing_config = configs[1:]
current_config = testing_config[0]

register('f110_gym:f110-cust-v0', entry_point='f110_gym.envs:F110_Cust_Env', max_episode_steps=10000)


env = gym.make('f110_gym:f110-cust-v0',config=current_config, num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False)
eval_env = gym.make('f110_gym:f110-cust-v0',config=current_config, num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False)

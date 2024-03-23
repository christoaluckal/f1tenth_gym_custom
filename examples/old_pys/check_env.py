from stable_baselines3.common.env_checker import check_env
import gym
from gym.envs.registration import register
from f110_gym.envs.base_classes import Integrator

map_config = {
        'map_ext': '.png',
        'map': '/home/caluckal/Developer/spring2024/thesis/f1tenth_gym/gym/f110_gym/unittest/maps/map_15_100_',
        'waypoints': '/home/caluckal/Developer/spring2024/thesis/f1tenth_gym/gym/f110_gym/unittest/centerline/map_15_100_.csv'
    }

register('f110_gym:f110-cust-v0', entry_point='f110_gym.envs:F110_Cust_Env', max_episode_steps=1000)
env = gym.make('f110_gym:f110-cust-v0',config=map_config, num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False)

check_env(env)
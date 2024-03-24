
from f110_gym.envs.base_classes import Integrator
import gym
from gym.envs.registration import register
import argparse
from pyglet.gl import GL_POINTS





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
    """
    main entry point
    """

    import os
    import matplotlib.pyplot as plt
    map_location = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','gym','f110_gym','unittest')

    import pickle
    with open("maps.pkl","rb") as f:
        maps = pickle.load(f)
        
    configs = maps
    eval_config = configs[0]
    current_config = configs[1:][args.config-1]
    
    register('f110_gym:f110-cust-v0', entry_point='f110_gym.envs:F110_Cust_Env', max_episode_steps=10000)
    
    if args.eval:
        eval_env = gym.make('f110_gym:f110-cust-v0',config=eval_config, num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False)
        eval_env.reset()
        while True:
            try:
                eval_env.render(mode='human')
            except KeyboardInterrupt:
                break
        return
    
    if args.test:
        env = gym.make('f110_gym:f110-cust-v0',config=current_config, num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False)
        obs = env.reset()
        while True:
            try:
                action = env.action_space.sample()
                obs, reward, done,trunc, info = env.step(action)
                env.render(mode='human')
                if done:
                    obs = env.reset()    
            except KeyboardInterrupt:
                break
        return

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the f1tenth gym environment')
    parser.add_argument('--config', type=int, default=1, help='Config to run')
    parser.add_argument('--eval', type=bool, default=False, help='Run evaluation mode')
    parser.add_argument('--test', type=bool, default=True, help='Run test mode')

    args = parser.parse_args()
    main()

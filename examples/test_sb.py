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

from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback,CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from stable_baselines3 import PPO,SAC
from torch.utils.tensorboard import SummaryWriter



class WeightedUpdate(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 0, 
                 config_type=1, 
                 modify_epoch=1, 
                 easy_policy_loc="",
                 base_policy_loc="", 
                 save_freq=1000, 
                 retain_ratio=0.8,
                 is_baseline=False,
                 policy_names=[],
                 eval_config=None,
                 own_policy_name="policy_1"
                 ):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.config_type = config_type
        self.modify_epoch = modify_epoch
        self.easy_policy_loc = easy_policy_loc
        self.base_policy_loc = base_policy_loc
        self.save_freq = save_freq
        self.retain_ratio = retain_ratio
        self.is_baseline = is_baseline
        self.modified = False
        self.has_saved = False
        self.policy_names = policy_names
        self.eval_config = eval_config
        self.own_policy_name = own_policy_name
        self.writer = SummaryWriter(log_dir=f"logs")
        self.temp_eval_env = gym.make('f110_gym:f110-cust-v0',config=self.eval_config, num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False, is_eval=True)
        self.temp_eval_env = Monitor(self.temp_eval_env)
        self.temp_model = SAC('MlpPolicy', self.temp_eval_env,verbose=0)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if not self.is_baseline:
            if self.n_calls % self.save_freq == 0:
                params = self.model.policy.state_dict()
                torch.save(params, f"logs/{self.own_policy_name}.pth")

            if self.n_calls % self.modify_epoch == 0:
                print(f"Evaluating for {self.own_policy_name}...")

                eval_results = []
                policies = [torch.load(f"logs/{self.own_policy_name}.pth")]
                policy_names = [self.own_policy_name]
                for policy_name in self.policy_names:
                    if policy_name != self.own_policy_name:
                        policy_names.append(policy_name)
                        policies.append(torch.load(f"logs/{policy_name}.pth"))

                for pol in policies:
                    self.temp_eval_env.reset()
                    self.temp_model.policy.load_state_dict(pol)
                    mean_reward, std_reward = evaluate_policy(self.temp_model, self.temp_eval_env, n_eval_episodes=3)
                    eval_results.append(mean_reward)

                probs = []

                if len(eval_results) > 1:
                    eval_others = eval_results[1:]
                    eval_others_exp = np.exp(eval_others)
                    eval_others_exp_sum = np.sum(eval_others_exp)
                    probs = eval_others_exp/eval_others_exp_sum
                    probs*=(1-self.retain_ratio)

                fin_probs = [self.retain_ratio] + list(probs)
                print(f"Policies: {policy_names}")
                print(f"Final probs: {fin_probs}")

                probabilities = {
                    
                }
                for i in range(len(policy_names)):
                    probabilities[policy_names[i]] = fin_probs[i]

                self.writer.add_scalars(f'probabilities_{self.own_policy_name}', probabilities, self.n_calls)

                keys = list(policies[0].keys())
                new_policy = {}
                for key in keys:
                    new_policy[key] = torch.zeros_like(policies[0][key])
                    for i in range(len(policies)):
                        new_policy[key] += policies[i][key]*fin_probs[i]
                    
                self.model.policy.load_state_dict(new_policy)

                self.modify_epoch *= 1.0
                self.modify_epoch = int(self.modify_epoch)
                

        
        
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

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

def weighedCombination(
    policies=["policy_1"],
    eval_config=None,
    exp=1,
    save_freq=1000,
    modify_epoch=1000,
    retain_ratio=0.8,
    is_baseline=False,
    own_policy_name="policy_1"
    ):
    # custom_cb = WeightedUpdate(
    #     config_type=args.config, 
    #     easy_policy_loc=f"logs/policy_{args.exp}.pth",
    #     base_policy_loc=f"logs/base_{args.exp}.pth", 
    #     save_freq=args.save_freq,
    #     modify_epoch=args.modify_epoch,
    #     retain_ratio=args.retain,
    #     is_baseline=True if args.base == 1 else False,
    #     policy_names = ["policy_1","policy_2","policy_3"],
    #     eval_config=eval_config,
    #     own_policy_name=f"policy_{args.config}"
    #     )
    weighted_cb = WeightedUpdate(
        config_type=exp, 
        easy_policy_loc=f"logs/policy_{exp}.pth",
        base_policy_loc=f"logs/base_{exp}.pth", 
        save_freq=save_freq,
        modify_epoch=modify_epoch,
        retain_ratio=retain_ratio,
        is_baseline=is_baseline,
        policy_names = policies,
        eval_config=eval_config,
        own_policy_name=own_policy_name
        )
    
    return weighted_cb

def main():
    """
    main entry point
    """

    import os
    import matplotlib.pyplot as plt
    map_location = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','gym','f110_gym','unittest')

    from f110_gym.unittest.collate import getConfigList
    
    config_dict = getConfigList(csv_f=os.path.join(map_location,'generated.csv'))
    eval_tr = config_dict['tr'][0]
    trs = config_dict['tr'][1:]
    scale = config_dict['scale']
    

    configs = []
    eval_config = {
        'map_ext': '.png',
        'map': os.path.join(map_location,f'maps/map_{eval_tr}_{scale}'),
        'waypoints': os.path.join(map_location,f'centerline/map_{eval_tr}_{scale}.csv'),
        'reset_pose': [0.0,0.0,np.pi/2]
    }
    
    for t in trs:
        map_config = {
            'map_ext': '.png',
            'map': os.path.join(map_location,f'maps/map_{t}_{scale}'),
            'waypoints': os.path.join(map_location,f'centerline/map_{t}_{scale}.csv'),
            'reset_pose': [0.0,0.0,np.pi/2]
        }
        plt.imshow(plt.imread(map_config['map']+'.png'))
        plt.show()
        yes = input("Is this the map you want to use? (y/n): ")
        if yes == 'y':
            configs.append(map_config)
        plt.close()    
    print(configs)
    exit(1)
    
    

    register('f110_gym:f110-cust-v0', entry_point='f110_gym.envs:F110_Cust_Env', max_episode_steps=10000)

    
    env = gym.make('f110_gym:f110-cust-v0',config=configs[args.config-1], num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False)
        

    retain_string = int(args.retain*100)

    
    experiment_name = f"{args.config}_{args.car_idx}_{retain_string}_{args.exp}"
    
    architecture = [256,256,256]
    
    
    model = SAC('MlpPolicy', env, verbose=args.verbose, tensorboard_log=f"logs/sac_{experiment_name}", policy_kwargs=dict(net_arch=architecture))

    
    model.learn(total_timesteps=args.total_timesteps, callback=custom_cb)

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the f1tenth gym environment')
    parser.add_argument('--config', type=int, default=1, help='Config to run')
    parser.add_argument('--car_idx', type=int, default=1, help='Car index to use')
    parser.add_argument('--base',type=int,default=1,help='Base or custom callback')
    parser.add_argument('--retain',type=float,default=1,help='Retain ratio for custom callback')
    parser.add_argument('--exp',type=int,default=1,help='Experiment number')
    parser.add_argument('--save_freq',type=int,default=500,help='Save frequency for custom callback')
    parser.add_argument('--modify_epoch',type=int,default=1000,help='Modify epoch for custom callback')
    parser.add_argument('--total_timesteps',type=int,default=1e6,help='Total timesteps for training')
    parser.add_argument('--verbose',type=int,default=0,help='Verbosity level')

    args = parser.parse_args()
    main()

import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import yaml

with open('../device.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

is_lab = data['is_lab']

def register_f110(idx=1):
    import os
    import pickle
    from gym.envs.registration import register
    import gym
    from f110_gym.envs.base_classes import Integrator
    with open("../maps.pkl","rb") as f:
        maps = pickle.load(f)
        
    configs = maps

    dir_path = os.path.dirname(os.path.realpath(__file__)).split('/')[:-2]
    dir_path = '/'.join(dir_path)

    for i in configs:
        base_map = i['map']
        base_wpt = i['waypoints']
        i['map']=dir_path+base_map
        i['waypoints']=dir_path+base_wpt

    testing_config = configs[1:]
    current_config = testing_config[idx-1]

    register('f110_gym:f110-cust-v0', entry_point='f110_gym.envs:F110_Cust_Env', max_episode_steps=10000)

    eval_env = gym.make('f110_gym:f110-cust-v0',config=configs[0], num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False)
    env = gym.make('f110_gym:f110-cust-v0',config=current_config, num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False)

    return env, eval_env

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=2000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=10, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--kl_scale', type=float, default=2)
parser.add_argument('--own_policy_idx',type=int,default=1)
parser.add_argument('--config', type=int, default=1)
parser.add_argument('--cup_flag', type=bool, default=False)
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
# env = gym.make(args.env_name)
# env.seed(args.seed)
# env.action_space.seed(args.seed)

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)

env,eval_env = register_f110(args.config)

eval_batch = eval_env.get_dummies()

own_policy_name = f"policy_{args.own_policy_idx}.pth"

# other_policy_name = ""

# if args.own_policy_idx == 1:
#     other_policy_name = "policy_2.pth"
# else:
#     other_policy_name = "policy_1.pth"

other_policies = [f"policy_{i}.pth" for i in range(1,4) if i!=args.own_policy_idx]

experiment = f"runs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.env_name}_{args.policy}_{'autotune' if args.automatic_entropy_tuning else ''}"

#Tensorboard
writer = SummaryWriter(experiment)

# Agent
agent = SAC(env.observation_space.shape[0], 
            env.action_space, 
            args,eval_batch,
            CUP_flag=args.cup_flag,
            other_policies=other_policies,
            own_idx=args.own_policy_idx
            )

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
update_freq = 25

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                # try:
                if i_episode % update_freq == 0:
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, kl, mu, sig, beta, idx = agent.update_parameters(memory, args.batch_size, updates,guided_itr=True)
                else:
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, kl, mu, sig, beta, idx = agent.update_parameters(memory, args.batch_size, updates)
                
                if i_episode % update_freq == 0:
                    if args.kl_scale > 0:
                        writer.add_scalar('div/kl_scaled', kl*beta, updates)
                        writer.add_scalar('div/beta_s', beta, updates)
                        writer.add_scalar('div/kl_original', kl, updates)
                        if idx is not None:
                            writer.add_scalar('div/idx',idx,updates)
                    else:
                        writer.add_scalar('div/kl_scaled', 0, updates)
                        writer.add_scalar('div/beta_s', 0, updates)
                        writer.add_scalar('div/kl_original', 0, updates)
                        if idx is not None:
                            writer.add_scalar('div/idx',idx,updates)

                    if type(mu) == torch.Tensor:
                        mu = mu.cpu().detach().numpy()
                        sig = sig.cpu().detach().numpy()

                        mean_mu = np.mean(mu,axis=0)
                        mean_sig = np.mean(sig,axis=0)

                        writer.add_scalar('div/steer_mu', mean_mu[0], updates)
                        writer.add_scalar('div/steer_sig', mean_sig[0], updates)
                        writer.add_scalar('div/speed_mu', mean_mu[1], updates)
                        writer.add_scalar('div/speed_sig', mean_sig[1], updates)

                if updates % update_freq == 0:
                    # writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    # writer.add_scalar('entropy_temprature/alpha', alpha, updates)

                    
                updates += 1

                # except Exception as e:
                #     print(e)
                #     continue

        
        next_state, reward, done, _, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = float(not done)

        if type(state)==tuple:
            state = state[0]

        if type(next_state)==tuple:
            next_state = next_state[0]

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Config: {}|{} Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(args.config,args.cup_flag,i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            if type(state)==tuple:
                state = state[0]
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Config: {}|{} Test Episodes: {}, Avg. Reward: {}".format(args.config,args.cup_flag,episodes, round(avg_reward, 2)))
        print("----------------------------------------")

    if i_episode % 10 == 0 and args.cup_flag:
        policy = agent.policy.state_dict()
        torch.save(policy, own_policy_name)
env.close()


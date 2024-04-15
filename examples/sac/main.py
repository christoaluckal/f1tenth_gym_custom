import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

def register_f110(idx=1):
    import os
    import pickle
    from gym.envs.registration import register
    import gym
    from f110_gym.envs.base_classes import Integrator
    with open("../maps.pkl","rb") as f:
        maps = pickle.load(f)
        
    configs = maps

    if not os.path.exists("logs"):
        os.makedirs("logs",exist_ok=True)


    print("Using home configs")
    for i in configs:
        i['map'] = i['map'].replace('/home/christo/Developer/thesis/f1tenth_gym_custom/examples','/home/caluckal/Developer/spring2024/thesis/f1tenth_gym_custom/examples')
        i['waypoints'] = i['waypoints'].replace('/home/christo/Developer/thesis/f1tenth_gym_custom/examples','/home/caluckal/Developer/spring2024/thesis/f1tenth_gym_custom/examples')


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
parser.add_argument('--kl_scale', type=float, default=10)
parser.add_argument('--own_policy_idx',type=int,default=1)
parser.add_argument('--config', type=int, default=1)
parser.add_argument('--cup_flag', type=bool, default=True)
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

other_policy_name = ""

if args.own_policy_idx == 1:
    other_policy_name = "policy_2.pth"
else:
    other_policy_name = "policy_1.pth"

# Agent
agent = SAC(env.observation_space.shape[0], 
            env.action_space, 
            args,eval_batch,
            CUP_flag=args.cup_flag,
            other_policy_name=other_policy_name)

#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

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
                if i_episode % 25 == 0:
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, kl = agent.update_parameters(memory, args.batch_size, updates,guided_itr=True)
                else:
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, kl = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                writer.add_scalar('div/kl_scaled', kl, updates)
                writer.add_scalar('div/kl_original', kl/args.kl_scale, updates)
                updates += 1

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


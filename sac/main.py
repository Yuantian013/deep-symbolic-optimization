import argparse
import datetime
import gym
import numpy as np
import itertools
import logger
import torch
from sac_rebuttal.sac import SAC
from torch.utils.tensorboard import SummaryWriter
from SAC.replay_memory import ReplayMemory
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Battery Management",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')

parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G', # default=0.99
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G', # 这个不要动了
                    help='learning rate (default: 0.0003)')
parser.add_argument('--step_size', type=float, default=10, metavar='G')
parser.add_argument('--alpha', type=float, default=0.5, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')

parser.add_argument('--mrl', type=bool, default=True, metavar='N',)

parser.add_argument('--seed', type=int, default=654321, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=10000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',  ###########初始是256
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=0, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')



parser.add_argument('--policy', default="Gaussian",
# parser.add_argument('--policy', default="Gaussian2",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')

args = parser.parse_args()

from envs.BatteryPack import BP_Management as env


########## 设置有几个电池 #####################
num_batteries=4

env = env(num_batteries,args.step_size)
env = env.unwrapped

torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Agent

# Train=False
Train=True

# Memory


# Training Loop
# total_numsteps = 0
updates = 0



High_Reward=0
High_Step=0
total_test_steps = 0



for rd_seeds in range(0,3):
    log_path = '/Users/tianyuan/polybox/Shared/Battery/rebuttal/4B/SAC/'+str(rd_seeds)
    memory = ReplayMemory(args.replay_size)
    logger.configure(dir=log_path, format_strs=['csv'])
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    total_numsteps = 0
    i_episode = -1

    while total_numsteps < 175000:
        i_episode+=1
        episode_reward = 0
        episode_steps = 0

        done = False

        state = env.reset()

        if Train == True:
            while not done:
                # print(total_numsteps)
                # if args.start_steps > total_numsteps / 100:
                #     action = env.action_space.sample()  # Sample random action
                #     a = action / sum(action)
                #
                # else:
                action = agent.select_action(state)  # Sample action from policy
                    # print(action)
                a = action / sum(action)

                ######## Train ####################
                if len(memory) > args.batch_size:

                    for i in range(args.updates_per_step):
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                             args.batch_size,
                                                                                                             updates)

                        updates += 1
                load_demand = np.random.rand()

                reward, next_state, done = env.step(a, load_demand)  # Step

                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == 2048 else float(not done)

                memory.push(state, action, reward, next_state, mask)  # Append transition to memory

                state = next_state
                if total_numsteps % 500 == 0 and args.eval is True:

                    avg_reward = 0.
                    avg_step = 0.
                    episodes = 1

                    for _ in range(1):
                        eval_state = env.eval_reset()

                        episode_reward = 0
                        episode_step = 0
                        eval_done = False

                        while not eval_done:
                            action = agent.select_action(eval_state, True)
                            a = action / sum(action)

                            load_demand = 0.6

                            reward, next_state, eval_done = env.step(a, load_demand)
                            episode_reward += reward
                            episode_step += 1

                            eval_state = next_state

                            # print(np.shape(RL_ratio))
                        avg_reward += episode_reward
                        avg_step += episode_step
                    avg_reward /= episodes
                    avg_step /= episodes
                    logger.logkv("total_episodes", i_episode)
                    logger.logkv("total_numsteps", total_numsteps)
                    logger.logkv("test_return", episode_step)
                    logger.dumpkvs()

                    print("----------------------------------------")
                    print("Test Episodes: {}, Avg. Reward: {}, Avg. Step: {}".format(i_episode, round(avg_reward, 2),
                                                                                     round(avg_step, 2)))

                    print("----------------------------------------")

    env.close()




import argparse
import time
import gym
import torch
import numpy as np
from itertools import count

import logging

import os
import os.path as osp
import json

from sac.replay_memory import ReplayMemory
from sac.sac import SAC
from model import EnsembleDynamicsModel
from predict_env import PredictEnv
from sample_env import EnvSampler
# from tf_models.constructor import construct_model, format_samples_for_training
# from utils.notebook_utils import save_video
import csv

def readParser():
    parser = argparse.ArgumentParser(description='MBPO')
    parser.add_argument('--env_name', default="Hopper-v2",
                        help='Mujoco Gym environment (default: Hopper-v2)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')

    parser.add_argument('--use_decay', type=bool, default=True, metavar='G',
                        help='discount factor for reward (default: 0.99)')

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')

    parser.add_argument('--num_networks', type=int, default=7, metavar='E',
                        help='ensemble size (default: 7)')
    parser.add_argument('--num_elites', type=int, default=5, metavar='E',
                        help='elite size (default: 5)')
    parser.add_argument('--pred_hidden_size', type=int, default=200, metavar='E',
                        help='hidden size for predictive model')
    parser.add_argument('--reward_size', type=int, default=1, metavar='E',
                        help='environment reward size')

    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--num_model_rollout', type=int, default=400, metavar='A',
                        help='model rollouts per step')
    parser.add_argument('--model_retain_epochs', type=int, default=1, metavar='A',
                        help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=250, metavar='A',
                        help='frequency of training')
    parser.add_argument('--rollout_batch_size', type=int, default=100000, metavar='A',
                        help='rollout number M')
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                        help='steps per epoch')
    parser.add_argument('--rollout_min_epoch', type=int, default=20, metavar='A',
                        help='rollout min epoch')
    parser.add_argument('--rollout_max_epoch', type=int, default=100, metavar='A',
                        help='rollout max epoch')
    parser.add_argument('--rollout_min_length', type=int, default=1, metavar='A',
                        help='rollout min length')
    parser.add_argument('--rollout_max_length', type=int, default=15, metavar='A',
                        help='rollout max length')
    parser.add_argument('--num_epoch', type=int, default=1000, metavar='A',
                        help='total number of epochs')
    parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
                        help='minimum pool size')
    parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
                        help='ratio of env samples / model samples')
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                        help='frequency of training policy')
    parser.add_argument('--num_train_repeat', type=int, default=20, metavar='A',
                        help='times to training policy per step')
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                        help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                        help='batch size for training policy')
    parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
                        help='exploration steps initially')

    parser.add_argument('--model_type', default='pytorch', metavar='A',
                        help='predict model -- pytorch or tensorflow')

    parser.add_argument('--cuda', default=True, action="store_true",
                        help='run on CUDA (default: True)')

    parser.add_argument('--exp_name', default='exp1',
                        help='your model save path')
    parser.add_argument('--save_dir', default='../exp/',
                        help='your model save path')

    # parser.add_argument('--model_select_method', default='random', metavar='A',
    #                     help='model_select_method -- random or weighted')

    ### exp for Sec1
    parser.add_argument('--reweight_model', default='none', metavar='A',
                        help='reweight_model -- none or delta or decay or delta_decay')

    ### exp for Sec2
    parser.add_argument('--reweight_rollout', default='none', metavar='A',
                        help='reweight_rollout -- none or delta or decay or delta_decay')
    parser.add_argument('--decay_rate', type=float, default=0.97, metavar='A',
                        help='sample by decay rate')
    parser.add_argument('--delta_clip', type=int, default=10000, metavar='A',
                        help='sample from top k buffer sorted by delta')
    parser.add_argument('--rollout_delta', default='none', metavar='A',
                        help='use delta to critic the quality of the model rollout sample')

    ### exp for Sec3
    parser.add_argument('--multiplier_batch', default=20, metavar='A',
                    help='batch size * multiplier = number of a batch which will be sampled and be weighted with KL results')
    parser.add_argument('--reweight_policy', default='none', metavar='A',
                        help='reweight replay buffer sampling model pool to train a policy -- none or delta or acc (measure accuracy using KL-div) ')

    return parser.parse_args()


def train(args, env_sampler, env_sampler_test, predict_env, agent, env_pool, model_pool, logger):
    total_step = 0
    reward_sum = 0
    rollout_length = 1
    exploration_before_start(args, env_sampler, env_pool, agent)
    # env_pool.update_decay_weights()

    decay_weight_list = []
    delta_score_list = []

    for epoch_step in range(args.num_epoch):
        start_step = total_step
        train_policy_steps = 0
        for i in count():
            cur_step = total_step - start_step

            if cur_step >= start_step + args.epoch_length and len(env_pool) > args.min_pool_size:
                break

            if cur_step > 0 and cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
                print("env_pool: {}, model_pool: {}".format(len(env_pool), len(model_pool)))
                # logger.info("env_pool: {}, model_pool: {}".format(len(env_pool), len(model_pool)))

                # update decay_weights and delta_score
                # env_pool.update_decay_weights()
                # env_pool.update_delta_score(agent, args)
                # decay_weights_list.append(env_pool.decay_weights)
                # delta_score_list.append(env_pool.delta_score)

                train_predict_model(args, env_pool, predict_env, logger)
                new_rollout_length = set_rollout_length(args, epoch_step)
                if rollout_length != new_rollout_length:
                    rollout_length = new_rollout_length
                    model_pool = resize_model_pool(args, rollout_length, model_pool)
                rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length)
                # logger.info("finish rollout - env_pool: {}, model_pool: {}".format(len(env_pool), len(model_pool)))
                # update delta_score and/or delta_weights for every transition in model_pool
                # model_pool.update_delta_score(agent,args)
                #model_pool update delta_score

            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
            env_pool.push(cur_state, action, reward, next_state, done)
            
            if len(env_pool) > args.min_pool_size:
                train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, cur_step, env_pool, model_pool, agent, logger)
                # logger.info("finish train policy")

            total_step += 1

            if total_step % 1000 == 0:
                env_sampler_test.current_state = None
                env_sampler_test.path_length = 0
                sum_reward = 0
                test_done = False
                # frames = []
                while not test_done:
                    # frame = env_sampler.env.render(mode='rgb_array')
                    test_cur_state, test_action, test_next_state, test_reward, test_done, test_info = env_sampler_test.sample(agent, eval_t=True)
                    sum_reward += test_reward
                    # frames.append(frame.copy())

                # log, print
                print("total_step: {}, sum_reward: {}".format(total_step, sum_reward))
                logger.info("total_step: {}, sum_reward: {}".format(total_step, sum_reward))

                # save model (option: save video)
                # model_file = os.path.join(args.exp_dir, 'model_last.pt')

                # torch.save({'Dynamics': predict_env.model.ensemble_model.state_dict(),
                #             'Policy': agent.policy.state_dict(),
                #             'Critic': agent.critic.state_dict(),
                #             }, model_file)

                # save_weights = True
                # if save_weights:
                #     # decay_weight_file = os.path.join(args.exp_dir, 'decay_weight_list.csv')
                #     # with open(decay_weight_file, 'w') as f1:
                #     #     write = csv.writer(f1)
                #     #     write.writerows(decay_weight_list)
                #     delta_score_file = os.path.join(args.exp_dir, 'delta_score_list.csv')
                #     with open(delta_score_file, 'w') as f2:
                #         write = csv.writer(f2)
                #         write.writerows(delta_score_list)


def exploration_before_start(args, env_sampler, env_pool, agent):
    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
        env_pool.push(cur_state, action, reward, next_state, done)
        if i % 1000 == 0:
            print('exploration_before_start: step {}/{}'.format(i, args.init_exploration_steps))


def set_rollout_length(args, epoch_step):
    rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
                              / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
                              args.rollout_min_length), args.rollout_max_length))
    return int(rollout_length)


def train_predict_model(args, env_pool, predict_env, logger):
    # Get all samples from environment
    state, action, reward, next_state, done = env_pool.sample(len(env_pool))
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

    if 'decay' in args.reweight_model:
        weights = np.array(env_pool.decay_weight)
        predict_env.model.train(inputs, labels, weights=weights, batch_size=256, holdout_ratio=0.2, logger=logger)
    else:
        predict_env.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2, logger=logger)


def resize_model_pool(args, rollout_length, model_pool):
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch

    sample_all = model_pool.return_all() 
    new_model_pool = ReplayMemory(new_pool_size)
    new_model_pool.push_batch(sample_all)

    return new_model_pool


def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length):
    # model_pool.KL = []
    if 'decay' in args.reweight_rollout:
        state, action, reward, next_state, done = env_pool.decayweightedsample_all_batch(args.rollout_batch_size)
    elif 'delta' in args.reweight_rollout:
        state, action, reward, next_state, done = env_pool.deltaweightedsample_all_batch(args.rollout_batch_size)
    else:
        state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size)
    for i in range(rollout_length):
        # TODO: Get a batch of actions
        action = agent.select_action(state)
        next_states, rewards, terminals, info, KL = predict_env.step(state, action)
        # next_states, rewards, terminals, info = predict_env.step(state, action)
        # TODO: Push a batch of samples
        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j], KL[j]) for j in range(state.shape[0])])
        # model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
        nonterm_mask = ~terminals.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]


def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent, logger):
    done_training = False
    if total_step % args.train_every_n_steps > 0:
        return 0

    if train_step > args.max_train_repeat_per_step * total_step:
        return 0

    for i in range(args.num_train_repeat):
        if i == args.num_train_repeat -1:
            done_training = True
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size

        env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))

        if model_batch_size > 0 and len(model_pool) > 0:
            if 'delta' in args.reweight_policy:
                model_state, model_action, model_reward, model_next_state, model_done = model_pool.deltaweightedsample_all_batch(int(model_batch_size))
            elif 'acc' in args.reweight_policy:
                model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch_KL(int(model_batch_size), int(model_batch_size)*args.multiplier_batch, done_training)
            else:
                model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), \
                                                                                    np.concatenate((env_action, model_action),
                                                                                                   axis=0), np.concatenate(
                (np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                                                                                    np.concatenate((env_next_state, model_next_state),
                                                                                                   axis=0), np.concatenate(
                (np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
        else:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)

        agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), args.policy_train_batch_size, i)

    return args.num_train_repeat


from gym.spaces import Box


class SingleEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleEnvWrapper, self).__init__(env)
        obs_dim = env.observation_space.shape[0]
        obs_dim += 2
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        torso_height, torso_ang = self.env.sim.data.qpos[1:3]  # Need this in the obs for determining when to stop
        obs = np.append(obs, [torso_height, torso_ang])

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        torso_height, torso_ang = self.env.sim.data.qpos[1:3]
        obs = np.append(obs, [torso_height, torso_ang])
        return obs


def main(args=None):
    if args is None:
        args = readParser()

    # exp path
    args.exp_dir = os.path.join(args.save_dir, args.exp_name)
    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    # logger
    log_file = os.path.join(args.exp_dir, '{}.txt'.format(args.exp_name))
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    print("exp name: " + args.exp_name)
    logger.info("exp name: " + args.exp_name)

    # Initial environment
    env = gym.make(args.env_name)
    env_test = gym.make(args.env_name)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    env_test.seed(args.seed)

    # Initial agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    if args.model_type == 'pytorch':
        env_model = EnsembleDynamicsModel(args.num_networks, args.num_elites, state_size, action_size, args.reward_size, args.pred_hidden_size,
                                          use_decay=args.use_decay)
    else:
        raise ValueError('this code blocked the tensorflow version of env_model')
        # env_model = construct_model(obs_dim=state_size, act_dim=action_size, hidden_dim=args.pred_hidden_size, num_networks=args.num_networks,
        #                             num_elites=args.num_elites)

    # Predict environments
    predict_env = PredictEnv(env_model, args.env_name, args.model_type, args)

    # Initial pool for env
    env_pool = ReplayMemory(args.replay_size)
    # Initial pool for model
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(1 * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    model_pool = ReplayMemory(new_pool_size)

    # Sampler of environment
    env_sampler = EnvSampler(env)
    env_sampler_test = EnvSampler(env_test)

    train(args, env_sampler, env_sampler_test, predict_env, agent, env_pool, model_pool, logger)


if __name__ == '__main__':
    main()

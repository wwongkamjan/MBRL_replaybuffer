import random
import numpy as np
from operator import itemgetter
import os
import os.path as osp
import csv

import torch
from torch.utils.data import WeightedRandomSampler


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

        # self.decay_weight = []
        self.delta_score = []
        self.delta_weight = []
        self.device = torch.device("cuda")

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            # self.decay_weight.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        # self.decay_weight[self.position] = None
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, batch):
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * append_len)

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.buffer[self.position : len(self.buffer)] = batch[:len(self.buffer) - self.position]
            self.buffer[:len(batch) - len(self.buffer) + self.position] = batch[len(self.buffer) - self.position:]
            self.position = len(batch) - len(self.buffer) + self.position

    # weights of new sample was set to None, this function will find all None and update it to a certain ratio
    def update_decay_weights(self, decay_rate=0.97):
        weight_arr = np.array(self.decay_weight, dtype=np.double)

        new_buffer_index = np.where(np.isnan(weight_arr))[0]
        if len(new_buffer_index) == len(weight_arr): # all new buffer
            value = 1
        else:
            weight_arr[new_buffer_index] = 0
            value = sum(weight_arr) * (1 / decay_rate - 1) / len(new_buffer_index)
            if value < self.decay_weight[0]:
                value = self.decay_weight[0]
        weight_arr[new_buffer_index] = value
        weight_arr = weight_arr * decay_rate

        self.decay_weight = weight_arr.tolist()

    def update_delta_score(self, agent):
        state, action, reward, next_state, done = map(np.stack, zip(*self.buffer))
        state_batch = torch.FloatTensor(state).to(self.device)
        next_state_batch = torch.FloatTensor(next_state).to(self.device)
        # action_batch = torch.FloatTensor(action).to(self.device)
        reward_batch = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(done).to(self.device).unsqueeze(1)

        q_state = self.get_Q_value(agent, state_batch)
        q_next_state = self.get_Q_value(agent, next_state_batch)
        score = reward_batch + q_next_state - q_state
        weight = (score - torch.min(score) + 0.001) / (torch.max(score) - torch.min(score))
        weight = torch.reshape(weight, (-1,))
        self.delta_score = score.detach().cpu().numpy().tolist()
        self.delta_weight = weight.detach().cpu().numpy().tolist()
        # print(self.delta_weight)

    def get_Q_value(self, agent, state):
        _, _, action = agent.policy.sample(state)
        qf1, qf2 = agent.critic(state, action)
        min_qf = torch.min(qf1, qf2)
        return min_qf

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, int(batch_size))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def weightedsample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        # batch = random.sample(self.buffer, int(batch_size))
        idx = np.array(list(WeightedRandomSampler(self.weight, batch_size, replacement=False)))
        batch = self.buffer[idx]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_all_batch(self, batch_size):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def weightedsample_all_batch(self, batch_size):
        # idxes = np.random.randint(0, len(self.buffer), batch_size)
        # weight = (score - torch.min(score) + 0.001) / (torch.max(score) - torch.min(score))
        idxes = np.array(list(WeightedRandomSampler(self.delta_weight, batch_size, replacement=True)))
        print(idxes)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def sample_all_batch_KL(self, batch_size, logger):
        KL_list = np.array([abs(t[-1]) for t in self.buffer])
        #flip
        sum_w = sum(KL_list)
        logger.info("finish KL list: {} and sumw: {}".format(KL_list.shape, sum_w))
        weight = np.array([sum_w - kl for kl in KL_list])
        logger.info("finish weight with shape {}".format(weight.shape))
        norm_weight = np.array([(float(w)/sum(weight)) for w in weight])
        logger.info("norm weight and then running idxes")
        idxes = np.array(list(WeightedRandomSampler(norm_weight, batch_size, replacement=True)))
        logger.info("finish idxes")
        batch = list(itemgetter(*idxes)(np.array(self.buffer, dtype=object)))
        state, action, reward, next_state, done, _ = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def return_all(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)

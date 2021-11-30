import random
import numpy as np
from operator import itemgetter
import os
import os.path as osp
import csv
import random
import torch
import heapq
from torch.utils.data import WeightedRandomSampler


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.KL = []
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
    def update_decay_weights(self, decay_rate=0.9):
        weight_arr = np.array(self.decay_weights, dtype=np.double)

        new_buffer_index = np.where(np.isnan(weight_arr))[0]
        if len(new_buffer_index) == len(weight_arr):  # all new buffer
            value = 1
        else:
            weight_arr[new_buffer_index] = 0
            value = sum(weight_arr) * (1 / decay_rate - 1) / len(new_buffer_index)
            if value < self.decay_weights[0]:  # ensure newly added sample have larger weights
                value = self.decay_weights[0]
        weight_arr[new_buffer_index] = value
        weight_arr = weight_arr * decay_rate

        self.decay_weights = weight_arr.tolist()

    def update_delta_score(self, agent, args):
        state, action, reward, next_state, done = map(np.stack, zip(*self.buffer))
        batch_size = 5000  # chop buffer to batches, to handle CUDA memory problem
        delta = np.empty(shape=0)
        for start_pos in range(0, len(self.buffer), batch_size):
            state_batch = torch.FloatTensor(state[start_pos: start_pos + batch_size]).to(self.device)
            next_state_batch = torch.FloatTensor(next_state[start_pos: start_pos + batch_size]).to(self.device)
            reward_batch = torch.FloatTensor(reward[start_pos: start_pos + batch_size]).to(self.device)

            q_state = self.get_Q_value(agent, state_batch)
            q_next_state = self.get_Q_value(agent, next_state_batch)
            score = reward_batch.squeeze() + q_next_state.squeeze() - q_state.squeeze()
            delta_batch = score.squeeze().detach().cpu().numpy()
            delta = np.append(delta, delta_batch)
        self.delta_score = delta.tolist()

        total_len = len(delta)
        delta_weights = np.zeros(total_len)
        top_number = int(total_len*args.delta_percent) # 10000
        delta_weights[np.argsort(delta)[-top_number:]] = 1

        self.delta_weights = delta_weights.tolist()


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

    def decayweightedsample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        # batch = random.sample(self.buffer, int(batch_size))
        idx = np.array(list(WeightedRandomSampler(self.decay_weights, batch_size, replacement=True)))
        batch = list(itemgetter(*idx)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def deltaweightedsample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        idx = np.array(list(WeightedRandomSampler(self.delta_weights, batch_size, replacement=True)))
        batch = list(itemgetter(*idx)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_all_batch(self, batch_size):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def decayweightedsample_all_batch(self, batch_size):
        # idxes = np.random.randint(0, len(self.buffer), batch_size)
        idxes = np.array(list(WeightedRandomSampler(self.decay_weights, batch_size, replacement=True)))
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def deltaweightedsample_all_batch(self, batch_size):
        idxes = np.array(list(WeightedRandomSampler(self.delta_weights, batch_size, replacement=True)))
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def sample_all_batch_KL(self, batch_size, sample_size, train_done):
        #idxes = np.random.randint(0, len(self.buffer), sample_size)
        idxes = np.random.randint(0, len(self.buffer), sample_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        if len(self.KL) ==0:
            KL_list = np.array([abs(t[-1]) for t in batch])
            max_w = [max(KL_list)]*len(batch)
            weight = np.array(max_w - KL_list)
            self.KL = weight
        # norm_weight = np.array([(float(w)/sum(weight)) for w in weight])
        new_batch = random.choices(batch,weights=self.KL,k=batch_size)
        state, action, reward, next_state, done, _ = map(np.stack, zip(*new_batch))
        if train_done:
            self.KL = []
        return state, action, reward, next_state, done

    def return_all(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)

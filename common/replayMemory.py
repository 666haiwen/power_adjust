import os
import random
import pickle
import numpy as np
from collections import namedtuple
from common.segment_tree import MinSegmentTree, SumSegmentTree

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
class ReplayMemory(object):
    """
        Save Transition into Memory to train.
        @param:
        capacity: the capacity of memory
        path: the path to save in order to read next time
    """
    def __init__(self, capacity, path):
        self.capacity = capacity
        self.memory = []
        self.postion = 0
        self.path = path
    
    def reset(self):
        self.memory = []
        self.postion = 0

    def push(self, *args):
        """ Save a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.postion] = Transition(*args)
        self.postion = (self.postion + 1) % self.capacity
        if self.postion == self.capacity - 1:
            self.save()
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def read(self, memory=None, rate=1):
        """ Read the data from disk by pickel"""
        if (not os.path.exists(self.path)) and (memory==None):
            return
        if memory != None:
            self.memory = []
            self.postion = 0
            for v in memory:
                if v.reward > 0:
                    self.push(v.state, v.action, v.next_state, v.reward, v.done)
            self.save()
        else:
            with open(self.path, 'rb') as fp:
                data = pickle.load(fp)
                self.memory = data['memory']
                self.postion = data['position']
                if rate < 1:
                    length = int(len(self.memory) * rate)
                    self.memory = self.memory[:length]
                    self.postion = len(self.memory)

    def get_state(self):
        return [m.state for m in self.memory]

    def save(self):
        with open(self.path, 'wb') as fp:
            pickle.dump({
                'memory': self.memory,
                'position': self.postion
            }, fp)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayMemory):
    def __init__(self, capacity, path, alpha=0.6):
        super(PrioritizedReplayBuffer, self).__init__(capacity, path)
        assert alpha >= 0
        self.alpha = alpha
        
        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self.it_sum = SumSegmentTree(it_capacity)
        self.it_min = MinSegmentTree(it_capacity)
        self.max_priority = 1.0

    def push(self, *args):
        position = self.postion
        super().push(*args)
        self.it_sum[position] = self.max_priority ** self.alpha
        self.it_min[position] = self.max_priority ** self.alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self.it_sum.sum(0, len(self.memory) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self.it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta=0.4):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        batch = [self.memory[i] for i in idxes]

        weights = []
        p_min = self.it_min.min() / self.it_sum.sum()
        max_weight = (p_min * len(self.memory)) ** (-beta)

        for idx in idxes:
            p_sample = self.it_sum[idx] / self.it_sum.sum()
            weight = (p_sample * len(self.memory)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)
        return batch, weights, idxes

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.memory)
            self.it_sum[idx] = priority ** self.alpha
            self.it_min[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

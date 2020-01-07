import tensorflow as tf
import numpy as np
import random
from collections import defaultdict

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

from collections import deque

class BaseReplayBuffer():
    """Simple interface (has no functionality) for a replay buffer"""

    #Add an item (experience, trajectory, episode etc) to the replay buffer.
    def add(self, item):
        pass

    #Returns n items from the replay buffer (behaviour as determined by the specific subclass).
    def get(self, n=1):
        pass

class SimpleReplayBuffer(BaseReplayBuffer):
    def __init__(self, max_storage):
        self.max_storage = max_storage
        self.storage = deque(maxlen=max_storage)
        self.length = 0

    def add_item(self, item):
        self.storage.append(item)
        self.length += 1

    def add(self, obs, act, next_obs, rew, done):
        self.add_item({"obs": obs, "act":act, "next_obs":next_obs, "rew":rew, "done": done})

    def get(self, n=1):
        return random.choices(self.storage, k=n)

    def sample(self, batch_size):
         LD = self.get(batch_size)

         return self.transform(LD)

    def transform(self, LD):
        DL = defaultdict(list)

        for item in LD:
            for k, v in item.items():
                DL[k].append(v)

        for k, v in DL.items():
            DL[k] = np.array(v, dtype="float32")

            if len(DL[k].shape) == 1:
                DL[k] = DL[k].reshape((-1, 1))

        return DL


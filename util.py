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

    # Add an item (experience, trajectory, episode etc) to the replay buffer.
    def add(self, item):
        pass

    # Returns n items from the replay buffer (behaviour as determined by the specific subclass).
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
        self.add_item({"obs": obs, "act": act, "next_obs": next_obs, "rew": rew, "done": done})

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


class VAEReplayJoiner:
    def __init__(self, VAE, replay_buffer, VAE_split):
        """

        :param VAE:
        :param replay_buffer:
        :param VAE_split: percentage of values to get from the VAE, instead of Replay buffer
        """
        self.VAE_split = VAE_split
        self.VAE = VAE
        self.replay_buffer = replay_buffer
        # self.split = self.split

    def sample(self, batch_size):
        from_vae = np.random.binomial(batch_size, self.VAE_split)
        from_rb = batch_size - from_vae

        if from_rb > 0:
            samples = self.replay_buffer.sample(from_rb)

        if from_vae > 0:
            samples2 = self.VAE(from_vae)

            if from_rb > 0:
                for k,v in samples.items():
                    samples[k] = np.concatenate((v,samples2[k]))
                return samples
            else:
                return samples2

        raise Exception("Shouldn't be here")

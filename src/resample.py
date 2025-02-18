import numpy as np
import torch

from abc import ABC, abstractmethod


class ScheduleSampler(ABC):

    @abstractmethod
    def weights(self):
        """ """

    def sample(self, batch_size, device):
        w = self.weights()
        p = w / np.sum(w)
        timesteps_np = np.random.choice(len(p), size=(batch_size,), p=p)
        timesteps = torch.from_numpy(timesteps_np).long().to(device)
        weights_np = 1 / (len(p) * p[timesteps_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return timesteps, weights

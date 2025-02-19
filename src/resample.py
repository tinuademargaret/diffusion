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


class UniformSampler(ScheduleSampler):

    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class LossSecondMomentResampler(ScheduleSampler):

    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros([diffusion.num_timesteps, history_per_term])
        self._loss_count = np.zeros(diffusion.num_timesteps)

    def weights(self):
        if not self._warmed_up:
            # return uniform weight if we don't have enough loss history
            return np.ones([self.diffusion.num_timesteps])

        weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            # check if timestep has filled loss history
            if self._loss_count[t] == self.history_per_term:
                # remove the oldest loss add the newest loss
                self._loss_history[t][:-1] = self.self._loss_history[t][1:]
                self._loss_history[t][-1] = loss
            else:
                # add new loss and update loss_count
                self._loss_history[t][self._loss_count[t]] = loss
                self._loss_count[t] += 1

    def _warmed_up(self):
        return (self._loss_count == self.history_per_term).all()

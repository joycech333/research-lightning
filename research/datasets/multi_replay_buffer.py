from typing import Union, Dict

import gym
import numpy as np
import torch

from research.utils import utils

from .replay_buffer.buffer import ReplayBuffer
from .simpler_dataset import SimplerDataset

class MultiReplayBuffer(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        batch_size: int,
        batch_weights: Union[Dict, str],
        **buffer_kwargs,
    ) -> None:
        self.batch_size = batch_size
        self.batch_weights = batch_weights
        if isinstance(self.batch_weights, str) and self.batch_weights.lower() == "uniform":
            self._batch_weights = {}
        elif isinstance(self.batch_weights, dict):
            self._batch_weights = self.batch_weights

        self.buffers = {}
        self.iters = {}
        self.buffer_keys = []

        for buffer_name, config in buffer_kwargs.items():
            buffer_name = buffer_name.replace("_kwargs", "")
            if buffer_name == "demos":
                self.buffers[buffer_name] = SimplerDataset(observation_space, action_space, **config)
            else:
                self.buffers[buffer_name] = ReplayBuffer(observation_space, action_space, **config)
            self.buffer_keys.append(buffer_name)

    def update_weights(self):
        if self.batch_weights != "uniform":
            return
        num_episodes = []
        for buffer_name, buffer in self.buffers.items():
            # Hacky, will fix: Hardcode num_episodes to 300 if buffer is 'demos'
            if buffer_name == "demos":
                num_episodes.append(300)
            else:
                if hasattr(buffer, "episode_filenames"):
                    num_episodes.append(len(buffer.episode_filenames))
                else:
                    num_episodes.append(0)
        total_episodes = sum(num_episodes)
        if total_episodes > 0:
            for i, (buffer_name, buffer) in enumerate(self.buffers.items()):
                # Use the corresponding episode count
                self._batch_weights[buffer_name] = num_episodes[i] / total_episodes

    def add(self, buffer_name, **kwargs):
        assert buffer_name in self.buffers, f"'{buffer_name}' not a buffer"
        self.buffers[buffer_name].add(**kwargs)

    def extend(self, buffer_name, **kwargs):
        assert buffer_name in self.buffers, f"'{buffer_name}' not a buffer"
        self.buffers[buffer_name].extend(**kwargs)

    def sample(self, batch_size, *args, **kwargs):
        self.update_weights()
        concatenated_samples = None
        for buffer_name, weight in self._batch_weights.items():
            buffer_batch_size = int(batch_size * weight)
            if buffer_batch_size > 0:
                sample = self.buffers[buffer_name].sample(*args, batch_size=buffer_batch_size, **kwargs)
                if concatenated_samples is None:
                    concatenated_samples = sample
                else:
                    concatenated_samples = utils.concatenate(concatenated_samples, sample, dim=0)
        return concatenated_samples

    def save(self, path):
        for _buffer_name, buffer in self.buffers.items():
            buffer.save(path, prefix=buffer.prefix)

    def __iter__(self):
        for buffer_name, buffer in self.buffers.items():
            self.iters[buffer_name] = iter(buffer)
            next(self.iters[buffer_name])
        while True:
            concatenated_batch = None
            self.update_weights()
            empty_iters = 0
            for buffer_name in self.buffer_keys:
                try:
                    og_batch_size = self.batch_size
                    new_batch_size = int(og_batch_size * self._batch_weights[buffer_name])
                    self.buffers[buffer_name].sample_fn.keywords["batch_size"] = new_batch_size
                    new_batch = next(self.iters[buffer_name])
                    concatenated_batch = (
                        new_batch
                        if concatenated_batch is None
                        else utils.concatenate(concatenated_batch, new_batch, dim=0)
                        if new_batch else concatenated_batch
                    )
                except StopIteration:
                    empty_iters += 1
            if empty_iters == len(self.iters):
                break
            if concatenated_batch is None:
                break

            yield concatenated_batch
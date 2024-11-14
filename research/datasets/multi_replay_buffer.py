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
        step_counts = {buffer_name: buffer._storage.size for buffer_name, buffer in self.buffers.items()}
        # Uniform sampling until the size of the online buffer reaches that of half the demos buffer
        if step_counts["online"] < (step_counts["demos"] / 2):
            total_steps = step_counts["online"] + step_counts["demos"]
            self._batch_weights = {
                "online": step_counts["online"] / total_steps,
                "demos": step_counts["demos"] / total_steps
            }
        # 50/50 sampling when the online buffer contains more steps
        else:
            self._batch_weights = {buffer_name: 1 / len(self.buffers) for buffer_name in self.buffers}

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
            remaining_batch_size = self.batch_size
            for i, buffer_name in self.buffer_keys:
                try:
                    # Calculate the batch size for the current buffer
                    if i < len(self.buffer_keys) - 1:
                        new_batch_size = int(self._batch_weights[buffer_name] * self.batch_size)
                    else:
                        # Ensure the sum of the batch sizes is equal to the total batch size
                        new_batch_size = remaining_batch_size
                    # Update remaining size
                    remaining_batch_size -= new_batch_size

                    self.buffers[buffer_name].sample_fn.keywords["batch_size"] = new_batch_size
                    new_batch = next(self.iters[buffer_name])
                    concatenated_batch = (
                        new_batch
                        if concatenated_batch is None
                        else utils.concatenate(concatenated_batch, new_batch, dim=0)
                        if new_batch
                        else concatenated_batch
                    )

                except StopIteration:
                    empty_iters += 1

            if empty_iters == len(self.iters) or concatenated_batch is None:
                break

            yield concatenated_batch
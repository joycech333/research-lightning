import random
from typing import Optional

import h5py
import numpy as np
import torch
import tensorflow as tf
import tensorflow_datasets as tfds

from research.utils import utils

from .replay_buffer.buffer import ReplayBuffer


class SimplerDataset(ReplayBuffer):
    """
    Simple Class that writes the data from the SimplerDatasets into a ReplayBuffer
    """

    def __init__(
        self, observation_space, action_space, *args, action_eps: Optional[float] = 1e-5, train=True, **kwargs
    ):
        self.action_eps = action_eps
        self.train = train
        # keys required to construct the full state
        self.state_keys = ['state', 'target_obj_pose', 'source_obj_pose', 'tcp_to_source_obj_pos']
        super().__init__(observation_space, action_space, *args, **kwargs)

    def _load_dataset(self):
        """
        Loads the dataset using `tfds.builder_from_directory`.
        """
        split = 'train' if self.train else 'validation'
        print(f"Loading data from: {self.dataset_path}")
        builder = tfds.builder_from_directory(self.dataset_path)
        dataset = builder.as_dataset(split=split)
        return dataset
    
    def _concatenate_state_keys(self, observation):
        """
        Concatenates state keys from observation (state, target_obj_pose, source_obj_pose, tcp_to_source_obj_pos).
        """
        state_tensors = []
        for key in self.state_keys:
            state_tensors.append(observation[key])
        
        return tf.concat(state_tensors, axis=-1)

    def _data_generator(self):
        dataset = self._load_dataset()

        for episode in dataset:
            observations = episode['steps']['observation']
            action = episode['steps']['action']
            reward = episode['steps']['reward']
            discount = episode['steps']['discount']

            # Concatenate state keys
            obs = [self._concatenate_state_keys(obs) for obs in observations]

            # Convert to numpy arrays
            obs = np.array([obs[i].numpy() for i in range(len(obs))])
            action = np.array([action[i].numpy() for i in range(len(action))])
            reward = np.array([reward[i].numpy() for i in range(len(reward))])
            dones = np.array([dones[i].numpy() for i in range(len(dones))])

            # TODO: Manually designed reward
            tcp_to_source_obj_pos = np.array([obs[i]['tcp_to_source_obj_pos'] for i in range(len(obs))])
            source_obj_pose = np.array([obs[i]['source_obj_pose'] for i in range(len(obs))])
            target_obj_pose = np.array([obs[i]['target_obj_pose'] for i in range(len(obs))])
            """
            reward = np.zeros(len(obs))
            # 1) Minimize tcp_to_source_obj_pos
            reward -= np.linalg.norm(tcp_to_source_obj_pos, axis=-1)
            # 2) Penalize if tcp_to_source_obj_pos increases
            for i in range(1, len(tcp_to_source_obj_pos)):
                reward[i] -= np.linalg.norm(tcp_to_source_obj_pos[i] - tcp_to_source_obj_pos[i-1], axis=-1)
            # 3) Minimize distance between source_obj_pose and target_obj_pose
            reward -= np.linalg.norm(source_obj_pose - target_obj_pose, axis=-1)
            """
            
            # NOTE: No dones collected in the dataset currently
            done = (1 - discount).astype(np.bool)

            obs_len = obs[next(iter(obs.keys()))].shape[0]
            assert all([len(obs[k]) == obs_len for k in obs.keys()])
            assert obs_len == len(action) == len(reward) == len(done) == len(discount)

            yield dict(obs=obs, action=action, reward=reward, done=done, discount=discount)

        f.close()  # Close the file handler.
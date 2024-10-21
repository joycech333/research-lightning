import random
from typing import Optional

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
        split = 'train' # if self.train else 'validation'
        print(f"Loading data from: {self.path}")
        builder = tfds.builder_from_directory(self.path)
        dataset = builder.as_dataset(split=split)
        return dataset
    
    def _concatenate_state_keys(self, observation):
        """
        Concatenates state keys from 'extra' dict from observation (state, target_obj_pose, source_obj_pose, tcp_to_source_obj_pos).
        """
        state_tensors = []
        for key in self.state_keys:
            state_tensors.append(observation[key])
        
        return tf.concat(state_tensors, axis=-1)

    def _data_generator(self):
        # Compute the worker info
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        dataset = self._load_dataset()

        # Shuffle the dataset
        dataset = dataset.shuffle(buffer_size=10000)

        # Split the dataset among workers
        dataset = dataset.shard(num_shards=num_workers, index=worker_id)

        for episode in dataset:
            obs_list = []
            action_list = []
            reward_list = []
            discount_list = []

            for step in episode['steps']:
                observation = step['observation']
                action = step['action']
                reward = step['reward']
                discount = step['discount']

                # Concatenate state keys
                obs_state = [self._concatenate_state_keys(observation)]

                obs_list.append(obs_state)
                action_list.append(action)
                reward_list.append(reward)
                discount_list.append(discount)

            # Convert to numpy arrays
            obs = np.array(obs_list)
            action = np.array(action_list)
            reward = np.array(reward_list)
            discount = np.array(discount_list)

            # TODO: Manually designed reward
            """
            tcp_to_source_obj_pos = np.array([obs[i]['tcp_to_source_obj_pos'] for i in range(len(obs))])
            source_obj_pose = np.array([obs[i]['source_obj_pose'] for i in range(len(obs))])
            target_obj_pose = np.array([obs[i]['target_obj_pose'] for i in range(len(obs))])
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
            done = (1 - discount).astype(np.bool_)

            obs_len = obs.shape[0]
            assert obs_len == len(action) == len(reward) == len(done) == len(discount)

            yield dict(obs=obs, action=action, reward=reward, done=done, discount=discount)
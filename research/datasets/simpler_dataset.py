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
        # Keys to include from object
        self.obj_keys = ['source_obj_pose', 'target_obj_pose', 'state', 'tcp_to_source_obj_pos']
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

    def _data_generator(self):
        # Compute the worker info
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        dataset = self._load_dataset()

        for episode in dataset:
            # Need dummy transition at start
            obs_list = []
            action_list = [self.dummy_action]
            reward_list = [0.0]
            discount_list = [1.0]
            done_list = [False]

            for step_ind, step in enumerate(episode['steps']):
                # Skip dummy step from RLDS
                if step_ind == len(episode['steps']) - 1:
                    break
                observation = step['observation']
                observation = {k: v.numpy() for k, v in observation.items() if k in self.obj_keys}
                action = step['action'].numpy()
                reward = step['reward'].numpy()
                discount = step['discount'].numpy()
                done = False

                obs_list.append(observation)
                action_list.append(action)
                reward_list.append(reward)
                discount_list.append(discount)
                done_list.append(done)

            # Add extra dummy observation (this doesn't get used)
            obs_list.append(obs_list[-1])
            done_list[-1] = True

            # Convert to numpy arrays
            obs = utils.concatenate(*utils.unsqueeze(obs_list, 0))
            action = np.array(action_list)
            reward = np.array(reward_list)
            discount = np.array(discount_list)
            done = np.array(done_list)

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
            
            obs_len = obs[next(iter(obs.keys()))].shape[0]
            assert all([len(obs[k]) == obs_len for k in obs.keys()])
            assert obs_len == len(action) == len(reward) == len(done) == len(discount)

            yield dict(obs=obs, action=action, reward=reward, done=done, discount=discount)
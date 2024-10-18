import numpy as np
import tensorflow as tf
from transforms3d.euler import euler2axangle
import gym

from experiments.utils import TemporalEnsembleWrapper, normalize_gripper_action
import simpler_env

def convert_maniskill(action):
    """
    Applies transforms to raw VLA action that Maniskill simpler_env env expects.
    Converts rotation to axis_angle.
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1] and binarizes.
    """
    assert action.shape[0] == 7

    # Change rotation to axis-angle
    action = action.copy()
    roll, pitch, yaw = action[3], action[4], action[5]
    action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
    action[3:6] = action_rotation_ax * action_rotation_angle

    # Binarize final gripper dimension & map to [-1...1]
    return normalize_gripper_action(action)


class SimplerEnvRLDSWrapper(gym.Wrapper):
    """Wraps observation to be compatible w/ RLDS, for a single Simpler task at a time.
    Will also follow this up with some number of init steps if specified (num_init_steps), before returning.
    """
    
    def __init__(self,
                 task,
                 initial_states_path=None,
                 resize_size=224):
        self.env = simpler_env.make(task)
        self.resize_size = resize_size
        self.episode_idx = 0
        if initial_states_path == "eval":
            self.seed = 999
        elif initial_states_path == "train":
            self.seed = -1
        else:
            raise ValueError("Unsupported initial states path")

    def _wrap_obs(self, obs):

        # Generate action with model.
        return {
            "state": obs["extra"]["tcp_pose"],
            "source_obj_pose": obs["extra"]["source_obj_pose"],
            "target_obj_pose": obs["extra"]["target_obj_pose"],
            "tcp_to_source_obj_pos": obs["extra"]["tcp_to_source_obj_pos"],
        }

    def step(self, action):
        action = convert_maniskill(action.copy())
        obs, reward, done, truncated, info = self.env.step(action)
        return self._wrap_obs(obs), float(reward), done, info

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        else:
            self.seed += 1

        obs, _ = self.env.reset(seed=self.seed)

        # wrap the resulting obs
        return self._wrap_obs(obs)


def get_simpler_env(task, model_family, initial_states_path=None, resize_size=224, num_init_steps=0):
    """Initializes and returns the Simpler environment along with the task description."""
    assert num_init_steps == 0, "SimplerEnv init steps not yet supported"
    env = simpler_env.make(task)
    env = SimplerEnvRLDSWrapper(env, initial_states_path=initial_states_path, resize_size=resize_size)
    # (For Octo only) Wrap the robot environment.
    if model_family == "octo":
        env = TemporalEnsembleWrapper(env, pred_horizon=4)
    return env


def get_simpler_dummy_action(model_family: str):
    if model_family == "octo":
        # TODO: don't hardcode the action horizon for Octo
        return np.tile(np.array([0, 0, 0, 0, 0, 0, -1])[None], (4, 1))
    else:
        return np.array([0, 0, 0, 0, 0, 0, -1])
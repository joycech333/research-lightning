from typing import Optional
import imageio
from PIL import Image, ImageDraw

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
                 resize_size=224,
                 terminate_early: bool = True,
                 horizon: Optional[int] = 60):
        self.env = simpler_env.make(task)
        self.resize_size = resize_size
        self.episode_idx = 0

        self.step_counter = 0 # Temporary counter for tracking steps
        self.frames = [] # Temporary storage for producing gifs

        if initial_states_path == "eval":
            self.seed = 999
        elif initial_states_path == "train":
            self.seed = -1
        else:
            raise ValueError("Unsupported initial states path")
        self.env.ignore_done = False
        if horizon is not None:
            self.env.horizon = horizon
        self.env._max_episode_steps = self.env.horizon
        self.terminate_early = terminate_early

    def _wrap_obs(self, obs):

        # Generate action with model.
        return {
            "state": obs["extra"]["tcp_pose"],
            "source_obj_pose": obs["extra"]["source_obj_pose"],
            "target_obj_pose": obs["extra"]["target_obj_pose"],
            "tcp_to_source_obj_pos": obs["extra"]["tcp_to_source_obj_pos"],
        }

    def step(self, action):
        self.step_counter += 1
        action = convert_maniskill(action.copy())
        obs, reward, done, truncated, info = self.env.step(action)

        # Capture the current observation as an image frame for the video
        """
        im = obs['image']['3rd_view_camera']['rgb']
        pil_im = Image.fromarray(im).resize((640, 640))
        draw = ImageDraw.Draw(pil_im)
        draw.text((10, 10), f'STEP {self.step_counter} | REWARD {reward:.2f}')
        draw.text((10, 30), str(list(action))) 
        self.frames.append(np.array(pil_im))
        """
        
        if self.terminate_early and self.step_counter >= self.env.horizon:
            done = True
            # print(f"Episode {self.episode_idx} has ended at step {self.step_counter}")

        """
        print("\nConverted action: ", action, " Done: ", done)
        print("\nRaw Observation: ", obs['extra'])
        print("\nWrapped Observation: ", self._wrap_obs(obs))

        # If episode done, save video
        if done:
            video_filename = f'rewards_carrot_vid{self.episode_idx}.mp4'
            imageio.mimwrite(video_filename, self.frames, fps=10)
            print(f"Video saved as {video_filename}")
        """

        return self._wrap_obs(obs), float(reward), done, info

    def reset(self, seed=None):
        self.step_counter = 0
        self.episode_idx += 1
        self.frames = []
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

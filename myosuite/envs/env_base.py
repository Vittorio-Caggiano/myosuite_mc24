"""=================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================="""

import os
import time as timer
from sys import platform
from typing import Optional

import mujoco
import numpy as np
import skvideo.io

import myosuite.utils.import_utils as import_utils
from myosuite.envs.env_variants import gym_registry_specs
from myosuite.envs.myo.myoedits.model_editor import ModelEditor
from myosuite.envs.obs_vec_dict import ObsVecDict
from myosuite.renderer.mj_renderer import MJRenderer
from myosuite.robot.robot import Robot
from myosuite.utils import gym, seed_envs, tensor_utils
from myosuite.utils.implement_for import implement_for
from myosuite.utils.prompt_utils import Prompt, prompt

# TODO
# remove rwd_mode
# convert obs_keys to obs_keys_wt
# batch images before passing them through the encoder
# should path methods(compute_path_rewards, truncate_paths, evaluate_success) be moved to paths_utils?


class MujocoEnv(gym.Env, gym.utils.EzPickle, ObsVecDict):
    """
    Superclass for all MuJoCo environments.
    """

    DEFAULT_CREDIT = """\
        MyoSuite: a collection of environments/tasks to be solved by musculoskeletal models | https://sites.google.com/view/myosuite
        Code: https://github.com/MyoHub/myosuite/stargazers (add a star to support the project)
    """

    def __init__(
        self,
        model_path,
        obsd_model_path=None,
        seed=None,
        edit_fn=None,
        env_credits=DEFAULT_CREDIT,
    ):
        """
        Create a gym env
        INPUTS:
            model_path: ground truth model
            obsd_model_path : observed model (useful for partially observed envs)
                            : observed model (useful to propagate noisy sensor through env)
                            : use model_path; if None
            seed: Random number generator seed

        """

        prompt("MyoSuite:> For environment credits, please cite -")
        prompt(env_credits, color="cyan", type=Prompt.ONCE)

        # Seed and initialize the random number generator
        self.seed(seed)
        self.model_path = model_path

        self.mj_spec: Optional[mujoco.MjSpec] = None
        if isinstance(model_path, str):
            self.mj_spec = self._get_spec(model_path, edit_fn)
            self.mj_model = self.mj_spec.compile()
        else:
            self.mj_model = model_path
        self.mj_data = mujoco.MjData(self.mj_model)

        self.obsd_mj_spec: Optional[mujoco.MjSpec] = None
        if obsd_model_path:
            if isinstance(obsd_model_path, str):
                self.obsd_mj_spec = self._get_spec(obsd_model_path, edit_fn)
                self.obsd_mj_model = self.obsd_mj_spec.compile()
            else:
                self.obsd_mj_model = obsd_model_path
            self.obsd_mj_data = mujoco.MjData(self.obsd_mj_model)
        else:
            self.obsd_mj_model = self.mj_model
            self.obsd_mj_data = self.mj_data

        self.mj_renderer = MJRenderer(self.mj_model, self.mj_data)

        mujoco.mj_forward(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.obsd_mj_model, self.obsd_mj_data)

        ObsVecDict.__init__(self)

    def _get_spec(self, model_path, edit_fn):
        if edit_fn is not None:
            # Load the model
            model_editor = ModelEditor(model_path)
            # Edit the model using an edit_fn
            # TODO: Reformat to be functional instead of creating side effect
            model_editor.edit_model(edit_fn)
            model_spec = model_editor.spec
        else:
            model_spec = mujoco.MjSpec.from_file(model_path)
        return model_spec

    def _setup(
        self,
        obs_keys: dict,  # Keys from obs_dict that forms the obs vector returned by get_obs()
        weighted_reward_keys: dict,  # Keys and weight that sums up to build the reward
        proprio_keys: list = None,  # Keys from obs_dict that forms the proprioception vector returned by get_proprio()
        visual_keys: list = None,  # Keys that specify visual_dict returned by get_visual()
        reward_mode: str = "dense",  # Configure env to return dense/sparse rewards
        frame_skip: int = 1,  # Number of mujoco frames/steps per env step
        normalize_act: bool = True,  # Ask env to normalize the action space
        obs_range: tuple = (
            -10,
            10,
        ),  # Permissible range of values in obs vector returned by get_obs()
        rwd_viz: bool = False,  # Visualize rewards (WIP, needs vtils)
        device_id: int = 0,  # Device id for rendering
        **kwargs,  # Additional arguments
    ):

        if self.mj_model is None or self.mj_data is None:
            raise TypeError(
                "mj_model and mj_data must be instantiated for setup to run"
            )

        # Resolve viewer
        self.mujoco_render_frames = False
        self.device_id = device_id
        self.rwd_viz = rwd_viz
        self.viewer_setup()

        # resolve robot config
        self.robot = Robot(
            mj_model=self.mj_model, random_generator=self.np_random, **kwargs
        )
        
        # FIXED: Synchronize MjData - make env.mj_data point to robot.mj_data to avoid duplication
        # This ensures that all updates to mj_data are consistent across env and robot
        self.mj_data = self.robot.mj_data
        
        # Also sync observed data if it was using the main data
        if self.obsd_mj_data is self.mj_data:
            self.obsd_mj_data = self.robot.mj_data

        # resolve action space
        self.frame_skip = frame_skip
        self.normalize_act = normalize_act
        act_low = (
            -np.ones(self.mj_model.nu)
            if self.normalize_act
            else self.mj_model.actuator_ctrlrange[:, 0].copy()
        )
        act_high = (
            np.ones(self.mj_model.nu)
            if self.normalize_act
            else self.mj_model.actuator_ctrlrange[:, 1].copy()
        )
        self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

        # resolve initial state
        self.init_qvel = self.mj_data.qvel.ravel().copy()
        self.init_qpos = (
            self.mj_data.qpos.ravel().copy()
        )  # has issues with initial jump during reset
        # self.init_qpos = np.mean(self.mj_model.actuator_ctrlrange, axis=1) if self.normalize_act else self.mj_data.qpos.ravel().copy() # has issues when nq!=nu
        # self.init_qpos[self.mj_model.jnt_dofadr] = np.mean(self.mj_model.jnt_range, axis=1) if self.normalize_act else self.mj_data.qpos.ravel().copy()
        if self.normalize_act:
            # find all linear+actuated joints. Use mean(jnt_range) as init position
            actuated_jnt_ids = self.mj_model.actuator_trnid[
                self.mj_model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT, 0
            ]  # dm
            linear_jnt_ids = np.logical_or(
                self.mj_model.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE,
                self.mj_model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE,
            )
            linear_jnt_ids = np.where(linear_jnt_ids == True)[0]
            linear_actuated_jnt_ids = np.intersect1d(actuated_jnt_ids, linear_jnt_ids)
            # assert np.any(actuated_jnt_ids==linear_actuated_jnt_ids), "Wooho: Great evidence that it was important to check for actuated_jnt_ids as well as linear_actuated_jnt_ids"
            linear_actuated_jnt_qposids = self.mj_model.jnt_qposadr[
                linear_actuated_jnt_ids
            ]
            self.init_qpos[linear_actuated_jnt_qposids] = np.mean(
                self.mj_model.jnt_range[linear_actuated_jnt_ids], axis=1
            )

        # resolve rewards
        self.rwd_dict = {}
        self.rwd_mode = reward_mode
        self.rwd_keys_wt = weighted_reward_keys

        # resolve observation keys
        self.obs_dict = {}
        self.obs_keys_wt = obs_keys
        self.proprio_keys = proprio_keys
        self.visual_keys = visual_keys
        self.obs_range = obs_range

        # Number of environment step
        self.env_step = 0

        # register for possible variants
        if hasattr(self.spec, "id"):
            gym_registry_specs[self.spec.id] = self.spec

        # Seeding
        self.seed(seed=None)

        print("MyoSuite:> Environment initialized using envs_base.py")

    def _seed(self, seed):
        if seed is None:
            seed = np.random.randint(low=1, high=2**32)
        self.seed_val = seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    seed = implement_for("gymnasium>=0.25", "<=0.26.0", _seed)

    def _get_obs_dict(self):
        """
        obs_dict: dictionary of all user defined observables for the env
        Note: get_obs() call must always come before the get_visual()
        This order is important as camera renders depend on obs_dict computations
        """
        obs_dict = {}
        return obs_dict

    def get_obs(self):
        """
        Return environment observation (vector)
        """
        self.obs_dict = self._get_obs_dict()  # get obs_dict
        t, obs = self.obsdict2obsvec(
            self.obs_dict, self.obs_keys_wt
        )  # get obs_vector
        return obs

    def get_obs_vec(self):
        """
        Return environment observation (vector)
        NOTE: identical to get_obs() but provided for clarity
        """
        return self.get_obs()

    def get_obs_dict(self):
        """
        Return environment observations (dictionary)
        """
        obs_dict = self._get_obs_dict()  # get obs_dict
        return obs_dict

    def get_proprio(self):
        """
        Return proprioceptive information
        """
        if self.proprio_keys:
            self.obs_dict = self._get_obs_dict()  # get obs_dict
            _, proprio = self.obsdict2obsvec(
                self.obs_dict, self.proprio_keys
            )  # get obs_vector
        else:
            proprio = self.get_obs()
        return proprio

    def get_visual(self, visual_keys=None):
        """
        Return visual observations
        NOTE: Following get_obs() call is very important as camera renders depend on obs_dict computations
        """
        visual_keys = visual_keys if visual_keys else self.visual_keys
        visual_dict = {}
        return visual_dict

    def is_done(self):
        """
        Returns if environment (episode) is done (bool). This could be done due to termination or truncation
        This function is implemented seperately from is_terminate() and is_truncate() with following requirements:
            - output from is_done() is equivalent to: is_terminate() or is_truncate()
            - is_done() could be efficient by directly using single condition instead of evaluating two conditions via the above equation.
            - is_done()/is_terminate()/is_truncate() are consistent
        """
        return False

    def is_terminate(self):
        """
        Returns if environment episode is terminated (bool). Termination is when task requirements have changed
        """
        return False

    def is_truncate(self):
        """
        Returns if environment episode is truncated (bool). Truncation is when task has timed-out
        """
        return False

    def reset_model(self):
        """
        resets model/ mujoco data
        """
        # resolve init pos
        init_qpos = self.init_qpos.copy()
        if hasattr(self, "init_qpos_noise"):
            if self.init_qpos_noise > 0:
                self.np_random.uniform(
                    low=-self.init_qpos_noise,
                    high=self.init_qpos_noise,
                    size=init_qpos.shape[0],
                )
                init_qpos = init_qpos + self.np_random.uniform(
                    low=-self.init_qpos_noise,
                    high=self.init_qpos_noise,
                    size=init_qpos.shape[0],
                )

        # resolve init vel
        init_qvel = self.init_qvel.copy()
        if hasattr(self, "init_qvel_noise"):
            if self.init_qvel_noise > 0:
                init_qvel = init_qvel + self.np_random.uniform(
                    low=-self.init_qvel_noise,
                    high=self.init_qvel_noise,
                    size=init_qvel.shape[0],
                )

        # set pos/vel
        self.robot.sync_sims(self.mj_model, self.mj_data)
        self.mj_data.qpos[:] = init_qpos
        self.mj_data.qvel[:] = init_qvel
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # update times
        self.env_step = 0
        return self.get_obs()

    def reset(self, seed=None, options=None):
        obs = self.reset_model()

        # TODO: Move back to  new gym paradigm
        if seed is None:
            return obs
        else:
            return obs, {}

    def get_reward_dict(self):
        """
        Compute rewards for environment
        """
        rwd_dict = {}
        return rwd_dict

    def is_success(self, paths):
        """
        Returns success status of path
        """
        return 0

    def step(self, a):
        """
        env step
        """
        pre_physics_time = self.mj_data.time
        pre_physics_step = self.env_step

        # apply action and step physics + robot
        for _ in range(self.frame_skip):
            if self.normalize_act:
                a = tensor_utils.denormalize(
                    a,
                    self.mj_model.actuator_ctrlrange[:, 0],
                    self.mj_model.actuator_ctrlrange[:, 1],
                )
            self.robot.step(a, self.mj_model, self.mj_data)
            mujoco.mj_step(self.mj_model, self.mj_data)

        # update times
        self.env_step += 1

        # observation is computed after simtime is incremented by the physics step
        obs = self.get_obs()

        # reward is computed after obs
        self.rwd_dict = self.get_reward_dict()
        reward = self.rwd_dict["rwd_total"] if "rwd_total" in self.rwd_dict.keys() else 0.0
        reward = reward.item() if hasattr(reward, 'item') else reward  # handle numpy scalars

        # termination and truncation
        done = self.is_done()
        terminated = self.is_terminate()
        truncated = self.is_truncate()

        # new gym paradigm, check #231
        info = dict(rwd_dict=self.rwd_dict,
                    obs_dict=self.obs_dict,
                    terminated=terminated,
                    truncated=truncated,
                    done=done)
        return obs, reward, done, info

    # Close env
    def close(self):
        if hasattr(self, "mj_renderer") and self.mj_renderer:
            self.mj_renderer.close()

    def viewer_setup(self):
        """
        Setup viewer for environments with visualization support
        """
        pass

    def set_camera(self, camera_name="cam_01"):
        """
        Set viewer to camera given camera name
        """
        pass

    def set_rgb_from_camera(self, img):
        img = np.flipud(img)
        self.viewer.display_image(img)

    def evaluate_success(self, paths, logger=None):
        """
        Evaluate paths and log metrics to logger.
        """
        num_success = 0
        num_paths = len(paths)

        # Record success if solved
        for path in paths:
            if np.mean(path["env_infos"]["solved"]) > 0.0:
                num_success += 1
        success_percentage = num_success * 100.0 / num_paths

        # Log metrics
        if logger:
            logger.log_kv("success_percentage", success_percentage)

        return success_percentage

    # record video of an episode
    def record_episode(self, episode_length=None, out_dir=None, prefix="MyoSuite_"):
        if out_dir is None:
            out_dir = os.getcwd()
        out_path = os.path.join(out_dir, prefix + str(timer.time()))

        if episode_length is None:
            episode_length = self.env_horizon

        frames = []
        obs = self.reset()
        for i in range(episode_length):
            self.env.render()
            frames.append(self.get_offscreen_rgb())
            obs, *_ = self.step(self.action_space.sample())

        skvideo.io.vwrite(out_path, np.asarray(frames), inputdict={"-r": str(1 / 0.01)})

    def get_offscreen_rgb(self, camera_name=None):
        # option1: opencv rendering (computationally slow)
        # from myosuite.renderer.recorder import opencv_renderer
        # offscreen_rgb = opencv_renderer.opencv_render(model=self.mj_model, data=self.mj_data, height=256, width=256, camera_name=camera_name)

        # option2: mujoco rendering (optimized)
        return self.mj_renderer.render_offscreen_rgb(camera_name=camera_name)
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
        
        # Create temporary mj_data for initialization - will be replaced by robot's mj_data in _setup
        self._temp_mj_data = mujoco.MjData(self.mj_model)

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
            self.obsd_mj_data = self._temp_mj_data

        self.mj_renderer = MJRenderer(self.mj_model, self._temp_mj_data)

        mujoco.mj_forward(self.mj_model, self._temp_mj_data)
        mujoco.mj_forward(self.obsd_mj_model, self.obsd_mj_data)

        ObsVecDict.__init__(self)

    @property
    def mj_data(self):
        """
        Return the robot's mj_data if available, otherwise return temporary data.
        This ensures env.mj_data and env.robot.mj_data point to the same reference.
        """
        if hasattr(self, 'robot') and self.robot is not None:
            return self.robot.mj_data
        else:
            return self._temp_mj_data

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

        if self.mj_model is None or self._temp_mj_data is None:
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

        # Update renderer to use robot's mj_data
        self.mj_renderer = MJRenderer(self.mj_model, self.mj_data)

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

        # resolve initial state using robot's mj_data
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

        # resolve observations
        self.obs_dict = {}
        self.obs_keys_wt = obs_keys
        self.obs_range = obs_range
        self.proprio_keys = proprio_keys
        self.visual_keys = visual_keys
        self.observation_space = self._get_obs_space()

        # resolve rwd viz
        if self.rwd_viz:
            import myosuite.utils.viz_utils as viz_utils

            self.rwd_viz = viz_utils.RwdViz(self)

        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed)

        # finalize setup
        self.reset()

    def step(self, a, **kwargs):
        """
        Apply action, return observation, reward, done, info
        """

        # apply action and step the sim
        for _ in range(self.frame_skip):
            if self.robot.is_hardware == True:
                self.robot.apply_commands(a, **kwargs)
                self.mj_data.qpos[:] = self.robot.get_sensors()["qpos"]
                self.mj_data.qvel[:] = self.robot.get_sensors()["qvel"]
                self.mj_data.act[:] = self.robot.get_sensors()["act"]
            else:
                self.robot.apply_commands(a, **kwargs)
                # step the sim forward
                mujoco.mj_step(self.mj_model, self.mj_data)

        # update time
        self.time_wall = timer.time() - self.time_start
        if self.robot.is_hardware == True:
            self.mj_data.time = self.time_wall  # hardware is real-time

        # observation
        obs = self.get_obs()

        # rewards
        self.expand_dims(
            self.obs_dict
        )  # required incase any key lacks a time axis. rwd_dict can be tricky
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        reward, rwd_info = self.get_reward(self.rwd_dict)

        # finalize step
        env_info = self.get_env_infos()
        env_info.update(rwd_info)

        # determine if done
        done = env_info["done"] if "done" in env_info.keys() else False

        return obs, reward, done, env_info

    def reset(
        self,
        reset_qpos=None,
        reset_qvel=None,
        reset_act=None,
        time_period=(0, 0),
        **kwargs
    ):
        """
        Reset the environment
        Default implemention provided. Override if custom reset needed.
        """
        # sample from time period and initialize accordingly
        time_sampled = (
            self.np_random.uniform(high=time_period[1], low=time_period[0])
            if time_period[1] > time_period[0]
            else 0.0
        )

        # set state
        qpos = self.init_qpos.copy() if reset_qpos is None else reset_qpos.copy()
        qvel = self.init_qvel.copy() if reset_qvel is None else reset_qvel.copy()
        self.robot.reset(reset_pos=qpos, reset_vel=qvel, **kwargs)

        # Set act
        if reset_act is not None:
            self.mj_data.act[:] = reset_act

        # forward the model to refresh data
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Take additional steps to initialize the simulation properly before the new episode begins
        if time_sampled > 0:
            for _ in range(int(time_sampled / self.mj_model.opt.timestep)):
                mujoco.mj_step(self.mj_model, self.mj_data)

        # observation
        obs = self.get_obs()

        # seed action space
        if hasattr(self, "action_space"):
            if hasattr(self.action_space, "seed"):
                self.action_space.seed(self.np_random.randint(0, 2**32 - 1))

        return obs

    def viewer_setup(self):
        """
        Due to specifics of new mujoco rendering, the viewer is setup
        by HumanRendering plugin.
        """
        pass

    def close(self):
        if self.mj_renderer:
            self.mj_renderer.close()

    # Utilities ===============================================================

    def mj_render(self):
        try:
            self.mj_renderer.render()
        except:
            print("WARNING: Rendering failed")

    def get_env_state(self):
        """
        Get full state of the environemnt
        Default implemention provided. Override if env has custom state
        """
        qp = self.mj_data.qpos.ravel().copy()
        qv = self.mj_data.qvel.ravel().copy()
        act = self.mj_data.act.ravel().copy() if self.mj_model.na > 0 else None
        mocap_pos = (
            self.mj_data.mocap_pos.ravel().copy()
            if self.mj_model.nmocap > 0
            else None
        )
        mocap_quat = (
            self.mj_data.mocap_quat.ravel().copy()
            if self.mj_model.nmocap > 0
            else None
        )
        return dict(
            qpos=qp, qvel=qv, act=act, mocap_pos=mocap_pos, mocap_quat=mocap_quat
        )

    def set_env_state(self, state_dict):
        """
        Set full state of the environemnt
        Default implemention provided. Override if env has custom state
        """
        qp = state_dict["qpos"]
        qv = state_dict["qvel"]
        act = state_dict["act"]
        mocap_pos = state_dict["mocap_pos"]
        mocap_quat = state_dict["mocap_quat"]
        self.mj_data.qpos[:] = qp
        self.mj_data.qvel[:] = qv
        if act is not None:
            self.mj_data.act[:] = act
        if mocap_pos is not None:
            self.mj_data.mocap_pos[:] = mocap_pos.reshape((-1, 3))
        if mocap_quat is not None:
            self.mj_data.mocap_quat[:] = mocap_quat.reshape((-1, 4))
        mujoco.mj_forward(self.mj_model, self.mj_data)

    # methods to override:
    def get_reward_dict(
        self,
        obs_dict: dict,
    ) -> dict:
        """
        Compute the reward components
        Args:
            obs_dict: current observation dictionary
        Returns:
            rwd_dict: dictionary of reward components
        """
        rwd_dict = {}
        # Example reward component:
        # rwd_dict['alive'] = np.array([1.0])
        return rwd_dict

    def get_randomization_dict(
        self,
    ) -> dict:
        """
        Compute the randomization dict
        Returns:
            randomization_dict: dictionary of DR variables
        """
        rnd_dict = {}
        # Example randomization:
        # rnd_dict['gravity'] = self.np_random.uniform(-10, -5)
        return rnd_dict

    def apply_randomization_dict(self, rnd_dict):
        """
        Apply the randomization to the env
        Args:
            rnd_dict: dictionary of randomization variables
        """
        # Example:
        # self.mj_model.opt.gravity = rnd_dict['gravity']
        pass

    def get_obs_dict(self, sim):
        """
        Get observation dictionary
        Args:
            sim: simulation object
        Returns:
            obs_dict: observation dictionary
        """
        # This method should be implemented by each specific env
        return {}

    def get_obs_vec(self):
        """
        Get observation vector
        Returns:
            obs_vec: ndarray of observations
        """
        self.obs_dict = self.get_obs_dict(self.mj_data)

        # process position based observations
        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys_wt)
        return obs

    def get_obs(self):
        """
        Get observation.
        Default implemention provided. Override if custom logic needed
        """
        obs_vec = self.get_obs_vec()
        obs_vec = np.clip(obs_vec, self.obs_range[0], self.obs_range[1])
        return obs_vec

    def get_reward(self, rwd_dict):
        """
        Compute reward from reward dictionary
        """
        rwd, info = self.rwd_dict2rwd_vec(rwd_dict, self.rwd_keys_wt, self.rwd_mode)
        return rwd, info

    def get_metrics(self, paths, **kwargs):
        """
        Evaluate rollouts and report metrics
        """
        # finalize paths
        if len(paths) == 0:
            return {}

        # compute metrics
        score = np.mean([np.sum(p["rewards"]) for p in paths])
        points = np.mean([np.sum(p["rewards"]) for p in paths])
        metrics = dict(
            score=score,
            points=points,
            num_paths=len(paths),
        )
        return metrics

    def get_env_infos(self):
        """
        Get information about the environment
        """
        env_info = {
            "time": self.mj_data.time,
            "rwd_dict": self.rwd_dict.copy(),
            "obs_dict": self.obs_dict.copy(),
        }
        return env_info

    @implement_for("gym", None, "0.24")  # gym<=0.24
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    @implement_for("gym", "0.25", None)  # gym>=0.25
    def seed(self, seed=None):
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
            return [seed]
        else:
            return []

    def render(
        self,
        mode="human",
        width=640,
        height=480,
        camera_id=-1,
        device_id=0,
    ):

        if mode in {"human", "rgb_array"}:
            if self.mj_renderer is None:
                self.mj_renderer = MJRenderer(self.mj_model, self.mj_data)

            self.mj_renderer.render(
                render_mode=mode,
                width=width,
                height=height,
                camera_id=camera_id,
                device_id=device_id,
            )
            if mode == "rgb_array":
                img = self.mj_renderer.get_pixels()
                return img

        elif mode == "rgb_array_list":
            return self.mujoco_render_frames

        else:
            raise ValueError(
                "mode must be either 'human', 'rgb_array', or 'rgb_array_list'"
            )

    def take_video(
        self,
        policy,
        filepath="myosuite_video",
        frame_size=(640, 480),
        camera_id=-1,
        device_id=0,
    ):
        """
        Take a video rollout using a policy
        """

        # Initialize video writer
        fps = int(1 / self.dt)
        fourcc = skvideo.io.vwriter.VideoWriter(
            "{}.mp4".format(filepath),
            outputdict={"-vcodec": "libx264", "-r": str(fps)},
        )

        # Take rollout
        self.mujoco_render_frames = True
        self.reset()
        frame = self.render(
            mode="rgb_array",
            width=frame_size[0],
            height=frame_size[1],
            camera_id=camera_id,
            device_id=device_id,
        )
        fourcc.writeFrame(frame)

        done = False
        while not done:
            action = policy(self.get_obs())
            self.step(action)
            frame = self.render(
                mode="rgb_array",
                width=frame_size[0],
                height=frame_size[1],
                camera_id=camera_id,
                device_id=device_id,
            )
            fourcc.writeFrame(frame)

            # env info
            done = self.get_env_infos()["done"] if "done" in self.get_env_infos() else False

        self.mujoco_render_frames = False
        fourcc.close()

    # Internal methods ========================================================

    # Seed and initialize the random number generator
    @property
    def dt(self):
        return self.mj_model.opt.timestep * self.frame_skip

    @property
    def random_generator(self):
        return self.np_random
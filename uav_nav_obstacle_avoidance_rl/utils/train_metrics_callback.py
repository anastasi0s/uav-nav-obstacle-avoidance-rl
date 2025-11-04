from typing import Dict, Optional

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

from uav_nav_obstacle_avoidance_rl import config

logger = config.logger


class TrainMetricsCallback(BaseCallback):
    """
    W&B callback that logs custom UAV metrics and standard SB3 metrics
    """

    def __init__(
        self,
        run_path: Optional[str] = None,
        model_save_path: Optional[str] = None,
        model_save_freq: int = 0,
        verbose: int = 0,
    ):
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        super().__init__(verbose)
        self.model_save_path = model_save_path
        self.model_save_freq = model_save_freq
        self.run_path = run_path

        # storage for whole train-run metrics
        self.train_run_metrics = {
            "success": [],
            "collision": [],
            "out_of_bounds": [],
        }

        # current episode tracking
        self.current_episode_data = {}

    def _on_training_start(self) -> None:
        """called at the start of training"""
        if self.verbose >= 1:
            print("Starting W&B UAV metrics collection...")

        # init episode tracking for each env
        n_envs = self.training_env.num_envs
        for i in range(n_envs):
            self._reset_current_episode(i)

    def _reset_current_episode(self, env_idx: int):
        """reset tracking for the next episode"""
        self.current_episode_data[env_idx] = {
            "positions": [],
            "velocities": [],
            "start_position": None,
            "target_position": None,
            "start_time": self.num_timesteps,
        }

    def _on_step(self) -> bool:
        """called after each environment step"""
        try:
            # extract step information
            done = self.locals["dones"]
            info = self.locals["infos"]  # collected by the Monitor wrapper

            # identify env
            if hasattr(self.training_env, "envs"):
                envs = self.training_env.envs
            else:
                envs = [self.training_env]

            for env_idx, env in enumerate(envs):
                # find the underlying UAV environment - peel wrappers to find the VectorVoyagerEnv
                underlying_env = env.unwrapped  # my custom env

                state = underlying_env.env.state(0)  # state from Aviary env
                lin_pos = state[3]  # [x, y, z] position
                lin_vel = state[2]  # [u, v, w] velocity

                self.current_episode_data[env_idx]["positions"].append(lin_pos.copy())
                self.current_episode_data[env_idx]["velocities"].append(lin_vel.copy())

                # set start position if first step
                if self.current_episode_data[env_idx]["start_position"] is None:
                    self.current_episode_data[env_idx]["start_position"] = (
                        lin_pos.copy()
                    )

                # get target position
                self.current_episode_data[env_idx]["target_position"] = (
                    underlying_env.waypoints.targets[0].copy()
                )  # NOTE adjust to multiple waypoints

            # process completed episode
            for env_idx, (done, info) in enumerate(zip(done, info)):
                if done:
                    self._process_completed_episode(info, env_idx)

        except Exception as e:
            logger.exception(f"Error in TrainMetrics collection step: {env_idx}: {e}")

        return True

    def _process_completed_episode(self, info: Dict, env_idx: int = 0):
        """
        process a completed episode and log metrics to W&B
        - episode_info keys=['out_of_bounds', 'collision', 'env_complete', 'num_targets_reached', 'TimeLimit.truncated', 'episode']
        - 'episode' is dic and contains logged info from the Monitor wrapper: r, l, t by default, plus any additional logged information (in this case)
        """
        try:
            # extract episode results from Monitor wrapper
            out_of_bounds = info["out_of_bounds"]
            collision = info["collision"]
            env_complete = info["env_complete"]
            targets_reached = info["num_targets_reached"]
            episode_reward = info["episode"]["r"]
            episode_length = info["episode"]["l"]

            # check for success = all targets reached, without collisions
            success = env_complete and not collision and not out_of_bounds

            # calculate custom metrics
            path_length = self._calculate_path_length(
                self.current_episode_data[env_idx]["positions"]
            )
            mean_velocity = self._calculate_average_velocity(
                self.current_episode_data[env_idx]["velocities"]
            )
            # calculate path efficiency only on successful episodes where the target is reached
            if success:
                # calculate path efficiency only on successful episodes where the target is reached
                path_efficiency = self._calculate_path_efficiency(
                    self.current_episode_data[env_idx]["start_position"],
                    self.current_episode_data[env_idx]["target_position"],
                    path_length,
                )
            else:
                # unsuccessful episodes (for consistent array length in self.episode_history)
                path_efficiency = 0.0

            # store in run-level episode history
            self.train_run_metrics["success"].append(int(success))
            self.train_run_metrics["collision"].append(int(collision))
            self.train_run_metrics["out_of_bounds"].append(int(out_of_bounds))

            # calculate performance ratios using run-level storage
            success_rate = np.mean(self.train_run_metrics["success"][-20:])
            collision_rate = np.mean(self.train_run_metrics["collision"][-20:])
            out_of_bounds_rate = np.mean(self.train_run_metrics["out_of_bounds"][-20:])

            # log metrics to W&B
            episode_metrics = {
                # performance ratios
                "train_uav/success_rate_rolling_20ep": success_rate,
                "train_uav/collision_rate_rolling_20ep": collision_rate,
                "train_uav/out_of_bounds_rate_rolling_20ep": out_of_bounds_rate,
                "train_uav/targets_reached": targets_reached,
                # movement metrics
                "train_uav/mean_velocity": mean_velocity,
                "train_uav/path_length": path_length,
                "train_uav/path_efficiency": path_efficiency,
                # episode basics
                "train_uav/episode_reward": episode_reward,
                "train_uav/episode_length": episode_length,
            }

            wandb.log(episode_metrics, step=self.num_timesteps)

            # reset episode tracking
            self._reset_current_episode(env_idx)

        except Exception as e:
            logger.exception(f"Error processing episode metrics: {e}")

    def _calculate_path_length(self, positions) -> float:
        """calculate total distance traveled"""
        if len(positions) < 2:
            return 0.0
        positions = np.array(positions)
        distances = np.linalg.norm(
            np.diff(positions, axis=0), axis=1
        )  # calc. difference between postions -> calc. norm (distance) of difference -> sum all distances between positions -> complete length of traveled path
        return float(np.sum(distances))

    def _calculate_average_velocity(self, velocities) -> float:
        """Calculate average velocity magnitude"""
        if len(velocities) == 0:
            return 0.0
        velocities = np.array(velocities)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        return float(np.mean(velocity_magnitudes))

    def _calculate_path_efficiency(
        self, start_position, target_position, path_length: float
    ) -> float:
        # direct distance to target
        direct_distance = np.linalg.norm(
            target_position - start_position
        )  # NOTE adjust this for multiple waypoint targets
        if direct_distance == 0:
            return 0.0

        return path_length / direct_distance  # efficiency ratio

    def _on_training_end(self) -> None:
        """called at the end of training"""
        # save final model if path specified
        if self.model_save_path:
            model_path = f"{self.model_save_path}/final_model.zip"
            self.model.save(model_path)

            # save as W&B artifact
            artifact = wandb.Artifact("final_model", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

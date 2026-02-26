from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from uav_nav_obstacle_avoidance_rl import config

logger = config.logger


class CustomEvalCallback(EvalCallback):
    """
    Custom EvalCallback that collects detailed UAV metrics during evaluation.
    Extends the standard EvalCallback to collect the same metrics as training.
    Initialize evaluation cycle in the _on_step() function.
    """

    def __init__(self, *args, **kwargs):
        self.exp_analysis = kwargs.pop("exp_analysis", True)

        # initialize parent with remaining kwargs
        super().__init__(*args, **kwargs)

        self.rng = np.random.default_rng(config.RANDOM_SEED)

        # store reference to underlying env (unwrap), vectorvoyager
        if hasattr(self.eval_env, "envs"):
            self.underlying_env = self.eval_env.envs[0].unwrapped
        else:
            self.underlying_env = self.eval_env.unwrapped

        self.current_episode_data = {}

    def _on_step(self) -> bool:
        """
        override the parent _on_step to use our custom callback
        this is actually not called in every step but only on the last step of the training cycle, just before the evaluation starts.
        the actual evaluation with evaluate_policy() is calling _log_eval_callback() on each step.
        """
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    from stable_baselines3.common.vec_env import sync_envs_normalization

                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # reset storage before evaluation-cycle
            self.current_eval_cycle_data = {
                "success": [],
                "collision": [],
                "out_of_bounds": [],
                "env_complete": [],
                "num_targets": [],
                "targets_reached": [],
                "episode_rewards": [],
                "episode_lengths": [],
                "mean_velocities": [],
                "path_lengths": [],
                "path_efficiencies": [],
                "num_obstacles": [],
                "positions_history": [],  # for trajectory plotting (nested array)
                "velocities_history": [],  # for trajectory plotting (nested array)
                "target_position_history": [],  # for trajectory plotting (nested array)
                "obstacles_history": [],  # for trajectory plotting
            }
            self._is_success_buffer = []

            # START EVALUATION
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_eval_callback,  # evaluate_policy itself calls the callback function and passes locals() and globals() to it
            )

            # log detailed eval metrics to W&B
            self._log_after_evaluation()

            # # Continue with standard EvalCallback logging
            # if self.log_path is not None:
            #     assert isinstance(episode_rewards, list)
            #     assert isinstance(episode_lengths, list)
            #     self.evaluations_timesteps.append(self.num_timesteps)
            #     self.evaluations_results.append(episode_rewards)
            #     self.evaluations_length.append(episode_lengths)

            #     kwargs = {}
            #     # Save success log if present
            #     if len(self._is_success_buffer) > 0:
            #         self.evaluations_successes.append(self._is_success_buffer)
            #         kwargs = dict(successes=self.evaluations_successes)

            #     np.savez(
            #         self.log_path,
            #         timesteps=self.evaluations_timesteps,
            #         results=self.evaluations_results,
            #         ep_lengths=self.evaluations_length,
            #         **kwargs,
            #     )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            # mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            # self.last_mean_reward = float(mean_reward)

            # if self.verbose >= 1:
            #     print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            #     print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            # # Add to current Logger
            # self.logger.record("eval/mean_reward", float(mean_reward))
            # self.logger.record("eval/mean_ep_length", mean_ep_length)

            # if len(self._is_success_buffer) > 0:
            #     success_rate = np.mean(self._is_success_buffer)
            #     if self.verbose >= 1:
            #         print(f"Success rate: {100 * success_rate:.2f}%")
            #     self.logger.record("eval/success_rate", success_rate)

            # # Dump log so the evaluation results are printed with the correct timestep
            # self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            # self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    import os

                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                self.best_mean_reward = float(mean_reward)

                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def _log_eval_callback(self, locals_dict: dict, globals_dict: dict) -> None:
        """
        custom callback for evaluate_policy that collects detailed metrics
        this gets called after each step during evaluation
        """
        try:
            # extract step information
            done = locals_dict["done"]
            info = locals_dict["info"]  # collected by the Monitor wrapper

            # initialize episode tracking if needed
            if not self.current_episode_data:
                self._reset_current_episode()

            state = self.underlying_env.env.state(0)  # capture state from Aviary env
            lin_pos = state[3]  # [x, y, z] position
            lin_vel = state[2]  # [u, v, w] velocity

            self.current_episode_data["positions"].append(lin_pos.copy())
            self.current_episode_data["velocities"].append(lin_vel.copy())

            # at episode start ->
            if self.current_episode_data["start_position"] is None:
                # -> capture start position
                self.current_episode_data["start_position"] = lin_pos.copy()

                # -> capture obstacle info
                if hasattr(self.underlying_env, 'obstacles') and self.underlying_env.obstacles:
                    obstacles_data = []
                    for obs_id in self.underlying_env.obstacles:
                        try:
                            # capture obstacle position and orientation
                            pos, orn = self.underlying_env.env.getBasePositionAndOrientation(obs_id)
                            
                            # capture collision shape data
                            collision_shape_info = self.underlying_env.env.getCollisionShapeData(obs_id, -1)

                            if collision_shape_info:
                                shape_info = collision_shape_info[0]  # first (and usually only) collision shape
                                obstacles_data.append({
                                    'id': obs_id,
                                    'position': np.array(pos),
                                    'orientation': np.array(orn),
                                    'shape_type': shape_info[2],  # geometry type (GEOM_SPHERE=2, GEOM_BOX=3, GEOM_CYLINDER=4)
                                    'dimensions': np.array(shape_info[3]),  # dimensions  (radius for sphere, half-extents for box, [height, radius] for cylinder)
                                    'local_frame_pos': np.array(shape_info[5]),  # local position  (local position relative to center of mass)
                                    'local_frame_orn': np.array(shape_info[6]),  # local orientation
                                })
                        except Exception as obs_error:
                            # skip obstacle if there are no data
                            if self.verbose >= 2:
                                logger.debug(f"Could not capture obstacle {obs_id}: {obs_error}")
                            continue

                    self.current_episode_data["obstacles"] = obstacles_data
                else:
                    # No obstacles in this episode
                    self.current_episode_data["obstacles"] = ["empty"]

            # at episode start ->
            if self.current_episode_data["target_position"] is None:
                # -> capture target position
                self.current_episode_data["target_position"] = (
                    self.underlying_env.waypoints.targets[0].copy()
                )  # XXX adjust for multiple waypoints

            # process completed episodes
            if done:
                self._process_completed_eval_episode(info)

        except Exception as e:
            if self.verbose >= 1:
                print(f"Error in _eval_callback: {e}")

    def _reset_current_episode(self):
        """reset tracking for a new episode"""
        self.current_episode_data = {
            "positions": [],
            "velocities": [],
            "start_position": None,
            "target_position": None,
            "start_time": self.num_timesteps,
            "obstacles": []
        }

    def _process_completed_eval_episode(self, info: dict):
        """
        process a completed episode and log metrics to W&B
        - episode_info keys=['out_of_bounds', 'collision', 'env_complete', 'num_targets_reached', 'TimeLimit.truncated', 'episode']
        - 'episode' is dic and contains logged info from the Monitor wrapper: r, l, t by default, plus any additional logged information (in this case)
        """
        try:
            ## GET METRICS
            # capture episode results from Monitor wrapper
            collision = info["collision"]
            out_of_bounds = info["out_of_bounds"]
            env_complete = info["env_complete"]
            targets_reached = info["num_targets_reached"]
            episode_reward = info["episode"]["r"]
            episode_length = info["episode"]["l"]

            # check for success = all targets reached, without collisions
            success = env_complete and not collision and not out_of_bounds
            
            # calculate custom metrics
            path_length = self._calculate_path_length(
                self.current_episode_data["positions"]
            )
            mean_velocity = self._calculate_average_velocity(
                self.current_episode_data["velocities"]
            )
            if success:
                # calculate path efficiency only on successful episodes where the target is reached
                path_efficiency = self._calculate_path_efficiency(
                    self.current_episode_data["start_position"],
                    self.current_episode_data["target_position"],
                    path_length,
                )
            else:
                # unsuccessful episodes (for consistent array length in self.episode_history)
                path_efficiency = 0.0

            ## STORE METRICS
            # store in evaluation-cycle-level episode history
            self.current_eval_cycle_data["success"].append(int(success))
            self.current_eval_cycle_data["collision"].append(int(collision))
            self.current_eval_cycle_data["out_of_bounds"].append(int(out_of_bounds))
            self.current_eval_cycle_data["env_complete"].append(int(env_complete))
            self.current_eval_cycle_data["targets_reached"].append(targets_reached)
            self.current_eval_cycle_data["num_targets"].append(len(self.current_episode_data["target_position"]))
            self.current_eval_cycle_data["episode_rewards"].append(episode_reward)
            self.current_eval_cycle_data["episode_lengths"].append(episode_length)
            self.current_eval_cycle_data["num_obstacles"].append(info["num_obstacles"])

            self.current_eval_cycle_data["mean_velocities"].append(mean_velocity)
            self.current_eval_cycle_data["path_lengths"].append(path_length)
            self.current_eval_cycle_data["path_efficiencies"].append(path_efficiency)

            self.current_eval_cycle_data["positions_history"].append(
                np.array(self.current_episode_data["positions"]).copy()
            )
            self.current_eval_cycle_data["velocities_history"].append(
                np.array(self.current_episode_data["velocities"]).copy()
            )
            self.current_eval_cycle_data["target_position_history"].append(
                np.array(self.current_episode_data["target_position"]).copy()
            )  # XXX adjust for mulitple waypoints

            self.current_eval_cycle_data["obstacles_history"].append(
                self.current_episode_data["obstacles"].copy()  # list of obstacles (dicts)
            )

            # RESET for next episode
            self._reset_current_episode()

        except Exception as e:
            if self.verbose >= 1:
                print(f"Error in detailed eval callback: {e}")

    def _calculate_path_length(self, positions) -> float:
        """Calculate total distance traveled"""
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

    def _calculate_path_efficiency(self, start_pos, target_pos, path_length) -> float:
        """Calculate path efficiency (path_length / direct_distance)"""
        direct_distance = np.linalg.norm(target_pos - start_pos)
        if direct_distance == 0:
            return 0.0

        return path_length / direct_distance

    def _log_after_evaluation(self):
        """log detailed metrics of a completed evaluation-round to W&B"""
        # calculate aggregate metrics
        eval_metrics = {
            "eval_uav/success_rate": np.mean(
                self.current_eval_cycle_data["success"]
            ),
            "eval_uav/collision_rate": np.mean(
                self.current_eval_cycle_data["collision"]
            ),
            "eval_uav/out_of_bounds_rate": np.mean(
                self.current_eval_cycle_data["out_of_bounds"]
            ),
            "eval_uav/completion_rate": np.mean(
                self.current_eval_cycle_data["env_complete"]
            ),
            "eval_uav/targets_reached_mean": np.mean(
                self.current_eval_cycle_data["targets_reached"]
            ),
            "eval_uav/avg_velocity_mean": np.mean(
                self.current_eval_cycle_data["mean_velocities"]
            ),
            "eval_uav/path_length_mean": np.mean(
                self.current_eval_cycle_data["path_lengths"]
            ),
            "eval_uav/path_efficiency_mean": np.mean(
                self.current_eval_cycle_data["path_efficiencies"]
            ),
            "eval_uav/ep_reward_mean": np.mean(
                self.current_eval_cycle_data["episode_rewards"]
            ),
            "eval_uav/ep_length_mean": np.mean(
                self.current_eval_cycle_data["episode_lengths"]
            ),
        }

        # log to W&B
        wandb.log(eval_metrics, step=self.num_timesteps)

        # log additional analysis plots
        if self.exp_analysis:
            self._log_success_analysis()
            self._log_trajectory_plot()
            self._log_correlation_matrix()

    def _log_success_analysis(self):
        """Create and log success analysis plots to W&B using Plotly"""
        try:
            # Create DataFrame from current evaluation cycle data
            df = pd.DataFrame(self.current_eval_cycle_data)

            # Convert success to boolean for filtering
            success_df = df[df["success"] == 1]
            failure_df = df[df["success"] == 0]

            if success_df.empty or failure_df.empty:
                if self.verbose >= 1:
                    logger.warning(
                        "Insufficient data for success analysis (need both success and failure episodes)"
                    )
                return

            # Create subplot layout
            fig = make_subplots(
                rows=3,
                cols=3,
                subplot_titles=(
                    "Path Length Distribution",
                    "Mean Velocity Distribution",
                    "Episode Reward Distribution",
                    "Episode Length Distribution",
                    "Metrics Box Plot Comparison",
                    "Success vs Failure Rate",
                    "",
                    "",
                    "Failure Reasons Distribution",
                ),
                specs=[
                    [
                        {"secondary_y": False},
                        {"secondary_y": False},
                        {"secondary_y": False},
                    ],
                    [
                        {"secondary_y": False},
                        {"secondary_y": False},
                        {"type": "domain"},
                    ],
                    [{"secondary_y": False, "colspan": 2}, None, {"type": "domain"}],
                ],
            )

            # 1. Path Length Distribution
            fig.add_trace(
                go.Histogram(
                    x=success_df["path_lengths"],
                    name="Success",
                    marker_color="green",
                    opacity=0.7,
                    nbinsx=20,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Histogram(
                    x=failure_df["path_lengths"],
                    name="Failure",
                    marker_color="red",
                    opacity=0.7,
                    nbinsx=20,
                ),
                row=1,
                col=1,
            )
            # Add mean lines for path length
            success_path_mean = success_df["path_lengths"].mean()
            failure_path_mean = failure_df["path_lengths"].mean()
            fig.add_shape(
                type="line",
                x0=success_path_mean,
                y0=0,
                x1=success_path_mean,
                y1=1,
                line=dict(color="darkgreen", width=2, dash="dash"),
                xref="x",
                yref="y domain",
                row=1,
                col=1,
            )
            fig.add_shape(
                type="line",
                x0=failure_path_mean,
                y0=0,
                x1=failure_path_mean,
                y1=1,
                line=dict(color="darkred", width=2, dash="dash"),
                xref="x",
                yref="y domain",
                row=1,
                col=1,
            )

            # 2. Velocity Distribution
            fig.add_trace(
                go.Histogram(
                    x=success_df["mean_velocities"],
                    name="Success",
                    marker_color="green",
                    opacity=0.7,
                    nbinsx=20,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Histogram(
                    x=failure_df["mean_velocities"],
                    name="Failure",
                    marker_color="red",
                    opacity=0.7,
                    nbinsx=20,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            # Add mean lines for velocity
            success_vel_mean = success_df["mean_velocities"].mean()
            failure_vel_mean = failure_df["mean_velocities"].mean()
            fig.add_shape(
                type="line",
                x0=success_vel_mean,
                y0=0,
                x1=success_vel_mean,
                y1=1,
                line=dict(color="darkgreen", width=2, dash="dash"),
                xref="x2",
                yref="y2 domain",
                row=1,
                col=2,
            )
            fig.add_shape(
                type="line",
                x0=failure_vel_mean,
                y0=0,
                x1=failure_vel_mean,
                y1=1,
                line=dict(color="darkred", width=2, dash="dash"),
                xref="x2",
                yref="y2 domain",
                row=1,
                col=2,
            )

            # 3. Episode Rewards Distribution
            fig.add_trace(
                go.Histogram(
                    x=success_df["episode_rewards"],
                    name="Success",
                    marker_color="green",
                    opacity=0.7,
                    nbinsx=20,
                    showlegend=False,
                ),
                row=1,
                col=3,
            )
            fig.add_trace(
                go.Histogram(
                    x=failure_df["episode_rewards"],
                    name="Failure",
                    marker_color="red",
                    opacity=0.7,
                    nbinsx=20,
                    showlegend=False,
                ),
                row=1,
                col=3,
            )
            # Add mean lines for episode rewards
            success_reward_mean = success_df["episode_rewards"].mean()
            failure_reward_mean = failure_df["episode_rewards"].mean()
            fig.add_shape(
                type="line",
                x0=success_reward_mean,
                y0=0,
                x1=success_reward_mean,
                y1=1,
                line=dict(color="darkgreen", width=2, dash="dash"),
                xref="x3",
                yref="y3 domain",
                row=1,
                col=3,
            )
            fig.add_shape(
                type="line",
                x0=failure_reward_mean,
                y0=0,
                x1=failure_reward_mean,
                y1=1,
                line=dict(color="darkred", width=2, dash="dash"),
                xref="x3",
                yref="y3 domain",
                row=1,
                col=3,
            )

            # 4. Episode Length Distribution
            fig.add_trace(
                go.Histogram(
                    x=success_df["episode_lengths"],
                    name="Success",
                    marker_color="green",
                    opacity=0.7,
                    nbinsx=20,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Histogram(
                    x=failure_df["episode_lengths"],
                    name="Failure",
                    marker_color="red",
                    opacity=0.7,
                    nbinsx=20,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            # Add mean lines for episode length
            success_length_mean = success_df["episode_lengths"].mean()
            failure_length_mean = failure_df["episode_lengths"].mean()
            fig.add_shape(
                type="line",
                x0=success_length_mean,
                y0=0,
                x1=success_length_mean,
                y1=1,
                line=dict(color="darkgreen", width=2, dash="dash"),
                xref="x4",
                yref="y4 domain",
                row=2,
                col=1,
            )
            fig.add_shape(
                type="line",
                x0=failure_length_mean,
                y0=0,
                x1=failure_length_mean,
                y1=1,
                line=dict(color="darkred", width=2, dash="dash"),
                xref="x4",
                yref="y4 domain",
                row=2,
                col=1,
            )

            # 5. Box plot comparison
            metrics_for_box = [
                "path_lengths",
                "mean_velocities",
                "episode_rewards",
                "episode_lengths",
            ]
            for metric in metrics_for_box:
                fig.add_trace(
                    go.Box(
                        y=success_df[metric],
                        name=f"{metric} (Success)",
                        marker_color="lightgreen",
                        showlegend=False,
                    ),
                    row=2,
                    col=2,
                )
                fig.add_trace(
                    go.Box(
                        y=failure_df[metric],
                        name=f"{metric} (Failure)",
                        marker_color="lightcoral",
                        showlegend=False,
                    ),
                    row=2,
                    col=2,
                )

            # 6. Success vs Failure Rate pie chart
            success_counts = df["success"].value_counts().sort_index()
            success_labels = [
                "Failure" if idx == 0 else "Success" for idx in success_counts.index
            ]
            success_colors = [
                "lightcoral" if idx == 0 else "lightgreen"
                for idx in success_counts.index
            ]

            if len(success_counts) > 0:
                fig.add_trace(
                    go.Pie(
                        labels=success_labels,
                        values=success_counts.values,
                        marker_colors=success_colors,
                        name="Success Rate",
                        showlegend=False,
                    ),
                    row=2,
                    col=3,
                )

            # 7. Failure reasons pie chart
            failure_reasons = {
                "Collision": failure_df["collision"].sum(),
                "Out of Bounds": failure_df["out_of_bounds"].sum(),
                "Timeout/Other": len(failure_df)
                - failure_df["collision"].sum()
                - failure_df["out_of_bounds"].sum(),
            }

            # Remove zero categories
            failure_reasons = {k: v for k, v in failure_reasons.items() if v > 0}

            if failure_reasons:
                fig.add_trace(
                    go.Pie(
                        labels=list(failure_reasons.keys()),
                        values=list(failure_reasons.values()),
                        name="Failure Reasons",
                        showlegend=False,
                    ),
                    row=3,
                    col=3,
                )

            # Update layout
            fig.update_layout(
                height=1200, title_text="Success Analysis Dashboard", showlegend=True
            )

            # Update x-axis labels
            fig.update_xaxes(title_text="Path Length (m)", row=1, col=1)
            fig.update_xaxes(title_text="Velocity (m/s)", row=1, col=2)
            fig.update_xaxes(title_text="Episode Reward", row=1, col=3)
            fig.update_xaxes(title_text="Steps per Episode", row=2, col=1)

            # Update y-axis labels
            fig.update_yaxes(title_text="Frequency", row=1, col=1)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=3)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)

            # Log to W&B
            wandb.log(
                {
                    "eval_analysis/success_analysis_dashboard": wandb.Html(
                        fig.to_html(include_plotlyjs="cdn")
                    )
                },
                step=self.num_timesteps,
            )

        except Exception as e:
            if self.verbose >= 1:
                logger.exception(f"Error creating success analysis plots: {e}")

    def _log_trajectory_plot(self):
        """Create and log evaluation trajectory plot for an episode"""
        try:
            # pick 3 random episodes
            episodes = self.current_eval_cycle_data["success"]
            picked_episodes = self.rng.choice(len(episodes), size=3, replace=False).tolist()

            for idx, ep_idx in enumerate(picked_episodes):
                positions = self.current_eval_cycle_data["positions_history"][ep_idx][:-1]  # take one position less ([:-1]), so that the trajectory is a oneway and not a closed loop
                if len(positions) < 2:
                    return

                velocities = self.current_eval_cycle_data["velocities_history"][ep_idx][:-1]
                reward = self.current_eval_cycle_data["episode_rewards"][ep_idx]
                success = self.current_eval_cycle_data["success"][ep_idx]
                collision = self.current_eval_cycle_data["collision"][ep_idx]
                target_position = self.current_eval_cycle_data["target_position_history"][ep_idx]
                targets_reached = self.current_eval_cycle_data["targets_reached"][ep_idx]

                # Calculate speed magnitudes
                speed_magnitudes = np.linalg.norm(velocities, axis=1)

                # Create 3D trajectory plot
                fig = go.Figure()

                # Add wireframe box representing the flight space boundaries
                # Get boundaries from occupancy grid (XY) and env (Z)
                x_min, x_max = self.underlying_env.occupancy_grid.x_min, self.underlying_env.occupancy_grid.x_max
                y_min, y_max = self.underlying_env.occupancy_grid.y_min, self.underlying_env.occupancy_grid.y_max
                z_min, z_max = 0.0, self.underlying_env.z_size

                # Define the 8 vertices of the box
                vertices = np.array([
                    [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
                    [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]
                ])

                # Define the 12 edges of the box (pairs of vertex indices)
                edges = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
                    [4, 5], [5, 6], [6, 7], [7, 4],  # top face
                    [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
                ]

                # Add edges as lines
                for edge in edges:
                    v1, v2 = vertices[edge[0]], vertices[edge[1]]
                    fig.add_trace(
                        go.Scatter3d(
                            x=[v1[0], v2[0]],
                            y=[v1[1], v2[1]],
                            z=[v1[2], v2[2]],
                            mode='lines',
                            line=dict(color='lightblue', width=4),
                            showlegend=False,
                            hoverinfo='skip',
                        )
                    )

                fig.add_trace(
                    go.Scatter3d(
                        x=positions[:, 0],
                        y=positions[:, 1],
                        z=positions[:, 2],
                        mode="lines",
                        line=dict(
                            color=speed_magnitudes,
                            colorscale="Viridis",
                            width=8,
                            colorbar=dict(title="Speed (m/s)"),
                        ),
                        name="UAV Trajectory",
                        text=[f"Speed: {speed:.2f} m/s" for speed in speed_magnitudes],
                        hovertemplate="<b>Position</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>%{text}<extra></extra>",
                    )
                )

                # Add start/end and target markers
                fig.add_trace(
                    go.Scatter3d(
                        x=[positions[0, 0]],
                        y=[positions[0, 1]],
                        z=[positions[0, 2]],
                        mode="markers",
                        marker=dict(size=8, color="green"),
                        name="Start",
                        showlegend=True,
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=[positions[-1, 0]],
                        y=[positions[-1, 1]],
                        z=[positions[-1, 2]],
                        mode="markers",
                        marker=dict(size=8, color="red"),
                        name="End",
                        showlegend=True,
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=[target_position[0]],
                        y=[target_position[1]],
                        z=[target_position[2]],
                        mode="markers",
                        marker=dict(size=10, color="gold", symbol="diamond"),
                        name="Target",
                        showlegend=True,
                    )
                )

                # plot obstacles using PyBullet collision shape data
                obstacles = self.current_eval_cycle_data["obstacles_history"][ep_idx]
                if obstacles:
                    for i, obs in enumerate(obstacles):
                        pos = obs['position']
                        shape_type = obs['shape_type']
                        dims = obs['dimensions']
                        
                        # GEOM_SPHERE = 2
                        if shape_type == 2:
                            radius = dims[0]
                            # Create sphere mesh
                            u = np.linspace(0, 2 * np.pi, 20)
                            v = np.linspace(0, np.pi, 20)
                            x = radius * np.outer(np.cos(u), np.sin(v)) + pos[0]
                            y = radius * np.outer(np.sin(u), np.sin(v)) + pos[1]
                            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
                            
                            fig.add_trace(go.Surface(
                                x=x, y=y, z=z,
                                opacity=0.5,
                                colorscale=[[0, 'red'], [1, 'red']],
                                showscale=False,
                                name=f'Obstacle {i+1}',
                                legendgroup=f'obs{i}',
                                hovertemplate=f'<b>Obstacle {i+1}</b><br>Type: Sphere<br>Radius: {radius:.2f}m<extra></extra>',
                            ))
                        
                        # GEOM_BOX = 3
                        elif shape_type == 3:
                            # dims contains half-extents [x, y, z]
                            half_extents = dims[:3]
                            
                            # Create 8 vertices of the box
                            vertices = np.array([
                                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
                            ]) * half_extents + pos
                            
                            # Create a mesh (faces) for the box for better visualization
                            # Define 6 faces (each face has 4 vertices forming 2 triangles)
                            i_faces = [0, 0, 2, 2, 4, 4, 1, 1, 3, 3, 5, 5]  # vertex indices for triangles
                            j_faces = [1, 3, 3, 1, 5, 7, 5, 0, 7, 2, 6, 4]
                            k_faces = [2, 1, 6, 3, 6, 5, 4, 5, 4, 6, 2, 7]
                            
                            fig.add_trace(go.Mesh3d(
                                x=vertices[:, 0],
                                y=vertices[:, 1],
                                z=vertices[:, 2],
                                i=i_faces,
                                j=j_faces,
                                k=k_faces,
                                opacity=0.5,
                                color='red',
                                name=f'Obstacle {i+1}',
                                legendgroup=f'obs{i}',
                                hovertemplate=f'<b>Obstacle {i+1}</b><br>Type: Box<br>Dims: {half_extents[0]:.2f}x{half_extents[1]:.2f}x{half_extents[2]:.2f}m<extra></extra>',
                            ))
                        
                        # GEOM_CYLINDER = 4 (or GEOM_CAPSULE = 7)
                        elif shape_type == 4:
                            # For cylinder: dims[0] = height, dims[1] = radius
                            height = dims[0]
                            radius = dims[1]
                            
                            # Create cylinder mesh
                            theta = np.linspace(0, 2*np.pi, 30)
                            z_cyl = np.linspace(-height/2, height/2, 20)
                            theta_grid, z_grid = np.meshgrid(theta, z_cyl)
                            x = radius * np.cos(theta_grid) + pos[0]
                            y = radius * np.sin(theta_grid) + pos[1]
                            z = z_grid + pos[2]
                            
                            fig.add_trace(go.Surface(
                                x=x, y=y, z=z,
                                opacity=0.5,
                                colorscale=[[0, 'red'], [1, 'red']],
                                showscale=False,
                                name=f'Obstacle {i+1}',
                                legendgroup=f'obs{i}',
                                hovertemplate=f'<b>Obstacle {i+1}</b><br>Type: Cylinder<br>Radius: {radius:.2f}m<br>Height: {height:.2f}m<extra></extra>',
                            ))

                # update layout
                fig.update_layout(
                    title=f"UAV Trajectory (Reward: {reward:.2f}, Success: {success}, Collision: {collision} Targets reached: {targets_reached})",
                    scene=dict(
                        xaxis_title="X Position (m)",
                        yaxis_title="Y Position (m)",
                        zaxis_title="Z Position (m)",
                        aspectmode="data",
                        xaxis=dict(range=[x_min, x_max]),
                        yaxis=dict(range=[y_min, y_max]),
                        zaxis=dict(range=[z_min, z_max]),
                    ),
                    legend=dict(
                        x=0.02,
                        y=0.98,
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="rgba(0,0,0,0.5)",
                        borderwidth=1,
                    ),
                )

                # Log to W&B
                wandb.log(
                    {
                        f"eval_trajectory/UAV_trajectory_{idx}": wandb.Html(
                            fig.to_html(include_plotlyjs="cdn")
                        ),
                    },
                    step=self.num_timesteps,
                )

        except Exception as e:
            if self.verbose >= 1:
                logger.exception(f"Error creating trajectory plot: {e}")

    def _log_correlation_matrix(self):
        """Create and log correlation matrix plot to W&B"""
        try:
            # Create DataFrame from current evaluation cycle data
            df = pd.DataFrame(self.current_eval_cycle_data)

            # remove useless columns
            df = df.drop(
                ["positions_history", "velocities_history", "target_position_history", "obstacles_history"],
                axis=1,
            )

            # remove rows where path_efficiency is 0 for better correlation
            df["path_efficiencies"] = df["path_efficiencies"].replace(0.0, np.nan)

            correlation_matrix = df.corr()

            # create Plotly heatmap
            fig = go.Figure(
                data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    colorscale="RdBu",
                    zmid=0,
                    text=correlation_matrix.round(3).values,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hoverongaps=False,
                    showscale=True,
                    colorbar=dict(title="Correlation"),
                )
            )

            fig.update_layout(
                title="Correlation Matrix of UAV Performance Metrics",
                width=700,
                height=600,
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                xaxis_zeroline=False,
                yaxis_zeroline=False,
            )

            # Log to W&B using native wandb logging
            wandb.log(
                {
                    "eval_analysis/correlation_matrix": wandb.Html(
                        fig.to_html(include_plotlyjs="cdn")
                    )
                },
                step=self.num_timesteps,
            )

        except Exception as e:
            if self.verbose >= 1:
                logger.exception(f"Error creating correlation matrix plot: {e}")

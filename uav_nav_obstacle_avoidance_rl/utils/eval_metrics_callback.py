import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from uav_nav_obstacle_avoidance_rl import config

logger = config.logger


class CustomEvalCallback(EvalCallback):
    """
    Custom EvalCallback that collects detailed UAV metrics during evaluation.
    Extends the standard EvalCallback to collect the same metrics as training.
    Initialize evaluation cycle in the _on_step() function.
    """

    # reward component keys emitted by RewardWrapper
    REWARD_COMPONENTS = [
        "reward_total", "reward_progress", "reward_obstacle", "reward_collision", "reward_target", "reward_step", "reward_distance",
        # obstacle sub-components
        "reward_obs_approaching", "reward_obs_weighting", "reward_obs_mean_approach_penalty", "reward_obs_proximity",
    ]

    def __init__(self, *args, **kwargs):
        self.exp_analysis = kwargs.pop("exp_analysis", True)
        seed = kwargs.pop("seed", None)

        # initialize parent with remaining kwargs
        super().__init__(*args, **kwargs)

        self.rng = np.random.default_rng(seed)

        #  get all parallel underlying_envs in a list
        if hasattr(self.eval_env, "envs"):
            self.underlying_envs = [e.unwrapped for e in self.eval_env.envs]
        else:
            self.underlying_envs = [self.eval_env.unwrapped]

        self.n_eval_envs = len(self.underlying_envs)
        self.current_episode_data = [{} for _ in range(self.n_eval_envs)]

        # best model tracking by success rate
        self.best_success_rate = -1.0

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
                "env_complete": [],
                "num_targets": [],
                "targets_reached": [],
                "episode_rewards": [],
                "episode_lengths": [],
                "mean_velocities": [],
                "path_lengths": [],
                "path_efficiencies": [],
                "num_obstacles": [],
                "positions_history": [],
                "velocities_history": [],
                "target_position_history": [],
                "obstacles_history": [],
                "step_rewards_history": [],
                # per-episode reward component sums
                **{f"{k}_sum": [] for k in self.REWARD_COMPONENTS},
                # per-episode per-step reward component histories (for trajectory plots)
                **{f"{k}_history": [] for k in self.REWARD_COMPONENTS},
            }
            self._is_success_buffer = []

            # reset all envs before evaluation
            for i in range(self.n_eval_envs):
                self._reset_current_episode(i)

            # START EVALUATION
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_eval_callback,
            )

            # log detailed eval metrics to W&B
            self._log_after_evaluation()

            success_rate = float(np.mean(self.current_eval_cycle_data["success"]))

            # save best model and generate analysis plots on new best success rate
            if success_rate > self.best_success_rate:
                logger.info(f"Saved new best model with success rate: {success_rate:.2%} (prev: {self.best_success_rate:.2%})")
                if self.best_model_save_path is not None:
                    import os

                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model_success_rate")
                    )
                self.best_success_rate = success_rate

                # generate analysis plots only when we achieve a new best
                if self.exp_analysis:
                    self._log_success_analysis()
                    self._log_trajectory_plot()
                    self._log_reward_decomposition_plot()
                    # self._log_correlation_matrix()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    # ──────────────────────────────────────────────────────────────
    #  Per-step callback (called by evaluate_policy on every step)
    # ──────────────────────────────────────────────────────────────
    def _log_eval_callback(self, locals_dict: dict, globals_dict: dict) -> None:
            """
            custom callback for evaluate_policy that collects detailed metrics.
            this gets called after each step during evaluation.
            """
            try:
                dones = locals_dict["dones"]
                infos = locals_dict["infos"]
                rewards = locals_dict["rewards"]

                for env_idx in range(self.n_eval_envs):
                    ep = self.current_episode_data[env_idx]
                    if not ep:
                        self._reset_current_episode(env_idx)
                        ep = self.current_episode_data[env_idx]

                    underlying = self.underlying_envs[env_idx]
                    state = underlying.env.state(0)
                    lin_pos = state[3]
                    lin_vel = state[2]

                    ep["positions"].append(lin_pos.copy())
                    ep["velocities"].append(lin_vel.copy())
                    ep["step_rewards"].append(float(rewards[env_idx]))

                    # accumulate per-step reward components from RewardWrapper
                    step_info = infos[env_idx]
                    for key in self.REWARD_COMPONENTS:
                        if key in step_info:
                            ep[key].append(float(step_info[key]))

                    # capture one-time data at episode start
                    if ep["start_position"] is None:
                        ep["start_position"] = lin_pos.copy()

                        # capture obstacle AABBs (shape-agnostic)
                        if hasattr(underlying, 'obstacles') and underlying.obstacles:
                            obstacles_data = []
                            for obs_id in underlying.obstacles:
                                try:
                                    aabb_min, aabb_max = underlying.env.getAABB(obs_id)
                                    obstacles_data.append({
                                        'aabb_min': np.array(aabb_min),
                                        'aabb_max': np.array(aabb_max),
                                    })
                                except Exception as obs_error:
                                    logger.debug(f"[EvalCallback] Could not capture obstacle {obs_id}: {obs_error}")
                                    continue
                            ep["obstacles"] = obstacles_data
                        else:
                            ep["obstacles"] = []

                    if ep["target_position"] is None:
                        ep["target_position"] = underlying.waypoints.targets[0].copy()

                    if dones[env_idx]:
                        self._process_completed_eval_episode(infos[env_idx], env_idx)

            except Exception as e:
                logger.error(f"[EvalCallback] Error in _log_eval_callback: {e}", exc_info=True)

    # ──────────────────────────────────────────────────────────────
    #  Episode processing
    # ──────────────────────────────────────────────────────────────
    def _process_completed_eval_episode(self, info: dict, env_idx: int):
        """
        process a completed episode and log metrics to W&B
        """
        try:
            ep = self.current_episode_data[env_idx]

            collision = info["collision"]
            env_complete = info["env_complete"]
            targets_reached = info["num_targets_reached"]
            episode_reward = info["episode"]["r"]
            episode_length = info["episode"]["l"]

            success = env_complete and not collision

            path_length = self._calculate_path_length(ep["positions"])
            mean_velocity = self._calculate_average_velocity(ep["velocities"])
            if success:
                path_efficiency = self._calculate_path_efficiency(
                    ep["start_position"],
                    ep["target_position"],
                    path_length,
                )
            else:
                path_efficiency = 0.0

            self.current_eval_cycle_data["success"].append(int(success))
            self.current_eval_cycle_data["collision"].append(int(collision))
            self.current_eval_cycle_data["env_complete"].append(int(env_complete))
            self.current_eval_cycle_data["targets_reached"].append(targets_reached)
            self.current_eval_cycle_data["num_targets"].append(len(ep["target_position"]))
            self.current_eval_cycle_data["episode_rewards"].append(episode_reward)
            self.current_eval_cycle_data["episode_lengths"].append(episode_length)
            self.current_eval_cycle_data["num_obstacles"].append(info["num_obstacles"])

            self.current_eval_cycle_data["mean_velocities"].append(mean_velocity)
            self.current_eval_cycle_data["path_lengths"].append(path_length)
            self.current_eval_cycle_data["path_efficiencies"].append(path_efficiency)

            self.current_eval_cycle_data["positions_history"].append(
                np.array(ep["positions"]).copy()
            )
            self.current_eval_cycle_data["velocities_history"].append(
                np.array(ep["velocities"]).copy()
            )
            self.current_eval_cycle_data["target_position_history"].append(
                np.array(ep["target_position"]).copy()
            )

            obstacles = ep["obstacles"]
            self.current_eval_cycle_data["obstacles_history"].append(
                obstacles.copy() if isinstance(obstacles, list) else []
            )

            self.current_eval_cycle_data["step_rewards_history"].append(
                np.array(ep["step_rewards"]).copy()
            )

            # store reward component episode sums and per-step histories
            for key in self.REWARD_COMPONENTS:
                values = ep[key]
                self.current_eval_cycle_data[f"{key}_sum"].append(
                    float(np.sum(values)) if values else 0.0
                )
                self.current_eval_cycle_data[f"{key}_history"].append(
                    np.array(values).copy() if values else np.array([])
                )

            # RESET for next episode
            self._reset_current_episode(env_idx)

        except Exception as e:
            logger.error(f"[EvalCallback] Error in _process_completed_eval_episode: {e}", exc_info=True)

    # ──────────────────────────────────────────────────────────────
    #  Logging (called once after all eval episodes complete)
    # ──────────────────────────────────────────────────────────────

    def _log_after_evaluation(self):
        """log detailed metrics of a completed evaluation-round to W&B"""
        n_episodes = len(self.current_eval_cycle_data["success"])
        if n_episodes == 0:
            logger.warning("[EvalCallback] No episodes were recorded during evaluation — skipping metrics logging.")
            return

        # reward component means across eval episodes
        reward_component_metrics = {}
        for key in self.REWARD_COMPONENTS:
            component_name = key.replace("reward_", "")
            sums = self.current_eval_cycle_data[f"{key}_sum"]
            if sums:
                # ep_sum_eval = per-episode reward sum, averaged over the eval cycle
                reward_component_metrics[f"eval_cycle_cum_reward/{component_name}_cycle"] = float(np.mean(sums))

        eval_metrics = {
            # success = all targets reached AND no collision
            # _cycle suffix = averaged over all episodes in this eval cycle
            "eval_cycle/success_rate": np.mean(self.current_eval_cycle_data["success"]),
            "eval_cycle/collision_rate": np.mean(self.current_eval_cycle_data["collision"]),
            # all_targets_reached_rate = env_complete (targets reached, regardless of collision)
            "eval_cycle/all_targets_reached_rate": np.mean(self.current_eval_cycle_data["env_complete"]),
            "eval_cycle/targets_reached": np.mean(self.current_eval_cycle_data["targets_reached"]),
            "eval_cycle/mean_velocity": np.mean(self.current_eval_cycle_data["mean_velocities"]),
            "eval_cycle/path_length": np.mean(self.current_eval_cycle_data["path_lengths"]),
            "eval_cycle/path_efficiency": np.mean(self.current_eval_cycle_data["path_efficiencies"]),
            "eval_cycle/ep_length": np.mean(self.current_eval_cycle_data["episode_lengths"]),
            # reward decomposition
            **reward_component_metrics,
        }

        wandb.log(eval_metrics, step=self.num_timesteps)
        logger.info(f"[EvalCallback] Eval cycle logged: {n_episodes} episodes, "
                     f"success_rate={eval_metrics['eval_cycle/success_rate']:.2f}, "
                     f"all_targets_reached_rate={eval_metrics['eval_cycle/all_targets_reached_rate']:.2f}")


    # ──────────────────────────────────────────────────────────────
    #  Analysis plots
    # ──────────────────────────────────────────────────────────────

    def _log_success_analysis(self):
        """Create and log success analysis plots to W&B using Plotly"""
        try:
            df = pd.DataFrame(self.current_eval_cycle_data)

            success_df = df[df["success"] == 1]
            failure_df = df[df["success"] == 0]

            if success_df.empty or failure_df.empty:
                logger.debug("[EvalCallback] Insufficient data for success analysis (need both success and failure episodes)")
                return

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
                    [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}, {"type": "domain"}],
                    [{"secondary_y": False, "colspan": 2}, None, {"type": "domain"}],
                ],
            )

            # --- histograms with mean lines ---
            hist_configs = [
                ("path_lengths",     "Path Length (m)",   1, 1, "x",  "y domain"),
                ("mean_velocities",  "Velocity (m/s)",    1, 2, "x2", "y2 domain"),
                ("episode_rewards",  "Episode Reward",    1, 3, "x3", "y3 domain"),
                ("episode_lengths",  "Steps per Episode", 2, 1, "x4", "y4 domain"),
            ]

            for i, (col, xlabel, row, c, xref, yref) in enumerate(hist_configs):
                show_legend = (i == 0)
                fig.add_trace(go.Histogram(x=success_df[col], name="Success", marker_color="green", opacity=0.7, nbinsx=20, showlegend=show_legend), row=row, col=c)
                fig.add_trace(go.Histogram(x=failure_df[col], name="Failure", marker_color="red", opacity=0.7, nbinsx=20, showlegend=show_legend), row=row, col=c)

                for val, color in [(success_df[col].mean(), "darkgreen"), (failure_df[col].mean(), "darkred")]:
                    fig.add_shape(type="line", x0=val, y0=0, x1=val, y1=1,
                                  line=dict(color=color, width=2, dash="dash"),
                                  xref=xref, yref=yref, row=row, col=c)

                fig.update_xaxes(title_text=xlabel, row=row, col=c)
                fig.update_yaxes(title_text="Frequency", row=row, col=c)

            # --- box plot comparison ---
            for metric in ["path_lengths", "mean_velocities", "episode_rewards", "episode_lengths"]:
                fig.add_trace(go.Box(y=success_df[metric], name=f"{metric} (S)", marker_color="lightgreen", showlegend=False), row=2, col=2)
                fig.add_trace(go.Box(y=failure_df[metric], name=f"{metric} (F)", marker_color="lightcoral", showlegend=False), row=2, col=2)

            # --- pie: success vs failure ---
            counts = df["success"].value_counts().sort_index()
            labels = ["Failure" if i == 0 else "Success" for i in counts.index]
            colors = ["lightcoral" if i == 0 else "lightgreen" for i in counts.index]
            if len(counts) > 0:
                fig.add_trace(go.Pie(labels=labels, values=counts.values, marker_colors=colors, showlegend=False), row=2, col=3)

            # --- pie: failure reasons ---
            n_collision = failure_df["collision"].sum()
            reasons = {"Collision": n_collision, "Timeout/Other": len(failure_df) - n_collision}
            reasons = {k: v for k, v in reasons.items() if v > 0}
            if reasons:
                fig.add_trace(go.Pie(labels=list(reasons.keys()), values=list(reasons.values()), showlegend=False), row=3, col=3)

            fig.update_layout(height=1200, title_text="Success Analysis Dashboard", showlegend=True)

            wandb.log(
                {"eval_analysis/success_analysis_dashboard": wandb.Html(fig.to_html(include_plotlyjs="cdn"))},
                step=self.num_timesteps,
            )

        except Exception as e:
            logger.exception(f"[EvalCallback] Error creating success analysis plots: {e}")

    # ──────────────────────────────────────────────────────────────
    #  Trajectory plot  (MODIFIED — deterministic picks + reward width)
    # ──────────────────────────────────────────────────────────────

    def _pick_trajectory_episodes(self) -> list[tuple[int, str, str]]:
        """
        Select up to 4 representative episodes for trajectory plotting:
          1. Successful   — highest reward among successful episodes
          2. Successful   — lowest  reward among successful episodes
          3. Unsuccessful — highest reward among failed episodes
          4. Unsuccessful — lowest  reward among failed episodes

        Returns:
            list of (episode_index, group_key, display_label) tuples
        """
        rewards = np.array(self.current_eval_cycle_data["episode_rewards"])
        successes = np.array(self.current_eval_cycle_data["success"])

        picks: list[tuple[int, str, str]] = []  # (ep_idx, group_key, display_label)

        # --- successful episodes ---
        success_mask = successes == 1
        if np.any(success_mask):
            success_indices = np.where(success_mask)[0]
            best_idx = int(success_indices[np.argmax(rewards[success_indices])])
            worst_idx = int(success_indices[np.argmin(rewards[success_indices])])
            picks.append((best_idx,  "1_Success_Highest_Reward", "Success | Highest Reward"))
            if worst_idx != best_idx:
                picks.append((worst_idx, "2_Success_Lowest_Reward", "Success | Lowest Reward"))

        # --- unsuccessful episodes ---
        fail_mask = successes == 0
        if np.any(fail_mask):
            fail_indices = np.where(fail_mask)[0]
            best_idx = int(fail_indices[np.argmax(rewards[fail_indices])])
            worst_idx = int(fail_indices[np.argmin(rewards[fail_indices])])
            picks.append((best_idx,  "3_Failure_Highest_Reward", "Failure | Highest Reward"))
            if worst_idx != best_idx:
                picks.append((worst_idx, "4_Failure_Lowest_Reward", "Failure | Lowest Reward"))

        return picks

    def _log_trajectory_plot(self):
        """Create and log evaluation trajectory plots for selected episodes.

        Episode selection: best successful, best unsuccessful, worst unsuccessful.
        Visual encoding:
          - Line COLOR  → speed (Viridis colorscale, unchanged)
          - Line WIDTH  → per-step reward (thin = low/negative, thick = high)
        """
        try:
            n_episodes = len(self.current_eval_cycle_data["success"])
            if n_episodes == 0:
                logger.warning("[EvalCallback] No episodes available for trajectory plotting.")
                return

            # ── deterministic episode selection ──────────────────────
            picked = self._pick_trajectory_episodes()
            if not picked:
                logger.warning("[EvalCallback] Could not pick any episodes for trajectory plotting.")
                return

            for ep_idx, group_key, label in picked:
                positions = self.current_eval_cycle_data["positions_history"][ep_idx][:-1]
                if len(positions) < 2:
                    continue

                velocities = self.current_eval_cycle_data["velocities_history"][ep_idx][:-1]
                step_rewards = self.current_eval_cycle_data["step_rewards_history"][ep_idx][:-1]
                reward = self.current_eval_cycle_data["episode_rewards"][ep_idx]
                success = self.current_eval_cycle_data["success"][ep_idx]
                collision = self.current_eval_cycle_data["collision"][ep_idx]
                target_position = self.current_eval_cycle_data["target_position_history"][ep_idx]
                targets_reached = self.current_eval_cycle_data["targets_reached"][ep_idx]

                speed_magnitudes = np.linalg.norm(velocities, axis=1)

                # ── map step rewards → line widths ──────────────────
                # normalize rewards to [MIN_WIDTH, MAX_WIDTH] range
                MIN_WIDTH, MAX_WIDTH = 2.0, 14.0
                r = np.array(step_rewards, dtype=float)
                r_min, r_max = r.min(), r.max()
                if r_max - r_min > 1e-8:
                    widths = MIN_WIDTH + (r - r_min) / (r_max - r_min) * (MAX_WIDTH - MIN_WIDTH)
                else:
                    widths = np.full_like(r, (MIN_WIDTH + MAX_WIDTH) / 2)

                fig = go.Figure()

                # ── boundary wireframe ───────────────────────────────
                x_min = self.underlying_envs[0].occupancy_grid.x_min
                x_max = self.underlying_envs[0].occupancy_grid.x_max
                y_min = self.underlying_envs[0].occupancy_grid.y_min
                y_max = self.underlying_envs[0].occupancy_grid.y_max
                z_min, z_max = 0.0, self.underlying_envs[0].z_size

                verts = np.array([
                    [x_min, y_min, z_min], [x_max, y_min, z_min],
                    [x_max, y_max, z_min], [x_min, y_max, z_min],
                    [x_min, y_min, z_max], [x_max, y_min, z_max],
                    [x_max, y_max, z_max], [x_min, y_max, z_max],
                ])
                edges = [
                    [0,1],[1,2],[2,3],[3,0],
                    [4,5],[5,6],[6,7],[7,4],
                    [0,4],[1,5],[2,6],[3,7],
                ]
                for edge in edges:
                    v1, v2 = verts[edge[0]], verts[edge[1]]
                    fig.add_trace(go.Scatter3d(
                        x=[v1[0], v2[0]], y=[v1[1], v2[1]], z=[v1[2], v2[2]],
                        mode='lines', line=dict(color='lightblue', width=4),
                        showlegend=False, hoverinfo='skip',
                    ))

                # ── trajectory segments (color=speed, width=reward) ──
                # Plotly Scatter3d only supports a scalar line.width,
                # so we draw one trace per segment to vary width.
                # To keep it manageable, we quantize widths into bins.
                N_BINS = 8
                bin_edges = np.linspace(MIN_WIDTH, MAX_WIDTH, N_BINS + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                seg_bins = np.digitize(widths, bin_edges) - 1
                seg_bins = np.clip(seg_bins, 0, N_BINS - 1)

                # build a Viridis-like color for each segment based on speed
                speed_min, speed_max = speed_magnitudes.min(), speed_magnitudes.max()

                import plotly.express as px
                viridis = px.colors.sequential.Viridis
                n_colors = len(viridis)

                def speed_to_color(s):
                    """Map a speed scalar to a Viridis hex color."""
                    if speed_max - speed_min > 1e-8:
                        t = (s - speed_min) / (speed_max - speed_min)
                    else:
                        t = 0.5
                    idx = int(t * (n_colors - 1))
                    idx = max(0, min(idx, n_colors - 1))
                    return viridis[idx]

                # group consecutive segments that share the same width-bin
                # to reduce trace count
                added_legend = False
                seg_idx = 0
                n_seg = len(positions) - 1
                while seg_idx < n_seg:
                    cur_bin = seg_bins[seg_idx]
                    # gather contiguous run of same bin
                    run_end = seg_idx + 1
                    while run_end < n_seg and seg_bins[run_end] == cur_bin:
                        run_end += 1

                    # point indices for this run: seg_idx .. run_end (inclusive endpoints)
                    pt_slice = slice(seg_idx, run_end + 1)
                    seg_slice = slice(seg_idx, run_end)

                    # average speed for color of this run
                    avg_speed = speed_magnitudes[seg_slice].mean()
                    color = speed_to_color(avg_speed)

                    hover_texts = [
                        f"Speed: {speed_magnitudes[k]:.2f} m/s<br>Reward: {step_rewards[k]:.3f}"
                        for k in range(seg_idx, run_end + 1)
                    ]

                    fig.add_trace(go.Scatter3d(
                        x=positions[pt_slice, 0],
                        y=positions[pt_slice, 1],
                        z=positions[pt_slice, 2],
                        mode="lines",
                        line=dict(color=color, width=bin_centers[cur_bin]),
                        name="UAV Trajectory" if not added_legend else None,
                        showlegend=not added_legend,
                        text=hover_texts,
                        hovertemplate=(
                            "<b>Position</b><br>"
                            "X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>"
                            "%{text}<extra></extra>"
                        ),
                    ))
                    added_legend = True
                    seg_idx = run_end

                # ── invisible trace for speed colorbar ───────────────
                fig.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode="markers",
                    marker=dict(
                        size=0.001,
                        color=[speed_min],
                        colorscale="Viridis",
                        cmin=speed_min, cmax=speed_max,
                        colorbar=dict(title="Speed (m/s)", x=1.05),
                        showscale=True,
                    ),
                    showlegend=False, hoverinfo="skip",
                ))

                # ── invisible trace for reward-width legend ──────────
                # (add annotation instead, since width can't have a colorbar)
                # placed at bottom-left to avoid overlapping the marker legend
                fig.add_annotation(
                    text=(
                        f"Line width = step reward<br>"
                        f"(thin={r_min:.2f}, thick={r_max:.2f})"
                    ),
                    xref="paper", yref="paper",
                    x=0.01, y=0.01,
                    xanchor="left", yanchor="bottom",
                    showarrow=False,
                    font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray", borderwidth=1,
                )

                # ── markers ──────────────────────────────────────────
                fig.add_trace(go.Scatter3d(
                    x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
                    mode="markers+text", marker=dict(size=8, color="black"),
                    text=["0"], textposition="top center",
                    textfont=dict(size=10, color="black"), name="Start",
                ))
                fig.add_trace(go.Scatter3d(
                    x=[positions[-1, 0]], y=[positions[-1, 1]], z=[positions[-1, 2]],
                    mode="markers+text", marker=dict(size=8, color="green"),
                    text=[str(len(positions) - 1)], textposition="top center",
                    textfont=dict(size=10, color="green"), name="End",
                ))
                fig.add_trace(go.Scatter3d(
                    x=[target_position[0]], y=[target_position[1]], z=[target_position[2]],
                    mode="markers", marker=dict(size=10, color="gold", symbol="diamond"),
                    name="Target",
                ))

                # ── step count markers along trajectory ──────────────
                n_pts = len(positions)
                n_markers = min(10, n_pts)
                if n_markers > 2:
                    marker_indices = np.linspace(1, n_pts - 2, n_markers - 2, dtype=int)
                    marker_indices = np.unique(marker_indices)  # deduplicate for short episodes
                    fig.add_trace(go.Scatter3d(
                        x=positions[marker_indices, 0],
                        y=positions[marker_indices, 1],
                        z=positions[marker_indices, 2],
                        mode="markers+text",
                        marker=dict(size=4, color="rgba(100,100,100,0.7)"),
                        text=[str(i) for i in marker_indices],
                        textposition="top center",
                        textfont=dict(size=9, color="rgba(80,80,80,0.9)"),
                        name="Step #",
                        hovertemplate="Step %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
                    ))

                # ── obstacles (AABB boxes) ───────────────────────────
                obstacles = self.current_eval_cycle_data["obstacles_history"][ep_idx]
                if obstacles:
                    i_f = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 4, 4]
                    j_f = [1, 3, 5, 7, 1, 4, 2, 5, 3, 6, 5, 0]
                    k_f = [2, 2, 6, 6, 5, 3, 6, 4, 7, 7, 1, 3]
                    for i, obs in enumerate(obstacles):
                        mn, mx = obs['aabb_min'], obs['aabb_max']
                        ov = np.array([
                            [mn[0],mn[1],mn[2]], [mx[0],mn[1],mn[2]],
                            [mx[0],mx[1],mn[2]], [mn[0],mx[1],mn[2]],
                            [mn[0],mn[1],mx[2]], [mx[0],mn[1],mx[2]],
                            [mx[0],mx[1],mx[2]], [mn[0],mx[1],mx[2]],
                        ])
                        fig.add_trace(go.Mesh3d(
                            x=ov[:,0], y=ov[:,1], z=ov[:,2],
                            i=i_f, j=j_f, k=k_f,
                            opacity=0.4, color='red', name=f'Obstacle {i+1}',
                            hovertemplate=f'<b>Obstacle {i+1}</b><extra></extra>',
                        ))

                fig.update_layout(
                    title=(
                        f"[{label}] UAV Trajectory "
                        f"(Reward: {reward:.2f}, Success: {success}, "
                        f"Collision: {collision}, Targets reached: {targets_reached})"
                    ),
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
                        x=0.02, y=0.98,
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="rgba(0,0,0,0.5)", borderwidth=1,
                    ),
                )

                wandb.log(
                    {f"eval_episode/{group_key}/trajectory": wandb.Html(fig.to_html(include_plotlyjs="cdn"))},
                    step=self.num_timesteps,
                )

        except Exception as e:
            logger.exception(f"[EvalCallback] Error creating trajectory plot: {e}")

    def _log_reward_decomposition_plot(self):
        """Create per-step reward decomposition line charts for picked trajectory episodes.

        For each representative episode, plots all reward components over time steps
        as separate lines, making it easy to see which components dominate and where
        penalties fire during the trajectory.
        """
        try:
            picked = self._pick_trajectory_episodes()
            if not picked:
                return

            COMPONENT_COLORS = {
                "reward_progress": "#2196F3",    # blue
                "reward_obstacle": "#FF9800",    # orange
                "reward_collision": "#F44336",   # red
                "reward_target": "#4CAF50",      # green
                "reward_step": "#9E9E9E",        # grey
                # obstacle sub-components — distinct hues, not just orange variants
                "reward_obs_approaching": "#E65100",           # deep orange
                "reward_obs_weighting": "#8E24AA",             # purple
                "reward_obs_mean_approach_penalty": "#BF360C", # brown-red
                "reward_obs_proximity": "#00897B",             # teal
            }
            DASH_STYLES = {
                "reward_obs_approaching": "dash",
                "reward_obs_weighting": "dot",
                "reward_obs_mean_approach_penalty": "dashdot",
                "reward_obs_proximity": "longdash",
            }

            for ep_idx, group_key, label in picked:
                step_rewards = self.current_eval_cycle_data["step_rewards_history"][ep_idx]
                if len(step_rewards) < 2:
                    continue

                n_steps = len(step_rewards)
                steps = np.arange(n_steps)

                fig = make_subplots(
                    rows=2, cols=1,
                    row_heights=[0.55, 0.45],
                    subplot_titles=("Per-step Reward Components", "Cumulative Reward Components"),
                    vertical_spacing=0.10,
                )

                # ── per-step lines ──
                for key in self.REWARD_COMPONENTS:
                    history = self.current_eval_cycle_data[f"{key}_history"][ep_idx]
                    if len(history) == 0:
                        continue
                    component_name = key.replace("reward_", "")
                    color = COMPONENT_COLORS.get(key, "gray")

                    dash = DASH_STYLES.get(key)
                    fig.add_trace(go.Scatter(
                        x=steps[:len(history)], y=history,
                        mode="lines", name=component_name,
                        line=dict(color=color, width=2, dash=dash),
                    ), row=1, col=1)

                    # ── cumulative lines ──
                    fig.add_trace(go.Scatter(
                        x=steps[:len(history)], y=np.cumsum(history),
                        mode="lines", name=component_name,
                        line=dict(color=color, width=2, dash=dash),
                        showlegend=False,
                    ), row=2, col=1)

                # total reward as dashed overlay
                fig.add_trace(go.Scatter(
                    x=steps, y=step_rewards,
                    mode="lines", name="total",
                    line=dict(color="black", width=1, dash="dot"),
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=steps, y=np.cumsum(step_rewards),
                    mode="lines", name="total",
                    line=dict(color="black", width=1, dash="dot"),
                    showlegend=False,
                ), row=2, col=1)

                reward = self.current_eval_cycle_data["episode_rewards"][ep_idx]
                success = self.current_eval_cycle_data["success"][ep_idx]
                collision = self.current_eval_cycle_data["collision"][ep_idx]

                fig.update_layout(
                    title=dict(
                        text=(
                            f"[{label}] Reward Decomposition "
                            f"(Total: {reward:.2f}, Success: {success}, Collision: {collision})"
                        ),
                        y=0.98,
                    ),
                    height=900,
                    width=900,
                    margin=dict(t=120, r=40),
                    legend=dict(orientation="h", y=1.10, x=0.5, xanchor="center"),
                )
                fig.update_xaxes(title_text="Step", row=2, col=1)
                fig.update_yaxes(title_text="Reward", row=1, col=1)
                fig.update_yaxes(title_text="Cumulative Reward", row=2, col=1)

                wandb.log(
                    {f"eval_episode/{group_key}/reward_decomposition": wandb.Html(fig.to_html(include_plotlyjs="cdn"))},
                    step=self.num_timesteps,
                )

        except Exception as e:
            logger.exception(f"[EvalCallback] Error creating reward decomposition plot: {e}")

    def _log_correlation_matrix(self):
        """Create and log correlation matrix plot to W&B"""
        try:
            df = pd.DataFrame(self.current_eval_cycle_data)
            history_cols = ["positions_history", "velocities_history", "target_position_history",
                           "obstacles_history", "step_rewards_history"]
            history_cols += [f"{k}_history" for k in self.REWARD_COMPONENTS]
            df = df.drop(columns=[c for c in history_cols if c in df.columns])

            if df.empty:
                logger.warning("[EvalCallback] Empty DataFrame for correlation matrix — skipping.")
                return

            df["path_efficiencies"] = df["path_efficiencies"].replace(0.0, np.nan)
            corr = df.corr()

            fig = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                colorscale="RdBu", zmid=0,
                text=corr.round(3).values, texttemplate="%{text}", textfont={"size": 10},
                hoverongaps=False, showscale=True, colorbar=dict(title="Correlation"),
            ))
            fig.update_layout(
                title="Correlation Matrix of UAV Performance Metrics",
                width=700, height=600,
                xaxis_showgrid=False, yaxis_showgrid=False,
                xaxis_zeroline=False, yaxis_zeroline=False,
            )

            wandb.log(
                {"eval_analysis/correlation_matrix": wandb.Html(fig.to_html(include_plotlyjs="cdn"))},
                step=self.num_timesteps,
            )

        except Exception as e:
            logger.exception(f"[EvalCallback] Error creating correlation matrix plot: {e}")

    # ──────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────

    def _reset_current_episode(self, env_idx=0):
        self.current_episode_data[env_idx] = {
            "positions": [],
            "velocities": [],
            "step_rewards": [],
            "start_position": None,
            "target_position": None,
            "start_time": self.num_timesteps,
            "obstacles": [],
            # per-step reward components (accumulated, then summed at episode end)
            **{k: [] for k in self.REWARD_COMPONENTS},
        }

    def _calculate_path_length(self, positions) -> float:
        """Calculate total distance traveled"""
        if len(positions) < 2:
            return 0.0
        positions = np.array(positions)
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return float(np.sum(distances))

    def _calculate_average_velocity(self, velocities) -> float:
        """Calculate average velocity magnitude"""
        if len(velocities) == 0:
            return 0.0
        velocities = np.array(velocities)
        return float(np.mean(np.linalg.norm(velocities, axis=1)))

    def _calculate_path_efficiency(self, start_pos, target_pos, path_length) -> float:
        """Calculate path efficiency (path_length / direct_distance)"""
        direct_distance = np.linalg.norm(target_pos - start_pos)
        if direct_distance == 0:
            return 0.0
        return path_length / direct_distance

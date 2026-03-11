"""
Reward function (single source of truth for all reward shaping):
    r_t = w_prog * r_prog + w_obs * r_obs + r_collision + r_target + r_step

    r_prog      = d_{t-1} - d_t                                              (progress toward target)
    r_obs       = -[max(0, exp(-k*d_min) - exp(-k*d_thresh))]^2               (obstacle avoidance)
    r_collision = -C  if collision, else 0                                    (terminal penalty)
    r_target    = +B  if target reached, else 0                               (waypoint bonus)
    r_step      = -P  every step                                              (alive/time penalty)

Config (added to YAML under `reward:`):
    k:                      float   # obstacle penalty steepness
    d_thresh:               float   # obstacle activation range
    w_prog:                 float   # weight for progress reward term
    w_obs:                  float   # weight for obstacle avoidance reward term
    collision_penalty:      float   # C
    target_reached_bonus:   float   # B
    step_penalty:           float   # P (subtracted each step to encourage speed)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

from uav_nav_obstacle_avoidance_rl import config

logger = config.logger

if TYPE_CHECKING:
    pass

class CustomRewardWrapper(gym.Wrapper):
    """
    lidar-aware reward wrapper (simplified squared-exponential obstacle penalty)

    uses agent-rate distance delta for progress reward and a single scalar
    function of d_min for obstacle avoidance — no per-ray computation, no
    previous lidar state needed

    wrapped env must populate these info keys every step:
        "collision"               bool
        "target_reached"          bool
        "sub_distance"            list[float]  (one per physics sub-step; last element used as agent-rate distance)
    """

    def __init__(
        self,
        env: gym.Env,
        k: float,
        d_thresh: float,
        w_prog: float,
        w_obs: float,
        collision_penalty: float,
        target_reached_bonus: float,
        step_penalty: float,
    ):
        super().__init__(env)

        assert k > 0, "k must be > 0"
        assert d_thresh > 0, "d_thresh must be > 0"

        self.k = k
        self.d_thresh = d_thresh
        self.w_prog = w_prog
        self.w_obs = w_obs
        self.collision_penalty = collision_penalty
        self.target_reached_bonus = target_reached_bonus
        self.step_penalty = step_penalty

        # precompute threshold exponential
        self._exp_thresh = np.exp(-k * d_thresh)

        # state tracking across steps
        self._prev_distance: float | None = None

    # –– gym stuff ––––––––––––––––––––––––––––––––––
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_distance = None
        return obs, info

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)

        # –– r_prog: simple agent-rate distance delta ––
        r_prog = self._compute_progress_reward(info["sub_distance"][-1])

        # –– r_obs: squared exponential of d_min ––
        r_obs = self._compute_obstacle_reward(obs['lidar'])

        # –– r_collision ––
        r_collision = -self.collision_penalty if info["collision"] else 0.0

        # –– r_target ––
        r_target = self.target_reached_bonus if info["target_reached"] else 0.0

        # –– r_step ––
        r_step = -self.step_penalty

        # –– total ––
        reward = self.w_prog * r_prog + self.w_obs * r_obs + r_collision + r_target + r_step

        # log components for wandb
        info["reward_step"] = r_step
        info["reward_progress"] = r_prog
        info["reward_obstacle"] = r_obs
        info["reward_collision"] = r_collision
        info["reward_target"] = r_target
        info["reward_total"] = reward

        return obs, reward, terminated, truncated, info

    # –– reward components ––––––––––––––––––––––––––––––––––––––
    def _compute_progress_reward(self, distance: float) -> float:
        """r_prog = d_{t-1} - d_t. Positive when getting closer."""
        if self._prev_distance is None:
            self._prev_distance = distance
            return 0.0

        r_prog = self._prev_distance - distance
        self._prev_distance = distance
        return r_prog

    def _compute_obstacle_reward(self, lidar: np.ndarray) -> float:
        """r_obs = -[max(0, exp(-k*d_min) - exp(-k*d_thresh))]^2"""
        d_min = float(np.min(lidar))
        inner = np.exp(-self.k * d_min) - self._exp_thresh
        if inner <= 0.0:
            return 0.0
        return -(inner * inner)


class PyFlytRewardWrapper(gym.Wrapper):
    """
    replicates the original reward logic of QuadXBaseEnv.step(), VectorVoyagerEnv.compute_term_trunc_reward()

    wrapped env must populate these info keys every step:
        "collision"               bool
        "target_reached"          bool
        "sub_progress"            list[float]  (one per physics sub-step)
        "sub_distance"            list[float]  (one per physics sub-step)
    """

    def __init__(
        self,
        env: gym.Env,
        step_penalty: float,
        collision_penalty: float,
        target_reached_bonus: float,
        sparse_reward: bool = False,
    ):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty
        self.target_reached_bonus = target_reached_bonus
        self.sparse_reward = sparse_reward

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)

        # ––– dense shaping: accumulate over all physics sub-steps (matches native behavior)
        r_step = -self.step_penalty
        r_progress = 0.0
        r_distance = 0.0
        r_yaw = 0.0
        if not self.sparse_reward:
            # physics sub-steps env-/agent-refresh-rate: 120Hz/30Hz
            for progress, distance in zip(info["sub_progress"], info["sub_distance"]):
                r_progress += max(3.0 * progress, 0.0)
                r_distance += 0.1 / distance
                # yaw_rate = abs(base_env.env.state(0)[0][2])  # angular velocity z-component
                # r_yaw = -0.01 * yaw_rate ** 2

        # collision overrides
        r_collision = -self.collision_penalty if info["collision"] else 0.0

        # target-reached overrides everything
        r_target = self.target_reached_bonus if info["target_reached"] else 0.0

        # –– total ––
        reward = r_step + r_progress + r_distance + r_yaw + r_collision + r_target

        # log components for wandb
        info["reward_step"] = r_step
        info["reward_progress"] = r_progress
        info["reward_distance"] = r_distance
        # info["reward_yaw"] = r_yaw
        info["reward_collision"] = r_collision
        info["reward_target"] = r_target
        info["reward_total"] = reward

        return obs, reward, terminated, truncated, info
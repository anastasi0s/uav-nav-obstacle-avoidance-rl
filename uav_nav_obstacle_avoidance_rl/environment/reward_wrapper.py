from uav_nav_obstacle_avoidance_rl import config

logger = config.logger

import gymnasium as gym


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

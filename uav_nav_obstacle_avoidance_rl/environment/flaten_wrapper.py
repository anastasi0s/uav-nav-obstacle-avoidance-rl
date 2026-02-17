from __future__ import annotations

from gymnasium.core import Env, ObservationWrapper
from gymnasium.spaces import Box
import numpy as np
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler

# Vendored and adopted from PyFlyt's `FlattenWaypointEnv`.
# Original source: https://github.com/jjshoots/PyFlyt/blob/master/PyFlyt/gym_envs/utils/flatten_waypoint_env.py 
# This wrapper flattens the waypoint-based observation space for compatibility with vector-based agents.
# Modifications:
# - Renamed class to `FlattenVectorVoyagerEnv`
# - Modified dtype of observation_space and targets
"""Wrapper class for flattening the waypoint envs to use homogeneous observation spaces."""
class FlattenVectorVoyagerEnv(ObservationWrapper):
    def __init__(self, env: Env, context_length):
        """__init__.

        Args:
            env (Env): a PyFlyt Waypoints environment.
            context_length: how many waypoints should be included in the flattened observation space.

        """
        super().__init__(env=env)
        if not hasattr(env, "waypoints") and not isinstance(
            env.unwrapped.waypoints,  # type: ignore[reportAttributeAccess]
            WaypointHandler,
        ):
            raise AttributeError(
                "Only a waypoints environment can be used with the `FlattenWaypointEnv` wrapper."
            )
        self.context_length = context_length
        self.attitude_shape = env.observation_space["attitude"].shape[0]  # type: ignore [reportGeneralTypeIssues]
        self.target_shape = env.observation_space["target_deltas"].feature_space.shape[  # type: ignore [reportGeneralTypeIssues]
            0
        ]  # type: ignore [reportGeneralTypeIssues]
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.attitude_shape + self.target_shape * self.context_length,),
            dtype=np.float32,
        )
    
    def observation(self, observation) -> np.ndarray:
        """Flattens an observation from the super env.

        Args:
            observation: a dictionary observation with an "attitude" and "target_deltas" keys.

        """
        num_targets = min(
            self.context_length, observation["target_deltas"].shape[0]
        )  # pyright: ignore[reportGeneralTypeIssues]

        targets = np.zeros((self.context_length, self.target_shape), dtype=np.float32)
        targets[:num_targets] = observation["target_deltas"][
            :num_targets
        ]  # pyright: ignore[reportGeneralTypeIssues]

        new_obs = np.concatenate(
            [observation["attitude"], *targets]
        )  # pyright: ignore[reportGeneralTypeIssues]

        return new_obs
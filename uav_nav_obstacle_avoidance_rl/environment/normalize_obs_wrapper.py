"""
Fixed observation normalization for VectorVoyagerEnv.

Scales each observation component by its known physical range so that
values seen by the policy network are approximately in [-1, 1].

Placement in wrapper chain (before flatten, after lidar if used):
    VectorVoyagerEnv -> RescaleAction -> [LidarObservationWrapper] -> NormalizeObservationWrapper -> Flatten
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NormalizeObservationWrapper(gym.ObservationWrapper):
    """
    fixed static observation normalization based on physical ranges. Inverting lidar scans.

    Attitude indices and their scales (derived from DJI Inspire 3 specs / env config):
        0   yaw angular velocity     ~+-5 rad/s        -> / 5.0 using 5.0 instead of 2.62 (obs space) gives headroom so the value doesn't saturate at 1.0 constantly during transients
        1   cos(yaw)                 [-1, 1]           -> / 1.0  (no-op)
        2   sin(yaw)                 [-1, 1]           -> / 1.0  (no-op)
        3   body-frame vel u         +-26 m/s          -> / 26.0
        4   body-frame vel v         +-26 m/s          -> / 26.0
        5   body-frame vel w         +-8 m/s           -> / 8.0
        6   height z                 [0, z_size]       -> shift + scale to [-1, 1]
        7   prev action u            +-26              -> / 26.0
        8   prev action v            +-26              -> / 26.0
        9   prev action vr           +-2.62            -> / 2.62
       10   prev action vz           +-8               -> / 8.0

    Lidar (if present):
        [min_range, max_range] -> [1, 0]  (inverted: 1 = obstacle at min_range, 0 = nothing detected)

    Target deltas:
        / max(grid_sizes) -> approximately [-1, 1]
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        base_env = env.unwrapped
        z_size = base_env.z_size

        # -- attitude: normalized = (value - offset) / scale --------  z = 2 × ((x - xmin) / (xmax - xmin)) - 1 -> [-1, 1]
        self._att_offset = np.zeros(11, dtype=np.float32)
        self._att_offset[6] = z_size / 2.0  # center height around 0

        self._att_scale = np.array([
            5.0,           # 0: yaw rate (rad/s)
            1.0,           # 1: cos(yaw)
            1.0,           # 2: sin(yaw)
            26.0,          # 3: body vel u (m/s)
            26.0,          # 4: body vel v (m/s)
            8.0,           # 5: body vel w (m/s)
            z_size / 2.0,  # 6: height -> [-1, 1]
            26.0,          # 7: prev action u
            26.0,          # 8: prev action v
            2.62,          # 9: prev action vr
            8.0,           # 10: prev action vz
        ], dtype=np.float32)

        # -- lidar: d_norm = (d_measured - min) / (max - min) -> [0, 1] ----------
        self._has_lidar = "lidar" in env.observation_space.spaces
        if self._has_lidar:
            lo = float(env.observation_space["lidar"].low[0])
            hi = float(env.observation_space["lidar"].high[0])
            self._lidar_min = lo
            self._lidar_range = hi - lo

        # -- target deltas: value / max_dim -> ~[-1, 1] ------------
        self._target_scale = float(max(base_env.grid_sizes))

        self._build_observation_space()

    def _build_observation_space(self):
        orig = self.env.observation_space
        new_spaces = {}

        new_spaces["attitude"] = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=orig["attitude"].shape, dtype=np.float32,
        )

        if self._has_lidar:
            new_spaces["lidar"] = spaces.Box(
                low=0.0, high=1.0,
                shape=orig["lidar"].shape, dtype=np.float32,
            )

        td = orig["target_deltas"]
        new_spaces["target_deltas"] = spaces.Sequence(
            space=spaces.Box(
                low=-2.0, high=2.0,
                shape=td.feature_space.shape,
                dtype=np.float32,
            ),
            stack=True,
        )

        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, obs):
        result = {
            "attitude": (obs["attitude"] - self._att_offset) / self._att_scale,
            # "attitude": obs["attitude"],
            "target_deltas": obs["target_deltas"] / self._target_scale,
            # "target_deltas": obs["target_deltas"],
        }

        if self._has_lidar and "lidar" in obs:
            result["lidar"] = (obs["lidar"] - self._lidar_min) / self._lidar_range

        return result

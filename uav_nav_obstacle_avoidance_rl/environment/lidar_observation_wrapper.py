from typing import Any, Literal, Tuple, Dict
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class LidarObservationWrapper(gym.ObservationWrapper):
    """
    adds raycasting-based obstacle detection to the observation space.
    it is simulating a LiDAR sensor with configurable range and resolution.
    casts rays from UAV position and returns normalised distances.

    Args:
        env: the base environment with a PyBullet client accessible via env.env
        num_rays_horizontal:
        num_rays_vertical: num of vertical layers (elevation). use 1 for 2D LIDAR
        max_range: max detection range in m
        min_range: blind zone
        fov_horizontal: horizontal field of view in degrees (360 for full sweep)
        fov_vertical: (centered around horizontal)
        ray_start_offset: offset from UAV center to start rays (avoids self-intersection)
        normalize_distances: If True, distances are normalized to [0, 1]
        add_to_obs: How to add lidar data to observations ('append', 'replace', 'separate')
    
    Observation Space:
        If add_to_obs='append': Lidar readings appended to 'attitude' in dict obs
        If add_to_obs='separate': New 'lidar' key added to dict observation
        If add_to_obs='replace': Only lidar observations returned (for testing)
    """

    def __init__(
            self,
            env: gym.Env,
            num_rays_horizontal: int = 36,
            num_rays_vertical: int = 1,
            max_range: float = 10.0,
            min_range: float = 0.1,
            fov_horizontal: float = 360.0,
            fov_vertical: float = 30.0,
            ray_start_offset: float = 0.15,
            normalize_distances: bool = True,
            add_to_obs: Literal["append", "separate", "replace"] = "separate",
    ):
        super().__init__(env)

        # LIDAR config.
        self.num_rays_horizontal = num_rays_horizontal
        self.num_rays_vertical = num_rays_vertical
        self.num_rays_total = num_rays_horizontal * num_rays_vertical
        self.max_range = max_range
        self.min_range = min_range
        self.fov_horizontal = math.radians(fov_horizontal)  # convert angles: degree -> radiants
        self.fov_vertical = math.radians(fov_vertical)
        self.ray_start_offset = ray_start_offset
        self.normalize_distances = normalize_distances
        self.add_to_obs = add_to_obs

        # precompute ray directions in local frame (body frame)
        self._ray_directions = self._compute_ray_directions()

        # modify observation space
        self._setup_observation_space()

    def _compute_ray_directions(self) -> np.ndarray:
        """
        precompute ray direction unit vectors in the body frame

        returns: array (num_rays_total, 3) with unit direction vectors
        """
        directions = []
        
        # horizontal angles (azimuth)
        if self.num_rays_horizontal == 1:
            h_angles = [0.0]  # one ray at the center
        else:
            # distribute rays evenly across horizontal FOV
            h_start = -self.fov_horizontal / 2
            h_step = self.fov_horizontal / self.num_rays_horizontal  # NOTE adjust this to: self.num_rays_horizontal - 1  if partial FOV is used!
            h_angles = [h_start + i * h_step for i in range(self.num_rays_horizontal)]
        
        # vertical angles (elevation)
        if self.num_rays_vertical == 1:
            v_angles = [0.0]  # single horizontal plane
        else:
            # distribute rays evenly across vertical FOV
            v_start = -self.fov_vertical / 2
            v_step = self.fov_vertical / (self.num_rays_vertical - 1)
            v_angles = [v_start + i * v_step for i in range(self.num_rays_vertical)]

        # generate all ray directions
        for v_angle in v_angles:
            for h_angle in h_angles:
                # spherical to cartesian conversion
                # x: forward, y: left, z: up (body frame)
                cos_v = math.cos(v_angle)  # cos(v_angle) -> use elevation angle (v_angle is the angle from the horizontal plane (elevation))
                x = cos_v * math.cos(h_angle)
                y = cos_v * math.sin(h_angle)
                z = math.sin(v_angle)
                directions.append([x, y, z])

        return np.array(directions, dtype=np.float32)
    
    def _setup_observation_space(self):
        """config the new observation space with LIDAR data"""
        lidar_space = spaces.Box(
            low=0.0,
            high=1.0 if self.normalize_distances else self.max_range,
            shape=(self.num_rays_total,),
            dtype=np.float32,
        )
        if self.add_to_obs == "separate":
            # add 'lidar' key to existing observation space
            new_spaces = dict(self.env.observation_space.spaces)
            new_spaces["lidar"] = lidar_space
            self.observation_space = spaces.Dict(new_spaces)
        # TODO add the other options ?

    def _get_uav_pose(self):
        """ger current UAV position and orientation
        
        returns:
            position: (3,) array of world coordiantes
            orientation: (4,) [x,y,z,w]
        """
        # access the aviary (PyBullet client) through the env
        aviary = self.env.env  # VectorVoyagerEnv -> Aviary
        drone_id = aviary.drones[0].Id
        
        # pos = [x, y, z] - position in meters
        # orn = [qx, qy, qz, qw] - orientation as quaternion
        pos, orn = aviary.getBasePositionAndOrientation(drone_id)  # PyBullet call
        return np.array(pos), np.array(orn)
    
    def _rotate_directions_to_world(
            self,
            directions: np.ndarray,
            orientation: np.ndarray,
    ):
        """
        rotate ray directions from body frame to world frame.

        Note on quaternion representaion of orientation in 3D:
            Rotation of θ degrees around axis (ax, ay, az):
                x = ax * sin(θ/2)  ─┐
                y = ay * sin(θ/2)   ├─ WHICH axis to rotate around
                z = az * sin(θ/2)  ─┘
                w = cos(θ/2)  ────────── HOW MUCH to rotate
            
        args:
            directions: (N, 3) array of direction vectors in body frame
            orientation (4,) quaternion [x, y, z, w]

        returns: (N, 3) array of direction vectors in world frame
        """
        aviary = self.env.env
        # get rotation matrix form quaternion
        rot_matrix = np.array(aviary.getMatrixFromQuaternion(orientation)).reshape(3, 3)

        # rotate all directions by applying the transformation 
        return np.dot(directions, rot_matrix.T)
    
    def _cast_rays(self):
        """
        perform ray casting and return distances to obstacles

        returns:
            array of shape (num_rays_total,) with distances (or normalized distances)
        """
        aviary = self.env.env

        # get UAV pose
        position, orientation = self._get_uav_pose()

        # rotate ray directions to world frame
        world_directions = self._rotate_directions_to_world(self._ray_directions, orientation)

        # compute ray start and end points
        ray_starts = position + world_directions * self.ray_start_offset
        ray_ends = position + world_directions * self.max_range

        # batch raycast (efficient)
        results = aviary.rayTestBatch(
            rayFromPositions=ray_starts.tolist(),
            rayToPositions=ray_ends.tolist(),
        )

        # extract distances from results
        distances = np.zeros(self.num_rays_total, dtype=np.float32)
        
        ray_length = self.max_range - self.ray_start_offset
        for i, result in enumerate(results):
            object_id, link_idx, hit_fraction, hit_pos, hit_normal = result

            if object_id == -1:
                # no hit -> return max range
                distances[i] = self.max_range
            else:
                # calculate the actual distance from the hit fraction
                distances[i] = self.ray_start_offset + hit_fraction * ray_length

        # apply minimum range (blind zone) # NOTE redundant!! just in case for smaller offsets
        distances = np.clip(distances, self.min_range, self.max_range)

        # normalize if requested 
        if self.normalize_distances:
            distances = (distances - self.min_range) / (self.max_range - self.min_range)
        
        return distances
    
    def observation(self, observation: Dict):
        """
        add LIDAR data to observation.

        Args:
            observation: original observation from base env

        returns:
            modified observation with LIDAR readings
        """

        lidar_data = self._cast_rays()

        if self.add_to_obs  == "separate":
            return {**observation, "lidar": lidar_data}
        # TODO add the other options ?

        
        return observation
    
    def get_ray_debug_lines(self):
        """
        get ray start/end position for debug visualization

        returns:
            ray_starts (N, 3)
            ray_ends (N, 3) positions at hit point or max range
        """
        position, orientation = self._get_uav_pose()
        world_directions = self._rotate_directions_to_world(self._ray_directions, orientation)

        ray_starts = position + world_directions * self.ray_start_offset
        ray_ends = position + world_directions * self.max_range
        
        return ray_starts, ray_ends
    
    def render_lidar_debug(self, duration: float = 0.1):
        """
        Render LiDAR rays as debug lines in PyBullet visualizer.
        
        Args:
            duration: How long the debug lines persist (seconds)
        """
        aviary = self.env.env
        position, orientation = self._get_uav_pose()
        world_directions = self._rotate_directions_to_world(
            self._ray_directions, orientation
        )
        
        ray_starts = position + world_directions * self.ray_start_offset
        ray_ends = position + world_directions * self.max_range
        
        # Cast rays to get hit points
        results = aviary.rayTestBatch(
            rayFromPositions=ray_starts.tolist(),
            rayToPositions=ray_ends.tolist(),
        )
        
        for i, result in enumerate(results):
            object_id, _, hit_fraction, hit_pos, _ = result
            
            start = ray_starts[i]
            
            if object_id == -1:
                # No hit - draw green ray to max range
                end = ray_ends[i]
                color = [0, 1, 0]  # Green
            else:
                # Hit - draw red ray to hit point
                end = hit_pos
                color = [1, 0, 0]  # Red
            
            aviary.addUserDebugLine(
                lineFromXYZ=start.tolist(),
                lineToXYZ=list(end),
                lineColorRGB=color,
                lineWidth=1,
                lifeTime=duration,
            )

class LidarFlattenWrapper(gym.ObservationWrapper):
    """
    flatten dict observations with LIDAR into a single vector
    """

    def __init__(self, env: gym.Env, context_length: int = 1):
        super().__init__(env)

        self.context_length = context_length

        # calculate observation size
        attitude_size = env.observation_space["attitude"].shape[0]
        lidar_size = env.observation_space["lidar"].shape[0]
        target_size = env.observation_space["target_deltas"].feature_space.shape[0]

        total_size = attitude_size + lidar_size + (target_size * context_length)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_size,),
            dtype=np.float32,
        )

        self._attitude_size = attitude_size
        self._lidar_size = lidar_size
        self._target_size = target_size

    def observation(self, observation: Dict):
        """flatten observation dict to vector"""
        # handle target deltas (pad if fewer than context_length)
        num_targets = min(self.context_length, observation["target_deltas"].shape[0])
        targets = np.zeros((self.context_length, self._target_size), dtype=np.float32)
        targets[:num_targets] = observation["target_deltas"][:num_targets]

        return np.concatenate([
            observation["attitude"],
            observation["lidar"],
            targets.flatten(),
        ]).astype(np.float32)
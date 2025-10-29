from typing import Any, Literal, Dict, Tuple

import gymnasium as gym
import numpy as np
from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler
from uav_nav_obstacle_avoidance_rl import config

logger = config.logger
rng = np.random.default_rng(config.RANDOM_SEED)

class VoxelGrid:
    """ 3d voxel grid for discretizing the space and placing obstacles"""

    def __init__(self, grid_sizes: Tuple[float, float, float], voxel_size: float):
        """
        initialize voxel grid

        args:
            grid_sizes: the size of each grid dimension in meter (x_size, y_size, z_size)
            voxel_size: size of each voxel cube in meter
        """
        self.x_size, self.y_size, self.z_size = grid_sizes
        
        # calculate min, max of each dimension
        x_magnitude, y_magnitude, z_magnitude = self.x_size / 2, self.y_size / 2, self.z_size / 2  # calculate magnitude from origin of each dimension
        self.x_min, self.x_max = -x_magnitude, x_magnitude
        self.y_min, self.y_max = -y_magnitude, y_magnitude
        self.z_min, self.z_max = -z_magnitude, z_magnitude
        self.voxel_size = voxel_size

        # claculate grid dimensions 
        self.nx = int(np.floor(self.x_size / self.voxel_size))  # floor division -> voxel grid might be slightly smaller than space
        self.ny = int(np.floor(self.y_size / self.voxel_size))
        self.nz = int(np.floor(self.z_size / self.voxel_size))

        # initialize occupancy grid
        self.grid = np.zeros((self.nx, self.ny, self.nz), dtype=bool)

        print(f"Voxel Grid initialized: {self.nx}x{self.ny}x{self.nz} voxels")
        print(f"Total voxels: {self.nx * self.ny * self.nz}")

    def world_to_voxel(self, position: np.ndarray) -> Tuple[int, int, int]:
        """convert 3d world coordinates into voxel indices"""
        x, y, z = position
        i = int(np.floor((x - self.x_min) / self.voxel_size))
        j = int(np.floor((y - self.y_min) / self.voxel_size))
        k = int(np.floor(z - self.z_min / self.voxel_size))
        return i, j, k  # voxel indices
    
    def voxel_to_world(self, voxel_idx: Tuple[int, int, int]) -> np.ndarray:
        """convert voxel idx to cartesian world position coordinates (+0.5: conter of voxel)"""
        i, j, k = voxel_idx
        x = self.x_min + (i + 0.5) * self.voxel_size
        y = self.y_min + (i + 0.5) * self.voxel_size
        z = self.z_min + (i + 0.5) * self.voxel_size
        return np.array([x, y, z])  # cartesian position
    
    def is_voxel_free(self, voxel_idx: Tuple[int, int, int]) -> bool:
        """check if a voxel is free (not occupied)"""
        i, j, k = voxel_idx
        # check if voxel_idx is inside bounds
        if (0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz):
            free = not self.grid[i, j, k]
        else:
            logger.debug(f"OUT OF BOUNDS INDEX: Trying to access out of bounds grid index: ({i}, {j}, {k})")
        return free
    
    def mark_voxel_occupied(self, voxel_idx: Tuple[int, int, int]):
        i, j, k = voxel_idx
        if (0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz):
            self.grid[i, j, k] = True
        else:
            logger.debug(f"OUT OF BOUNDS INDEX: Trying to access out of bounds grid index: ({i}, {j}, {k})")

    def get_random_free_voxel(self) -> Tuple[int, int, int]:
        """get free grid position"""
        free_voxels = np.argwhere(~self.grid)
        if len(free_voxels) == 0:
            logger.error("No free voxels available!")
            raise ValueError("No free voxels available!")
        idx = rng.choice(free_voxels)
        return tuple(idx)
    
    def get_random_free_position(self) -> np.ndarray:
        """ger free cartesian position"""
        voxel_idx = self.get_random_free_voxel()
        position_coordinates = self.voxel_to_world(voxel_idx)
        return position_coordinates


# my waypoint environment based on PyFlyt/QuadX-Waypoints-v3
class VectorVoyagerEnv(QuadXBaseEnv):
    def __init__(
        self,
        sparse_reward: bool = False,
        num_targets: int = 1,
        use_yaw_targets: bool = False,  # toggles whether the agent must also align its yaw (heading) to a per‐waypoint target before that waypoint is considered “reached,” and whether yaw error is included in the observation.
        flight_mode: int = 5,  # uav constrol mode 5: (u, v, vr, vz) -> u: local velocity forward in m/s, v: lateral velocity in m/s, vr: yaw in rad/s, vz: vertical velocity in m/s
        flight_dome_size: float = 5.0,  # 5.0 m = default, the radius of the “flight dome” within which the UAV must remain to avoid an out‑of‑bounds termination
        goal_reach_distance: float = 0.2,  # distance within which the target is considered reached
        goal_reach_angle=0.1,  # not in use since use_yaw_targets is not in use
        max_duration_seconds: float = 80.0,  # max simulation time of the env
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,  # looprate of the agent to environment interaction
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
        min_height: float = 0.1,  # min allowed hight, collision is detected if below that height
    ):
        # init parent class (QuadxBaseEnv)
        super().__init__(
            start_pos=np.array(
                [[0.0, 0.0, 1.0]]
            ),  # uav starting at pos [0, 0, 1] # TODO implement random spawning
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,  # its rquired in the base class but its not used in this env
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        self.num_targets = num_targets
        self.sparse_reward = sparse_reward
        self.flight_dome_size = flight_dome_size


        # define waypoint navigation – init waypoint handler
        self.waypoints = WaypointHandler(
            enable_render=self.render_mode is not None,
            num_targets=num_targets,
            use_yaw_targets=use_yaw_targets,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=goal_reach_angle,
            flight_dome_size=flight_dome_size,
            min_height=min_height,
            np_random=self.np_random,
        )

        # according to the DJI Inspire 3 specifications:
        #   u, v (horizontal velocities): ±26 m/s,
        #   vr (yaw rate): ±150°/s, which is ±2.62 rad/s,
        #   vz (vertical velocity, falling speed): ±8 m/s.
        # define action space
        self.action_space = gym.spaces.Box(  # u, v, vr, vz
            low=np.array([-26.0, -26.0, -2.62, -8.0], np.float32),
            high=np.array([26.0, 26.0, 2.62, 8.0], np.float32),
            dtype=np.float32,
        )

        # define observation space
        self.observation_space = gym.spaces.Dict(
            {
                "attitude": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        10,
                    ),  # 10 = 1 (yaw position) + 1 (yaw vel) + 3 (lin_vel) + 1 (height) + 4 (previous actions)
                    dtype=np.float32,
                ),
                "target_deltas": gym.spaces.Sequence(
                    space=gym.spaces.Box(
                        low=-2 * flight_dome_size,
                        high=2 * flight_dome_size,
                        shape=(4,) if use_yaw_targets else (3,),
                        dtype=np.float32,
                    ),
                    stack=True,  # True = convert the list of Box samples into one stacked NumPy array of shape (num_targets, 3), used with LSTM
                ),
            }
        )

    def reset(
        self,
        *,
        seed: None | int = None,
        options: None | dict[str, Any] = dict(),
        drone_options: None | dict[str, Any] = None,
    ) -> tuple[dict[Literal["attitude", "target_deltas"], np.ndarray], dict]:
        """
        rest the env: resets simulation state, the waypoint hnadler, and any other counters
        """
        # start reset procedure using the base env's methods
        super().begin_reset(seed, drone_options=drone_options)
        self.waypoints.reset(
            self.env, self.np_random
        )  # reset waypoint handler, which sets the current target

        # init tracked metrics
        self.info["num_targets_reached"] = 0
        super().end_reset()  # finish reset using base env producers

        return self.state, self.info

    # this is called in the step function of the QuadXBaseEnv parent class -> new observation
    def compute_state(self) -> None:
        # compute state of the uav, using the base env
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()

        # create empty array of type float32 and fill it
        attitude = np.empty(10, dtype=np.float32)
        attitude[0] = ang_vel[2]  # (1,) - yaw, rotation velocity
        attitude[1] = ang_pos[2]  # (1,) - yaw, rotation position
        attitude[2:5] = lin_vel  # (3,) - body frame linear velocity vector (u, v, w)
        attitude[5] = lin_pos[2]  # (1,) - z position
        attitude[6:10] = (self.action)  # (4,) - previous actions  # TODO check for other methods to capture temporal information of taken actions

        # # create empty array of type float32 -> compute the target deltas with the method from the waypointhandler -> fill array
        # target_deltas = np.empty((1,3), dtype=np.float32)
        # target_deltas[0, :] = self.waypoints.distance_to_targets(ang_pos, lin_pos, quaternion)
        target_deltas = self.waypoints.distance_to_targets(ang_pos, lin_pos, quaternion).astype(np.float32)

        # update the current state
        self.state = {"attitude": attitude, "target_deltas": target_deltas}

    # this is called in the step function of the QuadXBaseEnv parent class
    def compute_term_trunc_reward(self) -> None:
        """
        compute termination, truncation, and reward of the current timestep

        use reward and termination logic from pyflyt env (for now -> # TODO add custom reward function)
        """

        # call the base computation function
        super().compute_base_term_trunc_reward()

        # if not using sparse reward, add bonus rewards related to waypoint progression
        if not self.sparse_reward:
            self.reward += max(3.0 * self.waypoints.progress_to_next_target, 0.0)
            self.reward += 0.1 / self.waypoints.distance_to_next_target

        # on reaching the target waypoint
        if self.waypoints.target_reached:
            self.reward = 100.0  # large reward bonus
            
            # go to the next target if available
            self.waypoints.advance_targets()
            
            # update termination: if all targets are reached, signal environment completion
            self.truncation |= self.waypoints.all_targets_reached
            self.info["env_complete"] = self.waypoints.all_targets_reached
            self.info["num_targets_reached"] = self.waypoints.num_targets_reached


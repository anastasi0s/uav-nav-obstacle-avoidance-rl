from collections.abc import Sequence
from typing import Any, Literal
import os

import gymnasium as gym
import numpy as np
from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler

from uav_nav_obstacle_avoidance_rl import config
from uav_nav_obstacle_avoidance_rl.environment.voxel_grid import VoxelGrid

logger = config.logger


# my waypoint environment based on PyFlyt/QuadX-Waypoints-v3
class VectorVoyagerEnv(QuadXBaseEnv):
    def __init__(
        self,
        # boundary parameters
        grid_sizes: Sequence[float],  # (x, y, z)
        voxel_size: float,
        min_height: float = 0.0,  # min allowed hight, collision is detected if below that height
        # waypoint parameters
        num_targets: int = 1,
        sparse_reward: bool = False,
        use_yaw_targets: bool = False,  # toggles whether the agent must also align its yaw (heading) to a per‐waypoint target before that waypoint is considered “reached,” and whether yaw error is included in the observation.
        goal_reach_distance: float = 0.2,  # distance within which the target is considered reached
        goal_reach_angle=0.1,  # not in use since use_yaw_targets is not in use
        # obstacle parameters
        obstacle_strategy: str = "random",  # "random"
        num_obstacles: int = 3,
        visual_obstacles: bool = False,  # only for evaluation
        # simulation parameters
        max_duration_seconds: float = 80.0,  # max simulation time of the env
        flight_mode: int = 5,  # uav constrol mode 5: (u, v, vr, vz) -> u: local velocity forward in m/s, v: lateral velocity in m/s, vr: yaw in rad/s, vz: vertical velocity in m/s
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,  # looprate of the agent to environment interaction
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: Sequence[int] = (480, 480),
    ):
        # init parent class (QuadxBaseEnv)
        super().__init__(
            start_pos=np.array(
                [[0.0, 0.0, 1.0]]
            ),  # uav starting at pos [0, 0, 1]. Is randomized and overwritten in reset()
            flight_mode=flight_mode,
            # flight_dome_size=flight_dome_size,  # this is used only in a method of the parent class which has been overwritten by my custom env
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,  # its rquired in the base class but its not used in this env
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=(render_resolution[0], render_resolution[1]),
        )

        # boundary parameters
        self.grid_sizes = grid_sizes
        self.voxel_size = voxel_size
        self.min_height = min_height
        self.max_dimension_size = max(
            self.grid_sizes
        )  # tha dimension with the highest value
        # waypoint parameters
        self.num_targets = num_targets
        self.sparse_reward = sparse_reward
        # obstacle parameters
        self.obstacle_strategy = obstacle_strategy
        self.num_obstacles = num_obstacles
        self.visual_obstacles = visual_obstacles
        self.obstacles = []  # store PyBullet body IDs
        self.boundary_wall_ids = []  # # store PyBullet body IDs for walls

        # Note: obstacles are created in the first reset() call when self.env (Aviary/Pybullet client) is available. Pybullet connections from gym are created after env reset()

        # init voxel grid
        self.voxel_grid = VoxelGrid(grid_sizes, voxel_size, rng=self.np_random)

        # init waypoint handler
        self.waypoints = WaypointHandler(
            enable_render=self.render_mode is not None,
            num_targets=num_targets,
            use_yaw_targets=use_yaw_targets,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=goal_reach_angle,
            flight_dome_size=min(
                grid_sizes
            ),  # this is obsolete because _create_targets() is used for target setting
            min_height=min_height,
            np_random=self.np_random,
        )

        # define action space
        # according to the DJI Inspire 3 specifications:
        #   u, v (horizontal velocities): ±26 m/s,
        #   vr (yaw rate): ±150°/s, which is ±2.62 rad/s,
        #   vz (vertical velocity, falling speed): ±8 m/s.
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
                        low=-2 * self.max_dimension_size,
                        high=2 * self.max_dimension_size,
                        shape=(4,) if use_yaw_targets else (3,),
                        dtype=np.float32,
                    ),
                    stack=True,  # True = convert the list of Box samples into one stacked NumPy array of shape (num_targets, 3), used with LSTM
                ),
            }
        )

    def _create_targets(self) -> np.ndarray:
        """create waypoint targets that respect the voxel grid and obstacles"""
        targets = []
        for _ in range(self.num_targets):
            free_voxel = self.voxel_grid.get_random_free_voxel()
            new_position = self.voxel_grid.voxel_to_world(free_voxel)
            targets.append(new_position)

            # mark the voxel as occupied, this will prevent spawning obstacles in the same voxel
            self.voxel_grid.mark_voxels([free_voxel], occupied=True)

        return np.array(targets, dtype=np.float32)

    def _create_boundary_walls(self):
        """
        create invisavle collision walls at env boundaries so that lidar rays can detect them
        """
        vg = self.voxel_grid
        half_thickness = 0.01

        # (position, halfExtents) for each wall
        walls = [
            # -x wall
            (
                [vg.x_min - half_thickness, 0.0, vg.z_size / 2],
                [half_thickness, vg.y_size / 2, vg.z_size / 2],
            ),
            # +x wall
            (
                [vg.x_max + half_thickness, 0.0, vg.z_size / 2],
                [half_thickness, vg.y_size / 2, vg.z_size / 2],
            ),
            # -y wall
            (
                [0.0, vg.y_min - half_thickness, vg.z_size / 2],
                [vg.x_size / 2, half_thickness, vg.z_size / 2],
            ),
            # +y wall
            (
                [0.0, vg.y_min + half_thickness, vg.z_size / 2],
                [vg.x_size / 2, half_thickness, vg.z_size / 2],
            ),
            # celling
            (
                [0.0, 0.0, vg.z_max + half_thickness],
                [vg.x_size / 2, vg.y_size / 2, half_thickness],
            ),
        ]

        for pos, half_extents in walls:
            col_id = self.env.createCollisionShape(
                shapeType=self.env.GEOM_BOX,
                halfExtents=half_extents,
            )
            body_id = self.env.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=int(col_id),
                basePosition=pos,
                baseOrientation=[0, 0, 0, 1],
            )

            self.boundary_wall_ids.append(body_id)

    def _create_obstacles(self, num_obstacles: int):
        """
        init obstacles: create obstacle bodies
        """
        for _ in range(num_obstacles):
            if self.visual_obstacles:
                # create visual shape
                visual_id = self.env.createVisualShape(
                    shapeType=self.env.GEOM_CYLINDER,
                    radius=self.voxel_size / 2,
                    length=self.grid_sizes[
                        -1
                    ],  # obstacles have full hight of the environment (from ground to sealing)
                    rgbaColor=[0.8, 0.2, 0.2, 1.0],
                )

            # create collision shape
            collision_id = self.env.createCollisionShape(
                shapeType=self.env.GEOM_CYLINDER,
                radius=self.voxel_size / 2,
                height=self.grid_sizes[-1],
            )

            if self.visual_obstacles:
                # create visual body id
                object_id = self.env.createMultiBody(
                    baseMass=0.0,  # Mass=0 -> Static obstacles
                    baseVisualShapeIndex=int(visual_id),
                    baseCollisionShapeIndex=int(collision_id),
                    basePosition=[0, 0, -10],  # hide them at the beginning
                    baseOrientation=[0, 0, 0, 1],
                )
            else:
                # create only collision body id
                object_id = self.env.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=int(collision_id),
                    basePosition=[0, 0, -10],
                )

            self.obstacles.append(object_id)

    def _distribute_obstacles(self):
        """distribute obstacles at free-voxel positions"""
        for obstacle_id in self.obstacles:
            # get voxel occupancy
            occupancy_grid = (
                self.voxel_grid.get_occupancy()
            )  # 3d ndarray (x,y,z) 1=occupied, 0=free
            occupancy_mask = ~occupancy_grid  # invert: 0=occupied, 1=free

            # check for free columns -> pick one randomly
            col_free = np.all(
                occupancy_mask, axis=2
            )  # check occupancy along z axis -> returns array of shape (nx, ny) containing bool True if the whole nz was free and False otherwise.
            free_columns = np.argwhere(col_free)  # get (i, j) of free columns

            if len(free_columns) == 0:
                logger.warning("No free columns available for obstacles!")
                break
            column = self.np_random.choice(free_columns)
            i, j = column
            k = self.voxel_grid.nz // 2

            free_center_voxel = (
                i,
                j,
                k,
            )  # determine the voxel at the center of the free column -> expand (i, j) by k at the center
            free_position = self.voxel_grid.voxel_to_world(free_center_voxel)

            # reset obstacle position in pybullet
            self.env.resetBasePositionAndOrientation(
                obstacle_id,
                posObj=free_position,
                ornObj=[0, 0, 0, 1],
            )

            # reset obstacle velocity
            self.env.resetBaseVelocity(
                obstacle_id,
                linearVelocity=[0.0, 0.0, 0.0],
                angularVelocity=[0.0, 0.0, 0.0],
            )

            # mark the voxels of that column as occupied
            column_voxels = [
                (i, j, kk) for kk in range(self.voxel_grid.nz)
            ]  # get list of column voxels
            self.voxel_grid.mark_voxels(column_voxels, occupied=True)

    def reset(
        self,
        *,
        seed: None | int = None,
        options: None | dict[str, Any] = None,
        drone_options: None | dict[str, Any] = None,
    ) -> tuple[dict[Literal["attitude", "target_deltas"], np.ndarray], dict]:
        """
        rest the env: resets simulation state (obstacles, uav), the waypoint hnadler, and any other counters

        NOTE: begin_reset() creates a NEW Aviary each time, disconnecting the old one.
        This means all PyBullet bodies (including obstacles) are destroyed.
        Obstacles must be recreated after each begin_reset().
        """
        if options is None:
            options = {}

        # reset occupancy voxel grid
        self.voxel_grid.reset_occupancy()

        # create random UAV start position and set it
        free_voxel = self.voxel_grid.get_random_free_voxel()
        start_pos = self.voxel_grid.voxel_to_world(free_voxel)
        self.start_pos = start_pos.reshape(
            1, -1
        )  # reshape 1D array to 2D array with shape (1, 3). This overwrites the base env start_pos attribute
        self.voxel_grid.mark_voxels(
            [free_voxel], occupied=True
        )  # mark uav start position as occupied

        # start reset procedure using the base env's methods
        # suppress PyBullet's C-level "argv[0]=" stdout noise on each new Aviary connection -> clean terminal output
        with open(os.devnull, "w") as devnull:
            old_fd = os.dup(1)
            os.dup2(devnull.fileno(), 1)
            try:
                super().begin_reset(seed, drone_options=drone_options)  # Aviary is initialized, uav start position is set and all PyBullet bodies (including obstacles) are destroyed. Obstacles are recreated down the line.
            finally:
                os.dup2(old_fd, 1)
                os.close(old_fd)

        # clear old walls -> create new boundary walls
        self.boundary_wall_ids.clear()
        self._create_boundary_walls()

        # reset waypoint handler, which sets the current target -> create targets manually from voxel grid and overwrite the targets attribute
        self.waypoints.reset(self.env, self.np_random)
        self.waypoints.targets = self._create_targets()

        # cleat old obstacle IDs -> recreate new obstacles (after Aviary is initialized and self.env is available) -> reposition obstacles to random locations (they were created at hidden position [0,0,-10])
        self.obstacles.clear()
        self._create_obstacles(num_obstacles=self.num_obstacles)
        self._distribute_obstacles()

        # init tracked metrics
        self.info["num_targets_reached"] = 0
        self.info["num_obstacles"] = self.num_obstacles

        # finish reset
        super().end_reset()  # registers also all new pybullet bodies

        return self.state, self.info

    # called in the step function of the QuadXBaseEnv parent class -> new observation
    def compute_state(self) -> None:
        # compute state of the uav, using the base env
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()

        # create empty array of type float32 and fill it
        attitude = np.empty(10, dtype=np.float32)
        attitude[0] = ang_vel[2]  # (1,) - yaw, rotation velocity
        attitude[1] = ang_pos[2]  # (1,) - yaw, rotation position
        attitude[2:5] = lin_vel  # (3,) - body frame linear velocity vector (u, v, w)
        attitude[5] = lin_pos[2]  # (1,) - z position
        attitude[6:10] = (
            self.action
        )  # (4,) - previous action  # TODO check for other methods to capture temporal information of taken actions

        # # create empty array of type float32 -> compute the target deltas with the method from the waypointhandler -> fill array
        # target_deltas = np.empty((1,3), dtype=np.float32)
        # target_deltas[0, :] = self.waypoints.distance_to_targets(ang_pos, lin_pos, quaternion)
        target_deltas = self.waypoints.distance_to_targets(
            ang_pos, lin_pos, quaternion
        ).astype(np.float32)

        # update the current state
        self.state = {"attitude": attitude, "target_deltas": target_deltas}

    # called in the step function of the QuadXBaseEnv parent class
    def compute_term_trunc_reward(self) -> None:
        """
        1. compute termination, truncation, and reward of the current timestep in self.compute_base_term_trunc_reward().
        2. Computes rewards.

        use reward and termination logic from pyflyt env (for now -> # TODO add custom reward function)
        """

        # call my computation function that overwrites the base env function
        self.compute_base_term_trunc_reward()

        # if not using sparse reward, add bonus rewards related to waypoint progression
        if not self.sparse_reward:
            self.reward += max(3.0 * self.waypoints.progress_to_next_target, 0.0)
            self.reward += 0.1 / self.waypoints.distance_to_next_target

        # on reaching the target waypoint
        if self.waypoints.target_reached:
            self.reward = 100.0  # large reward bonus  # TODO increment reward instead of re-assigning

            # go to the next target if available
            self.waypoints.advance_targets()

            # update termination: if all targets are reached, signal environment completion
            self.truncation |= self.waypoints.all_targets_reached
            self.info["env_complete"] = self.waypoints.all_targets_reached
            self.info["num_targets_reached"] = self.waypoints.num_targets_reached

    # part of the compute_term_trunc_reward call above
    def compute_base_term_trunc_reward(self) -> None:
        """
        checks if:
        1. episode ends
        2. collision occurred
        Custom base trmination, truncation and reward computation method that is overwritting the parent class
        """
        # exceed step count
        if self.step_count > self.max_steps:
            self.truncation |= True

        # check for collisions between the drone and any other body in the space
        if np.any(self.env.contact_array[self.env.drones[0].Id]):
            self.reward = -100.0
            self.info["collision"] = True
            self.termination |= True

        # ## unnecessary since walls prevent out of bounds and collisions with them are detected above
        # # check exceed rectangular bounds on cartesian grid_sizes
        # lin_pos = self.env.state(0)[-1]  # get current position (lin_pos is at state[3])
        # x, y, z = lin_pos  # [x, y, z]

        # if (
        #     x < self.voxel_grid.x_min
        #     or x > self.voxel_grid.x_max
        #     or y < self.voxel_grid.y_min
        #     or y > self.voxel_grid.y_max
        #     or z
        #     > self.voxel_grid.z_max  # z constrains the max hight the uav is allowed to fly. collision with the floor is checked above already!
        # ):
        #     self.reward = -100.0
        #     self.info["out_of_bounds"] = True
        #     self.termination |= True

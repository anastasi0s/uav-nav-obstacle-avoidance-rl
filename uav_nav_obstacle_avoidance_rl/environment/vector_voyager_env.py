import os
from collections.abc import Sequence
from typing import Any, List, Literal, Optional

import gymnasium as gym
import numpy as np
from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler

from uav_nav_obstacle_avoidance_rl import config
from uav_nav_obstacle_avoidance_rl.environment.occupancy_grid import OccupancyGrid

logger = config.logger


# my waypoint environment based on PyFlyt/QuadX-Waypoints-v3
class VectorVoyagerEnv(QuadXBaseEnv):
    def __init__(
        self,
        start_pos: list,
        # boundary parameters
        grid_sizes: Sequence[float],
        cell_size: float,
        min_height: float,
        # waypoint parameters
        num_targets: int,
        sparse_reward: bool,
        use_yaw_targets: bool,
        goal_reach_distance: float,
        goal_reach_angle: float,
        # obstacle parameters
        num_obstacles: int,
        obstacle_strategy: str, # TODO remove ?
        visual_obstacles: bool,
        # simulation parameters
        max_duration_seconds: float, 
        flight_mode: int,
        angle_representation: Literal["euler", "quaternion"],
        agent_hz: int,
        render_mode: None | Literal["human", "rgb_array"],
        render_resolution: Sequence[int],
    ):
        # init parent class (QuadxBaseEnv)
        super().__init__(
            start_pos=np.array([start_pos]),  # uav starting at pos [0, 0, 1]. Is randomized and overwritten in reset()
            flight_mode=flight_mode,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,  # its rquired in the base class but its not used in this env
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=(render_resolution[0], render_resolution[1]),
        )
        # boundary parameters
        self.grid_sizes = grid_sizes
        self.z_size = grid_sizes[2]  # z dimension stored separately (not in occupancy grid)
        self.cell_size = cell_size
        self.min_height = min_height
        # waypoint parameters
        self.num_targets = num_targets
        self.target_distance_range = None  # [min_r, max_r] or None for unconstrained; set by curriculum
        self.sparse_reward = sparse_reward
        # obstacle parameters
        self.obstacle_strategy = obstacle_strategy
        self.num_obstacles = num_obstacles
        self.visual_obstacles = visual_obstacles
        self.obstacles = []  # store PyBullet body IDs
        self.boundary_wall_ids = []  # # store PyBullet body IDs for walls

        # Note: obstacles are created in the first reset() call when self.env (Aviary/Pybullet client) is available. Pybullet connections from gym are created after env reset()

        # init 2D occupancy grid (only XY plane)
        self.occupancy_grid = OccupancyGrid(grid_sizes[:2], cell_size, rng=self.np_random)

        # init waypoint handler
        self.waypoints = WaypointHandler(
            enable_render=self.render_mode is not None,
            num_targets=num_targets,
            use_yaw_targets=use_yaw_targets,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=goal_reach_angle,
            flight_dome_size=0.0,  # this is obsolete since _create_targets() is used for target setting
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
                    shape=(10,),  # 10 = 1 (yaw position) + 1 (yaw vel) + 3 (lin_vel) + 1 (height) + 4 (previous actions)
                    dtype=np.float32,
                ),
                "target_deltas": gym.spaces.Sequence(
                    space=gym.spaces.Box(
                        low=-2 * max(self.grid_sizes),
                        high=2 * max(self.grid_sizes),
                        shape=(4,) if use_yaw_targets else (3,),
                        dtype=np.float32,
                    ),
                    stack=True,  # True = convert the list of Box samples into one stacked NumPy array of shape (num_targets, 3), used with LSTM
                ),
            }
        )

    # called in reset()
    def _create_targets(self) -> np.ndarray:
        """create waypoint targets within a square annulus around the UAV start position.
        Falls back to random free cell placement when target_distance_range is None.
        """
        og = self.occupancy_grid
        start_x, start_y = self.start_pos[0, 0], self.start_pos[0, 1]
        start_i, start_j = og.world_to_cell(start_x, start_y)

        targets = []
        for _ in range(self.num_targets):
            if self.target_distance_range is None:
                free_cell = og.get_random_free_cell()
            else:
                r_min, r_max = self.target_distance_range
                # convert radii from world units to cell units
                cells_inner = int(np.floor(r_min / og.cell_size))
                cells_outer = int(np.ceil(r_max / og.cell_size))

                # outer square bounds, clamped to grid
                i_lo = max(start_i - cells_outer, 0)
                i_hi = min(start_i + cells_outer, og.nx - 1)
                j_lo = max(start_j - cells_outer, 0)
                j_hi = min(start_j + cells_outer, og.ny - 1)

                # all cell indices inside the outer square
                ii, jj = np.mgrid[i_lo:i_hi + 1, j_lo:j_hi + 1]
                # exclude inner square (cells too close to start)
                outside_inner = (np.abs(ii - start_i) >= cells_inner) | (np.abs(jj - start_j) >= cells_inner)
                # exclude occupied cells
                free = ~og.grid[i_lo:i_hi + 1, j_lo:j_hi + 1]

                valid_mask = outside_inner & free
                valid_cells = np.argwhere(valid_mask)

                if len(valid_cells) == 0:
                    raise ValueError("No free cells in target distance range")

                pick = self.np_random.integers(len(valid_cells))
                # convert local sub-grid indices back to global grid indices
                free_cell = (int(valid_cells[pick, 0] + i_lo), int(valid_cells[pick, 1] + j_lo))

            x, y = og.cell_to_world(free_cell)
            z = self.np_random.uniform(self.min_height + 0.5, self.z_size)
            targets.append([x, y, z])
            og.mark_cells([free_cell], occupied=True)

        return np.array(targets, dtype=np.float32)

    # called in reset()
    def _create_boundary_walls(self):
        """
        create invisavle collision walls at env boundaries so that lidar rays can detect them
        """
        og = self.occupancy_grid
        half_thickness = 0.01
        z_half = self.z_size / 2

        # (position, halfExtents) for each wall
        walls = [
            # -x wall
            (
                [og.x_min - half_thickness, 0.0, z_half],
                [half_thickness, og.y_size / 2, z_half],
            ),
            # +x wall
            (
                [og.x_max + half_thickness, 0.0, z_half],
                [half_thickness, og.y_size / 2, z_half],
            ),
            # -y wall
            (
                [0.0, og.y_min - half_thickness, z_half],
                [og.x_size / 2, half_thickness, z_half],
            ),
            # +y wall
            (
                [0.0, og.y_max + half_thickness, z_half],
                [og.x_size / 2, half_thickness, z_half],
            ),
            # ceiling
            (
                [0.0, 0.0, self.z_size + half_thickness],
                [og.x_size / 2, og.y_size / 2, half_thickness],
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

    def _create_object_shape(
            self, 
            object_type: Literal['cylinder', 'sphere', 'box'], 
            scaled_radius: float=1.0, 
            height: float=1.0,
            half_extents: Optional[List[float]]=None,  # define half sizes to get full sized objects 
        ):
        """
        create the object shapes that can be spawned with _spawn_object() on multiple positions, without having to recreate the shapes
        Args:
            object_type:    'cylinder' or 'box' or sphere
            scaled_radius:  fraction of *cell_size* used as cylinder radius [0-1]
            height:         fraction of environment z-height  (1.0 = floor -> ceiling) [0-1]
            half_extents:   [sx, sy, sz] — fractions of cell_size (xy) and
                            grid z-height (z) used as **full-size** scalars;
                            the method halves them for PyBullet's halfExtents. [0-1]

        Returns:
            (collision_id, visual_id)   visual_id is -1 when visual_obstacles=False
        """
        visual_id = -1  # PyBullet convention: no visual shape
        if object_type == 'cylinder':
            r = self.cell_size * scaled_radius
            h = self.grid_sizes[-1] * height
            
            # create collision shape
            collision_id = self.env.createCollisionShape(
                shapeType=self.env.GEOM_CYLINDER, radius=r, height=h,
            )
            if self.visual_obstacles:
                # create visual shape
                visual_id = self.env.createVisualShape(
                    shapeType=self.env.GEOM_CYLINDER, radius=r, length=h,  # obstacles have full hight of the environment (from ground to sealing)
                    rgbaColor=[0.8, 0.2, 0.2, 1.0],
                )

        elif object_type == 'sphere':
            r = self.cell_size * scaled_radius
            collision_id = self.env.createCollisionShape(
                shapeType=self.env.GEOM_SPHERE, radius=r,
            )

            if self.visual_obstacles:
                visual_id = self.env.createVisualShape(
                    shapeType=self.env.GEOM_SPHERE, radius=r,
                    rgbaColor=[0.8, 0.2, 0.2, 1.0],
                )

        elif object_type == 'box':
            if half_extents is None:
                half_extents = [0.5, 0.5, 0.5]
            he = [
                self.cell_size * half_extents[0] / 2.0,  # define half size to get full sized objects
                self.cell_size * half_extents[1] / 2.0,
                self.grid_sizes[-1] * half_extents[2] / 2.0,
            ]
            collision_id = self.env.createCollisionShape(
                shapeType=self.env.GEOM_BOX, halfExtents=he,
            )
            if self.visual_obstacles:
                visual_id = self.env.createVisualShape(
                    shapeType=self.env.GEOM_BOX, halfExtents=he,
                    rgbaColor=[0.8, 0.2, 0.2, 1.0],
                )

        return int(collision_id), int(visual_id)

    def _spawn_object(
            self, 
            collision_id, 
            base_position: List[int],
            base_orientation: List[int]=[0, 0, 0, 1],
            visual_id=None,
        ):
        """
        position object body in environment.

        Args:
            collision_id: PyBullet collision shape id
            base_position: [x, y, z] world coordinates of the objects center position
            base_orientation: [x, y, z, w] quaternion representation of 3d rotation
        
        """
        # create body id
        object_id = self.env.createMultiBody(
            baseMass=0.0,  # Mass=0 -> Static obstacles
            baseVisualShapeIndex=int(visual_id if visual_id else -1),
            baseCollisionShapeIndex=int(collision_id),
            basePosition=base_position,
            baseOrientation=base_orientation,
            )
        
        self.obstacles.append(object_id)  # collect object ids to destroy the objects at the end of the episode

    def _generate_obstacles(self, num_obstacles: int):
        """
        create obstacles and position them in the environment
        
        """
        for obj in range(num_obstacles):
            # find a free cell in the 2D occupancy grid
            free_cell = self.occupancy_grid.get_random_free_cell()
            x, y = self.occupancy_grid.cell_to_world(free_cell)
            z = self.z_size / 2  # center obstacle vertically (floor-to-ceiling columns)
            free_position = [x, y, z]

            # reset obstacle position in pybullet
            self.env.resetBasePositionAndOrientation(
                object_id,
                posObj=free_position,
                ornObj=[0, 0, 0, 1],
            )

            # reset obstacle velocity
            self.env.resetBaseVelocity(
                object_id,
                linearVelocity=[0.0, 0.0, 0.0],
                angularVelocity=[0.0, 0.0, 0.0],
            )

            # mark cell as occupied
            self.occupancy_grid.mark_cells([free_cell], occupied=True)

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

        # reset occupancy grid
        self.occupancy_grid.reset_occupancy()

        # create random UAV start position and set it
        free_cell = self.occupancy_grid.get_random_free_cell()
        x, y = self.occupancy_grid.cell_to_world(free_cell)
        z = self.np_random.uniform(self.min_height + 0.5, self.z_size)
        self.start_pos = np.array([[x, y, z]])  # shape (1, 3). This overwrites the base env start_pos attribute
        self.occupancy_grid.mark_cells([free_cell], occupied=True)  # mark uav start cell as occupied

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

        # cleat old obstacle IDs -> recreate new obstacles (after Aviary is initialized and self.env is available)
        self.obstacles.clear()
        self._generate_obstacles(num_obstacles=self.num_obstacles)

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
        attitude[6:10] = (self.action)  # (4,) - previous action  # TODO check for other methods to capture temporal information of taken actions

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

        reward flow:
        reward = -0.1                          # base env sets beeing alive penalty
        reward += dense_shaping                # compute_term_trunc_reward adds progress/distance bonuses
        reward = -100.0  (if collision)        # overwrites everything
        reward = 100.0   (if target reached)   # also overwrites everything
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
        #     x < self.occupancy_grid.x_min
        #     or x > self.occupancy_grid.x_max
        #     or y < self.occupancy_grid.y_min
        #     or y > self.occupancy_grid.y_max
        #     or z
        #     > self.occupancy_grid.z_max  # z constrains the max hight the uav is allowed to fly. collision with the floor is checked above already!
        # ):
        #     self.reward = -100.0
        #     self.info["out_of_bounds"] = True
        #     self.termination |= True

    def set_difficulty(self, stage):
        """interface method that adjusts the environments difficulty level according to the current stage of the curriculum_callback"""

        # adjust num of spawned obstacles, targets 
        self.num_obstacles = stage['num_obstacles']
        self.num_targets = stage['num_targets']  # ??? can policy handle varying num o targets? check state space
        self.target_distance_range = stage['target_distance_range']

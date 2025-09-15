from typing import Any, Literal

import gymnasium as gym
import numpy as np
from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler


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
        # Obstacle parameters
        enable_obstacles: bool = True,
        visual_obstacles: bool = False,  # visual representation of obstacles used e.g. in evaluation when recording video
        num_obstacles: tuple[int, int] = (
            0,
            3,
        ),  # range of number obstacles that will be spawned in the env
        obstacle_types: list[str] = ["sphere", "box", "cylinder"],
        obstacle_size_range: tuple[float, float] = (0.1, 0.8),
        obstacle_min_distance_from_start: float = 1.0,  # min distance from uav start point to spawn obstacle from in meter
        obstacle_hight_range: tuple[float, float] = (0.1, 5.0),
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

        # obstacle config
        self.enable_obstacles = enable_obstacles
        self.visual_obstacles = visual_obstacles
        self.num_obstacles = num_obstacles
        self.obstacle_types = obstacle_types
        self.obstacle_size_range = obstacle_size_range
        self.obstacle_min_distance_from_start = obstacle_min_distance_from_start
        self.obstacle_spawn_radius = (
            self.flight_dome_size - 1.0
        )  # 1m less than spawn radius
        self.obstacle_hight_range = obstacle_hight_range
        self.obstacle_colors = [
            [0.8, 0.2, 0.2, 1.0],  # Red
            [0.2, 0.8, 0.2, 1.0],  # Green
            [0.2, 0.2, 0.8, 1.0],  # Blue
            [0.8, 0.8, 0.2, 1.0],  # Yellow
            [0.8, 0.2, 0.8, 1.0],  # Magenta
            [0.2, 0.8, 0.8, 1.0],  # Cyan
        ]
        # track spawned obstacles
        self.obstacle_ids = []
        self.obstacle_collision_shapes = []
        self.obstacle_visual_shapes = []

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

        # spawn and register new random obstacle bodies
        self._spawn_obstacles()
        self.env.register_all_new_bodies()

        # init tracked metrics
        self.info["num_targets_reached"] = 0
        self.info["num_obstacles_spawned"] = len(self.obstacle_ids)
        super().end_reset()  # finish reset using base env producers

        return self.state, self.info

    def _remove_obstacles(self):
        pass

    def _spawn_obstacles(self):
        """spawn random obstacles in the environment"""
        # determine num of obstacles to spawn
        min_obs, max_obs = self.num_obstacles
        num_obstacles = self.np_random.integers(min_obs, max_obs + 1)

        uav_start_pos = self.start_pos[0]

        for i in range(num_obstacles):
            # generate random obstacle properties
            obstacle_type = self.np_random.choice(self.obstacle_types)
            position = self._generate_obstacle_position(uav_start_pos)
            orientation = [0, 0, 0, 1]
            size = self._generate_obstacle_size(obstacle_type)
            color = self.obstacle_colors[i % len(self.obstacle_colors)]

            # create collision and visual shapes
            collision_id = self._create_collision_id(obstacle_type, size)
            if self.visual_obstacles:
                visual_id = self._create_visual_id(obstacle_type, size, color)
            else:
                visual_id = -1  # invisible objects

            object_id = self.env.createMultiBody(
                baseMass=0.0,  # 0.0: static obstacles, >0: dynamic obstacles
                baseVisualShapeIndex=int(visual_id),
                baseCollisionShapeIndex=int(collision_id),
                basePosition=position,
                baseOrientation=orientation,
            )

            # track spawned obstacles
            self.obstacle_ids.append(object_id)
            self.obstacle_collision_shapes.append(collision_id)
            self.obstacle_visual_shapes.append(visual_id)

    def _generate_obstacle_position(self, uav_start_pos):
        """generate random position for an obstacle inside hemisphere"""
        max_attempts = 50

        for _ in range(max_attempts):
            # determine random spawn position inside dome boundaries
            # θ = [0, 2pi], φ = [0, pi/2]
            # x = r * sin(φ) * cos(θ), y = r * cos(φ) * sin(θ), z = r * cos(φ)
            theta = self.np_random.uniform(0, np.pi * 2)  # azimuth angle
            phi = self.np_random.uniform(0, np.pi / 2)  # polar angle for hemisphere
            radius = self.np_random.uniform(
                self.obstacle_min_distance_from_start, self.obstacle_spawn_radius
            )  # distance from origin

            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)

            # min hight constraint
            z = max(z, self.obstacle_hight_range[0])

            position = [x, y, z]

            # Check distance from UAV start position
            distance_to_uav = np.linalg.norm(np.array(position) - uav_start_pos)

            if distance_to_uav >= self.obstacle_min_distance_from_start:
                return position

            # fallback position if all attempts failed
            return [self.obstacle_spawn_radius * 0.8, 0, 1.5]

    def _generate_obstacle_size(self, obstacle_type):
        """generate random size parameters for each obstacle type"""
        base_size = self.np_random.uniform(*self.obstacle_size_range)

        if obstacle_type == "sphere":
            return base_size  # radius
        elif obstacle_type == "box":
            # half extents [x, y, z]
            return [
                base_size * self.np_random.uniform(0.5, 2.0),
                base_size * self.np_random.uniform(0.5, 2.0),
                base_size * self.np_random.uniform(0.5, 2.0),
            ]
        elif obstacle_type == "cylinder":
            # radius, hight
            return [
                base_size,
                base_size * self.np_random.uniform(0.5, 2.0),
            ]
        else:
            return base_size

    def _create_collision_id(self, obstacle_type, size):
        """create collision and visual shapes for an obstacle"""

        if obstacle_type == "sphere":
            collision_id = self.env.createCollisionShape(
                shapeType=self.env.GEOM_SPHERE,
                radius=size,
            )

        elif obstacle_type == "box":
            collision_id = self.env.createCollisionShape(
                shapeType=self.env.GEOM_BOX, halfExtents=size
            )

        elif obstacle_type == "cylinder":
            radius, height = size
            collision_id = self.env.createCollisionShape(
                shapeType=self.env.GEOM_CYLINDER, radius=radius, height=height
            )

        return collision_id

    def _create_visual_id(self, obstacle_type, size, color):
        """create collision and visual shapes for an obstacle"""
        if obstacle_type == "sphere":
            visual_id = self.env.createVisualShape(
                shapeType=self.env.GEOM_SPHERE,
                radius=size,
                rgbaColor=color,
            )

        elif obstacle_type == "box":
            visual_id = self.env.createVisualShape(
                shapeType=self.env.GEOM_BOX, halfExtents=size, rgbaColor=color
            )

        elif obstacle_type == "cylinder":
            radius, height = size
            visual_id = self.env.createVisualShape(
                shapeType=self.env.GEOM_CYLINDER,
                radius=radius,
                length=height,  # for visual shape, use length instead of height
                rgbaColor=color,
            )

        return visual_id

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
        attitude[6:10] = (
            self.action
        )  # (4,) - previous actions  # TODO check for other methods to capture temporal information of taken actions

        # # create empty array of type float32 -> compute the target deltas with the method from the waypointhandler -> fill array
        # target_deltas = np.empty((1,3), dtype=np.float32)
        # target_deltas[0, :] = self.waypoints.distance_to_targets(ang_pos, lin_pos, quaternion)
        target_deltas = self.waypoints.distance_to_targets(
            ang_pos, lin_pos, quaternion
        ).astype(np.float32)

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

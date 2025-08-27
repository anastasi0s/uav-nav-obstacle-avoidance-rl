from typing import Any

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
            num_targets: int = 1,  # NOTE starting with one waypoint for now
            use_yaw_targets: bool = False,  # toggles whether the agent must also align its yaw (heading) to a per‐waypoint target before that waypoint is considered “reached,” and whether yaw error is included in the observation.
            flight_mode: int = 5,  # uav constrol mode 5: (u, v, vr, vz) -> u: local velocity forward in m/s, v: lateral velocity in m/s, vr: yaw in rad/s, vz: vertical velocity in m/s
            flight_dome_size: float = 5.0,  # (5.0 = default) maximum allowed flying distance from the start position of the uav in meters. (the radius of the “flight dome” within which the UAV must remain to avoid an out‑of‑bounds termination)
            goal_reach_distance: float = 0.2,  # distance within which the target is considered reached
            goal_reach_angle=0.1,  # not in use since use_yaw_targets is not in use
            max_duration_seconds: float = 80.0,  # max simulation time of the env
            angle_representation: Literal["euler", "quaternion"] = "quaternion",
            agent_hz: int = 30,  # looprate of the agent to environment interaction
            render_mode: None | Literal["human", "rgb_array"] = None,
            render_resolution: tuple[int, int] = (480, 480),
            min_height: float = 0.1,  # min allowed hight, to avoid crashing on the floor. # NOTE (might take this out in the future to let the agent learn it itself to not crash on the floor)
    ):  
        # init the quadx base env
        super().__init__(
            start_pos=np.array([[0.0, 0.0, 1.0]]),  # uav starting at pos [0, 0, 1] # TODO implement random spawning
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,  # its rquired in the base class but its not used in this env
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        # define reward structure # TODO for now use the reward and termination logic from PyFlyt
        self.sparse_reward = sparse_reward
        self.num_targets = num_targets
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
                    shape=(10,),  # 10 = 1 (yaw position) + 1 (yaw vel) + 3 (lin_vel) + 1 (height) + 4 (previous actions)
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
                )
            }
        )



    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict(), drone_options: None | dict[str, Any] = None,
    ) -> tuple[dict[Literal["attitude", "target_deltas"], np.ndarray], dict]:
        """
        rest the env: resets simulation state, the waypoint hnadler, and any other counters
        """
        # start reset procedure using the base env's methods
        super().begin_reset(seed, drone_options=drone_options)
        self.waypoints.reset(self.env, self.np_random)  # reset waypoint handler, which sets the current target
        # init tracked metrics
        self.info["num_targets_reached"] = 0
        super().end_reset()  # finish reset using base env producers
        
        return self.state, self.info
    

    # this is the step function -> new observation
    def compute_state(self) -> None:
        # compute state of the uav, using the base env
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        
        # create empty array of type float32 and fill it
        attitude = np.empty(10, dtype=np.float32)
        attitude[0] = ang_vel[2]        # (1,) - yaw, rotation velocity
        attitude[1] = ang_pos[2]        # (1,) - yaw, rotation position
        attitude[2:5] = lin_vel         # (3,) - body frame linear velocity vector (u, v, w)
        attitude[5] = lin_pos[2]        # (1,) - z position
        attitude[6:10] = self.action    # (4,) - previous actions  # TODO check for other methods to capture temporal information of taken actions

        # # create empty array of type float32 -> compute the target deltas with the method from the waypointhandler -> fill array
        # target_deltas = np.empty((1,3), dtype=np.float32)
        # target_deltas[0, :] = self.waypoints.distance_to_targets(ang_pos, lin_pos, quaternion)
        target_deltas = self.waypoints.distance_to_targets(ang_pos, lin_pos, quaternion).astype(np.float32)

        # update the current state
        self.state = {
            "attitude": attitude,
            "target_deltas": target_deltas
        }
        

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

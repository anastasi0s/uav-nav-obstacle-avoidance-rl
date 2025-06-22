import json
from pathlib import Path
from typing import Dict, Optional
from loguru import logger

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from sympy import EX

from uav_nav_obstacle_avoidance_rl import config


class UAVMetricsCallback(BaseCallback):
    """
    Custom callback for collecting UAV performance metrics during training.
    
    Tracks:
    - Path length: Total distance traveled by UAV
    - Average velocity: Mean velocity during flight
    - Path efficiency ratio: Ratio of actual path to direct distance
    - Success rate: Proportion of successful flights
    """
    
    def __init__(self, 
                 log_dir: Optional[str] = None,
                 save_freq: int = 1000,
                 verbose: int = 0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        self.log_dir = Path(log_dir) if log_dir else None
        self.save_freq = save_freq
        
        # Episode-level metrics storage
        self.episode_metrics = {
            'path_lengths': [],
            'average_velocities': [],
            'path_efficiency_ratios': [],
            'success_rates': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'collisions': [],
            'out_of_bounds': [],
            'targets_reached': []
        }
        
        # Current episode tracking for each environment
        self.current_episodes = {}
        
        # Running statistics
        self.total_episodes = 0
        self.successful_episodes = 0

    def _on_training_start(self) -> None:
        """called before the first rollout starts"""
        if self.verbose > 0:
            logger.info("Starting UAV metrics collection...")
        
        # create log dir
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # init episode tracking for each env
        n_envs = getattr(self.training_env, "num_envs", 1)
        for i in range(n_envs):
            self._reset_episode_tracking(i)

    def _on_step(self) -> bool:
        """called after each env step"""
        # get the underlying env to access full state
        if hasattr(self.training_env, 'envs'):
            # vac_env case
            envs = self.training_env.envs
        else:
            # single env case
            envs = [self.training_env]
        
        # process each env
        for env_idx, env in enumerate(envs):
            # get underlying VectorVoyagerEnv
            underlying_env = self._get_underlying_env(env)
            
            if underlying_env is not None:
                try:
                    # check if the aviary state is accessible
                    if hasattr(underlying_env, 'env') and hasattr(underlying_env.env, 'state'):
                        ## extract position and velocity from env
                        # state is (4, 3) array: [ang_vel, ang_pos, lin_vel, lin_pos]
                        state = underlying_env.env.state(0)  # 0 idx for accessing the first drone
                        lin_pos = state[3]  # [x, y, z] position
                        lin_vel = state[2]  # [u, v, w] velocity

                        # store position and velocity for this step
                        self.current_episodes[env_idx]['position'].append(lin_pos.copy())
                        self.current_episodes[env_idx]['velocities'].append(lin_vel.copy())

                        # for now, set start position at the hard coded one: start_pos=np.array([[0.0, 0.0, 1.0]]) # TODO adjust when random start position changes 
                        self.current_episodes[env_idx]['start_position'] = [0.0, 0.0, 1.0]

                        # get target position from waypoint handler 
                        if (hasattr(underlying_env, 'waypoints') and 
                            hasattr(underlying_env.waypoints, 'targets') and
                            len(underlying_env.waypoints.targets) > 0):
                            self.current_episodes[env_idx]['target_position'] = underlying_env.waypoints.targets[0].copy()  # TODO adjust this when adding multiple waypoints

                        ## extract info from env
                        # check if episode has ended
                        for env_idx, info in enumerate(underlying_env.info):
                            if info['env_complete'] or # TODO CONTINUO FROM HERE

            
                except Exception as e:
                    if self.verbose > 1:
                        logger.warning(f"Could not extract data for env {env_idx}: {e}")

        # check for episode completion


    def _get_underlying_env(self, env):
        """peel wrappeers to find the VectorVoyagerEnv"""
        current = env
        while hasattr(current, 'env'):
            # check if its VectorVoyagerEnv
            if hasattr(current, 'waypoints') and hasattr(current, 'compute_attitude'):
                return current
            # get env on lower level
            current = current.env
        return None
    
    def _reset_episode_tracking(self, env_idx: int):
        """reset tracking for the next episode"""
        self.current_episodes[env_idx] = {
            'positions': [],
            'start_position': None,
            'target_position': None,
            'velocities': [],
            'start_time': self.num_timesteps
        }
        


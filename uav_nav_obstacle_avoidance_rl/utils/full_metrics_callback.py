# uav_nav_obstacle_avoidance_rl/utils/metrics_callback.py

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List, Optional
import json
from pathlib import Path


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
        """Called before the first rollout starts."""
        if self.verbose > 0:
            print("Starting UAV metrics collection...")
            
        # Create log directory if specified
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize episode tracking for each environment
        n_envs = getattr(self.training_env, 'num_envs', 1)
        for i in range(n_envs):
            self._reset_episode_tracking(i)
    
    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Get the underlying environments to access full state
        if hasattr(self.training_env, 'envs'):
            # VecEnv case
            envs = self.training_env.envs
        else:
            # Single env case
            envs = [self.training_env]
        
        # Process each environment
        for env_idx, env in enumerate(envs):
            # Get the underlying VectorVoyagerEnv
            underlying_env = self._get_underlying_env(env)
            
            if underlying_env is not None:
                # Extract position and velocity from the underlying environment
                try:
                    # Get the full state from the aviary
                    if hasattr(underlying_env, 'env') and hasattr(underlying_env.env, 'state'):
                        # state is (4, 3) array: [ang_vel, ang_pos, lin_vel, lin_pos]
                        state = underlying_env.env.state(0)
                        lin_pos = state[3]  # [x, y, z] position
                        lin_vel = state[2]  # [u, v, w] velocity
                        
                        # Store position and velocity for this step
                        self.current_episodes[env_idx]['positions'].append(lin_pos.copy())
                        self.current_episodes[env_idx]['velocities'].append(lin_vel.copy())
                        
                        # Set start position if this is the first step of episode
                        if self.current_episodes[env_idx]['start_position'] is None:
                            self.current_episodes[env_idx]['start_position'] = lin_pos.copy()
                            
                        # Try to get target position from waypoint handler
                        if (hasattr(underlying_env, 'waypoints') and 
                            hasattr(underlying_env.waypoints, 'targets') and
                            len(underlying_env.waypoints.targets) > 0):
                            self.current_episodes[env_idx]['target_position'] = underlying_env.waypoints.targets[0].copy()
                            
                except Exception as e:
                    if self.verbose > 1:
                        print(f"Warning: Could not extract state data for env {env_idx}: {e}")
        
        # Check for episode completion
        if hasattr(self, 'locals') and 'infos' in self.locals:
            infos = self.locals['infos']
            
            for env_idx, info in enumerate(infos):
                if 'episode' in info:
                    # Episode completed - process it
                    self._process_completed_episode(info, env_idx)
                elif ('collision' in info or 'out_of_bounds' in info or 'env_complete' in info):
                    # Check if this indicates episode termination
                    if hasattr(self, 'locals') and 'dones' in self.locals:
                        dones = self.locals['dones']
                        if env_idx < len(dones) and dones[env_idx]:
                            # Create episode info for processing
                            episode_info = {
                                'episode': {
                                    'r': getattr(info, 'episode_reward', 0),
                                    'l': getattr(info, 'episode_length', len(self.current_episodes[env_idx]['positions']))
                                },
                                **info
                            }
                            self._process_completed_episode(episode_info, env_idx)
        
        return True
    
    def _get_underlying_env(self, env):
        """Navigate through wrappers to find the VectorVoyagerEnv."""
        current = env
        while hasattr(current, 'env'):
            current = current.env
            # Check if this is our VectorVoyagerEnv
            if hasattr(current, 'waypoints') and hasattr(current, 'compute_attitude'):
                return current
        
        # Check if the current env itself is VectorVoyagerEnv
        if hasattr(current, 'waypoints') and hasattr(current, 'compute_attitude'):
            return current
            
        return None
    
    def _process_completed_episode(self, episode_info: Dict, env_idx: int = 0):
        """Process a completed episode and calculate metrics."""
        try:
            # Get episode data from Monitor or create defaults
            if 'episode' in episode_info:
                episode_reward = episode_info['episode']['r']
                episode_length = episode_info['episode']['l']
            else:
                episode_reward = 0
                episode_length = len(self.current_episodes[env_idx]['positions'])
            
            # Get custom metrics from info
            collision = episode_info.get('collision', False)
            out_of_bounds = episode_info.get('out_of_bounds', False)
            env_complete = episode_info.get('env_complete', False)
            targets_reached = episode_info.get('num_targets_reached', 0)
            
            # Calculate custom metrics
            path_length = self._calculate_path_length(env_idx)
            avg_velocity = self._calculate_average_velocity(env_idx)
            efficiency_ratio = self._calculate_path_efficiency(env_idx)
            
            # Determine if episode was successful
            success = env_complete and not collision and not out_of_bounds
            
            # Store metrics
            self.episode_metrics['path_lengths'].append(path_length)
            self.episode_metrics['average_velocities'].append(avg_velocity)
            self.episode_metrics['path_efficiency_ratios'].append(efficiency_ratio)
            self.episode_metrics['episode_rewards'].append(episode_reward)
            self.episode_metrics['episode_lengths'].append(episode_length)
            self.episode_metrics['collisions'].append(collision)
            self.episode_metrics['out_of_bounds'].append(out_of_bounds)
            self.episode_metrics['targets_reached'].append(targets_reached)
            
            # Update success tracking
            self.total_episodes += 1
            if success:
                self.successful_episodes += 1
            
            # Calculate running success rate
            success_rate = self.successful_episodes / self.total_episodes if self.total_episodes > 0 else 0
            self.episode_metrics['success_rates'].append(success_rate)
            
            # Log to tensorboard if available
            self._log_metrics_to_tensorboard(path_length, avg_velocity, efficiency_ratio, success_rate)
            
            # Reset current episode tracking
            self._reset_episode_tracking(env_idx)
            
            if self.verbose > 1:
                print(f"Episode {self.total_episodes}: path_length={path_length:.3f}, "
                      f"avg_velocity={avg_velocity:.3f}, efficiency={efficiency_ratio:.3f}, "
                      f"success={success}")
            
        except Exception as e:
            if self.verbose > 1:
                print(f"Error processing episode: {e}")
    
    def _calculate_path_length(self, env_idx: int) -> float:
        """Calculate total distance traveled."""
        positions = self.current_episodes[env_idx]['positions']
        if len(positions) < 2:
            return 0.0
        
        positions = np.array(positions)
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return float(np.sum(distances))
    
    def _calculate_average_velocity(self, env_idx: int) -> float:
        """Calculate average velocity magnitude."""
        velocities = self.current_episodes[env_idx]['velocities']
        if not velocities:
            return 0.0
        
        velocities = np.array(velocities)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        return float(np.mean(velocity_magnitudes))
    
    def _calculate_path_efficiency(self, env_idx: int) -> float:
        """Calculate path efficiency ratio (actual path / direct distance)."""
        episode_data = self.current_episodes[env_idx]
        
        if (episode_data['start_position'] is None or 
            not episode_data['positions'] or
            len(episode_data['positions']) < 2):
            return 0.0
        
        path_length = self._calculate_path_length(env_idx)
        
        # Calculate direct distance to target if available
        if episode_data['target_position'] is not None:
            direct_distance = np.linalg.norm(
                episode_data['target_position'] - episode_data['start_position']
            )
        else:
            # Fallback: use end position
            end_position = episode_data['positions'][-1]
            direct_distance = np.linalg.norm(
                end_position - episode_data['start_position']
            )
        
        if direct_distance == 0:
            return 1.0 if path_length == 0 else float('inf')
        
        return path_length / direct_distance
    
    def _reset_episode_tracking(self, env_idx: int):
        """Reset tracking for the next episode."""
        self.current_episodes[env_idx] = {
            'positions': [],
            'start_position': None,
            'target_position': None,
            'velocities': [],
            'start_time': self.num_timesteps
        }
    
    def _log_metrics_to_tensorboard(self, path_length: float, avg_velocity: float, 
                                   efficiency_ratio: float, success_rate: float):
        """Log metrics to tensorboard if logger is available."""
        if hasattr(self.model, 'logger') and self.model.logger:
            self.model.logger.record('uav_metrics/path_length', path_length)
            self.model.logger.record('uav_metrics/average_velocity', avg_velocity)
            self.model.logger.record('uav_metrics/path_efficiency', efficiency_ratio)
            self.model.logger.record('uav_metrics/success_rate', success_rate)
    
    def save_metrics(self, filepath: Optional[str] = None):
        """Save collected metrics to file."""
        if filepath is None and self.log_dir:
            filepath = self.log_dir / f"uav_metrics_{self.num_timesteps}.json"
        elif filepath is None:
            return
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats()
        
        # Prepare data for saving
        data = {
            'summary': summary,
            'episode_data': self.episode_metrics,
            'total_episodes': self.total_episodes,
            'successful_episodes': self.successful_episodes,
            'num_timesteps': self.num_timesteps
        }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        if self.verbose > 0:
            print(f"Metrics saved to {filepath}")
    
    def _calculate_summary_stats(self) -> Dict:
        """Calculate summary statistics."""
        summary = {}
        
        for metric_name, values in self.episode_metrics.items():
            if values and metric_name not in ['collisions', 'out_of_bounds']:
                summary[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        
        # Special handling for boolean metrics
        if self.episode_metrics['collisions']:
            summary['collision_rate'] = float(np.mean(self.episode_metrics['collisions']))
        if self.episode_metrics['out_of_bounds']:
            summary['out_of_bounds_rate'] = float(np.mean(self.episode_metrics['out_of_bounds']))
        
        return summary
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.log_dir:
            self.save_metrics()
        
        if self.verbose > 0:
            print("\nTraining completed. Final metrics summary:")
            summary = self._calculate_summary_stats()
            for metric, stats in summary.items():
                if isinstance(stats, dict):
                    print(f"{metric}: mean={stats['mean']:.3f} ± {stats['std']:.3f}")
                else:
                    print(f"{metric}: {stats:.3f}")
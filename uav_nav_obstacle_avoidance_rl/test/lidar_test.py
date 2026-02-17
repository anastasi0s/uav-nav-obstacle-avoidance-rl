import numpy as np
import gymnasium as gym

from uav_nav_obstacle_avoidance_rl.environment.vector_voyager_env import VectorVoyagerEnv
from uav_nav_obstacle_avoidance_rl.environment.lidar_observation_wrapper import (
    LidarObservationWrapper,
    LidarFlattenWrapper,
)


def create_env_with_lidar(
    render: bool = False,
    num_rays: int = 36,
    max_range: float = 8.0,
    flatten: bool = False,
) -> gym.Env:
    """
    Factory function to create VectorVoyagerEnv with LiDAR observations.
    
    Args:
        render: Whether to render the environment
        num_rays: Number of horizontal LiDAR rays
        max_range: Maximum LiDAR detection range
        flatten: Whether to flatten observations for MLP policies
    
    Returns:
        Wrapped environment with LiDAR observations
    """
    # Create base environment
    env = VectorVoyagerEnv(
        grid_sizes=(20.0, 20.0, 10.0),
        voxel_size=2.0,
        min_height=0.5,
        num_targets=3,
        num_obstacles=5,
        visual_obstacles=render,  # Only visualize obstacles when rendering
        max_duration_seconds=60.0,
        render_mode="human" if render else None,
    )
    
    # Wrap with LiDAR observation
    env = LidarObservationWrapper(
        env,
        num_rays_horizontal=num_rays,
        num_rays_vertical=1,  # 2D LiDAR (horizontal plane only)
        max_range=max_range,
        min_range=0.2,
        fov_horizontal=360.0,  # Full 360° sweep
        fov_vertical=30.0,
        ray_start_offset=0.15,  # Start rays slightly away from UAV center
        normalize_distances=True,
        add_to_obs="separate",  # Adds 'lidar' key to observation dict
    )
    
    # Optionally flatten for algorithms that need vector observations
    if flatten:
        env = LidarFlattenWrapper(env, context_length=2)
    
    return env


def demo_lidar_visualization():
    """Demo showing LiDAR rays in the PyBullet visualizer."""
    print("Creating environment with LiDAR visualization...")
    
    env = create_env_with_lidar(render=True, num_rays=36, max_range=8.0)
    obs, info = env.reset()
    
    print(f"\nObservation keys: {obs.keys()}")
    print(f"Attitude shape: {obs['attitude'].shape}")
    print(f"LiDAR shape: {obs['lidar'].shape}")
    print(f"Target deltas shape: {obs['target_deltas'].shape}")
    
    print("\nRunning simulation with LiDAR debug visualization...")
    print("Watch the colored rays: Green = no obstacle, Red = obstacle detected")
    
    for step in range(500):
        # Random action for demo
        action = env.action_space.sample() * 0.3  # Scale down for safety
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render LiDAR rays every 10 steps
        if step % 10 == 0:
            env.render_lidar_debug(duration=0.3)
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"  Terminated: {terminated}, Truncated: {truncated}")
            print(f"  Info: {info}")
            obs, info = env.reset()
    
    env.close()


def demo_lidar_data():
    """Demo showing LiDAR data values."""
    print("Creating environment (no rendering)...")
    
    env = create_env_with_lidar(render=False, num_rays=12, max_range=10.0)
    obs, info = env.reset()
    
    print(f"\nLiDAR configuration:")
    print(f"  Number of rays: {env.num_rays_total}")
    print(f"  Max range: {env.max_range}m")
    print(f"  Normalized: {env.normalize_distances}")
    
    print(f"\nInitial LiDAR readings (normalized distances):")
    print(f"  {obs['lidar']}")
    
    # Take a few steps
    for i in range(5):
        action = env.action_space.sample() * 0.2
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {i+1} LiDAR readings:")
        print(f"  Min distance: {obs['lidar'].min():.3f}")
        print(f"  Max distance: {obs['lidar'].max():.3f}")
        print(f"  Mean distance: {obs['lidar'].mean():.3f}")
        
        # Find closest obstacle direction
        closest_idx = np.argmin(obs['lidar'])
        angle = closest_idx * (360 / env.num_rays_horizontal)
        print(f"  Closest obstacle at ~{angle:.0f}° (ray {closest_idx})")
        
        if terminated or truncated:
            break
    
    env.close()


def demo_flattened_obs():
    """Demo showing flattened observations for RL training."""
    print("Creating environment with flattened observations...")
    
    env = create_env_with_lidar(
        render=False, 
        num_rays=24, 
        max_range=8.0, 
        flatten=True
    )
    
    obs, info = env.reset()
    
    print(f"\nFlattened observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    
    # Verify it works with stable-baselines3 style training loop
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == env.observation_space.shape, "Shape mismatch!"
        assert obs.dtype == np.float32, "Wrong dtype!"
        
        if terminated or truncated:
            obs, info = env.reset()
    
    print("✓ Environment compatible with flat observation policies")
    env.close()


def demo_3d_lidar():
    """Demo showing 3D LiDAR with multiple vertical layers."""
    print("Creating environment with 3D LiDAR...")
    
    env = VectorVoyagerEnv(
        grid_sizes=(20.0, 20.0, 10.0),
        voxel_size=2.0,
        num_obstacles=5,
        render_mode="human",
        visual_obstacles=True,
    )
    
    # 3D LiDAR: 24 horizontal × 3 vertical = 72 rays
    env = LidarObservationWrapper(
        env,
        num_rays_horizontal=24,
        num_rays_vertical=3,  # 3 vertical layers
        max_range=10.0,
        fov_horizontal=360.0,
        fov_vertical=60.0,  # ±30° from horizontal
        normalize_distances=True,
        add_to_obs="separate",
    )
    
    obs, info = env.reset()
    
    print(f"\n3D LiDAR configuration:")
    print(f"  Horizontal rays: {env.num_rays_horizontal}")
    print(f"  Vertical layers: {env.num_rays_vertical}")
    print(f"  Total rays: {env.num_rays_total}")
    print(f"  LiDAR observation shape: {obs['lidar'].shape}")
    
    print("\nRunning 3D LiDAR visualization...")
    
    for step in range(300):
        action = env.action_space.sample() * 0.3
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 15 == 0:
            env.render_lidar_debug(duration=0.4)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()


if __name__ == "__main__":
    import sys
    
    demos = {
        "vis": demo_lidar_visualization,
        "data": demo_lidar_data,
        "flat": demo_flattened_obs,
        "3d": demo_3d_lidar,
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in demos:
        demos[sys.argv[1]]()
    else:
        print("Available demos:")
        print("  python example_lidar_usage.py vis   - Visualize LiDAR rays")
        print("  python example_lidar_usage.py data  - Show LiDAR data values")
        print("  python example_lidar_usage.py flat  - Test flattened observations")
        print("  python example_lidar_usage.py 3d    - 3D LiDAR with multiple layers")
        print("\nRunning default demo (data)...")
        demo_lidar_data()

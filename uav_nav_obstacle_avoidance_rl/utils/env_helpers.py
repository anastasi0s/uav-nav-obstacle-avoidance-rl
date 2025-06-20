from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

from uav_rl_navigation.environment.vector_voyager_env import VectorVoyagerEnv
from uav_rl_navigation.vendor.pyflyt import FlattenVectorVoyagerEnv
from uav_rl_navigation.environment.cam_wrapper import ThirdPersonCamWrapper
from uav_rl_navigation.environment.metrics_wrapper import UAVMetricsWrapper


def make_flat_voyager(num_waypoints: int = 1, with_metrics: bool = False, **env_kwargs):
    """
    Create a flattened Vector Voyager environment.
    
    Args:
        num_waypoints: Number of waypoints for the environment
        with_metrics: Whether to include metrics collection wrapper
        **env_kwargs: Additional arguments for the environment
    """
    # Create base environment
    env = VectorVoyagerEnv(num_targets=num_waypoints, **env_kwargs)
    
    # Add flattening wrapper
    env = FlattenVectorVoyagerEnv(env, context_length=num_waypoints)
    
    # Optionally add metrics wrapper
    if with_metrics:
        env = UAVMetricsWrapper(env)
    
    return env


def make_voyager_for_recording(video_folder, video_length, video_name, num_waypoints: int = 1, with_metrics: bool = False, **env_kwargs):
    """
    Create environment for video recording with third-person camera and optional metrics.
    """
    # create vectorized env and use render_mode="rgb_array" (env.rander() returns image array instead of poping up a window)
    # add third person camera wrapper
    env = make_vec_env(
        lambda **kw: ThirdPersonCamWrapper(make_flat_voyager(**kw)), 
        n_envs=1, 
        env_kwargs=dict(num_waypoints=num_waypoints, with_metrics=with_metrics,  render_mode="rgb_array")
    )

    # wrap with the VecVideoRecorder
    env = VecVideoRecorder(
        env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=video_name,
    )

    return env




import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

from uav_nav_obstacle_avoidance_rl.config import logger
from uav_nav_obstacle_avoidance_rl.environment.cam_wrapper import ThirdPersonCamWrapper
from uav_nav_obstacle_avoidance_rl.environment.vector_voyager_env import (
    VectorVoyagerEnv,
)
from uav_nav_obstacle_avoidance_rl.environment.flaten_wrapper import FlattenVectorVoyagerEnv
from uav_nav_obstacle_avoidance_rl.environment.lidar_observation_wrapper import (
    LidarObservationWrapper,
    LidarFlattenWrapper,
)


# create flat pyflyt environment
def make_flat_voyager(**env_kwargs):
    """
    Create a flattened Vector Voyager environment

    Args:
        **env_kwargs: Additional arguments for the environment (including num_targets)
    """
    # extract num_targets from env_kwargs
    num_targets = env_kwargs.get("num_targets", 1)

    # extract wrapper-specific parameters (not accepted by VectorVoyagerEnv)
    perception_mode = env_kwargs.pop("perception_mode", "none")
    lidar_kwargs = {
        "num_rays_horizontal": env_kwargs.pop("num_rays_horizontal", 36),
        "num_rays_vertical": env_kwargs.pop("num_rays_vertical", 1),
        "max_range": env_kwargs.pop("max_range", 10.0),
        "min_range": env_kwargs.pop("min_range", 0.1),
        "fov_horizontal": env_kwargs.pop("fov_horizontal", 360.0),
        "fov_vertical": env_kwargs.pop("fov_vertical", 30.0),
        "ray_start_offset": env_kwargs.pop("ray_start_offset", 0.15),
        "normalize_distances": env_kwargs.pop("normalize_distances", True),
        "add_to_obs": env_kwargs.pop("add_to_obs", "separate"),
    }

    # create base environment
    env = VectorVoyagerEnv(**env_kwargs)

    # wrap with LiDAR observation
    if perception_mode == "lidar":
        env = LidarObservationWrapper(env, **lidar_kwargs)
        env = LidarFlattenWrapper(env, context_length=num_targets)
    else:
        # use standard flattening wrapper
        env = FlattenVectorVoyagerEnv(env, context_length=num_targets)

    return env


# create vectorized environment for video recording
def make_voyager_for_recording(video_folder, video_length, video_name, **env_kwargs,):
    """
    Create environment for video recording with third-person camera and optional metrics.
    """
    # create vectorized env and use render_mode="rgb_array" (env.rander() returns image array instead of poping up a window)
    # add third person camera wrapper
    env = make_vec_env(
        env_id=lambda **env_kwargs: ThirdPersonCamWrapper(make_flat_voyager(**env_kwargs)),
        n_envs=1,
        env_kwargs=env_kwargs,
    )

    # wrap with the VecVideoRecorder
    env = VecVideoRecorder(
        env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step % video_length == 0,
        video_length=video_length,
        name_prefix=video_name,
    )

    return env




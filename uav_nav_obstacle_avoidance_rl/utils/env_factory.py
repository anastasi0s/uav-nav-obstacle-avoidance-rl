from gymnasium.wrappers import RescaleAction
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

from uav_nav_obstacle_avoidance_rl.environment.cam_wrapper import ThirdPersonCamWrapper
from uav_nav_obstacle_avoidance_rl.environment.flaten_wrapper import (
    FlattenVectorVoyagerEnv,
)
from uav_nav_obstacle_avoidance_rl.environment.lidar_wrapper import (
    LidarFlattenWrapper,
    LidarObservationWrapper,
)
from uav_nav_obstacle_avoidance_rl.environment.normalize_obs_wrapper import (
    NormalizeObservationWrapper,
)
from uav_nav_obstacle_avoidance_rl.environment.reward_wrapper import PyFlytRewardWrapper
from uav_nav_obstacle_avoidance_rl.environment.vector_voyager_env import (
    VectorVoyagerEnv,
)


# Env chain: Aviary (PyBullet) -> QuadXBaseEnv -> VectorVoyagerEnv -> LidarObservationWrapper -> CustomRewardWrapper -> RescaleAction -> NormalizeObservationWrapper -> LidarFlattenWrapper -> DummyVecEnv
def make_flat_voyager(**env_kwargs):
    """
    Create a flattened Vector Voyager environment

    Args:
        **env_kwargs: Additional arguments for the environment (including num_targets).
            perception_mode: "lidar" or "none" (default "none")
            lidar: dict of LidarObservationWrapper kwargs
            reward: dict of reward wrapper kwargs
    """
    # extract wrapper-specific parameters (not accepted by VectorVoyagerEnv)
    perception_mode = env_kwargs.pop("perception_mode")
    lidar_kwargs = env_kwargs.pop("lidar")
    
    reward_cfg = env_kwargs.pop("reward")
    reward_type = reward_cfg["type"]
    reward_kwargs = reward_cfg[reward_type]  # select the active sub-dict

    # 1. base environment
    env = VectorVoyagerEnv(**env_kwargs)

    # 2. lidar wrapper 
    if perception_mode == "lidar":
        env = LidarObservationWrapper(env, **lidar_kwargs)

    # # 3.reward wrappers
    if reward_type == "pyflyt":
        env = PyFlytRewardWrapper(env, **reward_kwargs)
    # elif reward_type == "custom":
    #     # works only with ray cast (lidar) measurements !!!
    #     env = CustomRewardWrapper(env, **reward_kwargs)

    # 4. normalization wrappers - normalize actions and observations (works with and without lidar)
    # env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    # env = NormalizeObservationWrapper(env)

    # 5. flatten env
    if perception_mode == "lidar":
        env = LidarFlattenWrapper(env, context_length=env_kwargs.get("num_targets"))
    else:
        env = FlattenVectorVoyagerEnv(env, context_length=env_kwargs.get("num_targets"))

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




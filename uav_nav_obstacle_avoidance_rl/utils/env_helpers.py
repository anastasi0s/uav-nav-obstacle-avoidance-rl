import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

from uav_nav_obstacle_avoidance_rl.config import logger
from uav_nav_obstacle_avoidance_rl.environment.cam_wrapper import ThirdPersonCamWrapper
from uav_nav_obstacle_avoidance_rl.environment.vector_voyager_env import (
    VectorVoyagerEnv,
)
from uav_nav_obstacle_avoidance_rl.vendor.pyflyt import FlattenVectorVoyagerEnv


# create flat pyflyt environment
def make_flat_voyager(num_waypoints: int = 1, **env_kwargs):
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

    return env


# create vectorized environment for video recording
def make_voyager_for_recording(
    video_folder,
    video_length,
    video_name,
    num_waypoints: int = 1,
    with_metrics: bool = False,
    **env_kwargs,
):
    """
    Create environment for video recording with third-person camera and optional metrics.
    """
    # create vectorized env and use render_mode="rgb_array" (env.rander() returns image array instead of poping up a window)
    # add third person camera wrapper
    env = make_vec_env(
        lambda **kw: ThirdPersonCamWrapper(make_flat_voyager(**kw)),
        n_envs=1,
        env_kwargs=dict(
            num_waypoints=num_waypoints,
            with_metrics=with_metrics,
            render_mode="rgb_array",
        ),
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


# analyse environment characteristics
def analyse_env(env):
    # reset
    out = env.reset()
    if hasattr(env, "num_envs"):
        obs = out  # VecEnv only returns obs
        n = env.num_envs
        # build batch of actions
        actions = np.stack([env.action_space.sample() for _ in range(n)])
        batch = env.step(actions)  # get batch of multiple stacked envs
        # VecEnv.step returns (obs, rewards, dones, infos)
        obs2, rews, dones, infos = batch
        logger.info(
            f"VecEnv:\n\nObservation space:\n{env.observation_space}\n\nStacked observations sample:\n{obs}\n\nAction space:\n{env.action_space}\n\nStacked actions sample:\n{actions}\n\nStacked reward samples:\n{rews}\n\nAdditional stacked monitored info:\n{infos}\n"
        )
    else:
        obs, info = out  # single env returns (obs, info)
        action = env.action_space.sample()
        obs2, rew, term, trunc, info2 = env.step(action)
        logger.info(
            f"Single Env:\n\nObservation space:\n{env.observation_space}\n\nObservation sample:\n{obs}\n\nAction space:\n{env.action_space}\n\nAction sample:\n{action}\n\nReward sample:\n{rew}\n\nAdditional monitored infos:\n{info2}\n"
        )

    # only vec environments have the env.envs attribute
    if hasattr(env, "envs"):
        inner_env = env.envs[0]  # grab the first inner env
    else:
        inner_env = env.env

    chain = []
    chain.append(type(env).__name__)  # add the name of the outer most env

    # now peel off the rest of the wrappers
    current = inner_env
    while True:
        chain.append(type(current).__name__)
        # most wrappers keep the inner env in .env
        if not hasattr(current, "env"):
            break
        current = current.env

    logger.info(f"Wrapper chain (outer <- inner): {' <- '.join(chain)}")

import numpy as np
import typer

from uav_nav_obstacle_avoidance_rl.config import logger

app = typer.Typer()


def analyse_env(env):
    """Analyse environment characteristics - helper function that can be reused"""
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


@app.command()
def analyse():
    """Analyse environment characteristics using the analyse_env function"""
    from stable_baselines3.common.env_util import make_vec_env

    from uav_nav_obstacle_avoidance_rl.utils import env_helpers

    # create env
    vec_env = make_vec_env(
        env_helpers.make_flat_voyager,
        n_envs=1,
        env_kwargs=dict(
            num_targets=1,
            grid_sizes=(10.0, 10.0, 5.0),
            voxel_size=0.5,
        ),
    )

    analyse_env(vec_env)
    vec_env.close()


@app.command()
def test_environment_access():
    """quick debug function to test environment state access. check if the Observations are aligned with the state values (velocities and position) from the base env"""
    from stable_baselines3.common.env_util import make_vec_env

    from uav_nav_obstacle_avoidance_rl.utils import env_helpers

    # create env
    vec_env = make_vec_env(
        env_helpers.make_flat_voyager,
        n_envs=1,
        env_kwargs=dict(
            num_targets=1,
            grid_sizes=(10.0, 10.0, 5.0),
            voxel_size=0.5,
        ),
    )

    # reset env
    obs = vec_env.reset()
    logger.info(f"Vec Env: {type(vec_env)}")
    logger.info(
        f"Observation shape: {obs.shape}. [0] rotation vel, [1] rotation position, [2:5] linear vel (u,v,w), [5] z position, [6:10] previous action, [11:13] target deltas"
    )
    logger.info(f"Observation sample: {obs[0][:]}")  # First 10 elements (attitude)

    # access underlying wrapper (only vec_envs have 'envs' attribute)
    if hasattr(vec_env, "envs"):
        env = vec_env.envs[0]
        logger.info(f"First wrapper: {type(env)}")

        # peel off wrappers
        current = env
        while hasattr(current, "env"):
            current = current.env
            logger.info(f"Next wrapper: {type(current)}")

            # check if this is VectorVoyagerEnv
            if hasattr(current, "waypoints") and hasattr(current, "compute_attitude"):
                logger.info("Found VectorVoyagerEnv!")

                # check if the aviary state is accessible
                if hasattr(current, "env") and hasattr(current.env, "state"):
                    state = current.env.state(0)
                    logger.info(f"Aviary state shape: {state.shape}")
                    logger.info(f"Velocity, state[2] (lin_vel): {state[2]}")
                    logger.info(f"Position, state[3] (lin_pos): {state[3]}")

                    # check waypoints
                    if hasattr(current.waypoints, "targets"):
                        logger.info(
                            f"Number of targets: {len(current.waypoints.targets)}"
                        )
                        if len(current.waypoints.targets) > 0:
                            logger.info(f"First target: {current.waypoints.targets[0]}")
                break

    # take a few steps
    for i in range(5):
        # vec envs need a batch of actions
        action = np.array([vec_env.action_space.sample()])  # Shape: (1, 4) for n_envs=1
        obs, reward, done, info = vec_env.step(action)
        logger.info(f"Action {i} shape: {action.shape}, action: {action}\n")
        logger.info(
            f"Step {i}: reward={reward}, done={done}, info keys={list(info[0].keys())} info values={list(info[0].values())}\n"
        )
        logger.info(f"Observation {i}: {obs[0][:]}")  # First 10 elements (attitude)

        # peel off wrappers to get values from base env
        if hasattr(vec_env, "envs"):
            env = vec_env.envs[0]
            current = env
        else:
            current = vec_env
        while hasattr(current, "env"):
            current = current.env
            # check if this is VectorVoyagerEnv
            if hasattr(current, "waypoints") and hasattr(current, "compute_attitude"):
                # check if the aviary state is accessible
                if hasattr(current, "env") and hasattr(current.env, "state"):
                    state = current.env.state(0)
                    logger.info(f"Velocity, state[2] (lin_vel): {state[2]}")
                    logger.info(f"Position, state[3] (lin_pos): {state[3]}\n\n")
                break

    vec_env.close()


if __name__ == "__main__":
    app()

import typer

from uav_nav_obstacle_avoidance_rl import config

logger = config.logger
app = typer.Typer()


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

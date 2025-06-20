from pathlib import Path

from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import typer

from uav_rl_navigation import config
from uav_rl_navigation.utils import env_helpers

app = typer.Typer()


@app.command()
def main(
    model_path: Path = config.MODELS_DIR / "ppo_waypoint_v0",
    timesteps: int = 10_000,
):

    logger.info("Training agent...")

    # instantiate vec_env
    vec_env = make_vec_env(  # -> vec env
        env_helpers.make_flat_voyager, 
        n_envs=1, 
        env_kwargs=dict(num_waypoints=1, with_metrics=False), 
        monitor_dir= f"{config.TRAIN_RESULTS}",
        monitor_kwargs=dict(info_keywords=("out_of_bounds", "collision", "env_complete", "num_targets_reached")),  # collect extra information to log, from the information return of env.step() - ep reward, ep length, ep time length  
    ) 

    # TRAIN agent
    model = PPO("MlpPolicy", vec_env, verbose=1).learn(total_timesteps=timesteps)

    # sample an observation from the env
    obs = model.env.observation_space.sample()

    # predict an action, BEFORE saving the model
    logger.info("prediction, BEFOR saving model:", model.predict(obs, deterministic=True))

    # save model
    model.save(model_path)

    del model

    # load model
    model = PPO.load(model_path)

    # predict an action, AFTER loading the model (use the same observation)
    logger.info("prediction, AFTER loading model:", model.predict(obs, deterministic=True))

    logger.success("Agent training complete.")


if __name__ == "__main__":
    app()

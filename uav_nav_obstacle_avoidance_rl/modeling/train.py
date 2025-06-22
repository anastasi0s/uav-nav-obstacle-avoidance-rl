from pathlib import Path
import numpy as np

from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import typer

from uav_nav_obstacle_avoidance_rl import config
from uav_nav_obstacle_avoidance_rl.utils import env_helpers, metrics_callback

app = typer.Typer()


@app.command()
def main(
    exp_name: str = "waypoint_v0",
    timesteps: int = 100_000,
    eval_freq: int = 1000,
    n_envs: int = 1,
):
    logger.info(f"Training agent {exp_name}")
    
    exp_report_dir = config.REPORTS_DIR / exp_name / "train"
    exp_report_dir.mkdir(parents=True, exist_ok=True)  # create dir 

    # instantiate vec_env
    vec_env = make_vec_env(  # -> vec env
        env_helpers.make_flat_voyager, 
        n_envs=n_envs, 
        env_kwargs=dict(num_waypoints=1, with_metrics=False), 
        monitor_dir= exp_report_dir.as_posix(),
        monitor_kwargs=dict(info_keywords=("out_of_bounds", "collision", "env_complete", "num_targets_reached")),  # collect extra information to log, returned by compute_base_term_trunc_reward and compute_term_trunc_reward
    )

    # analyse env
    env_helpers.analyse_env(vec_env)

    # create callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=max(100000 // n_envs, 1),
            save_path=(exp_report_dir / 'checkpoints').as_posix(),
            name_prefix=exp_name,
        ),
        metrics_callback.UAVMetricsCallback(
            log_dir=exp_report_dir.as_posix(),
            save_freq=1000,
            verbose=1,
            )
    ]

    # TRAIN agent
    model = PPO(
        "MlpPolicy", 
        vec_env,
        verbose=1, 
        tensorboard_log=(exp_report_dir / 'tensorboard').as_posix()
        ).learn(
            total_timesteps=timesteps, 
            callback=callbacks,
            )

    # # sample an observation from the env
    # obs = model.env.observation_space.sample()

    # # predict an action, BEFORE saving the model
    # pred_before = model.predict(obs, deterministic=True)
    # logger.info(f"prediction, BEFOR saving model: {pred_before}")

    # # save -> re-load -> check model outputs
    # model.save(config.MODELS_DIR / exp_name)
    # del model
    # # load model
    # model = PPO.load(config.MODELS_DIR / exp_name)

    # # predict an action, AFTER loading the model (use the same observation)
    # pred_after = model.predict(obs, deterministic=True)
    # logger.info(f"prediction, AFTER re-loading model: {pred_after}")
    # if np.array_equal(pred_before[0], pred_after[0]):
    #     logger.info("Saved model is OK")
    # else:
    #     logger.info("Saved model is NOT OK! Model outputs don't match.")

    logger.success("Agent training complete.")

if __name__ == "__main__":
    app()

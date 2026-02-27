from pathlib import Path

import typer
import yaml
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

from uav_nav_obstacle_avoidance_rl import config
from uav_nav_obstacle_avoidance_rl.utils import env_helpers
from uav_nav_obstacle_avoidance_rl.test import base_env_test
from uav_nav_obstacle_avoidance_rl.utils.eval_metrics_callback import CustomEvalCallback
from uav_nav_obstacle_avoidance_rl.utils.train_metrics_callback import TrainMetricsCallback

logger = config.logger
app = typer.Typer()

CONFIG_PATH = Path(__file__).resolve().parent / "config-defaults.yaml"

MONITOR_INFO_KEYWORDS = ("out_of_bounds", "collision", "env_complete", "num_targets_reached", "num_obstacles")


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load experiment config from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)

@app.command()
def run_exp(
    exp_name: str,
    timesteps: int,
    eval_freq: int,
    n_envs: int = 2,
    exp_analysis: bool = True,
    wandb_project: str = "uav-nav-obstacle-avoidance-rl",
    wandb_tags: list = [],
    ):
    """
    training script with W&B integration for experiment tracking
    """

    # ----- 1. load config & init W&B --------

    exp_config = load_config()
    with wandb.init(
        project=wandb_project,
        name=exp_name,
        tags=wandb_tags,
        config=exp_config,
        dir=config.REPORTS_DIR.as_posix(),
        monitor_gym=True,  # auto-upload videos
        ) as run:

        env_config = dict(run.config["env"])
        ppo_config = dict(run.config["ppo"])
        monitor_info = {"info_keywords": MONITOR_INFO_KEYWORDS}

        # ----- 2. create environments --------
        # create training environment
        vec_env = make_vec_env(
            env_helpers.make_flat_voyager,
            n_envs=n_envs,
            env_kwargs={**env_config, "visual_obstacles": False},  # train without visual obstacles
            monitor_kwargs=monitor_info,
            )
        
        # # analyze train environment
        base_env_test.analyse_env(vec_env)

        # create separate evaluation environment
        vec_env_eval = make_vec_env(
            env_helpers.make_flat_voyager,
            n_envs=1,  # single env for evaluation - simple and clean
            env_kwargs=env_config,
            monitor_kwargs=monitor_info,
        )

        # ----- 3. callbacks --------
        callbacks = []
        # add W&B callback
        wandb_callback = WandbCallback(
            # gradient_save_freq=1000,
            verbose=0,
        )

        # add custom train metrics callback
        train_callback = TrainMetricsCallback(
            run_path=run.dir,
            # model_save_path=f"{run.dir}/models",
            verbose=0,
        )

        # calculate eval frequency based on parallel environments -> eval_freq = actual time-steps
        eval_freq = eval_freq // n_envs
        # add custom evaluation metrics callback
        eval_callback = CustomEvalCallback(
            vec_env_eval,
            best_model_save_path=f"{run.dir}/models",
            log_path=run.dir,
            eval_freq=eval_freq,
            n_eval_episodes=40,
            deterministic=True,
            render=False,
            verbose=0,
            exp_analysis=exp_analysis,
        )
        
        callbacks.append(wandb_callback)
        callbacks.append(train_callback)
        callbacks.append(eval_callback)

        # ----- 4. define and train model --------
        model = PPO(
            "MlpPolicy",
            vec_env,
            **ppo_config,
            verbose=0,
            tensorboard_log=f"{run.dir}/tensorboard",
            seed=config.RANDOM_SEED,  # the seed is passed through the chain: (PPO → Gymnasium → QuadXBaseEnv → Aviary → VectorVoyagerEnv → VoxelGrid)
        )

        # train model
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True,
        )


@app.command()
def sweep():
    """
    Run a hyperparameter sweep using W&B.
    """
    pass


if __name__ == "__main__":
    app()

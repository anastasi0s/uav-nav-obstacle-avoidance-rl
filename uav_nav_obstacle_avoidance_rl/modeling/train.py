from pathlib import Path

import typer
import wandb
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

from uav_nav_obstacle_avoidance_rl import config
from uav_nav_obstacle_avoidance_rl.test import base_env_test
from uav_nav_obstacle_avoidance_rl.utils import env_helpers
from uav_nav_obstacle_avoidance_rl.utils.curriculum_callback import CurriculumCallback
from uav_nav_obstacle_avoidance_rl.utils.eval_metrics_callback import CustomEvalCallback
from uav_nav_obstacle_avoidance_rl.utils.train_metrics_callback import TrainMetricsCallback

app = typer.Typer()
logger = config.logger

MONITOR_INFO_KEYWORDS = ("collision", "env_complete", "num_targets_reached", "num_obstacles")


def load_config(path: Path = config.EXP_CONFIG_PATH) -> dict:
    """Load experiment config from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)

@app.command()
def run_exp(
    exp_name: str = "exp_0",
    timesteps: int = 500000,
    eval_freq: int = 100000,
    n_envs: int = 2,
    n_eval_episodes: int = 30,
    log_interval: int = 10,
    seed: int = config.RANDOM_SEED,
    exp_analysis: bool = True,
    wandb_project: str = "uav-nav-obstacle-avoidance-rl",
    wandb_tags: list[str] | None = None,
    verbose: int = 0,
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
        settings=wandb.Settings(x_disable_stats=True)
        ) as run:

        env_config = dict(run.config["env"])
        ppo_config = dict(run.config["ppo"])
        curriculum_config = dict(run.config["curriculum"])
        monitor_info = {"info_keywords": MONITOR_INFO_KEYWORDS}

        # ----- 2. create environments --------
        # create training environment (never visual obstacles for performance)
        vec_env = make_vec_env(
            env_helpers.make_flat_voyager,
            n_envs=n_envs,
            env_kwargs={**env_config, "visual_obstacles": False},
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
            verbose=verbose,
        )

        # add custom train metrics callback
        train_callback = TrainMetricsCallback(
            run_path=run.dir,
            # model_save_path=f"{run.dir}/models",
            verbose=verbose,
        )

        eval_freq = eval_freq // n_envs  # calculate eval frequency based on parallel environments -> eval_freq = actual time-steps
        # add custom evaluation metrics callback
        eval_callback = CustomEvalCallback(
            vec_env_eval,
            best_model_save_path=f"{run.dir}/models",
            log_path=run.dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=verbose,
            exp_analysis=exp_analysis,
        )
        
        callbacks.append(wandb_callback)
        callbacks.append(train_callback)
        callbacks.append(eval_callback)

        # add curriculum callback
        if curriculum_config.pop('enabled', False):
            curriculum_callback = CurriculumCallback(**curriculum_config, verbose=verbose, eval_env=vec_env_eval)
            callbacks.append(curriculum_callback)

        # ----- 4. define and train model --------
        model = PPO(
            "MlpPolicy",
            vec_env,
            **ppo_config,
            verbose=verbose,
            tensorboard_log=f"{run.dir}/tensorboard",
            seed=seed,  # the seed is passed through the chain: (PPO → Gymnasium → QuadXBaseEnv → Aviary → VectorVoyagerEnv → VoxelGrid)
        )

        # train model
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=log_interval,
        )


@app.command()
def sweep():
    """
    Run a hyperparameter sweep using W&B.
    """
    pass


if __name__ == "__main__":
    app()

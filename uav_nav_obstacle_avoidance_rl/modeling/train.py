from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer
import wandb
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from wandb.integration.sb3 import WandbCallback

from uav_nav_obstacle_avoidance_rl import config
from uav_nav_obstacle_avoidance_rl.test import base_env_test
from uav_nav_obstacle_avoidance_rl.utils import env_factory
from uav_nav_obstacle_avoidance_rl.utils.curriculum_callback import CurriculumCallback
from uav_nav_obstacle_avoidance_rl.utils.eval_metrics_callback import CustomEvalCallback
from uav_nav_obstacle_avoidance_rl.utils.train_metrics_callback import TrainMetricsCallback

app = typer.Typer()
logger = config.logger

MONITOR_INFO_KEYWORDS = ("collision", "env_complete", "num_targets_reached", "num_obstacles")


@dataclass
class TrainParams:
    timesteps: int = 2_000_000
    eval_freq: int = 200_000
    n_envs: int = 2
    n_eval_episodes: int = 30
    log_interval: int = 10
    seed: int = 9
    verbose: int = 0


def _load_config(path: Path = config.EXP_CONFIG_PATH) -> dict:
    """Load experiment config from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def _train(
    run: wandb.sdk.wandb_run.Run,
    params: TrainParams,
    exp_analysis: bool,
):
    """training logic used by both run_train and sweep agents"""
    env_config = dict(run.config["env"])
    ppo_config = dict(run.config["ppo"])
    curriculum_config = dict(run.config["curriculum"])
    monitor_info = {"info_keywords": MONITOR_INFO_KEYWORDS}

    # ----- create environments --------
    vec_env = make_vec_env(
        env_factory.make_flat_voyager,
        n_envs=params.n_envs,
        env_kwargs={**env_config, "visual_obstacles": False},
        monitor_kwargs=monitor_info,
    )

    base_env_test.analyse_env(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    base_env_test.analyse_env(vec_env)

    # create separate evaluation env
    vec_env_eval = make_vec_env(
        env_factory.make_flat_voyager,
        n_envs=params.n_envs,
        env_kwargs=env_config,
        monitor_kwargs=monitor_info,
    )

    vec_env_eval = VecNormalize(vec_env_eval, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ----- callbacks --------
    callbacks = []
    wandb_callback = WandbCallback(verbose=params.verbose)

    train_callback = TrainMetricsCallback(
        run_path=run.dir,
        verbose=params.verbose,
    )

    adj_eval_freq = params.eval_freq // params.n_envs
    adj_n_eval_episodes = params.n_eval_episodes // params.n_envs
    eval_callback = CustomEvalCallback(
        vec_env_eval,
        best_model_save_path=f"{run.dir}/models",
        log_path=run.dir,
        eval_freq=adj_eval_freq,
        n_eval_episodes=adj_n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=params.verbose,
        exp_analysis=exp_analysis,
        seed=params.seed,
    )

    callbacks.append(wandb_callback)
    callbacks.append(train_callback)
    callbacks.append(eval_callback)

    if curriculum_config.pop('enabled', False):
        curriculum_callback = CurriculumCallback(**curriculum_config, verbose=params.verbose, eval_env=vec_env_eval)
        callbacks.append(curriculum_callback)

    model = PPO(
        "MlpPolicy",
        vec_env,
        **ppo_config,
        verbose=params.verbose,
        tensorboard_log=f"{run.dir}/tensorboard",
        seed=params.seed,
        device="cpu",
    )

    model.learn(
        total_timesteps=params.timesteps,
        callback=callbacks,
        progress_bar=True,
        log_interval=params.log_interval,
    )

# uv run python -m uav_nav_obstacle_avoidance_rl.modeling.train run-train --exp-name "exp" --wandb-tags --timesteps 2_000_000 --eval-freq 200_000                                        
@app.command()
def run_train(
    exp_name: str = "exp",
    timesteps: int = TrainParams.timesteps,
    eval_freq: int = TrainParams.eval_freq,
    n_envs: int = TrainParams.n_envs,
    n_eval_episodes: int = TrainParams.n_eval_episodes,
    log_interval: int = TrainParams.log_interval,
    seed: int = TrainParams.seed,
    verbose: int = TrainParams.verbose,
    exp_analysis: bool = True,
    wandb_project: str = "uav-nav-obstacle-avoidance-rl",
    wandb_tags: list[str] | None = None,
):
    """
    training script with W&B integration for experiment tracking
    """
    exp_config = _load_config()
    with wandb.init(
        project=wandb_project,
        name=exp_name,
        tags=wandb_tags,
        config=exp_config,
        dir=config.REPORTS_DIR.as_posix(),
        monitor_gym=True,
        save_code=True,
        settings=wandb.Settings(x_disable_stats=True),
    ) as run:
        _train(
            run=run,
            params=TrainParams(
                timesteps=timesteps,
                eval_freq=eval_freq,
                n_envs=n_envs,
                n_eval_episodes=n_eval_episodes,
                log_interval=log_interval,
                seed=seed,
                verbose=verbose,
            ),
            exp_analysis=exp_analysis,
        )


# # create sweep from file
# uv run python -m uav_nav_obstacle_avoidance_rl.modeling.train sweep \
#   --sweep-config-path uav_nav_obstacle_avoidance_rl/modeling/exp-5-sweep.yaml \
#   --count 25

# # terminal 2: join the same sweep
# uv run python -m uav_nav_obstacle_avoidance_rl.modeling.train sweep \
#   --sweep-id abc123 \
#   --count 25
@app.command()
def sweep(
    wandb_project: str = "uav-nav-obstacle-avoidance-rl",
    count: int = 20,
    timesteps: int = TrainParams.timesteps,
    eval_freq: int = TrainParams.eval_freq,
    n_envs: int = TrainParams.n_envs,
    n_eval_episodes: int = TrainParams.n_eval_episodes,
    log_interval: int = TrainParams.log_interval,
    seed: int = TrainParams.seed,
    verbose: int = TrainParams.verbose,
    sweep_id: Optional[str] = None,
    sweep_config_path: Optional[Path] = None,
):
    """
    wandb sweep using Bayesian optimization

    create new sweep (or resume existing --sweep-id) and launches
    an agent that runs `count` training runs each with different hyperparameters sampled by the sweep controller
    """
    exp_config = _load_config()

    def _sweep_train():
        with wandb.init(
            config=exp_config,
            group=sweep_id,
            dir=config.REPORTS_DIR.as_posix(),
            save_code=True,
            settings=wandb.Settings(x_disable_stats=True),
        ) as run:
            _train(
                run=run,
                params=TrainParams(
                    timesteps=timesteps,
                    eval_freq=eval_freq,
                    n_envs=n_envs,
                    n_eval_episodes=n_eval_episodes,
                    log_interval=log_interval,
                    seed=seed,
                    verbose=verbose,
                ),
                exp_analysis=False,
            )

    if sweep_id is None:
        cfg_path = Path(sweep_config_path) if sweep_config_path else config.SWEEP_CONFIG_PATH
        sweep_config = _load_config(cfg_path)
        sweep_id = wandb.sweep(sweep=sweep_config, project=wandb_project)
        logger.info(f"Created sweep with ID: {sweep_id}")

    logger.info(f"Starting sweep agent (sweep_id={sweep_id}, count={count})")
    wandb.agent(sweep_id, function=_sweep_train, count=count, project=wandb_project)

#  uv run python -m uav_nav_obstacle_avoidance_rl.modeling.train seed-sweep --exp-name "exp" --wandb-tags ["tag",] --timesteps 2_000_000 --eval-freq 200_000 
@app.command()
def seed_sweep(
    exp_name: str = "exp",
    wandb_project: str = "uav-nav-obstacle-avoidance-rl",
    timesteps: int = TrainParams.timesteps,
    eval_freq: int = TrainParams.eval_freq,
    n_envs: int = TrainParams.n_envs,
    n_eval_episodes: int = TrainParams.n_eval_episodes,
    log_interval: int = TrainParams.log_interval,
    verbose: int = TrainParams.verbose,
    sweep_id: Optional[str] = None,
    wandb_tags: list[str] | None = None,
):
    """
    Run the same experiment with multiple seeds using a W&B grid sweep.

    Seeds are defined in seed-sweep.yaml. Each seed becomes a separate
    W&B run, grouped together for easy comparison and aggregation.
    """
    exp_config = _load_config()

    def _seed_sweep_train():
        with wandb.init(
            config=exp_config,
            group=sweep_name,
            dir=config.REPORTS_DIR.as_posix(),
            save_code=True,
            settings=wandb.Settings(x_disable_stats=True),
            tags=wandb_tags,
        ) as run:
            run.name = f"{exp_name}-{run.config['seed']}"
            _train(
                run=run,
                params=TrainParams(
                    timesteps=timesteps,
                    eval_freq=eval_freq,
                    n_envs=n_envs,
                    n_eval_episodes=n_eval_episodes,
                    log_interval=log_interval,
                    seed=run.config["seed"],
                    verbose=verbose,
                ),
                exp_analysis=False,
            )

    sweep_name = exp_name
    if sweep_id is None:
        sweep_config = _load_config(config.SEED_SWEEP_CONFIG_PATH)
        sweep_config["name"] = sweep_name
        sweep_id = wandb.sweep(sweep=sweep_config, project=wandb_project)
        logger.info(f"Created seed sweep with ID: {sweep_id}")

    logger.info(f"Starting seed sweep agent (sweep_id={sweep_id})")
    wandb.agent(sweep_id, function=_seed_sweep_train, project=wandb_project)


if __name__ == "__main__":
    app()
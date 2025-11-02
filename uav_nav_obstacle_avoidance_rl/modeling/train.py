import time

import typer
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

from uav_nav_obstacle_avoidance_rl import config
from uav_nav_obstacle_avoidance_rl.utils import env_helpers
from uav_nav_obstacle_avoidance_rl.utils.eval_metrics_callback import CustomEvalCallback
from uav_nav_obstacle_avoidance_rl.utils.train_metrics_callback import TrainMetricsCallback

logger = config.logger
app = typer.Typer()


@app.command()
def run_exp(
    exp_name: str = f"run_{int(time.time())}",
    timesteps: int = 15_000,
    eval_freq: int = 5_000,
    n_envs: int = 2,
    exp_analysis: bool = True,
    wandb_project: str = "uav-nav-obstacle-avoidance-rl",
    wandb_tags: list[str] | None = None,
    
    # Environment parameters
    num_targets: int = 1,
    flight_dome_size: float = 5.0,
    max_duration_seconds: float = 80.0,
    enable_obstacles: bool = True,
    visual_obstacles: bool = True,
    num_obstacles: tuple[int, int] = (0, 3),
    obstacle_types: list[str] = ["sphere", "box", "cylinder"],
    obstacle_size_range: tuple[float, float] = (0.1, 0.8),
    obstacle_min_distance_from_start: float = 1.0,
    obstacle_hight_range: tuple[float, float] = (0.1, 5.0),
    
    # PPO hyperparameters
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_steps: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
):
    """
    training script with W&B integration for experiment tracking
    """
    logger.info(f"Running Experiment: {exp_name}")

    # calculate eval frequency based on parallel environments -> eval_freq = actual time-steps
    eval_freq = eval_freq // n_envs

    # Prepare configuration for tracking
    config_dict = {
        "algorithm": "PPO",
        "total_timesteps": timesteps,
        "eval_freq": eval_freq,
        "n_envs": n_envs,
        # Environment params
        "num_targets": num_targets,
        "flight_dome_size": flight_dome_size,
        "max_duration_seconds": max_duration_seconds,
        "enable_obstacles": enable_obstacles,
        "visual_obstacles": visual_obstacles,
        "num_obstacles": num_obstacles,
        "obstacle_types": obstacle_types,
        "obstacle_size_range": obstacle_size_range,
        "obstacle_min_distance_from_start": obstacle_min_distance_from_start,
        "obstacle_hight_range": obstacle_hight_range,
        # PPO hyperparameters
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
    }

    # init W&B
    wandb_run = wandb.init(
        project=wandb_project,
        name=exp_name,
        config=config_dict,
        tags=wandb_tags or ["training_and_eval", f"dome_{flight_dome_size}m"],
        dir=config.REPORTS_DIR.as_posix(),
        sync_tensorboard=False,  # auto-upload tensorboard metrics
        monitor_gym=True,  # auto-upload videos
        save_code=True,  # save code for reproducibility
    )

    # train environment configuration
    env_kwargs_train = {
        "num_targets": num_targets,
        "flight_dome_size": flight_dome_size,
        "max_duration_seconds": max_duration_seconds,
        "enable_obstacles": enable_obstacles,
        "visual_obstacles": False,
        "num_obstacles": num_obstacles,
        "obstacle_types": obstacle_types,
        "obstacle_size_range": obstacle_size_range,
        "obstacle_min_distance_from_start": obstacle_min_distance_from_start,
        "obstacle_hight_range": obstacle_hight_range,
    }

    # create training environment
    vec_env = make_vec_env(
        env_helpers.make_flat_voyager,
        n_envs=n_envs,
        env_kwargs=env_kwargs_train,
        monitor_kwargs=dict(
            info_keywords=(
                "out_of_bounds",
                "collision",
                "env_complete",
                "num_targets_reached",
                "num_obstacles_spawned",
            )
        ),
    )

    # separate eval environment configuration
    env_kwargs_val = {
        "num_targets": num_targets,
        "flight_dome_size": flight_dome_size,
        "max_duration_seconds": max_duration_seconds,
        "enable_obstacles": enable_obstacles,
        "visual_obstacles": visual_obstacles,
        "num_obstacles": num_obstacles,
        "obstacle_types": obstacle_types,
        "obstacle_size_range": obstacle_size_range,
        "obstacle_min_distance_from_start":obstacle_min_distance_from_start,
        "obstacle_hight_range": obstacle_hight_range,
    }

    # separete evaluation environment
    vec_env_eval = make_vec_env(
        env_helpers.make_flat_voyager,
        n_envs=1,  # single env for evaluation - simple and clean
        env_kwargs=env_kwargs_val,
        monitor_kwargs=dict(
            info_keywords=(
                "out_of_bounds",
                "collision",
                "env_complete",
                "num_targets_reached",
                "num_obstacles_spawned",
            )
        ),
    )

    # analyze environment
    env_helpers.analyse_env(vec_env)

    # Setup callbacks
    callbacks = []

    # add W&B callback
    wandb_callback = WandbCallback(
        # gradient_save_freq=1000,
        verbose=2,
    )
    callbacks.append(wandb_callback)

    # add custom train metrics callback
    train_callback = TrainMetricsCallback(
        run_path=wandb_run.dir,
        # model_save_path=f"{wandb_run.dir}/models",
        verbose=2,
    )
    callbacks.append(train_callback)

    # add custom evaluation metrics callback
    eval_callback = CustomEvalCallback(
        vec_env_eval,
        best_model_save_path=f"{wandb_run.dir}/models",
        log_path=wandb_run.dir,
        eval_freq=eval_freq,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        exp_analysis=exp_analysis,
    )
    callbacks.append(eval_callback)

    # create model
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=0,
        tensorboard_log=f"{wandb_run.dir}/tensorboard",
        seed=config.RANDOM_SEED,
    )

    # train model
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Cleanup
    wandb.finish()

    logger.info(f"Completed experiment: {exp_name}")


@app.command()
def sweep():
    """
    Run a hyperparameter sweep using W&B.
    """
    # Define sweep configuration
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "rollout/ep_rew_mean", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"values": [1e-4, 3e-4, 1e-3]},
            "batch_size": {"values": [32, 64, 128]},
            "gamma": {"values": [0.95, 0.99, 0.995]},
            "flight_dome_size": {"values": [3.0, 5.0, 10.0]},
            "n_steps": {"values": [1024, 2048, 4096]},
        },
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="uav-nav-rl-sweep")

    def train_sweep():
        with wandb.init() as run:
            config = run.config
            run_exp(
                exp_name=f"sweep_{run.id}",
                timesteps=50_000,  # Shorter for sweeps
                wandb_project="uav-nav-rl-sweep",
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                gamma=config.gamma,
                flight_dome_size=config.flight_dome_size,
                n_steps=config.n_steps,
                exp_analysis=False,  # W&B handles plotting
            )

    # Run sweep
    wandb.agent(sweep_id, train_sweep, count=10)


if __name__ == "__main__":
    app()

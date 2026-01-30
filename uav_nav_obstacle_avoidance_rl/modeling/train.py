from tabnanny import verbose
import time

import typer
from typing import Any, Literal, Tuple, Dict
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
    ### Experiment parameters ###
    exp_name: str = f"run_{int(time.time())}",
    timesteps: int = 15_000,
    eval_freq: int = 5_000,
    n_envs: int = 2,
    exp_analysis: bool = True,
    wandb_project: str = "uav-nav-obstacle-avoidance-rl",
    wandb_tags: list[str] | None = None,

    ### Environment parameters ###
    # boundary parameters
    grid_sizes: tuple[float, float, float] = (10.0, 10.0, 5.0),  # (x, y, z)
    voxel_size: float = 1.0,
    min_height: float = 0.0,  # default: 0.1,  min allowed hight, collision is detected if below that height

    # waypoint parameters
    num_targets: int = 1,
    sparse_reward: bool = False,
    use_yaw_targets: bool = False,  # toggles whether the agent must also align its yaw (heading) to a per‐waypoint target before that waypoint is considered "reached," and whether yaw error is included in the observation.
    goal_reach_distance: float = 0.2,  # distance within which the target is considered reached
    goal_reach_angle: float = 0.1,  # not in use since use_yaw_targets is not in use

    # obstacle parameters
    obstacle_strategy: str = "random",  # "random",
    num_obstacles: int = 6,
    visual_obstacles: bool = False,  # only for evaluation

    # observation parameters
    perception_mode: Literal["none", "lidar"] = "lidar",
    num_rays_horizontal: int = 36,
    num_rays_vertical: int = 1,
    max_range: float = 10.0,
    min_range: float = 0.1,
    fov_horizontal: float = 360.0,
    fov_vertical: float = 30.0,
    ray_start_offset: float = 0.15,
    normalize_distances: bool = True,
    add_to_obs: Literal["append", "separate", "replace"] = "separate",

    # simulation parameters
    max_duration_seconds: float = 80.0,  # max simulation time of the env
    flight_mode: int = 5,  # uav constrol mode 5: (u, v, vr, vz) -> u: local velocity forward in m/s, v: lateral velocity in m/s, vr: yaw in rad/s, vz: vertical velocity in m/s
    angle_representation: str = "quaternion",
    agent_hz: int = 30,  # looprate of the agent to environment interaction
    render_mode: str | None = None,
    render_resolution: tuple[int, int] = (480, 480),

    ### PPO hyperparameters ###
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

    # Prepare configuration for w&b tracking
    config_dict = {
        "algorithm": "PPO",
        "total_timesteps": timesteps,
        "eval_freq": eval_freq,
        "n_envs": n_envs,
        # Environment params
        "num_targets": num_targets,
        "grid_sizes": grid_sizes,
        "voxel_size": voxel_size,
        "max_duration_seconds": max_duration_seconds,
        "visual_obstacles": visual_obstacles,
        "num_obstacles": num_obstacles,
        "obstacle_strategy": obstacle_strategy,
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
        tags=wandb_tags or ["training_and_eval", f"grid_space_{grid_sizes}m"],
        dir=config.REPORTS_DIR.as_posix(),
        sync_tensorboard=False,  # auto-upload tensorboard metrics
        monitor_gym=True,  # auto-upload videos
        save_code=True,  # save code for reproducibility
    )

    monitored_info = dict(
            info_keywords=(
                "out_of_bounds",
                "collision",
                "env_complete",
                "num_targets_reached",
                "num_obstacles",
                )
            )
    
    # train environment configuration
    env_kwargs_train = {
        "num_targets": num_targets,
        "grid_sizes": grid_sizes,
        "voxel_size": voxel_size,
        "min_height": min_height,
        "sparse_reward": sparse_reward,
        "use_yaw_targets": use_yaw_targets,
        "goal_reach_distance": goal_reach_distance,
        "goal_reach_angle": goal_reach_angle,
        "max_duration_seconds": max_duration_seconds,
        "visual_obstacles": False,
        "num_obstacles": num_obstacles,
        "obstacle_strategy": obstacle_strategy,
        "flight_mode": flight_mode,
        "angle_representation": angle_representation,
        "agent_hz": agent_hz,
        "render_mode": render_mode,
        "render_resolution": render_resolution,
        "perception_mode": perception_mode,
        "num_rays_horizontal": num_rays_horizontal,
        "num_rays_vertical": num_rays_vertical,
        "max_range": max_range,
        "min_range": min_range,
        "fov_horizontal": fov_horizontal,
        "fov_vertical": fov_vertical,
        "ray_start_offset": ray_start_offset,
        "normalize_distances": normalize_distances,
        "add_to_obs": add_to_obs,
    }

    # create training environment
    vec_env = make_vec_env(
        env_helpers.make_flat_voyager,
        n_envs=n_envs,
        env_kwargs=env_kwargs_train,
        monitor_kwargs=monitored_info,
        )

    # separate eval environment configuration
    env_kwargs_val = {
        "num_targets": num_targets,
        "grid_sizes": grid_sizes,
        "voxel_size": voxel_size,
        "min_height": min_height,
        "sparse_reward": sparse_reward,
        "use_yaw_targets": use_yaw_targets,
        "goal_reach_distance": goal_reach_distance,
        "goal_reach_angle": goal_reach_angle,
        "max_duration_seconds": max_duration_seconds,
        "visual_obstacles": visual_obstacles,
        "num_obstacles": num_obstacles,
        "obstacle_strategy": obstacle_strategy,
        "flight_mode": flight_mode,
        "angle_representation": angle_representation,
        "agent_hz": agent_hz,
        "render_mode": render_mode,
        "render_resolution": render_resolution,
        "perception_mode": perception_mode,
        "num_rays_horizontal": num_rays_horizontal,
        "num_rays_vertical": num_rays_vertical,
        "max_range": max_range,
        "min_range": min_range,
        "fov_horizontal": fov_horizontal,
        "fov_vertical": fov_vertical,
        "ray_start_offset": ray_start_offset,
        "normalize_distances": normalize_distances,
        "add_to_obs": add_to_obs,
    }

    # separete evaluation environment
    vec_env_eval = make_vec_env(
        env_helpers.make_flat_voyager,
        n_envs=1,  # single env for evaluation - simple and clean
        env_kwargs=env_kwargs_val,
        monitor_kwargs=monitored_info,
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
        verbose=2,
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
        verbose=2,
        tensorboard_log=f"{wandb_run.dir}/tensorboard",
        seed=config.RANDOM_SEED,  # the seed is passed through the chain: (PPO → Gymnasium → QuadXBaseEnv → Aviary → VectorVoyagerEnv → VoxelGrid)
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
                grid_sizes=(10.0, 10.0, 5.0),
                n_steps=config.n_steps,
                exp_analysis=False,  # W&B handles plotting
            )

    # Run sweep
    wandb.agent(sweep_id, train_sweep, count=10)


if __name__ == "__main__":
    app()

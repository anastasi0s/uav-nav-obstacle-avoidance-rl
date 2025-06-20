from pathlib import Path
from typing import Any, Callable, Optional, Union
import warnings
import json

import gymnasium as gym
from loguru import logger
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common import type_aliases
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
import typer

from uav_rl_navigation import config
from uav_rl_navigation.utils.env_helpers import make_flat_voyager
from uav_rl_navigation.environment.metrics_wrapper import UAVMetricsWrapper

app = typer.Typer()


# Enhanced evaluation function that uses the metrics wrapper
def evaluate_with_metrics(
    model,
    env_factory,
    n_eval_episodes: int = 100,
    deterministic: bool = True,
    render: bool = False,
    print_summary: bool = True
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """
    evaluate policy with custom UAV metrics collection.
    
    Args:
        model: Trained RL model
        env_factory: Function that creates the environment
        n_eval_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        render: Whether to render during evaluation
        print_summary: Whether to print metrics summary
    
    Returns:
        Tuple of (summary_metrics, episode_metrics)
    """
    # Create environment with metrics wrapper
    env = env_factory()
    env = UAVMetricsWrapper(env)
    
    episode_rewards = []
    
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Completed {episode + 1}/{n_eval_episodes} episodes")
    
    # Get metrics
    summary_metrics = env.get_metrics_summary()
    episode_metrics = env.get_episode_metrics()
    
    # Add reward statistics to summary
    summary_metrics['avg_episode_reward'] = np.mean(episode_rewards)
    summary_metrics['std_episode_reward'] = np.std(episode_rewards)
    
    if print_summary:
        env.print_metrics_summary()
        print(f"Average Episode Reward: {summary_metrics['avg_episode_reward']:.3f} ± {summary_metrics['std_episode_reward']:.3f}")
    
    env.close()
    
    return summary_metrics, episode_metrics


@app.command()
def evaluate_with_custom_metrics(
    model_path: Path = config.MODELS_DIR / "model.zip",
    n_episodes: int = 100,
    deterministic: bool = True,
    render: bool = False,
    save_results: bool = True,
    results_dir: Path = config.REPORTS_DIR / "evaluation_results",
):
    """
    evaluate model with custom UAV performance metrics.
    
    Args:
        model_path: Path to the trained model
        n_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        render: Whether to render during evaluation
        save_results: Whether to save results to files
        results_dir: Directory to save results
    """
    logger.info(f"Evaluating agent with custom metrics for {n_episodes} episodes...")
    
    # Create results directory if saving
    if save_results:
        results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trained model (then create env inside evaluate_with_metrics)
    temp_env = make_flat_voyager(num_waypoints=1)
    model = PPO.load(str(model_path), env=temp_env)
    temp_env.close()
    
    # Run evaluation with metrics
    summary_metrics, episode_metrics = evaluate_with_metrics(
        model=model,
        env_factory=lambda: make_flat_voyager(num_waypoints=1),
        n_eval_episodes=n_episodes,
        deterministic=deterministic,
        render=render,
        print_summary=True
    )
    
    # Log key metrics
    logger.info("Key Performance Metrics:")
    logger.info(f"  Success Rate: {summary_metrics['success_rate']:.3f}")
    logger.info(f"  Average Path Length: {summary_metrics['avg_path_length']:.3f} m")
    logger.info(f"  Average Velocity: {summary_metrics['avg_velocity']:.3f} m/s")
    logger.info(f"  Path Efficiency Ratio: {summary_metrics['avg_path_efficiency_ratio']:.3f}")
    logger.info(f"  Collision Rate: {summary_metrics['collision_rate']:.3f}")
    
    if save_results:
        # Save summary metrics as JSON
        summary_file = results_dir / f"summary_metrics_{model_path.stem}.json"
        with open(summary_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in summary_metrics.items()}
            json.dump(json_metrics, f, indent=2)
        
        # Save detailed episode metrics as CSV
        episodes_file = results_dir / f"episode_metrics_{model_path.stem}.csv"
        df = pd.DataFrame(episode_metrics)
        df.to_csv(episodes_file, index=False)
        
        logger.info(f"Results saved to {results_dir}")
        logger.info(f"  Summary: {summary_file}")
        logger.info(f"  Episodes: {episodes_file}")
    
    logger.success("Evaluation complete.")
    return summary_metrics, episode_metrics


@app.command()
def compare_models(
    model_paths: list[Path],
    n_episodes: int = 50,
    deterministic: bool = True,
    save_comparison: bool = True,
    results_dir: Path = config.REPORTS_DIR / "evaluation_results",
):
    """
    compare multiple models using custom metrics.
    
    Args:
        model_paths: List of paths to trained models
        n_episodes: Number of episodes to evaluate each model
        deterministic: Whether to use deterministic actions
        save_comparison: Whether to save comparison results
        results_dir: Directory to save results
    """
    logger.info(f"Comparing {len(model_paths)} models...")
    
    if save_comparison:
        results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model_path in model_paths:
        logger.info(f"Evaluating {model_path.name}...")
        
        # load and evaluate model
        temp_env = make_flat_voyager(num_waypoints=1)
        model = PPO.load(str(model_path), env=temp_env)
        temp_env.close()
        
        summary_metrics, _ = evaluate_with_metrics(
            model=model,
            env_factory=lambda: make_flat_voyager(num_waypoints=1),
            n_eval_episodes=n_episodes,
            deterministic=deterministic,
            render=False,
            print_summary=False
        )
        
        all_results[model_path.stem] = summary_metrics
        
        logger.info(f"  Success Rate: {summary_metrics['success_rate']:.3f}")
        logger.info(f"  Path Efficiency: {summary_metrics['avg_path_efficiency_ratio']:.3f}")
    
    # create comparison DataFrame
    comparison_df = pd.DataFrame(all_results).T
    
    # sort by success rate (descending)
    comparison_df = comparison_df.sort_values('success_rate', ascending=False)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(comparison_df[['success_rate', 'avg_path_length', 'avg_velocity', 
                        'avg_path_efficiency_ratio', 'collision_rate']].round(3))
    print("="*80)
    
    if save_comparison:
        comparison_file = results_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_file)
        logger.info(f"Comparison saved to {comparison_file}")
    
    logger.success("Model comparison complete.")
    return comparison_df


@app.command()
def main(
    model_name: str = 'model',
    n_episodes: int = 100,
    compare_mode: bool = False,
):
    """
    main evaluation. choose between single model evaluation or comparison mode.
    """
    model_path = config.MODELS_DIR / model_name
    
    if compare_mode:
        # Look for all model files in the models directory
        model_files = list(config.MODELS_DIR.glob("*.zip"))
        if len(model_files) > 1:
            compare_models(model_files, n_episodes=n_episodes)
        else:
            logger.warning("less than 2 models found for comparison -> running single evaluation.")
            evaluate_with_custom_metrics(model_path, n_episodes=n_episodes)
    else:
        evaluate_with_custom_metrics(model_path, n_episodes=n_episodes)


if __name__ == "__main__":
    app()
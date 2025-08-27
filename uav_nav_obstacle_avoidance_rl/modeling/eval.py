from gc import callbacks
from pathlib import Path
import json

import numpy as np
import pandas as pd
from typing import Union
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
import typer

from uav_nav_obstacle_avoidance_rl import config
from uav_nav_obstacle_avoidance_rl.config import logger
from uav_nav_obstacle_avoidance_rl.utils import env_helpers, custom_callbacks

app = typer.Typer()


@app.command()
def evaluate_agent(
    model_path: Union[Path, str],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    save_results: bool = True,
    results_dir: Path = config.REPORTS_DIR / "evaluation_results",
):
    """
    Args:
        model_path: Path to the trained model
        n_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        render: Whether to render during evaluation
        save_results: Whether to save results to files
        results_dir: Directory to save results
    """
    logger.info(f"Evaluating agent with custom metrics for {n_eval_episodes} episodes...")
    
    # load trained model
    model = PPO.load(str(model_path))

    # init vec_env
    vec_env = make_vec_env(  # -> vec env
        env_helpers.make_flat_voyager,
        env_kwargs=dict(num_waypoints=1), 
        monitor_kwargs=dict(info_keywords=("out_of_bounds", "collision", "env_complete", "num_targets_reached")),
    )

    # create callbacks
    callbacks = [
        # collect performance metrics
        custom_callbacks.UAVMetricsCallback(
            log_dir=results_dir.as_posix(),
            save_freq=1000,
            verbose=1,
            ),
        # log hyperparameters
        custom_callbacks.HParamCallback(),
    ]

    # evaluate models
    evaluate_policy(model, vec_env, n_eval_episodes, render=render, callback=callbacks)

    temp_env.close()
    
    
    if save_results:
        results_dir.mkdir(parents=True, exist_ok=True)

        # save summary metrics as JSON
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
    return summary_metrics, episode_metric

@app.command()
def compare_agents(
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
    model_dir: Union[str, Path],
    n_eval_episodes: int = 20,
):  
    model_dir = Path(model_dir)

    # Look for all model files in directory
    model_files = list(model_dir.glob("*.zip"))
    if len(model_files) > 1:
        compare_agents(model_files, n_episodes=n_eval_episodes)
    else:
        logger.warning("less than 2 models found for comparison -> running single evaluation.")
        evaluate_agent(model_dir, n_eval_episodes=n_eval_episodes)


if __name__ == "__main__":
    app()
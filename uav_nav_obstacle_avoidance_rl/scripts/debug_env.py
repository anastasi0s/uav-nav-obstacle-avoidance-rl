from uav_nav_obstacle_avoidance_rl import config
from loguru import logger
import numpy as np
import typer

app = typer.Typer()

@app.command()
def debug_environment_access():
    """quick debug function to test environment state access. check if the Observations are aligned with the state values (velocities and position) from the base env"""
    from uav_nav_obstacle_avoidance_rl.utils import env_helpers
    from stable_baselines3.common.env_util import make_vec_env
    
    # create env
    vec_env = make_vec_env(
        env_helpers.make_flat_voyager, 
        n_envs=1, 
        env_kwargs=dict(num_waypoints=1, with_metrics=False)
    )
    
    # reset env
    obs = vec_env.reset()
    logger.info(f"Vec Env: {type(vec_env)}")
    logger.info(f"Observation shape: {obs.shape}. [0] rotation vel, [1] rotation position, [2:5] linear vel (u,v,w), [5] z position, [6:10] previous action, [11:13] target deltas")
    logger.info(f"Observation sample: {obs[0][:]}")  # First 10 elements (attitude)
    
    # access underlying wrapper (only vec_envs have 'envs' attribute)
    if hasattr(vec_env, 'envs'):
        env = vec_env.envs[0]
        logger.info(f"First wrapper: {type(env)}")
        
        # peal off wrappers
        current = env
        while hasattr(current, 'env'):
            current = current.env
            logger.info(f"Next wrapper: {type(current)}")
            
            # check if this is VectorVoyagerEnv
            if hasattr(current, 'waypoints') and hasattr(current, 'compute_attitude'):
                logger.info("Found VectorVoyagerEnv!")
                
                # check if the aviary state is accessible
                if hasattr(current, 'env') and hasattr(current.env, 'state'):
                    state = current.env.state(0)
                    logger.info(f"Aviary state shape: {state.shape}")
                    logger.info(f"Velocity, state[2] (lin_vel): {state[2]}")
                    logger.info(f"Position, state[3] (lin_pos): {state[3]}")
                    
                    # check waypoints
                    if hasattr(current.waypoints, 'targets'):
                        logger.info(f"Number of targets: {len(current.waypoints.targets)}")
                        if len(current.waypoints.targets) > 0:
                            logger.info(f"First target: {current.waypoints.targets[0]}")
                break
    
    # take a few steps
    for i in range(5):
        # vec envs need a batch of actions
        action = np.array([vec_env.action_space.sample()])  # Shape: (1, 4) for n_envs=1
        obs, reward, done, info = vec_env.step(action)
        logger.info(f"Action {i} shape: {action.shape}, action: {action}\n")
        logger.info(f"Step {i}: reward={reward}, done={done}, info keys={list(info[0].keys())} info values={list(info[0].values())}\n")
        logger.info(f"Observation {i}: {obs[0][:]}")  # First 10 elements (attitude)

        # peal off wrappers to get values from base env
        if hasattr(vec_env, 'envs'):
            env = vec_env.envs[0]
        current = env
        while hasattr(current, 'env'):
            current = current.env
            # check if this is VectorVoyagerEnv
            if hasattr(current, 'waypoints') and hasattr(current, 'compute_attitude'):
                # check if the aviary state is accessible
                if hasattr(current, 'env') and hasattr(current.env, 'state'):
                    state = current.env.state(0)
                    logger.info(f"Velocity, state[2] (lin_vel): {state[2]}")
                    logger.info(f"Position, state[3] (lin_pos): {state[3]}\n\n")
                break

    vec_env.close()

if __name__ == "__main__":
    app()
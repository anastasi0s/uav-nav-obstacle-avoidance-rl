"""
record evaluation videos of a trained model.
uses SB3 VecVideoRecorder + ThirdPersonCamWrapper.

Usage:
    uv run python -m uav_nav_obstacle_avoidance_rl.scripts.record_videos record \
        --model-path /path/to/best_model.zip \
        --num-episodes 2

    # with a specific curriculum stage
    uv run python -m uav_nav_obstacle_avoidance_rl.scripts.record_video record \
        best_model_success_rate.zip \
        --num-episodes 2 \
        --curriculum-stage 3

    # record a showcase video for each curriculum stage (3rd-person view of full env)
    uv run python -m uav_nav_obstacle_avoidance_rl.scripts.record_video record-stages \
        best_model_success_rate.zip \
        --stages 0 1 3 5

"""
from pathlib import Path
from typing import List, Optional

import pybullet as p
import typer
import yaml
from stable_baselines3 import PPO

from uav_nav_obstacle_avoidance_rl import config
from uav_nav_obstacle_avoidance_rl.utils import env_factory

logger = config.logger
app = typer.Typer()


def load_config(path: Path = config.EXP_CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_curriculum_stage(env_config: dict, stages: list[dict], stage_idx: int) -> dict:
    """Override env_config with the specified curriculum stage parameters."""
    if stage_idx < 0 or stage_idx >= len(stages):
        raise ValueError(f"Stage {stage_idx} not found (available: 0-{len(stages) - 1})")

    stage = stages[stage_idx]
    logger.info(f"Applying curriculum stage {stage_idx}: {stage.get('desc', '')}")

    # only override keys that are not None in the stage definition
    for key in ["num_obstacles_range", "obstacle_shapes", "obstacle_size_range",
                "num_targets", "target_distance_range"]:
        if key in stage and stage[key] is not None:
            env_config[key] = stage[key]

    return env_config


@app.command()
def record(
    model_path: Path = typer.Argument(..., help="Path to trained model .zip file"),
    num_episodes: int = typer.Option(5, help="Number of episodes to record"),
    video_length: int = typer.Option(2500, help="Max steps per video (safety cutoff)"),
    output_dir: Path = typer.Option(None, help="Output directory (default: next to model)"),
    curriculum_stage: int = typer.Option(-1, help="Curriculum stage to use (-1 = use config defaults)"),
    config_path: Path = typer.Option(config.EXP_CONFIG_PATH, help="Path to config YAML"),
):
    """Record evaluation videos of a trained PPO model."""

    # ── resolve output directory ──
    if output_dir is None:
        output_dir = model_path.parent.parent / "media" / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Videos will be saved to: {output_dir}")

    # ── load config ──
    full_config = load_config(config_path)
    env_config = dict(full_config["env"])

    # apply curriculum stage if requested
    if curriculum_stage >= 0:
        curriculum_stages = full_config["curriculum"]["stages"]
        env_config = apply_curriculum_stage(env_config, curriculum_stages, curriculum_stage)

    # override for recording: visual obstacles on, rgb_array mode
    env_config["visual_obstacles"] = True
    env_config["render_mode"] = "rgb_array"

    # ── load model ──
    logger.info(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    # ── record episodes ──
    for episode in range(num_episodes):
        logger.info(f"Recording episode {episode + 1}/{num_episodes}")

        video_env = env_factory.make_voyager_for_recording(
            video_folder=str(output_dir),
            video_length=video_length,
            video_name=f"ep-{episode}",
            **env_config,
        )

        obs = video_env.reset()
        done = False
        step_count = 0

        while not done and step_count < video_length:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = video_env.step(action)
            done = dones[0] if hasattr(dones, '__len__') else dones
            step_count += 1

        video_env.close()

        # extract episode outcome from last info
        info = infos[0] if isinstance(infos, list) else infos
        collision = info.get("collision", False)
        env_complete = info.get("env_complete", False)
        outcome = "SUCCESS" if (env_complete and not collision) else "COLLISION" if collision else "TIMEOUT"
        logger.info(f"  Episode {episode + 1}: {outcome} in {step_count} steps")

    logger.info(f"Done — {num_episodes} videos saved to {output_dir}")


@app.command()
def record_pybullet(
    model_path: Path = typer.Argument(..., help="Path to trained model .zip file"),
    num_episodes: int = typer.Option(1, help="Number of episodes to record"),
    max_steps: int = typer.Option(2500, help="Max steps per episode (safety cutoff)"),
    output_dir: Path = typer.Option(None, help="Output directory (default: next to model)"),
    curriculum_stage: int = typer.Option(-1, help="Curriculum stage to use (-1 = use config defaults)"),
    config_path: Path = typer.Option(config.EXP_CONFIG_PATH, help="Path to config YAML"),
):
    """Record MP4 using PyBullet's built-in STATE_LOGGING_VIDEO_MP4 (requires GUI mode)."""

    if output_dir is None:
        output_dir = model_path.parent.parent / "media" / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    full_config = load_config(config_path)
    env_config = dict(full_config["env"])

    if curriculum_stage >= 0:
        curriculum_stages = full_config["curriculum"]["stages"]
        env_config = apply_curriculum_stage(env_config, curriculum_stages, curriculum_stage)

    # PyBullet MP4 logging requires GUI rendering
    env_config["visual_obstacles"] = True
    env_config["render_mode"] = "human"

    logger.info(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    # create a single unwrapped env (no VecEnv / VecVideoRecorder needed)
    env = env_factory.make_flat_voyager(**env_config)

    for episode in range(num_episodes):
        obs, _ = env.reset()

        # access PyBullet-backed Aviary (extends BulletClient) from the base env
        aviary = env.unwrapped.env  # type: ignore[attr-defined]

        video_path = str(output_dir / f"pybullet_ep-{episode}.mp4")
        logger.info(f"Recording episode {episode + 1}/{num_episodes} -> {video_path}")
        log_id = aviary.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)

        done = False
        step_count = 0
        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

        aviary.stopStateLogging(log_id)

        collision = info.get("collision", False)
        env_complete = info.get("env_complete", False)
        outcome = "SUCCESS" if (env_complete and not collision) else "COLLISION" if collision else "TIMEOUT"
        logger.info(f"  Episode {episode + 1}: {outcome} in {step_count} steps -> {video_path}")

    env.close()
    logger.info(f"Done — {num_episodes} videos saved to {output_dir}")


def _get_base_env(env):
    """Peel off wrappers to get the VectorVoyagerEnv instance."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def _create_visible_walls(aviary, og, z_size):
    """Create semi-transparent visual boundary walls so they show in the recording."""
    half_thickness = 0.05
    z_half = z_size / 2
    wall_color = [0.4, 0.6, 0.9, 0.3]  # light blue, semi-transparent

    walls = [
        ([og.x_min - half_thickness, 0.0, z_half], [half_thickness, og.y_size / 2, z_half]),
        ([og.x_max + half_thickness, 0.0, z_half], [half_thickness, og.y_size / 2, z_half]),
        ([0.0, og.y_min - half_thickness, z_half], [og.x_size / 2, half_thickness, z_half]),
        ([0.0, og.y_max + half_thickness, z_half], [og.x_size / 2, half_thickness, z_half]),
        ([0.0, 0.0, z_size + half_thickness], [og.x_size / 2, og.y_size / 2, half_thickness]),
    ]
    for pos, he in walls:
        vis_id = aviary.createVisualShape(
            shapeType=p.GEOM_BOX, halfExtents=he, rgbaColor=wall_color,
        )
        aviary.createMultiBody(baseMass=0.0, baseVisualShapeIndex=vis_id, basePosition=pos)


def _setup_camera(aviary, base_env, distance=8.0, yaw=45, pitch=-35):
    """Point the debug visualizer camera at the drone start position."""
    start = base_env.start_pos[0]
    aviary.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=[start[0], start[1], start[2] * 0.5],
    )
    aviary.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)


@app.command()
def record_stages(
    model_path: Path = typer.Argument(..., help="Path to trained model .zip file"),
    stages: Optional[List[int]] = typer.Option(None, help="Stage indices to record, repeat flag per value e.g. --stages 0 --stages 1 (default: all)"),
    max_steps: int = typer.Option(2500, help="Max steps per episode"),
    output_dir: Path = typer.Option(None, help="Output directory (default: next to model)"),
    config_path: Path = typer.Option(config.EXP_CONFIG_PATH, help="Path to config YAML"),
    camera_distance: float = typer.Option(8.0, help="Camera distance from target"),
    camera_yaw: float = typer.Option(45.0, help="Camera yaw angle"),
    camera_pitch: float = typer.Option(-35.0, help="Camera pitch angle"),
    trail: bool = typer.Option(True, help="Draw trajectory trail behind drone"),
):
    """Record one MP4 per curriculum stage showing the full 3D environment (bird's-eye view)."""

    if output_dir is None:
        output_dir = model_path.parent.parent / "media" / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    full_config = load_config(config_path)
    env_config = dict(full_config["env"])
    curriculum_stages = full_config["curriculum"]["stages"]

    # default: record all stages
    stage_indices = stages if stages else list(range(len(curriculum_stages)))

    logger.info(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    for stage_idx in stage_indices:
        if stage_idx < 0 or stage_idx >= len(curriculum_stages):
            logger.warning(f"Skipping invalid stage {stage_idx}")
            continue

        stage_cfg = curriculum_stages[stage_idx]
        desc = stage_cfg.get("desc", "")
        logger.info(f"── Stage {stage_idx}: {desc} ──")

        # build env config for this stage
        stage_env_config = dict(env_config)
        stage_env_config["visual_obstacles"] = True
        stage_env_config["render_mode"] = "human"

        # create a fresh flat env
        env = env_factory.make_flat_voyager(**stage_env_config)
        base_env = _get_base_env(env)

        # apply curriculum stage parameters via set_stage()
        base_env.set_stage(stage_cfg)
        base_env.visual_obstacles = True  # ensure obstacles are visible

        obs, _ = env.reset()
        aviary = base_env.env  # PyBullet / Aviary client

        # add visible boundary walls for stages that have them
        walls_enabled = stage_cfg.get("walls_enabled", False)
        if walls_enabled:
            _create_visible_walls(aviary, base_env.occupancy_grid, base_env.z_size)

        # position camera to see the whole environment
        _setup_camera(aviary, base_env, camera_distance, camera_yaw, camera_pitch)

        # start recording
        video_path = str(output_dir / f"stage_{stage_idx}.mp4")
        aviary.setRealTimeSimulation(0)
        log_id = aviary.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)

        prev_pos = None
        done = False
        step_count = 0
        info = {}

        try:
            while not done and step_count < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1

                # draw trajectory trail
                if trail:
                    try:
                        pos_orn = aviary.getBasePositionAndOrientation(aviary.drones[0].Id)
                        if pos_orn is not None:
                            cur_pos = pos_orn[0]
                            if prev_pos is not None:
                                aviary.addUserDebugLine(
                                    prev_pos, cur_pos,
                                    lineColorRGB=[0, 1, 0],
                                    lineWidth=2,
                                    lifeTime=0,
                                )
                            prev_pos = cur_pos
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"  Stage {stage_idx}: simulation error at step {step_count}: {e}")

        aviary.stopStateLogging(log_id)

        collision = info.get("collision", False) if info else False
        env_complete = info.get("env_complete", False) if info else False
        outcome = "SUCCESS" if (env_complete and not collision) else "COLLISION" if collision else "TIMEOUT"
        logger.info(f"  Stage {stage_idx}: {outcome} in {step_count} steps -> {video_path}")

        env.close()

    logger.info(f"Done — videos saved to {output_dir}")


if __name__ == "__main__":
    app()

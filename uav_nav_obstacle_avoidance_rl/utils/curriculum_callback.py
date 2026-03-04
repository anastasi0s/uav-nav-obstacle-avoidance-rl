from collections import deque
from typing import List

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

from uav_nav_obstacle_avoidance_rl import config

logger = config.logger


class CurriculumCallback(BaseCallback):
    """
    Args:
        stages:             list of stage config dicts from YAML
        advance_threshold:  success rate above which the agent may advance
        regress_threshold:  success rate below which the agent may regress
        advance_patience:   num or rollout-end evals above threshold before advancing
        regress_patience:   num of rollout-end evals below threshold before regressing
        window_size:        num of recent episodes used for rolling success rate
    """
    def __init__(
        self,
        stages: List[dict],
        advance_threshold: float,
        regress_threshold: float,
        advance_patience: int,
        regress_patience: int,
        window_size: int,
        verbose: int,
        eval_env=None,
    ) -> None:
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        
        self.eval_env = eval_env  # sync difficult with eval env

        # stage definitions
        self.stages = stages
        self.num_stages = len(stages)
        self.current_stage_idx = 0

        # threshold and patients
        self.advance_threshold = advance_threshold
        self.regress_threshold = regress_threshold
        self.advance_patience = advance_patience
        self.regress_patience = regress_patience
        
        # rolling success window
        self.window_size = window_size
        self.success_window: deque = deque(maxlen=window_size)  # deque automatically removes old entries/episodes from the window

        # patience counters for rollout-end evaluations that meet the threshold. reset to 0 when the condition is not met
        self._advance_streak = 0
        self._regress_streak = 0
    
    def _on_step(self):
        """collect episode outcomes"""
        dones = self.locals['dones']
        infos = self.locals['infos']
        
        for done, info in zip(dones, infos):
            if done:
                # check for success: all targets reached, without collisions
                collision = info["collision"]
                env_complete = info["env_complete"]
                success = env_complete and not collision

                self.success_window.append(int(success))

        return True
    
    def _on_training_start(self):
        """apply initial stage to all sub envs"""
        self._transition_stage(new_idx=self.current_stage_idx, force_reset=True)
        logger.info(f"[Curriculum] Training starts at stage {self.stages[self.current_stage_idx]}")

    def _on_rollout_end(self):
        """evaluate whether to advance or regress th curriculum stage. on rollout-end because the rollout buffer is complete"""
        # collect enough episodes to make decisions
        if len(self.success_window) < self.window_size:
            return
        success_rate = np.mean(self.success_window)

        # log wandb
        self._log_metrics(success_rate)

        # try advance
        if success_rate >= self.advance_threshold:
            self._advance_streak += 1
            self._regress_streak = 0
            if self._advance_streak >= self.advance_patience:
                # advance if patience is met
                if self.current_stage_idx >= self.num_stages - 1:
                    # reached max stage
                    logger.debug("[Curriculum] Already at final stage — cannot advance.")
                    return
                old_idx = self.current_stage_idx
                self.current_stage_idx += 1
                self._transition_stage(old_idx=old_idx, new_idx=self.current_stage_idx, force_reset=True)
        # try regress
        elif success_rate <= self.regress_threshold:
            self._regress_streak += 1
            self._advance_streak = 0
            if self._regress_streak >= self.regress_patience:
                # regress if patience is met
                if self.current_stage_idx <= 0:
                    # reached min stage
                    logger.debug("[Curriculum] Already at lowest stage — cannot regress.")
                    return
                old_idx = self.current_stage_idx
                self.current_stage_idx -= 1
                self._transition_stage(old_idx=old_idx, new_idx=self.current_stage_idx, force_reset=True)
        # in between thresholds - reset both
        else:
            self._advance_streak = 0
            self._regress_streak = 0
    
    def _transition_stage(self, new_idx: int, old_idx: int=0, force_reset: bool=False):
        """ push new stage to all sub envs via VecEnv"""
        self._advance_streak = 0  # prevent immediate retriggering of stage change
        self._regress_streak = 0
        self.success_window.clear()  # episodes from old difficulty are meaningless for evaluations in new stage
        
        # apply stage to all train envs
        stage = self.stages[new_idx]
        self.training_env.env_method('set_stage', stage)

        # sync eval env to same difficulty
        if self.eval_env is not None:
            self.eval_env.env_method('set_stage', stage)

        # reset envs
        if force_reset:
            # triggers reset on all sub envs, ensuring all new episodes start under new difficulty
            self.training_env.env_method('reset')

        logger.info(f"[Curriculum] Stage Transition: {old_idx} → {new_idx}")

        # log wandb
        wandb.log({"curriculum/stage": new_idx}, step=self.num_timesteps)

    def _log_metrics(self, success_rate):
        """Log curriculum metrics to W&B"""
        wandb.log(
            {
                "curriculum/stage": self.current_stage_idx,
                "curriculum/success_rate": success_rate,
                "curriculum/advance_streak": self._advance_streak,
                "curriculum/regress_streak": self._regress_streak,
            },
            step=self.num_timesteps,
        )
        logger.debug(
            f"[Curriculum] stage={self.current_stage_idx} success_rate={success_rate:.3f} "
            f"advance_streak={self._advance_streak} regress_streak={self._regress_streak}"
        )
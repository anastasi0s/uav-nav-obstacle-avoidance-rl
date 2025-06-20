import gymnasium as gym
import numpy as np


# wrapper to adjust the uav camera position when recording a video
class ThirdPersonCamWrapper(gym.Wrapper):
    def __init__(self, env, offset=(-1.3, 0, .5)):
        super().__init__(env)
        self.offset = np.asarray(offset, dtype=np.float32)

    def _apply_patch(self):
        base = self.env
        while hasattr(base, "env"):             # peel off wrappers
            base = base.env
        
        # reset cam settings in the base env
        cam = base.drones[0].camera
        cam.has_camera_offset = True
        cam.camera_position_offset = self.offset
        cam.camera_angle_degrees   = -15        # or negative if gimballed
        cam.is_tracking_camera = False          # use either tracking_camera OR gimbal
        cam.use_gimbal= True
        cam.cinematic = False

    def reset(self, **kw):
        out = self.env.reset(**kw)
        self._apply_patch()                     # patch immediately after every reset
        return out
    
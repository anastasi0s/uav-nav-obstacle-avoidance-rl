from typing import Tuple

import numpy as np

from uav_nav_obstacle_avoidance_rl import config

logger = config.logger


class VoxelGrid:
    """3d voxel grid for discretizing the space and placing obstacles"""

    def __init__(self, grid_sizes: Tuple[float, float, float], voxel_size: float, rng):
        """
        initialize voxel grid

        args:
            grid_sizes: the size of each grid dimension in meter (x_size, y_size, z_size)
            voxel_size: size of each voxel cube in meter
        """
        self.x_size, self.y_size, self.z_size = grid_sizes  # used also by object instances
        self.rng = rng

        # calculate min, max of each dimension
        x_magnitude, y_magnitude = (self.x_size / 2, self.y_size / 2)  # calculate magnitude from origin of x and y dimensions
        
        self.x_min, self.x_max = -x_magnitude, x_magnitude
        self.y_min, self.y_max = -y_magnitude, y_magnitude
        self.z_min, self.z_max = 0.0, self.z_size  # z dim (hight) can only be positive
        self.voxel_size = voxel_size

        # claculate grid dimensions
        self.nx = int(np.floor(self.x_size / self.voxel_size))  # floor division -> voxel grid might be slightly smaller than space
        self.ny = int(np.floor(self.y_size / self.voxel_size))
        self.nz = int(np.floor(self.z_size / self.voxel_size))

        # initialize occupancy grid
        self.grid = np.zeros((self.nx, self.ny, self.nz), dtype=bool)

    def world_to_voxel(self, position: np.ndarray) -> Tuple[int, int, int]:
        """convert 3d world coordinates into voxel indices"""
        x, y, z = position
        i = int(np.floor((x - self.x_min) / self.voxel_size))
        j = int(np.floor((y - self.y_min) / self.voxel_size))
        k = int(np.floor((z - self.z_min) / self.voxel_size))
        return i, j, k  # voxel indices

    def voxel_to_world(self, voxel_idx: Tuple[int, int, int]) -> np.ndarray:
        """convert voxel idx to cartesian world position coordinates (+0.5: conter of voxel)"""
        i, j, k = voxel_idx
        x = self.x_min + (i + 0.5) * self.voxel_size
        y = self.y_min + (j + 0.5) * self.voxel_size
        z = self.z_min + (k + 0.5) * self.voxel_size
        return np.array([x, y, z])  # cartesian position

    def is_voxel_free(self, voxel_idx: Tuple[int, int, int]) -> bool:
        """check if a voxel is free (not occupied)"""
        i, j, k = voxel_idx
        # check if voxel_idx is inside bounds
        if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
            free = not self.grid[i, j, k]
        else:
            # checking voxel_idx outside bounds
            free = False
        return free

    def mark_voxel_occupied(self, voxel_idx: Tuple[int, int, int]):
        i, j, k = voxel_idx
        if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
            self.grid[i, j, k] = True

    def mark_voxel_free(self, voxel_idx: Tuple[int, int, int]):
        i, j, k = voxel_idx
        if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
            self.grid[i, j, k] = False

    def get_random_free_voxel(self) -> Tuple[int, int, int]:
        """get free grid position"""
        free_voxels = np.argwhere(~self.grid)
        if len(free_voxels) == 0:
            logger.error("No free voxels available!")
            raise ValueError("No free voxels available!")
        idx = self.rng.choice(free_voxels)
        return tuple(idx)

    def get_random_free_position(self) -> np.ndarray:
        """ger free cartesian position"""
        voxel_idx = self.get_random_free_voxel()
        position_coordinates = self.voxel_to_world(voxel_idx)
        return position_coordinates
    
    def reset_occupancy(self):
        """reset all voxels to free (unoccupied) state"""
        self.grid.fill(False)


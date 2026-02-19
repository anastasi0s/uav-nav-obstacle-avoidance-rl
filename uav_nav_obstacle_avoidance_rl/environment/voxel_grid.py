from typing import List, Tuple

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
            rng: np random generator
        """
        self.x_size, self.y_size, self.z_size = (
            grid_sizes  # used also by object instances
        )
        self.rng = rng

        # calculate min, max of each dimension
        x_magnitude, y_magnitude = (
            self.x_size / 2,
            self.y_size / 2,
        )  # calculate magnitude from origin of x and y dimensions

        self.x_min, self.x_max = -x_magnitude, x_magnitude
        self.y_min, self.y_max = -y_magnitude, y_magnitude
        self.z_min, self.z_max = 0.0, self.z_size  # z dim (hight) can only be positive
        self.voxel_size = voxel_size

        # claculate grid dimensions
        self.nx = int(
            np.floor(self.x_size / self.voxel_size)
        )  # floor division -> voxel grid might be slightly smaller than space
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

    def are_voxels_free(self, voxel_indices: List[Tuple[int, int, int]]) -> List[bool]:
        """
        check if voxels are free (not occupied)

        args:
            voxel_indices: list of (i, j, k) tuples

        return:
            list of bools
        """
        result = []
        for i, j, k in voxel_indices:
            if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
                result.append(not self.grid[i, j, k])
            else:
                result.append(False)
        return result

    def get_occupancy(self):
        """
        Return:
            grid of current occupancy status of all voxels in 3d space: ndarray
        """
        return self.grid

    def get_random_free_voxel(self) -> Tuple[int, int, int]:
        """
        get free grid position

        args:
            voxel_idx: i, j, k
        """
        free_voxels = np.argwhere(~self.grid)  # invert occupancy grid -> get indices of free elements
        if len(free_voxels) == 0:
            logger.error("No free voxels available!")
            raise ValueError("No free voxels available!")
        idx = self.rng.choice(free_voxels)
        return tuple(idx)

    def get_random_free_position(self) -> np.ndarray:
        """ger random free cartesian position"""
        voxel_idx = self.get_random_free_voxel()
        position_coordinates = self.voxel_to_world(voxel_idx)
        return position_coordinates

    def mark_voxels(self, voxel_indices: List[Tuple[int, int, int]], occupied: bool):
        """
        mark voxels as occupied or free

        args:
            voxel_indices: list of (i, j, k) tuples
            occupied: bool
        """
        for i, j, k in voxel_indices:
            if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
                self.grid[i, j, k] = occupied

    def reset_occupancy(self):
        """reset all voxels to free (unoccupied) state"""
        self.grid.fill(False)


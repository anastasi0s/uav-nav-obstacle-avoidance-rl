from collections.abc import Sequence
from typing import List, Tuple

import numpy as np

from uav_nav_obstacle_avoidance_rl import config

logger = config.logger


class OccupancyGrid:
    """2d grid for discretizing the XY plane and placing obstacles"""

    def __init__(self, grid_sizes: Sequence[float], cell_size: float, rng):
        """
        initialize grid

        args:
            grid_sizes: the size of each grid dimension in meter (x_size, y_size)
            cell_size: size of each cell in meter
            rng: np random generator
        """
        self.x_size, self.y_size = grid_sizes
        self.rng = rng
        self.cell_size = cell_size

        # calculate min, max of each dimension
        x_magnitude, y_magnitude = (
            self.x_size / 2,
            self.y_size / 2,
        )

        self.x_min, self.x_max = -x_magnitude, x_magnitude
        self.y_min, self.y_max = -y_magnitude, y_magnitude

        # calculate grid dimensions
        self.nx = int(np.floor(self.x_size / self.cell_size))  # floor division -> grid might be slightly smaller than space
        self.ny = int(np.floor(self.y_size / self.cell_size))

        # initialize 2D occupancy grid
        self.grid = np.zeros((self.nx, self.ny), dtype=bool)

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """convert world XY coordinates into grid cell indices"""
        i = int(np.floor((x - self.x_min) / self.cell_size))
        j = int(np.floor((y - self.y_min) / self.cell_size))
        return i, j

    def cell_to_world(self, cell_idx: Tuple[int, int]) -> Tuple[float, float]:
        """convert cell index to world XY coordinates (center of cell)"""
        i, j = cell_idx
        x = self.x_min + (i + 0.5) * self.cell_size
        y = self.y_min + (j + 0.5) * self.cell_size
        return x, y

    def are_cells_free(self, cell_indices: List[Tuple[int, int]]) -> List[bool]:
        """
        check if cells are free (not occupied)

        args:
            cell_indices: list of (i, j) tuples

        return:
            list of bools
        """
        result = []
        for i, j in cell_indices:
            if 0 <= i < self.nx and 0 <= j < self.ny:
                result.append(not self.grid[i, j])
            else:
                result.append(False)
        return result

    def get_occupancy(self):
        """
        Return:
            grid of current occupancy status of all cells in 2D: ndarray
        """
        return self.grid

    def get_random_free_cell(self) -> Tuple[int, int]:
        """get a random free grid cell"""
        free_cells = np.argwhere(~self.grid)
        if len(free_cells) == 0:
            logger.error("No free cells available!")
            raise ValueError("No free cells available!")
        idx = self.rng.choice(len(free_cells))
        return tuple(free_cells[idx])

    def get_random_free_position(self) -> Tuple[float, float]:
        """get random free XY world position"""
        cell_idx = self.get_random_free_cell()
        return self.cell_to_world(cell_idx)

    def mark_cells(self, cell_indices: List[Tuple[int, int]], occupied: bool):
        """
        mark cells as occupied or free

        args:
            cell_indices: list of (i, j) tuples
            occupied: bool
        """
        for i, j in cell_indices:
            if 0 <= i < self.nx and 0 <= j < self.ny:
                self.grid[i, j] = occupied

    def reset_occupancy(self):
        """reset all cells to free (unoccupied) state"""
        self.grid.fill(False)
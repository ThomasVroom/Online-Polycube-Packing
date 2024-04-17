from src.heuristics.heuristic import Heuristic
from overrides import override
import numpy as np

class HeightMapMinimization(Heuristic):

    def __init__(self, axis=1):
        '''
        Initialize the HeightMapMinimization heuristic.

        Parameters
        ----------
            `axis` : int
                the axis along which to calculate the height map.
        '''
        self.axis = axis

    @override
    def get_score(self, matrix):
        # get the height map
        height_map = np.sum(matrix, axis=self.axis)

        # calculate the filled area of the height map
        filled_area = np.count_nonzero(height_map)

        # return the normalized percentage of filled area (inversed)
        return 1 - filled_area / height_map.size

from src.heuristics import Heuristic
from overrides import override
import numpy as np

class BLBF(Heuristic):
    '''
    Bottom-Left-Back-Fill (BLBF).
    '''
    
    @override
    def get_score(self, matrix):
        # get the center of mass of the matrix
        center_of_mass = np.mean(np.argwhere(matrix), axis=0)

        # return the normalized distance from the CoM to the bottom-left-back corner (inversed)
        return 1 - np.linalg.norm(center_of_mass) / np.linalg.norm(np.array(matrix.shape) - 1)

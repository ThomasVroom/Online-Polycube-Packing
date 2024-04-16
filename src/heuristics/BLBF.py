from src.heuristics.heuristic import Heuristic
import numpy as np

class BLBF(Heuristic):
    '''
    Bottom-Left-Back-Fill (BLBF).
    '''
    
    def get_score(self, matrix):
        # get the center of mass of the container
        center_of_mass = np.mean(np.argwhere(matrix), axis=0)

        # return the normalized distance from the CoM to the bottom-left-back corner
        return 1 - np.linalg.norm(center_of_mass) / np.linalg.norm(np.array(matrix.shape) - 1)

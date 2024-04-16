from src.heuristics.heuristic import Heuristic
import numpy as np

class HAPE(Heuristic):
    '''
    Heuristic Algorithm based on the principle of minimum total Potential Energy (HAPE).
    '''
    
    def get_score(self, matrix):
        # get the center of mass of the matrix
        center_of_mass = np.mean(np.argwhere(matrix), axis=0)
        
        # return the normalized distance from the (vertical) CoM to the top of the container
        return (matrix.shape[1] - center_of_mass[1]) / matrix.shape[1]

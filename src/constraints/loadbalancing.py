from src.constraints import Constraint
from overrides import override
import numpy as np

class LoadBalancing(Constraint):

    def __init__(self, margin: float=1.0):
        '''
        Create a constraint that checks if the load of the matrix is balanced.

        Parameters
        ----------
            `margin` : float, optional
                how far off the center the center of mass can be.
        '''
        self.margin = margin

    @override
    def apply(self, matrix):
        pass # not relevant

    @override
    def is_satisfied(self, matrix) -> bool:
        # get the dimensions of the matrix
        width, _, depth = matrix.shape

        # get the positions of the shapes in the matrix
        blocks = np.argwhere(matrix)

        # if there are no blocks, the constraint is satisfied
        if len(blocks) == 0:
            return True

        # get the center of mass
        center_of_mass = np.mean(blocks, axis=0)

        # check if the center of mass is within the margin
        return (center_of_mass[0] - width / 2.0)**2 + (center_of_mass[2] - depth / 2.0)**2 <= self.margin**2

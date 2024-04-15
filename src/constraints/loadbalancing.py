from src.constraints.constraint import Constraint
import numpy as np

class LoadBalancing(Constraint):

    def __init__(self, margin=1.0):
        '''
        Create a constraint that checks if the load of the container is balanced.

        Parameters
        ----------
            `margin` : float, optional
                how far off the center the center of mass can be.
        '''
        self.margin = margin

    def apply(self, container):
        pass # not relevant

    def is_satisfied(self, container):
        '''
        Check if the load of the container is balanced.
        
        Parameters
        ----------
            `container` : `Container`
                the container to check.
        
        Returns
        -------
            bool : whether the constraint is satisfied.
        '''
        
        # get the dimensions of the container
        width, _, depth = container.get_dimensions()

        # get the positions of the shapes in the container
        blocks = np.argwhere(container.matrix)

        # if there are no blocks, the constraint is satisfied
        if len(blocks) == 0:
            return True

        # get the center of mass
        center_of_mass = np.mean(blocks, axis=0)

        # check if the center of mass is within the margin
        return (center_of_mass[0] - width / 2.0)**2 + (center_of_mass[2] - depth / 2.0)**2 <= self.margin**2

from abc import ABC, abstractmethod
import numpy as np

class Constraint(ABC):

    @abstractmethod
    def apply(self, matrix: np.ndarray):
        '''
        Apply the constraint to the matrix.
        
        Parameters
        ----------
            `matrix` : `np.ndarray`
                the matrix to apply the constraint to.
        '''
        pass

    @abstractmethod
    def is_satisfied(self, matrix: np.ndarray) -> bool:
        '''
        Check if the constraint is satisfied by the matrix.
        
        Parameters
        ----------
            `matrix` : `np.ndarray`
                the matrix to check the constraint against.
        
        Returns
        -------
            bool : whether the constraint is satisfied.
        '''
        pass

from abc import ABC, abstractmethod

class Constraint(ABC):

    @abstractmethod
    def apply(self, matrix):
        '''
        Apply the constraint to the matrix.
        
        Parameters
        ----------
            `matrix` : `Matrix`
                the matrix to apply the constraint to.
        '''
        pass

    @abstractmethod
    def is_satisfied(self, matrix):
        '''
        Check if the constraint is satisfied by the matrix.
        
        Parameters
        ----------
            `matrix` : `Matrix`
                the matrix to check the constraint against.
        
        Returns
        -------
            bool : whether the constraint is satisfied.
        '''
        pass

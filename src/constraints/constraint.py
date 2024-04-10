from abc import ABC, abstractmethod

class Constraint(ABC):

    @abstractmethod
    def apply(self, container):
        '''
        Apply the constraint to the container.
        
        Parameters
        ----------
            `container` : `Container`
                the container to apply the constraint to.
        '''        
        pass

    @abstractmethod
    def is_satisfied(self, container):
        '''
        Check if the constraint is satisfied by the container.
        
        Parameters
        ----------
            `container` : `Container`
                the container to check.
        
        Returns
        -------
            bool : whether the constraint is satisfied.
        '''
        pass

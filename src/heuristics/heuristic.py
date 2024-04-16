from abc import ABC, abstractmethod

class Heuristic(ABC):

    @abstractmethod
    def get_score(self, matrix):
        '''
        Get the score of the container.
        Range = [0, 1].
        A higher score means a better container.

        Parameters
        ----------
            `matrix` : np.array
                the matrix representation of the container.
        
        Returns
        -------
            `float`
                the score of the container.
        '''
        pass

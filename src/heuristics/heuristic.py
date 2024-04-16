from abc import ABC, abstractmethod

class Heuristic(ABC):

    @abstractmethod
    def get_score(self, matrix):
        '''
        Get the normalized `[0, 1]` score of the matrix.
        A higher score means a better matrix.

        Parameters
        ----------
            `matrix` : np.array
                the matrix representation of the container.
        
        Returns
        -------
            `float`
                the score of the matrix.
        '''
        pass

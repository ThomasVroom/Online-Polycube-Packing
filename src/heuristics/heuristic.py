from abc import ABC, abstractmethod
import numpy as np

class Heuristic(ABC):

    @abstractmethod
    def get_score(self, matrix: np.ndarray) -> float:
        '''
        Get the normalized `[0, 1]` score of the matrix.
        A higher score means a better matrix.

        Parameters
        ----------
            `matrix` : `np.ndarray`
                the matrix representation of the container.
        
        Returns
        -------
            `float`
                the score of the matrix.
        '''
        pass

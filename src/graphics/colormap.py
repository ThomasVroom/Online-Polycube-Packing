import numpy as np

class ColorMap:

    def __init__(self):
        '''
        Create a color map object that maps integers to colors.
        '''
        self.map = {}

    def get_color(self, int: int, rng: np.random.Generator=None) -> np.ndarray:
        '''
        Returns a random color for the given integer.
        Querying the same integer will return the same color.

        Parameters
        ----------
            `int` : int
                the integer to query.
            `rng` : `np.random.Generator`, optional
                the random number generator to use.
        
        Returns
        -------
            `np.ndarray`
                the color corresponding to the integer.
        '''

        if int not in self.map:
            if rng is None:
                self.map[int] = np.random.rand(3)
            else:
                self.map[int] = rng.random(3)
        return self.map[int]

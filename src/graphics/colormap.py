import numpy as np

class ColorMap:

    def __init__(self):
        '''
        Create a color map object that maps integers to colors.
        '''

        # empty map
        self.map = {}

    def get_color(self, int):
        '''
        Returns a random color for the given integer.
        Querying the same integer will return the same color.

        Parameters
        ----------
            `int` : int
                the integer to query.
        '''

        if int not in self.map:
            self.map[int] = np.random.rand(3)
        return self.map[int]

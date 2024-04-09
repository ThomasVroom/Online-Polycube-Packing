from shapes.polycube import Polycube
import numpy as np
import os

class ShapeGenerator:

    def __init__(self, upper_bound):
        '''
        Create a shape generator that can generate random polycubes
        (https://en.wikipedia.org/wiki/Polycube).

        Parameters
        ----------
            `upper_bound` : int
                the maximum size of the polycube (3 <= upper_bound <= 10).  
        '''

        # check if the upper bound is valid
        assert upper_bound >= 3, "The minimum size of the polycube is 3."
        assert upper_bound <= 10, "The maximum size of the polycube is 10."

        # empty list to store the polycubes
        self.polycubes = np.array([])

        while upper_bound >= 3:
            # check if the cache exist
            cache_path = os.path.join('resources', 'polycubes', f'cubes_{upper_bound}.npy')
            assert os.path.exists(cache_path), "cache was not found."

            # load the cache (source: https://github.com/mikepound/cubes)
            print(f"\rLoading polycubes n={upper_bound} from cache: ", end = "")
            self.polycubes = np.concatenate((self.polycubes, np.load(cache_path, allow_pickle=True)))
            print(f"{len(self.polycubes)} shapes")

            # decrement the upper bound
            upper_bound -= 1

    def get_random_polycube(self, idx=None):
        '''
        Get a random polycube.

        Parameters
        ----------
            `idx` : int, optional
                the index of the polycube to get. If None, a random polycube is returned.
        
        Returns
        -------
            `Polycube` : a Polycube object
                a random polycube.
        '''

        if idx is None:
            # get a random index
            idx = np.random.randint(0, len(self.polycubes))

        # get the corresponding polycube
        return Polycube(self.polycubes[idx] * (idx + 1))
    
    def create_sequence(self, length):
        '''
        Create a sequence of polycubes.

        Parameters
        ----------
            `length` : int
                the length of the sequence.
        
        Returns
        -------
            `iter` : an iterator of random Polycube objects
                a sequence of polycubes.
        '''

        # create the sequence
        return iter([self.get_random_polycube() for _ in range(length)])

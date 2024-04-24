import os
import json
import numpy as np
from src.environment.shapes import Polycube

class ShapeGenerator:

    def __init__(self, upper_bound: int, cache_path: str='resources/polycubes'):
        '''
        Create a shape generator that can generate random polycubes.

        Parameters
        ----------
            `upper_bound` : int
                the maximum size of the polycube (3 <= upper_bound <= 10).
            `cache_path` : str, optional
                the path to the cache of polycubes.
        '''

        # check if the upper bound is valid
        assert upper_bound >= 3, "The minimum size of the polycube is 3."
        assert upper_bound <= 10, "The maximum size of the polycube is 10."
        self.upper_bound = upper_bound

        # empty list to store the polycubes
        self.polycubes = np.array([])

        while upper_bound >= 3:
            # check if the cache exist
            path = os.path.join(cache_path, f'cubes_{upper_bound}.npy')
            assert os.path.exists(path), "cache was not found."

            # load the cache (source: https://github.com/mikepound/cubes)
            print(f"\rLoading polycubes n={upper_bound} from cache: ", end = "")
            self.polycubes = np.concatenate((self.polycubes, np.load(path, allow_pickle=True)))
            print(f"{len(self.polycubes)} shapes")

            # decrement the upper bound
            upper_bound -= 1

    def get_random_polycube(self, idx: int=None, rng: np.random.Generator=None) -> Polycube:
        '''
        Get a random polycube.

        Parameters
        ----------
            `idx` : int, optional
                the index of the polycube to get. If `None`, a random polycube is returned.
            `rng` : `np.random.Generator`, optional
                a random number generator.
        
        Returns
        -------
            `Polycube` : a Polycube object
                a random polycube.
        '''

        if idx is None:
            # get a random index
            if rng is not None:
                idx = rng.integers(0, len(self.polycubes))
            else:
                idx = np.random.randint(0, len(self.polycubes))

        # get the corresponding polycube
        return Polycube(self.polycubes[idx] * (idx + 1))
    
    def create_sequence(self, length: int, rng: np.random.Generator=None) -> list[Polycube]:
        '''
        Create a sequence of polycubes.

        Parameters
        ----------
            `length` : int
                the length of the sequence.
            `rng` : `np.random.Generator`, optional
                a random number generator.
        
        Returns
        -------
            `list[Polycube]` : a list of random Polycube objects
                a sequence of polycubes.
        '''
        return [self.get_random_polycube(rng=rng) for _ in range(length)]

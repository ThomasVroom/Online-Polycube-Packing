from shapes.polycube import Polycube
import numpy as np
import os
import json

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
        self.upper_bound = upper_bound

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
    
    def create_sequence(self, length, file_path=None):
        '''
        Create a sequence of polycubes.

        Parameters
        ----------
            `length` : int
                the length of the sequence.
            `file_path` : str, optional
                the path to save the sequence to.
        
        Returns
        -------
            `list` : a list of random Polycube objects
                a sequence of polycubes.
        '''

        # create a random sequence
        if file_path is None:
            return [self.get_random_polycube() for _ in range(length)]

        # create a random sequence and save it to a file
        seq = np.random.choice(range(len(self.polycubes)), length).tolist()
        with open(file_path, 'w') as f:
            json.dump({'upper_bound':self.upper_bound, 'length':length, 'sequence':seq}, f)

        # return the sequence from the file
        return self.load_sequence(file_path)
    
    def load_sequence(self, file_path):
        '''
        Load a sequence of polycubes from a file.

        Parameters
        ----------
            `file_path` : str
                the path to the file.
        
        Returns
        -------
            `list` : a list of Polycube objects
                a sequence of polycubes.
        '''

        # load the sequence
        with open(file_path, 'r') as f:
            seq = json.load(f)['sequence']

        # convert the sequence to polycubes
        return [self.get_random_polycube(idx) for idx in seq]

import numpy as np
import os
import json

class ShapeGenerator:

    def __init__(self, upper_bound):
        '''
        Create a shape generator that can generate random polycubes.

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
                the index of the polycube to get. If `None`, a random polycube is returned.
        
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

class Polycube:
    
    def __init__(self, matrix):
        '''
        Create a [polycube](https://en.wikipedia.org/wiki/Polycube) object.
        
        Parameters
        ----------
            `matrix` : 3-dimensional matrix
                the matrix of the polycube.
        '''
        
        # set the polycube
        self.matrix = matrix
        self.id = np.amax(matrix).astype(int)
    
    def __str__(self):
        '''
        Get the string representation of the polycube.
        
        Returns
        -------
            str : the string representation of the polycube.
        '''
        
        return '---------------\n' + str(self.matrix) + '\n---------------'

    def increment_id(self):
        '''
        Increment the id of the polycube.
        '''
        
        # add the value to the matrix and the id
        self.matrix[self.matrix != 0] += 1
        self.id += 1

    def get_rotations(self):
        '''
        Get all the unique rotations of the polycube.

        Returns
        -------
            list : 4-dimensional array
                all unique rotations of the polycube.
        '''

        # source: https://stackoverflow.com/a/33190472
        def rotations24(polycube):
            """List all 24 rotations of the given 3d array."""
            def rotations4(polycube, axes):
                """List the four rotations of the given 3d array in the plane spanned by the given axes."""
                for i in range(4):
                    yield np.rot90(polycube, i, axes)

            # imagine shape is pointing in axis 0 (up)

            # 4 rotations about axis 0
            yield from rotations4(polycube, (1,2))

            # rotate 180 about axis 1, now shape is pointing down in axis 0
            # 4 rotations about axis 0
            yield from rotations4(np.rot90(polycube, 2, axes=(0,2)), (1,2))

            # rotate 90 or 270 about axis 1, now shape is pointing in axis 2
            # 8 rotations about axis 2
            yield from rotations4(np.rot90(polycube, axes=(0,2)), (0,1))
            yield from rotations4(np.rot90(polycube, -1, axes=(0,2)), (0,1))

            # rotate about axis 2, now shape is pointing in axis 1
            # 8 rotations about axis 1
            yield from rotations4(np.rot90(polycube, axes=(0,1)), (0,2))
            yield from rotations4(np.rot90(polycube, -1, axes=(0,1)), (0,2))

        # get the rotations
        rot = list(rotations24(self.matrix))

        # remove duplicates
        unique = []
        for r in rot:
            if not any(np.array_equal(r, u.matrix) for u in unique):
                unique.append(Polycube(r))
        return unique

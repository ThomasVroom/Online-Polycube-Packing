import numpy as np

class Polycube:
    
    def __init__(self, matrix):
        '''
        Create a polycube object.
        
        Parameters
        ----------
            `matrix` : 3-dimensional matrix
                the matrix of the polycube.
        '''
        
        # set the polycube
        self.matrix = matrix
        self.id = np.amax(matrix).astype(int)

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

        def rotations24(polycube): # source: https://stackoverflow.com/a/33190472
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

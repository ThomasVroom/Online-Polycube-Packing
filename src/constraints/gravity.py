from src.constraints.constraint import Constraint
import numpy as np

class Gravity(Constraint):

    def __init__(self, connected=True):
        '''
        Create a constraint that applies gravity to the container.

        Parameters
        ----------
            `connected` : bool, optional
                whether connected components should be considered as a single piece.
        '''
        self.connected = connected

    def apply(self, container):
        '''
        Apply the constraint to the container.

        Parameters
        ----------
            `container` : `Container`
                the container to apply the constraint to.
        '''

        # apply gravity to the container
        if self.connected:
            self.apply_connected_gravity(container.matrix)
        else:
            self.apply_disconnected_gravity(container.matrix)

    def apply_connected_gravity(self, matrix):
        '''
        Apply gravity to connected components in the matrix.

        Parameters
        ----------
            `matrix` : numpy array
                the matrix to apply gravity to.
        '''
        
        # get the dimensions of the matrix
        width, height, depth = matrix.shape
        
        # keep track of the ids that have been moved
        moved_ids = []

        # apply gravity to each connected component
        for h in range(height):
            for x in range(width):
                for y in range(depth):
                    id = matrix[x, h, y]
                    if id != 0 and id not in moved_ids:
                        piece = (matrix == id) * 1 # get the piece matrix
                        h1 = h # current height
                        while h1 > 0 and \
                            all([(matrix[p[0], p[1] - 1, p[2]] == 0 or piece[p[0], p[1] - 1, p[2]]) for p in np.argwhere(piece)]):
                            # move down one level until we hit something
                            piece = np.roll(piece, -1, axis=1)
                            h1 -= 1
                        
                        # add piece to matrix
                        matrix[matrix == id] = 0
                        matrix[piece == 1] = id

                        # update moved ids
                        moved_ids.append(id)
    
    def apply_disconnected_gravity(self, matrix):
        '''
        Apply gravity to disconnected components in the matrix.

        Parameters
        ----------
            `matrix` : numpy array
                the matrix to apply gravity to.
        '''
        
        # get the dimensions of the matrix
        width, height, depth = matrix.shape
        
        # apply gravity to each disconnected component
        for h in range(1, height):
            for x in range(width):
                for y in range(depth):
                    if matrix[x, h, y] != 0: # non-empty square
                        h1 = h # current height
                        while h1 > 0 and matrix[x, h1 - 1, y] == 0: # move down until the square is on the ground
                            matrix[x, h1 - 1, y] = matrix[x, h1, y]
                            matrix[x, h1, y] = 0
                            h1 -= 1

    def is_satisfied(self, container):
        return True # not relevant

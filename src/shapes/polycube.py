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
    
    def rotate(self, axis, angle): # TODO : UNTESTED
        '''
        Rotate the polycube.
        
        Parameters
        ----------
            `axis` : int
                the axis of rotation.
                0: x-axis, 1: y-axis, 2: z-axis
            `angle` : int
                the angle of rotation (in multiples of 90).
                0: 0 degrees, 1: 90 degrees, 2: 180 degrees, 3: 270 degrees
        '''
        
        # rotate the polycube
        self.matrix = np.rot90(self.matrix, k=angle, axes=(axis, (axis + 1) % 3))

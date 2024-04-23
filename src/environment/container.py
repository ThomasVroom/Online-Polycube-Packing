import numpy as np
from src.constraints import Constraint
from src.environment.shapes import Polycube

class Container:

    def __init__(self, width: int, height: int, depth: int, constraints: list[Constraint]=None):
        '''
        Create a container object.
        
        Parameters
        ----------
            `width` : int
                the width of the container.
            `height` : int
                the height of the container.
            `depth` : int
                the depth of the container.
            `constraints` : `list[Constraint]`, optional
                a list of constraints that the container must satisfy.
        '''
        
        # set the container
        self.width = width
        self.height = height
        self.depth = depth
        self.matrix = np.zeros((width, height, depth))
        self.constraints = [] if constraints is None else constraints

    def get_dimensions(self) -> tuple[int, int, int]:
        '''
        Get the dimensions of the container.
        Format: (width, height, depth).
        
        Returns
        -------
            `tuple[int, int, int]` : the dimensions of the container.
        '''
        return (self.width, self.height, self.depth)
    
    def get_ids(self) -> np.ndarray:
        '''
        Get the unique ids of the shapes in the container.
        
        Returns
        -------
            `np.ndarray` : the unique ids of the shapes in the container.
        '''

        unique = np.unique(self.matrix)
        return unique[np.where(unique != 0)]

    def reset(self):
        '''
        Reset the container to a blank state.
        '''
        self.matrix = np.zeros(self.get_dimensions())
    
    def fits(self, polycube: Polycube, position: tuple[int, int, int]) -> bool:
        '''
        Check if a polycube fits in the container.

        Parameters
        ----------
            `polycube` : `Polycube`
                the polycube to be checked.
            `position` : `tuple[int, int, int]`
                the position of the polycube.
        
        Returns
        -------
            bool : True if the polycube fits in the container, otherwise False.
        '''

        # get the dimensions of the polycube
        shape_width, shape_height, shape_depth = polycube.matrix.shape

        # check if the polycube fits in the container
        if position[0] + shape_width > self.width:
            return False
        if position[1] + shape_height > self.height:
            return False
        if position[2] + shape_depth > self.depth:
            return False
        
        # check for overlap      
        mx = np.ma.masked_array(self.matrix[position[0]:position[0] + shape_width,
                                            position[1]:position[1] + shape_height,
                                            position[2]:position[2] + shape_depth],
                                            mask=(polycube.matrix == 0))
        if np.any(mx):
            return False
        
        # add the polycube to the container
        self.matrix[position[0]:position[0] + shape_width,
                    position[1]:position[1] + shape_height,
                    position[2]:position[2] + shape_depth] += polycube.matrix
        
        # check if the constraints are satisfied
        for constraint in self.constraints:
            if not constraint.is_satisfied(self.matrix):
                # remove the polycube from the container
                self.matrix[position[0]:position[0] + shape_width,
                            position[1]:position[1] + shape_height,
                            position[2]:position[2] + shape_depth] -= polycube.matrix
                return False
        
        # remove the polycube from the container
        self.matrix[position[0]:position[0] + shape_width,
                    position[1]:position[1] + shape_height,
                    position[2]:position[2] + shape_depth] -= polycube.matrix
        return True

    def add(self, polycube: Polycube, position: tuple[int, int, int]) -> bool:
        '''
        Add a polycube to the container.
        
        Parameters
        ----------
            `polycube` : `Polycube`
                the polycube to be added.
            `position` : `tuple[int, int, int]`
                the position of the polycube.
            
        Returns
        -------
            bool : True if the shape was successfully added, otherwise False 
            (in this case the container will not be modified).
        '''

        # check if the polycube fits in the container
        if not self.fits(polycube, position):
            return False
        
        # check if the id is already taken
        while np.isin(polycube.id, self.matrix):
            polycube.increment_id()

        # add the polycube to the container
        shape_width, shape_height, shape_depth = polycube.matrix.shape
        self.matrix[position[0]:position[0] + shape_width,
                    position[1]:position[1] + shape_height,
                    position[2]:position[2] + shape_depth] += polycube.matrix
        
        # apply constraints
        for constraint in self.constraints:
            constraint.apply(self.matrix)
        
        return True

    def get_feasible_mask(self, polycube: Polycube) -> np.ndarray:
        '''
        Get a mask of the container where the polycube can fit.

        Parameters
        ----------
            `polycube` : `Polycube`
                the polycube to be checked (locked rotation).
        
        Returns
        -------
            `np.ndarray` : a 3D mask of the container where the shape can fit.
        '''

        # create an empty copy of the container
        mask = np.full(self.get_dimensions(), False, dtype=bool)
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    if self.fits(polycube, (x, y, z)):
                        mask[x, y, z] = True
        return mask

    def get_dummy_container(self, polycube: Polycube, position: tuple[int, int, int]) -> np.ndarray:
        ''''
        Get a copy of the container with the polycube added.
        Note that this copy does not have unique ids for the shapes.

        Parameters
        ----------
            `polycube` : `Polycube`
                the polycube to be added.
            `position` : `tuple[int, int, int]`
                the position of the polycube.
        
        Returns
        -------
            `np.ndarray` : a copy of the container with the polycube added.
        '''

        # create a copy of the container
        dummy_container = self.matrix.copy()

        # add the polycube to the container
        shape_width, shape_height, shape_depth = polycube.matrix.shape
        dummy_container[position[0]:position[0] + shape_width,
                        position[1]:position[1] + shape_height,
                        position[2]:position[2] + shape_depth] += polycube.matrix
        
        # apply constraints
        for constraint in self.constraints:
            constraint.apply(dummy_container)
        
        return dummy_container

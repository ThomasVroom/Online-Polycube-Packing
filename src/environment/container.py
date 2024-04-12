import numpy as np

class Container:

    def __init__(self, width, height, depth, constraints=None):
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
            `constraints` : list of constraints, optional
                a list of constraints that the container must satisfy.
        '''
        
        # set the container
        self.width = width
        self.height = height
        self.depth = depth
        self.matrix = np.zeros((width, height, depth))
        self.constraints = [] if constraints is None else constraints

    def get_dimensions(self):
        '''
        Get the dimensions of the container.
        Format: (width, height, depth).
        
        Returns
        -------
            tuple : the dimensions of the container.
        '''
        
        return (self.width, self.height, self.depth)
    
    def get_ids(self):
        '''
        Get the unique ids of the shapes in the container.
        
        Returns
        -------
            list : the unique ids of the shapes in the container.
        '''
        
        return np.unique(self.matrix)[1:]

    def add(self, shape, position):
        '''
        Add a shape to the container.
        
        Parameters
        ----------
            `shape` : `Polycube`
                the shape to be added.
            `position` : 3-dimensional integer vector
                the position of the shape.
            
        Returns
        -------
            bool : True if the shape was successfully added, otherwise False 
            (in this case the container will not be modified).
        '''
        
        # get the dimensions of the shape
        shape_width, shape_height, shape_depth = shape.matrix.shape

        # check if the shape fits in the container
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
                                            mask=(shape.matrix == 0))
        if np.any(mx):
            return False

        # check if the id is already taken
        while np.isin(shape.id, self.matrix):
            shape.increment_id()
        
        # add the shape to the container
        self.matrix[position[0]:position[0] + shape_width,
                    position[1]:position[1] + shape_height,
                    position[2]:position[2] + shape_depth] += shape.matrix
        
        # check if the constraints are satisfied
        for constraint in self.constraints:
            if not constraint.is_satisfied(self):
                # remove the shape from the container
                self.matrix[position[0]:position[0] + shape_width,
                            position[1]:position[1] + shape_height,
                            position[2]:position[2] + shape_depth] -= shape.matrix
                return False
        
        # apply constraints
        for constraint in self.constraints:
            constraint.apply(self)
        
        return True
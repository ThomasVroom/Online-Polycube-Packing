from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self, container):
        '''
        Create an agent that can pack polycubes into a container.

        Parameters
        ----------
            `container` : Container
                the container object.
        '''
        
        self.container = container
        self.dimensions = container.get_dimensions()
    
    def reset(self):
        '''
        Reset the environment.
        '''
        self.container.reset()

    @abstractmethod
    def step(self, shape):
        '''
        Add a shape to the container.

        Parameters
        ----------
            `shape` : Polycube
                the shape to be added.
        
        Returns
        -------
            `bool`
                whether the shape was added successfully.
        '''
        pass

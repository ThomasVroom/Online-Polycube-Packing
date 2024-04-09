from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self, visualizer):
        '''
        Create an agent that can pack polycubes into a container.

        Parameters
        ----------
            `visualizer` : Visualizer
                the visualizer object.
        '''
        self.vis = visualizer
    
    @abstractmethod
    def pack(self, container, sequence):
        '''
        Pack a sequence of polycubes into the container.

        Parameters
        ----------
            `container` : Container
                the container object.
            `sequence` : list
                the sequence of polycubes to pack.
        '''
        pass

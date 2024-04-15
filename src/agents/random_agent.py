import numpy as np
from agents.agent import Agent

class RandomAgent(Agent):

    def __init__(self, container, max_tries=100):
        '''
        Create a random agent that can pack polycubes into a container.

        Parameters
        ----------
            `container` : Container
                the container object.
            `max_tries` : int, optional
                the maximum number of tries to fit a polycube before giving up.
        '''

        self.max_tries = max_tries
        super().__init__(container)

    def step(self, shape):
        # get all the rotations of the shape
        rotations = shape.get_rotations()

        # state variables
        options_tried = 0
        successful_fit = False

        # try to add the polycube to the container
        while not (successful_fit or options_tried >= self.max_tries):
            # select a random rotation
            p = np.random.choice(rotations)

            # get random position
            x = np.random.randint(0, self.dimensions[0] - 1)
            y = np.random.randint(0, self.dimensions[1] - 1)
            z = np.random.randint(0, self.dimensions[2] - 1)
            print(f'Polycube {p.id} : Attempt {options_tried + 1} at ({x}, {y}, {z})...', end='\r')

            # try to fit the polycube
            if self.container.add(p, (x, y, z)):
                successful_fit = True
            
            # increment options tried
            options_tried += 1

        # print the result
        print('', end='\n', flush=True)
        print(f'Number fitted: {len(self.container.get_ids())}')
        return successful_fit

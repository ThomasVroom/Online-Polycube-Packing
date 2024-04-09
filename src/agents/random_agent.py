import numpy as np

class RandomAgent:

    def __init__(self, visualizer):
        '''
        Create a random agent that can pack polycubes into a container.

        Parameters
        ----------
            `visualizer` : Visualizer
                the visualizer object.
        '''
        self.vis = visualizer

    def pack(self, container, sequence, max_tries=100):
        '''
        Pack a sequence of polycubes into the container.

        Parameters
        ----------
            `container` : Container
                the container object.
            `sequence` : list
                the sequence of polycubes to pack (will be mutated!).
            `max_tries` : int, optional
                the maximum number of tries to fit a polycube before giving up.
        '''

        # state variables
        no_more_options = False
        options_tried = 0

        # await start signal
        self.vis.await_start()

        # loop until sequence is empty or agent exhausted all options
        while not (len(sequence) <= 0 or no_more_options):
            # get the next polycube
            p = sequence.pop(0)
            rotations = p.get_rotations()
            
            # try to add the polycube to the container
            options_tried = 0
            while not no_more_options:
                # select a random rotation
                p = np.random.choice(rotations)

                # get random position
                x = np.random.randint(0, container.get_dimensions()[0] - 1)
                y = np.random.randint(0, container.get_dimensions()[1] - 1)
                z = np.random.randint(0, container.get_dimensions()[2] - 1)
                print(f'Polycube {p.id} : Attempt {options_tried + 1} at ({x}, {y}, {z})...', end='\r')

                # try to fit the polycube
                if container.fit(p, (x, y, z)):
                    container.add(p, (x, y, z))
                    break

                # check if all options are exhausted
                options_tried += 1
                if options_tried >= max_tries:
                    no_more_options = True
            print('', end='\n', flush=True)
        
        # update the visualizer
        self.vis.update(container)
        print(f'Number fitted: {len(container.get_ids())}')

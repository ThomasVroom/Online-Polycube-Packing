import gymnasium as gym
from gymnasium import spaces
import numpy as np
from overrides import override

class PackingEnv(gym.Env):
    
    def __init__(self, container, shape_generator=None, sequence_length=100, sequence_path=None):
        '''
        Create a packing environment.

        Parameters
        ----------
            `container` : Container
                the container that needs to be packed.
            `shape_generator` : ShapeGenerator, optional
                a shape generator that can generate random polycubes.
            `sequence_length` : int, optional
                the length of the sequence of polycubes to pack (only relevant if `shape_generator` is used).
            `sequence_path` : str, optional
                the path to a sequence of polycubes to pack (overwrites `shape_generator`).
        '''

        # set the environment variables
        self.container = container
        self.generator = shape_generator
        self.sequence_length = sequence_length
        self.sequence_path = sequence_path
        self.sequence = None
        self.dimensions = container.get_dimensions()
        self.action_space_nvec = np.append([24], self.dimensions)

        # the observation space is defined as the combination of the (current) container and the (next) polycube.
        # the container is represented as a binary tensor, where 1 indicates an occupied cell.
        # the polycube is represented as a (padded) binary tensor, where 1 indicates the presence of a cube.
        # note that this space is only dependent on the size of the container.
        self.observation_space = spaces.Dict(
            {
                'container': spaces.MultiBinary(self.dimensions),
                'polycube': spaces.MultiBinary(self.dimensions)
            }
        )

        # the action space is defined as the product of the rotation and position of the polycube.
        # e.g. a 5x5x5 container with 24 rotations has an action space of 24 * 5 * 5 * 5 = 3000.
        # each number encodes a unique rotation and position of the polycube.
        # note that this space is only dependent on the size of the container.
        self.action_space = spaces.Discrete(np.prod(self.action_space_nvec))
    
    def _get_obs(self):
        '''
        Translate the current state of the environment to an observation.

        Returns
        -------
            dict : the observation of the environment.
        '''

        # transform the container and polycube to binary tensors
        binary_container = np.where(self.container.matrix > 0, 1, 0)
        binary_polycube = np.where(self.sequence[-1].matrix > 0, 1, 0)

        # pad the polycube to the size of the container
        binary_polycube = np.pad(binary_polycube, [(0, self.dimensions[0] - binary_polycube.shape[0]),
                                                   (0, self.dimensions[1] - binary_polycube.shape[1]),
                                                   (0, self.dimensions[2] - binary_polycube.shape[2])])

        # return the observation
        return {'container': binary_container, 'polycube': binary_polycube}
    
    def _get_info(self):
        pass

    @override
    def reset(self, seed=None, options=None):
        # reset np.random to the correct seed
        super().reset(seed=seed)

        # reset the container
        self.container.reset()

        # generate a new sequence
        if self.sequence_path is None:
            self.sequence = self.generator.create_sequence(self.sequence_length)
        else:
            self.sequence = self.generator.load_sequence(self.sequence_path)
        
        # return the current observation
        return self._get_obs(), self._get_info()

    @override
    def step(self, action):
        # decode the action
        rot, pos = self.decode_action(action)

        # get the polycube
        polycube = self.sequence.pop().get_rotations()[rot]

        # add the polycube to the container
        self.container.add(polycube, (pos[0], pos[1], pos[2]))

        # check if the episode is done
        terminated = len(self.sequence) == 0 or len(self.get_feasible_positions()) == 0

        # give binary sparse rewards
        reward = 1 if terminated else 0 # TODO: or the heuristic scores?

        # return the next observation, reward, terminated, truncated and info
        return self._get_obs(), reward, terminated, False, self._get_info()
    
    def decode_action(self, action):
        '''
        Decode the action to a rotation and position of the polycube.

        Parameters
        ----------
            `action` : int
                the action to decode.

        Returns
        -------
            Tuple[int, [int, int, int]] : the rotation and position of the polycube.
        '''

        # decode the action
        action = np.unravel_index(action, self.action_space_nvec)

        # return the rotation and position of the polycube
        return action[0], (action[1], action[2], action[3])
    
    def get_feasible_positions(self):
        '''
        Get the feasible positions for the current polycube.

        Returns
        -------
            List[Tuple[int, int, int, int]] : the feasible positions for the current polycube (format: r, x, y, z).
        '''

        # get all rotations of the current polycube
        rotations = self.sequence[-1].get_rotations()

        # get all feasible positions for the current polycube (format: r, x, y, z)
        return np.argwhere(np.array([self.container.get_feasible_mask(r) for r in rotations]))
    
    def action_masks(self):
        '''
        Get the action masks for the current state of the environment.

        Returns
        -------
            List[bool] : the action masks for the current state of the environment (True if the action is valid).
        '''

        # get all feasible positions for the current polycube (format: r, x, y, z)
        feasible_positions = self.get_feasible_positions()

        # set the action mask
        action_mask = np.full(np.prod(self.action_space_nvec), False, dtype=bool)
        for pos in feasible_positions: # encode the positions as a single number
            action_mask[np.ravel_multi_index(pos, self.action_space_nvec)] = True
        
        # return the action mask
        return action_mask

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from overrides import override
from src.environment import Container
from src.environment import ShapeGenerator
from src.environment.shapes import Polycube
from src.heuristics import Heuristic

class PackingEnv(gym.Env):
    
    def __init__(
            self,container: Container,
            upper_bound: int,
            seq_length: int=100,
            cache_path: str='resources/polycubes',
            seed: int=None
        ):
        '''
        Create a packing environment.

        Parameters
        ----------
            `container` : `Container`
                the container that needs to be packed.
            `upper_bound` : int
                an upper bound for the size of the polycubes to pack.
            `seq_length` : int, optional
                the length of the sequence of polycubes to pack.
            `cache_path` : str, optional
                the path to the cache of polycubes.
            `seed` : int, optional
                the seed for the random number generator (used when packing through UI).
        '''

        # set the environment variables
        self.container = container
        assert upper_bound <= max(container.get_dimensions()), 'polycubes cannot be larger than the container'
        self.generator = ShapeGenerator(upper_bound, cache_path)
        self.sequence_length = seq_length
        self.seed = seed
        self.sequence = None
        self.dimensions = container.get_dimensions()
        self.action_space_nvec = np.append([24], self.dimensions)
        self.feasible_positions = None
        self.heuristics = None
        self.heuristics_n = None
        self.obs_cache = None

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

    def _get_obs_cache(self) -> dict:
        '''
        Translate the current state of the environment to an observation.
        This method uses a cache of the obervation computed in the previous step,
        making it faster when used consecutively.

        Returns
        -------
            `dict[container, polycube]` : the observation of the environment.
        '''
        return self.obs_cache
    
    def _get_obs(self) -> dict:
        '''
        Translate the current state of the environment to an observation.

        Returns
        -------
            `dict[container, polycube]` : the observation of the environment.
        '''

        # transform the container and polycube to binary tensors
        binary_container = np.where(self.container.matrix > 0, 1, 0)
        binary_polycube = np.where(self.get_current_polycube().matrix > 0, 1, 0)

        # pad the polycube to the size of the container
        binary_polycube = np.pad(binary_polycube, [(0, self.dimensions[0] - binary_polycube.shape[0]),
                                                   (0, self.dimensions[1] - binary_polycube.shape[1]),
                                                   (0, self.dimensions[2] - binary_polycube.shape[2])])

        # return the observation
        return {'container': binary_container, 'polycube': binary_polycube}
    
    def _get_info(self) -> dict:
        '''
        Get additional information about the environment.

        Returns
        -------
            `dict` : additional information about the environment.
        '''
        return {}

    @override
    def reset(self, seed: int=None, options=None):
        # reset np.random to the correct seed
        super().reset(seed=seed if seed is not None else self.seed, options=options)

        # reset the container
        self.container.reset()

        # generate a new sequence
        self.sequence = self.generator.create_sequence(self.sequence_length, rng=self.np_random)

        # set the feasible positions
        self.feasible_positions = self.find_feasible_positions()

        # update the observation cache
        self.obs_cache = self._get_obs()
        
        # return the current observation
        return self._get_obs_cache(), self._get_info()

    @override
    def step(self, action: int):
        # decode the action
        rot, pos = self.decode_action(action)

        # get the polycube
        polycube = self.sequence.pop().get_rotations()[rot]

        # add the polycube to the container
        self.container.add(polycube, (pos[0], pos[1], pos[2]))

        # set the feasible positions
        self.feasible_positions = self.find_feasible_positions()

        # check if the episode is done
        terminated = self.is_terminal()

        # get reward
        reward = 0 if not terminated else len(self.container.get_ids())

        # update the observation cache
        self.obs_cache = self._get_obs()

        # return the next observation, reward, terminated, truncated and info
        return self._get_obs_cache(), reward, terminated, False, self._get_info()
    
    def is_terminal(self) -> bool:
        '''
        Check if the current state of the environment is terminal.

        Returns
        -------
            bool : True if the state is terminal, False otherwise.
        '''

        # the state is terminal if the sequence is empty or no feasible positions are available
        return len(self.sequence) == 0 or len(self.feasible_positions) == 0
    
    def get_current_polycube(self) -> Polycube:
        '''
        Get the current polycube to pack.

        Returns
        -------
            `Polycube` : the current polycube to pack.
        '''
        return self.sequence[-1]
    
    def decode_action(self, action: int) -> tuple[int, tuple[int, int, int]]:
        '''
        Decode the action to a rotation and position of the polycube.

        Parameters
        ----------
            `action` : int
                the action to decode.

        Returns
        -------
            `tuple[int, tuple[int, int, int]]` : the rotation and position of the polycube.
        '''

        # decode the action
        action = np.unravel_index(action, self.action_space_nvec)

        # return the rotation and position of the polycube
        return action[0], (action[1], action[2], action[3])
    
    def encode_action(self, rot: int, pos: tuple[int, int, int]) -> int:
        '''
        Encode the rotation and position of the polycube to an action.

        Parameters
        ----------
            `rot` : int
                the rotation of the polycube.
            `pos` : `tuple[int, int, int]`
                the position of the polycube.

        Returns
        -------
            int : the action.
        '''

        # encode the rotation and position of the polycube
        return np.ravel_multi_index([rot, pos[0], pos[1], pos[2]], self.action_space_nvec)
    
    def find_feasible_positions(self) -> list[tuple[int, int, int, int]]:
        '''
        Find the feasible positions for the current polycube.

        Returns
        -------
            `list[tuple[int, int, int, int]]` : the feasible positions for the current polycube (format: r, x, y, z).
        '''

        # get all rotations of the current polycube
        rotations = self.get_current_polycube().get_rotations()

        # get all feasible positions for the current polycube (format: r, x, y, z)
        return np.argwhere(np.array([self.container.get_feasible_mask(r) for r in rotations]))
    
    def action_masks(self) -> list[bool]:
        '''
        Get the action masks for the current state of the environment.

        Returns
        -------
            `list[bool]` : the action masks for the current state of the environment (True if the action is valid).
        '''

        # check if heuristics are set
        if self.heuristics is not None:
            return self.get_heuristic_mask()

        # create the action mask
        action_mask = np.full(np.prod(self.action_space_nvec), False, dtype=bool)
        for pos in self.feasible_positions: # encode the positions as a single number
            action_mask[self.encode_action(pos[0], (pos[1], pos[2], pos[3]))] = True
        
        # return the action mask
        return action_mask
    
    def set_heuristics(self, heuristics: list[Heuristic], n: int):
        '''
        Set the heuristics for the environment.
        If set, this will alter the `action_masks` method to consider the `n` best positions according to the heuristics.

        Parameters
        ----------
            `heuristics` : `list[Heuristic]`
                the heuristics to use.
            `n` : int
                the number of best positions to consider.
        '''
        self.heuristics = heuristics
        self.heuristics_n = n
    
    def get_heuristic_mask(self) -> list[bool]:
        '''
        Get the heuristic mask for the current state of the environment.

        Returns
        -------
            `list[bool]` : the heuristic mask for the current state of the environment (True if the action is valid).
        '''

        # create the heuristic mask
        heuristic_mask = np.full(np.prod(self.action_space_nvec), False, dtype=bool)

        # get all rotations of the current polycube
        rotations = self.get_current_polycube().get_rotations()

        # create score table for heuristics
        scores = np.zeros((len(self.heuristics), len(self.feasible_positions)))

        # get the scores for every feasible position
        for j, (r, x, y, z) in enumerate(self.feasible_positions):
            for i, h in enumerate(self.heuristics):
                scores[i, j] = h.get_score(self.container.get_dummy_container(rotations[r], (x, y, z)))

        # average the scores
        avg_scores = np.average(scores, axis=0)

        # get the best positions
        sorted_feasible_positions = np.argsort(avg_scores)[-self.heuristics_n:]
        for p in sorted_feasible_positions:
            pos = self.feasible_positions[p]
            heuristic_mask[self.encode_action(pos[0], (pos[1], pos[2], pos[3]))] = True

        # return the heuristic mask
        return heuristic_mask

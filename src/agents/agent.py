from abc import ABC, abstractmethod
from src.environment import PackingEnv

class Agent(ABC):

    @abstractmethod
    def get_action(self, env: PackingEnv) -> int:
        '''
        Get the next action to perform in the environment.

        Parameters
        ----------
            `env` : `PackingEnv`
                the environment to get the action for.
        
        Returns
        -------
            int : the action to perform.
        '''
        pass

from src.agents import Agent
from overrides import override
from sb3_contrib import MaskablePPO

class PPOAgent(Agent):

    def __init__(self, model: MaskablePPO):
        '''
        Create a PPO agent that can pack using a trained PPO model.

        Parameters
        ----------
            `model` : `MaskablePPO`
                the model to use for the agent.
        '''
        self.model = model
    
    @override
    def get_action(self, env) -> int:
        # get observation and action mask
        obs = env._get_obs()
        action_mask = env.action_masks()

        # get action
        action, _ = self.model.predict(obs, action_masks=action_mask)
        return action

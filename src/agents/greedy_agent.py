from src.agents import Agent
from overrides import override
from src.heuristics import Heuristic
import numpy as np

class GreedyAgent(Agent):

    def __init__(self, heuristics: list[Heuristic], weights: list[float]=None):
        '''
        Create a greedy agent that can pack using heuristics.

        Parameters
        ----------
            `heuristics` : `list[Heuristic]`
                the heuristics to use for the agent.
            `weights` : `list[float]`, optional
                the weights to use for the heuristics (if not provided, the weights will be equal).
        '''

        self.heuristics = heuristics
        self.weights = weights

    @override
    def get_action(self, env) -> int:
        # get current polycube id
        id = env.get_current_polycube().id

        # get all rotations of the current polycube
        rotations = env.get_current_polycube().get_rotations()

        # get the feasible positions for the current polycube
        feasible_positions = env.feasible_positions
        print(f'found {len(feasible_positions)} feasible positions for polycube {id}')

        # create score table for heuristics
        scores = np.zeros((len(self.heuristics), len(feasible_positions)))

        # get the scores for every feasible position
        for j, (r, x, y, z) in enumerate(feasible_positions):
            for i, h in enumerate(self.heuristics):
                scores[i, j] = h.get_score(env.container.get_dummy_container(rotations[r], (x, y, z)))
        
        # average the scores
        avg_scores = np.average(scores, axis=0, weights=self.weights)

        # get the best position
        best_idx = np.argmax(avg_scores)
        best_pos = feasible_positions[best_idx]
        print(f'adding polycube {id} at position {best_pos[1], best_pos[2], best_pos[3]} with a score of {avg_scores[best_idx]}')

        # return the action
        return env.encode_action(best_pos[0], (best_pos[1], best_pos[2], best_pos[3]))

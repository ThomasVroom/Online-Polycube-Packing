from src.agents.agent import Agent
import numpy as np

class GreedyAgent(Agent):

    def __init__(self, container, heuristics, weights=None):
        '''
        Create a greedy agent that can pack polycubes into a container using heuristics.

        Parameters
        ----------
            `container` : Container
                the container object.
            `heuristics` : List[`Heuristic`]
                the heuristics to use for the agent.
            `weights` : List[float], optional
                the weights to use for the heuristics (if not provided, the weights will be equal).
        '''

        self.heuristics = heuristics
        self.weights = weights
        super().__init__(container)

    def step(self, shape):
        # get all the rotations of the shape
        rotations = shape.get_rotations()

        # get all feasible positions (format: r, x, y, z)
        feasible_positions = np.argwhere(np.array([self.container.get_feasible_mask(r) for r in rotations]))
        print(f'found {len(feasible_positions)} feasible positions for polycube {shape.id}')
        if len(feasible_positions) == 0:
            return False

        # create score tracker for every heuristic
        scores = np.zeros((len(self.heuristics), len(feasible_positions)))
        
        # get the scores for every feasible position
        for j, (r, x, y, z) in enumerate(feasible_positions):
            for i, h in enumerate(self.heuristics):
                scores[i, j] = h.get_score(self.container.get_dummy_container(rotations[r], (x, y, z)))
        
        # average the scores
        avg_scores = np.average(scores, axis=0, weights=self.weights)

        # get the best position
        best_idx = np.argmax(avg_scores)
        best_pos = feasible_positions[best_idx]
        print(f'adding polycube {shape.id} at position {best_pos[1], best_pos[2], best_pos[3]} with a score of {avg_scores[best_idx]}')
        return self.container.add(rotations[best_pos[0]], (best_pos[1], best_pos[2], best_pos[3]))

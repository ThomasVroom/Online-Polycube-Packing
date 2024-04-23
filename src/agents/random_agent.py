from src.agents import Agent
from overrides import override

class RandomAgent(Agent):

    @override
    def get_action(self, env) -> int:
        # get current polycube id
        id = env.get_current_polycube().id

        # get the feasible positions for the current polycube
        feasible_positions = env.feasible_positions
        print(f'found {len(feasible_positions)} feasible positions for polycube {id}')

        # select a random feasible position
        pos = env.np_random.choice(feasible_positions)
        print(f'adding polycube {id} at position {pos[1], pos[2], pos[3]}')

        # return the action
        return env.encode_action(pos[0], (pos[1], pos[2], pos[3]))

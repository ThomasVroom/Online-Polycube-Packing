from src.graphics import Visualizer
from src.environment import ShapeGenerator
from src.environment import Container
from src.agents import RandomAgent

if __name__ == '__main__':

    # set up environment
    c = Container(5, 5, 5)
    g = ShapeGenerator(upper_bound=5)
    seq = g.create_sequence(50)
    agent = RandomAgent(c)

    # start the UI
    vis = Visualizer(c, agent_sequence_pair=(agent, seq))
    vis.start()

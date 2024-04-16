from src.graphics.visualizer import Visualizer
from src.environment.shapes import ShapeGenerator
from src.environment.container import Container
from src.agents.random_agent import RandomAgent
from src.constraints.gravity import Gravity
from src.constraints.loadbalancing import LoadBalancing

if __name__ == '__main__':

    # set up environment
    c = Container(5, 5, 5, constraints=[Gravity()])
    g = ShapeGenerator(upper_bound=5)
    seq = g.create_sequence(50)
    agent = RandomAgent(c)

    # start the UI
    vis = Visualizer(c, agent_sequence_pair=(agent, seq))
    vis.start()

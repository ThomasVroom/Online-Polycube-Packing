from src.graphics.visualizer import Visualizer
from src.environment.shapes import ShapeGenerator
from src.environment.container import Container
from src.agents.greedy_agent import GreedyAgent
from src.heuristics.HAPE import HAPE
from src.heuristics.BLBF import BLBF
from src.heuristics.HM import HeightMapMinimization
from src.constraints.gravity import Gravity
from threading import Thread

if __name__ == '__main__':

    # set up environment
    c = Container(5, 5, 5, constraints=[])
    g = ShapeGenerator(upper_bound=5)
    seq = g.create_sequence(50)
    agent = GreedyAgent(c, heuristics=[HAPE(), BLBF(), HeightMapMinimization()])

    # start the UI
    vis = Visualizer(c, agent_sequence_pair=(agent, seq))
    vis.start()

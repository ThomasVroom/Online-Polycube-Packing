from src.graphics import Visualizer
from src.environment import ShapeGenerator
from src.environment import Container
from src.agents import GreedyAgent
from src.heuristics import *
from threading import Thread

if __name__ == '__main__':

    c = Container(5, 5, 5)
    g = ShapeGenerator(upper_bound=5)
    seq = g.create_sequence(50)
    agent = GreedyAgent(c, heuristics=[HeightMapMinimization(axis=0),
                                       HeightMapMinimization(axis=1),
                                       HeightMapMinimization(axis=2),
                                       HAPE()])

    # pack the shapes
    success = True
    while success and len(seq) > 0:
        success = agent.step(seq.pop())

    # start the UI
    vis = Visualizer(c)
    def update_ui():
        vis.await_start()
        vis.update()
    Thread(target=update_ui).start()
    vis.start()

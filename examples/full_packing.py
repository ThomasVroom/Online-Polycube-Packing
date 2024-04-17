from src.graphics.visualizer import Visualizer
from src.environment.shapes import ShapeGenerator
from src.environment.container import Container
from src.agents.random_agent import RandomAgent
from threading import Thread

if __name__ == '__main__':

    # set up environment
    c = Container(5, 5, 5)
    g = ShapeGenerator(upper_bound=5)
    seq = g.create_sequence(50)
    agent = RandomAgent(c)

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

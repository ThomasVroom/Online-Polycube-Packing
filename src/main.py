from graphics.visualizer import Visualizer
from environment.shapes import ShapeGenerator
from environment.container import Container
from threading import Thread
from agents.random_agent import RandomAgent

if __name__ == '__main__':

    # set up environment
    c = Container(5, 5, 5)
    vis = Visualizer(c.get_dimensions())
    g = ShapeGenerator(upper_bound=5)
    seq = g.create_sequence(50)

    # create a random agent
    agent = RandomAgent(vis)

    # start the packing
    Thread(target=agent.pack, args=(c, seq)).start()
    vis.start()

from src.graphics.visualizer import Visualizer
from src.environment.shapes import ShapeGenerator
from src.environment.container import Container
from src.agents.greedy_agent import GreedyAgent
from src.heuristics.HAPE import HAPE
from src.heuristics.BLBF import BLBF
from src.heuristics.HM import HeightMapMinimization
from threading import Thread

if __name__ == '__main__':

    # set up environment
    c = Container(5, 5, 5)
    g = ShapeGenerator(upper_bound=5)
    seq = g.create_sequence(50)
    agent = GreedyAgent(c, heuristics=[HeightMapMinimization(), HAPE(), BLBF()])

    # pack the shapes
    success = True
    while success and len(seq) > 0:
        success = agent.step(seq.pop())
    print(f'Total fitted: {len(c.get_ids())}')

    # start the UI
    vis = Visualizer(c)
    def update_ui():
        vis.await_start()
        vis.update()
    Thread(target=update_ui).start()
    vis.start()

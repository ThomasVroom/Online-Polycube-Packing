from src.graphics import Visualizer
from src.environment import Container
from src.environment import PackingEnv
from src.agents import GreedyAgent
from src.heuristics import *

if __name__ == '__main__':

    # set up environment
    c = Container(5, 5, 5)
    env = PackingEnv(c, upper_bound=5, seed=42)
    agent = GreedyAgent(heuristics=[BLBF(), HAPE()])

    # start the UI
    vis = Visualizer(env, agent=agent)
    env.reset()
    vis.start()

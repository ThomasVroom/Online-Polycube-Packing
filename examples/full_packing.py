from src.graphics import Visualizer
from src.environment import Container
from src.environment import PackingEnv
from src.agents import GreedyAgent
from src.heuristics import *
from threading import Thread

if __name__ == '__main__':

    # set up environment
    c = Container(5, 5, 5)
    env = PackingEnv(c, upper_bound=5, seed=42)
    agent = GreedyAgent(heuristics=[BLBF(), HAPE()])

    # pack the shapes
    env.reset()
    while not env.is_terminal():
        env.step(agent.get_action(env))

    # start the UI
    vis = Visualizer(env)
    def update_ui():
        vis.await_start()
        vis.update()
    Thread(target=update_ui).start()
    vis.start(labels=False)

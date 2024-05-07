from src.environment import PackingEnv
from src.environment import Container
from sb3_contrib import MaskablePPO
from src.graphics import Visualizer
from src.agents import PPOAgent

if __name__ == '__main__':

    # set up environment
    c = Container(3, 3, 3)
    env = PackingEnv(c, upper_bound=3, seed=42)

    # load model
    model = MaskablePPO('MultiInputPolicy', env, device='cuda')
    model.set_parameters('resources/models/without_heuristics/3x3x3.zip')
    agent = PPOAgent(model)

    # start the UI
    vis = Visualizer(env, agent=agent)
    env.reset()
    vis.start()

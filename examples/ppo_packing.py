from src.environment import PackingEnv
from src.environment import Container
from sb3_contrib import MaskablePPO
from src.graphics import Visualizer
from src.agents import PPOAgent

if __name__ == '__main__':

    # set up environment
    c = Container(5, 5, 5)
    env = PackingEnv(c, upper_bound=5, seed=42)

    # load model
    model = MaskablePPO('MultiInputPolicy', env, seed=42, device='cuda')
    model.set_parameters('resources/models/5x5x5-5-1_100000_steps.zip')
    agent = PPOAgent(model)

    # start the UI
    vis = Visualizer(env, agent=agent)
    env.reset()
    vis.start()

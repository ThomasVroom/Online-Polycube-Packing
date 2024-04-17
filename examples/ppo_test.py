from src.environment.packing_environment import PackingEnv
from src.environment.container import Container
from src.environment.shapes import ShapeGenerator
from sb3_contrib import MaskablePPO
from src.graphics.visualizer import Visualizer
from threading import Thread

if __name__ == '__main__':

    c = Container(5, 5, 5)
    env = PackingEnv(c, ShapeGenerator(upper_bound=5))
    model = MaskablePPO('MultiInputPolicy', env, seed=42, verbose=1)

    obs, _ = env.reset()
    action_mask = env.action_masks()
    action, _ = model.predict(obs, action_masks=action_mask)
    print('action:', action)
    obs, reward, terminated, _, _ = env.step(action)
    print('reward:', reward)
    print('terminated:', terminated)

    vis = Visualizer(c)
    def update_ui():
        vis.await_start()
        vis.update()
    Thread(target=update_ui).start()
    vis.start()

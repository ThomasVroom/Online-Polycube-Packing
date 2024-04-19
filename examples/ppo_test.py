from src.environment import PackingEnv
from src.environment import Container
from sb3_contrib import MaskablePPO
from src.graphics import Visualizer
from threading import Thread

if __name__ == '__main__':

    c = Container(5, 5, 5)
    env = PackingEnv(c, upper_bound=5, seed=42)
    model = MaskablePPO('MultiInputPolicy', env, seed=42, verbose=1)

    obs, _ = env.reset()
    action_mask = env.action_masks()
    action, _ = model.predict(obs, action_masks=action_mask)
    print('action:', action)
    obs, reward, terminated, _, _ = env.step(action)
    print('reward:', reward)
    print('terminated:', terminated)

    vis = Visualizer(env)
    def update_ui():
        vis.await_start()
        vis.update()
    Thread(target=update_ui).start()
    vis.start()

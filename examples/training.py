from src.environment import PackingEnv
from src.environment import Container
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

if __name__ == '__main__':

    # variables
    container_dim = (5, 5, 5) # dimensions of the container (width, height, depth)
    upper_bound = 5 # upper bound of the polycube size
    callback_freq = 1000 # how often the model should be evaluated (in steps)
    n_eval_episodes = 10 # number of episodes to evaluate the model
    name = '5x5x5-5' # name of the model
    run = 1 # run number
    update_freq = 100 # how often the policy should be updated (in steps)
    total_timesteps = 100000 # total number of steps to train the model
    checkpoint = '' # path to the model to continue training

    # create callbacks
    eval_callback = MaskableEvalCallback(
        eval_env=PackingEnv(Container(container_dim[0], container_dim[1], container_dim[2]), upper_bound=upper_bound),
        eval_freq=callback_freq,
        n_eval_episodes=n_eval_episodes,
        verbose=1,
        warn=False
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=callback_freq,
        save_path='resources/models/',
        name_prefix=name + '-' + str(run),
        save_replay_buffer=True,
        verbose=2
    )
    callback = CallbackList([eval_callback, checkpoint_callback])

    # create model
    model = MaskablePPO(
        policy='MultiInputPolicy',
        env=PackingEnv(Container(container_dim[0], container_dim[1], container_dim[2]), upper_bound=upper_bound),
        n_steps=update_freq,
        tensorboard_log='resources/logs/', # to see logs, run `tensorboard --logdir resources/logs/`
        device='cuda'
    )
    if checkpoint: # load model from checkpoint
        model.set_parameters(checkpoint)

    # train model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name=name,
        reset_num_timesteps=True,
        progress_bar=True
    )

from src.environment import PackingEnv
from src.environment import Container
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from src.heuristics import *

if __name__ == '__main__':

    # variables
    container_dim = (3, 3, 3) # dimensions of the container (width, height, depth)
    run = 0 # run number
    checkpoint = '' # path to a model to continue training
    expected_packed = 8 # expected number of polycubes that will be packed (used to normalize rewards)

    # create environment
    env = PackingEnv(
        Container(container_dim[0], container_dim[1], container_dim[2]),
        upper_bound=max(container_dim),
        exp_packed=expected_packed
    )

    # create callbacks
    eval_callback = MaskableEvalCallback(
        eval_env=env,
        eval_freq=50000, # how often the model should be evaluated (in steps)
        n_eval_episodes=25, # how many episodes to evaluate the model
        verbose=1,
        warn=False
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, # how often the model should be saved (in steps)
        save_path='resources/models/without_heuristics/',
        name_prefix=f'{container_dim[0]}x{container_dim[1]}x{container_dim[2]}-{run}',
        verbose=2
    )
    callback = CallbackList([eval_callback, checkpoint_callback])

    # create model
    model = MaskablePPO(
        policy='MultiInputPolicy',
        env=env,
        learning_rate=0.0003,
        n_steps=2500, # number of steps to collect samples for each policy update
        batch_size=64, # number of samples per training batch (policy update)
        n_epochs=10, # number of epochs when updating the policy
        gamma=0.99,
        gae_lambda=0.95, # factor for trade-off of bias vs variance for Generalized Advantage Estimator
        clip_range=0.2, # clip range for the policy loss
        clip_range_vf=None, # clip range for the value function
        normalize_advantage=True,
        ent_coef=0.001, # entropy coefficient for the loss calculation
        vf_coef=0.5, # value function coefficient for the loss calculation
        max_grad_norm=0.5, # max gradient norm
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        target_kl=None,
        tensorboard_log='resources/logs/without_heuristics/', 
        # to see logs, run `tensorboard --logdir resources/logs/without_heuristics/`
        device='cuda'
    )
    if checkpoint: # load model from checkpoint
        model.set_parameters(checkpoint)

    # train model
    model.learn( # torch bug: https://github.com/DLR-RM/stable-baselines3/issues/1596
        total_timesteps=500000, # total number of steps to train the model
        callback=callback,
        tb_log_name=f'{container_dim[0]}x{container_dim[1]}x{container_dim[2]}-{run}',
        reset_num_timesteps=False,
        progress_bar=True
    )

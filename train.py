import argparse
import gymnasium as gym
import torch
import os
import logging
import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize
from gymnasium.wrappers import AtariPreprocessing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['ALE/Pong-v5'], default='ALE/Pong-v5')
parser.add_argument('--evaluate_freq', type=int, default=50, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=10, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'ALE/Pong-v5': config.Pong  # Use the Pong configuration
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env, frameskip=1)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, noop_max=30, scale_obs=True)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    online_dqn = DQN(env_config=env_config).to(device)
    target_dqn = DQN(env_config=env_config).to(device)
    target_dqn.load_state_dict(online_dqn.state_dict())
    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])
    obs_stack_size = 4  # Stacking 4 frames for input
    # Initialize optimizer used for training the DQN.
    optimizer = torch.optim.Adam(online_dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")
    epsilon_decay_step = 0
    step_counter = 0


    #################################################### LOGGING  ##############################################################################

# Set the directory in Google Drive where models will be saved
drive_model_dir = '/content/drive/MyDrive/DQN_v2'


# Set up logging
# Create a named logger to prevent duplicate handlers
logger = logging.getLogger('DQN_Pong')
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent logs from being passed to the root logger

# Prevent adding multiple handlers if the cell is rerun
if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
    # Create file handler which logs messages to a file
    log_file_path = os.path.join(drive_model_dir, 'training_log.txt')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)

    # Create console handler with the same logging level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

logger.info("Logging setup complete. Starting training...")
# Create the directory if it doesn't exist
if not os.path.exists(drive_model_dir):
    os.makedirs(drive_model_dir)

    for episode in range(env_config['n_episodes']):
        #print(f'Starting episode: {episode}/10')
        terminated = False
        truncated = False
        # Reset environment and preprocess observation
        obs, _ = env.reset()
        obs = preprocess(obs, env=args.env)
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
        else:
            obs = obs.clone().detach().to(device)  # Shape: [84, 84]

        # Initialize the observation stack with 4 identical frames
        obs_stack = torch.stack([obs for _ in range(obs_stack_size)], dim=0).to(device)  # Shape: [4, 84, 84]

        
        #print("Loaded and preprocessed env.")
        #print("Starting play.")
        while not (terminated or truncated):
            # Get action from DQN
            action = online_dqn.act(obs_stack.unsqueeze(0), epsilon_decay_step)  # Shape: [1, 4, 84, 84]

            # Act in the true environment
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Preprocess incoming observation
            next_obs = preprocess(next_obs, env=args.env)
            if not torch.is_tensor(next_obs):
                next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
            else:
                next_obs = next_obs.clone().detach().to(device)  # Shape: [84, 84]

            # Update the observation stack
            next_obs_stack = torch.cat((obs_stack[1:], next_obs.unsqueeze(0)), dim=0)  # Shape: [4, 84, 84]


            # Compute 'done' flag
            done = terminated or truncated

            # Add the transition to the replay memory
            memory.push(obs_stack, action, next_obs_stack, reward, done)

            # Move to the next state
            obs_stack = next_obs_stack
            #print("Moving to optimization.")
            # Run optimization
            if step_counter % env_config['train_frequency'] == 0:
                optimize(online_dqn, target_dqn=target_dqn, memory=memory, optimizer=optimizer, device=device)

            # Update the target network
            if step_counter % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(online_dqn.state_dict())

            #print("Optimization done, moving to next step.")
            step_counter += 1
            epsilon_decay_step += 1
        #print("Episode done.")
        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(online_dqn, env, args, n_episodes=args.evaluation_episodes)
            logger.info(f"Evaluation after {episode} episodes: Average reward: {mean_return}")
            logger.info(f"Steps done: {step_counter}")

            # Save current agent if it has the best performance so far.
            # Save current agent if it has the best performance so far.
            if mean_return > best_mean_return:
                best_mean_return = mean_return
                print(f'Best performance so far! Saving model.')
                # Save the model directly to Google Drive
                torch.save(online_dqn.state_dict(), os.path.join(drive_model_dir, f'{episode}_best.pt'))
                model_save_path = os.path.join(drive_model_dir, f'best_checkpoint.pth')
                logger.info(f"Saved best checkpoint at episode {episode} to {model_save_path}")


    # Close environment after training is completed.
    env.close()

import argparse
import gymnasium as gym
import torch
import os
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# Import additional libraries for rendering
import matplotlib.pyplot as plt

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same action mapping as during training
ACTION_SPACE = [2, 5]  # NOOP, UP, DOWN
n_actions = len(ACTION_SPACE)

# Hyperparameter configurations for Pong
env_config = {
    "batch_size": 32,
    "gamma": 0.99,
    "eps_start": 1.0,
    "eps_end": 0.1,
    "anneal_length": 1000000,
    "n_actions": n_actions,  # Set n_actions to 3
}

# Define the DQN architecture (same as during training)
class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Define the same network architecture as during training
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

    def forward(self, x):
        x = x / 255.0  # Normalize pixel values
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def act(self, observation, step, exploit=False):
        # Epsilon-greedy policy
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * max(0, (self.anneal_length - step) / self.anneal_length)
        if exploit or random.random() > epsilon:
            with torch.no_grad():
                q_values = self.forward(observation)
                return torch.argmax(q_values).item()
        else:
            return random.randrange(self.n_actions)

def preprocess(obs, env):
    obs = np.array(obs)  # Convert observation to numpy array
    if not torch.is_tensor(obs):
        obs = torch.tensor(obs, dtype=torch.float32)
    return obs

def evaluate_policy(dqn, env, args, n_episodes, render=False, verbose=False):
    """Runs {n_episodes} episodes to evaluate the current policy."""
    total_return = 0

    for i in range(n_episodes):
        obs, _ = env.reset()
        obs = preprocess(obs, env=args.env)
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
        else:
            obs = obs.clone().detach().to(device)  # Shape: [84, 84]

        # Initialize the observation stack with 4 identical frames
        obs_stack = torch.stack([obs for _ in range(4)], dim=0).to(device)  # Shape: [4, 84, 84]

        terminated = False
        truncated = False
        episode_return = 0

        while not (terminated or truncated):
            # Add a batch dimension to obs_stack: Shape becomes [1, 4, 84, 84]
            action_index = dqn.act(obs_stack.unsqueeze(0), step=0, exploit=True)

            # Map action index to actual action
            action = ACTION_SPACE[action_index]

            # Act in the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = preprocess(next_obs, env=args.env)
            if not torch.is_tensor(next_obs):
                next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
            else:
                next_obs = next_obs.clone().detach().to(device)  # Shape: [84, 84]

            # Update the observation stack with the new frame
            obs_stack = torch.cat((obs_stack[1:], next_obs.unsqueeze(0)), dim=0)  # Shape: [4, 84, 84]

            episode_return += reward

            if render:
                # Render the environment
                frame = env.render()
                if frame is not None:
                    plt.imshow(frame)
                    plt.axis('off')
                    plt.show(block=False)
                    plt.pause(0.001)
                    plt.clf()

        total_return += episode_return

        if verbose:
            print(f'Finished episode {i + 1} with a total return of {episode_return}')

    return total_return / n_episodes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['ALE/Pong-v5'], default='ALE/Pong-v5')
    parser.add_argument('--path', type=str, help='Path to stored DQN model.')
    parser.add_argument('--n_eval_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')
    parser.add_argument('--render', dest='render', action='store_true', help='Render the environment.')
    parser.add_argument('--save_video', dest='save_video', action='store_true', help='Save the episodes as video.')
    parser.set_defaults(render=False)
    parser.set_defaults(save_video=False)

    args = parser.parse_args()

    # Determine the render mode based on arguments
    if args.save_video or args.render:
        render_mode = 'rgb_array'
    else:
        render_mode = None

    # Initialize environment with the appropriate render_mode
    env = gym.make(args.env, render_mode=render_mode)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30, scale_obs=False)
    env = FrameStack(env, num_stack=4)

    if args.save_video:
        # Wrap the environment to save video
        env = gym.wrappers.RecordVideo(env, './video/', episode_trigger=lambda episode_id: True)
        # Suppress the warning about overwriting existing videos
        import logging
        logging.getLogger('gymnasium').setLevel(logging.ERROR)

    # Create an instance of DQN with the updated env_config
    dqn = DQN(env_config=env_config).to(device)

    # Sanitize the environment name and set the model path
    safe_env_name = args.env.replace('/', '_')
    if args.path is None:
        args.path = f'models/{safe_env_name}_best.pt'

    # Load model from provided path
    state_dict = torch.load(args.path, map_location=device)
    dqn.load_state_dict(state_dict)
    dqn.eval()

    # Evaluate the policy
    mean_return = evaluate_policy(dqn, env, args, args.n_eval_episodes, render=args.render, verbose=True)
    print(f'The policy got a mean return of {mean_return} over {args.n_eval_episodes} episodes.')
    env.close()
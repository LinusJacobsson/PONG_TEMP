import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward, terminated):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (obs, action, next_obs, reward, terminated)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.n_actions)



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


    def act(self, observation, step, exploit=False):
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * max(0, (self.anneal_length - step) / self.anneal_length)
        if exploit or random.random() > epsilon:
            with torch.no_grad():
                q_values = self.forward(observation)  # observation is already shaped correctly
                return torch.argmax(q_values).item()
        else:
            return random.randrange(self.n_actions)

            
def optimize(dqn, target_dqn, memory, optimizer, device):
    if len(memory) < dqn.batch_size:
        return

    obs, action, next_obs, reward, done = memory.sample(batch_size=dqn.batch_size)
    
    # Stack observations into batches
    obs_batch = torch.stack(obs).to(device)         # Shape: [batch_size, 4, 84, 84]
    next_obs_batch = torch.stack(next_obs).to(device)
    action_batch = torch.tensor(action, device=device, dtype=torch.int64).unsqueeze(1)
    reward_batch = torch.tensor(reward, device=device, dtype=torch.float32)
    done_batch = torch.tensor(done, device=device, dtype=torch.float32)

    q_values = dqn(obs_batch).gather(1, action_batch).squeeze()

    with torch.no_grad():
        max_next_q_values = target_dqn(next_obs_batch).max(1)[0]
        q_value_targets = reward_batch + (1 - done_batch) * (dqn.gamma * max_next_q_values)

    loss = F.mse_loss(q_values, q_value_targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


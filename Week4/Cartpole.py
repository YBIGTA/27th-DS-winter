import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import gymnasium as gym
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = torch.from_numpy(np.stack([x[0] for x in data], axis=0)).to(dtype=torch.float32)
        action = torch.from_numpy(np.stack([x[1] for x in data], axis=0)).to(dtype=torch.int32)
        reward = torch.from_numpy(np.stack([x[2] for x in data], axis=0)).to(dtype=torch.float32)
        next_state = None # TODO
        done = torch.from_numpy(np.stack([x[4] for x in data], axis=0)).to(dtype=torch.float32)
        return state, action, reward, next_state, done

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128) 
        self.fc2 = nn.Linear(128, 128) 
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x) :
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet()
        self.qnet_target = QNet()
        self.optimizer = optim.Adam(self.qnet.parameters(), self.lr)
        self.criterion = nn.MSELoss()

    def sync_qnet(self):
        self.qnet_target = None # TODO

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state).unsqueeze(0)
            qs = self.qnet(state) # (1, 2)
            return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        # 경험데이터 버퍼에 추가하고
        self.replay_buffer.add(state, action, reward, next_state, done)
        # 만약 배치사이즈만큼 안찼으면 업데이트 불가
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        # Q(s,a) 계산
        qs = self.qnet(state) # (B, 2)
        q = qs[torch.arange(self.batch_size), action] # (B,)
        # target 계산
        next_qs = None # (B, 2) # TODO
        next_q = next_qs.max(dim=1).values.detach() # (B,)
        target = reward + (1 - done) * self.gamma * next_q
        # 학습
        loss = self.criterion(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

episodes = 300

sync_interval = 20
env = gym.make('CartPole-v0', render_mode='rgb_array')
agent = DQNAgent()
reward_history = []

# 시뮬레이션 & 훈련루프
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = int(terminated or truncated)

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        pass # TODO

    reward_history.append(total_reward)

# 결과 plot
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.plot(reward_history)
plt.show()

# 결과 시각화
env = gym.make('CartPole-v0', render_mode='human')
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = agent.get_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated | truncated
    state = next_state
    total_reward += reward
    env.render()
print('Total Reward:', total_reward)

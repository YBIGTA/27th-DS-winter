import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from dezero import Model, optimizers
import dezero.functions as F
import dezero.layers as L
from collections import deque
import random

# 1. Actor 네트워크 정의 (Deterministic Policy)
class Actor(Model):
    def __init__(self, action_size, action_limit):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(64)
        self.l3 = L.Linear(action_size)
        self.action_limit = action_limit

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # TODO: tanh를 사용하여 -1~1 범위로 만들고 action_limit을 곱하세요.
        # 힌트: F.tanh(self.l3(x)) * self.action_limit
        x = None  # TODO
        return x

# 2. Critic 네트워크 정의 (Q-function)
class Critic(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(64)
        self.l3 = L.Linear(1)

    def forward(self, state, action):
        # state와 action을 concat
        # TODO: state와 action을 concat하세요.
        # 힌트: F.concat(state, action, axis=1)
        x = None  # TODO
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

# 3. Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def size(self):
        return len(self.buffer)

# 4. DDPG 에이전트 클래스
class DDPGAgent:
    def __init__(self, state_size, action_size, action_limit):
        self.gamma = 0.99
        self.lr_actor = 0.0001
        self.lr_critic = 0.001
        self.tau = 0.005  # soft update 계수
        self.action_size = action_size
        self.action_limit = action_limit

        # Actor와 Critic 네트워크 (main + target)
        self.actor = Actor(action_size, action_limit)
        self.actor_target = Actor(action_size, action_limit)
        self.critic = Critic()
        self.critic_target = Critic()

        # Target 네트워크 초기화 (main과 동일하게)
        self.update_target(tau=1.0)

        # Optimizer 설정
        self.actor_optimizer = optimizers.Adam(self.lr_actor)
        self.actor_optimizer.setup(self.actor)
        self.critic_optimizer = optimizers.Adam(self.lr_critic)
        self.critic_optimizer.setup(self.critic)

        # Replay buffer
        self.buffer = ReplayBuffer()

    def update_target(self, tau=None):
        """Target 네트워크를 soft update"""
        if tau is None:
            tau = self.tau

        # Actor target 업데이트
        for main_param, target_param in zip(self.actor.params(), self.actor_target.params()):
            target_param.data = tau * main_param.data + (1 - tau) * target_param.data

        # Critic target 업데이트
        for main_param, target_param in zip(self.critic.params(), self.critic_target.params()):
            target_param.data = tau * main_param.data + (1 - tau) * target_param.data

    def get_action(self, state, noise_scale=0.1):
        """Exploration을 위한 noise 추가"""
        state = state[np.newaxis, :]
        action = self.actor(state)
        action = action.data[0]

        # TODO: exploration을 위해 가우시안 노이즈를 추가하고 범위를 제한하세요.
        # 힌트: noise = np.random.randn(self.action_size) * noise_scale
        # 힌트: action_with_noise = np.clip(action + noise, -self.action_limit, self.action_limit)
        action_with_noise = None  # TODO
        return action_with_noise

    def update(self, batch_size=64):
        if self.buffer.size() < batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        # Critic 업데이트
        # TODO: target actor로 next action 계산
        # 힌트: self.actor_target(next_states)
        next_actions = None  # TODO

        # TODO: target critic으로 target Q-value 계산
        # 힌트: self.critic_target(next_states, next_actions)
        target_q = None  # TODO

        # TODO: TD target 계산
        # 힌트: rewards와 dones의 shape를 (batch_size, 1)로 맞춰야 합니다.
        # td_target = rewards.reshape(-1, 1) + self.gamma * target_q.data * (1 - dones.reshape(-1, 1))
        td_target = None  # TODO

        self.critic.cleargrads()
        # TODO: 현재 Q-value 계산
        # 힌트: self.critic(states, actions)
        current_q = None  # TODO

        # TODO: Critic loss 계산 (MSE)
        # 힌트: F.mean_squared_error(current_q, td_target)
        # 주의: td_target은 numpy array이지만 DeZero가 자동으로 Variable로 변환합니다.
        critic_loss = None  # TODO
        critic_loss.backward()
        self.critic_optimizer.update()

        # Actor 업데이트
        self.actor.cleargrads()
        # TODO: actor로 action 생성
        # 힌트: self.actor(states)
        actions_pred = None  # TODO

        # TODO: Q-value를 최대화하는 방향으로 Actor 업데이트
        # 힌트: -F.mean(self.critic(states, actions_pred))
        actor_loss = None  # TODO
        actor_loss.backward()
        self.actor_optimizer.update()

        # Target 네트워크 soft update
        self.update_target()

# 5. 학습 루프 및 시각화
episodes = 300
env = gym.make('Pendulum-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
action_limit = env.action_space.high[0]

agent = DDPGAgent(state_size, action_size, action_limit)
reward_history = []

for episode in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    noise_scale = max(0.1, 1.0 - episode / 200)  # 점차 noise 감소

    while not done:
        action = agent.get_action(state, noise_scale)
        next_state, reward, terminated, truncated, info = env.step([action])
        done = terminated or truncated

        # Buffer에 경험 저장
        agent.buffer.add(state, action, reward, next_state, done)

        # 충분한 경험이 쌓이면 학습
        if agent.buffer.size() > 1000:
            agent.update(batch_size=64)

        state = next_state
        total_reward += reward

    reward_history.append(total_reward)
    if episode % 20 == 0:
        print(f"episode: {episode}, total reward: {total_reward:.1f}, noise: {noise_scale:.3f}")

# 결과 시각화
target_reward = -200
plt.plot(reward_history)
plt.title("DDPG (Pendulum-v1)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.axhline(y=target_reward, color='r', linestyle='--', label='Target')
plt.legend()
plt.savefig("ddpg.png", dpi=150, bbox_inches="tight")
plt.show()

# [과제 답변란]
'''
질문 3: DDPG에서 Experience Replay와 Target Network를 사용하는 이유를 설명하시오.
답변:
'''

import numpy as np
import gymnasium as gym
from dezero import Model, optimizers
import dezero.functions as F
import dezero.layers as L
import matplotlib.pyplot as plt

# 1. 정책 네트워크 정의
class Policy(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x

# 2. 에이전트 정의
class Agent:
    def __init__(self):
        # TODO: 학습이 불안정하면 lr을 더 작게 시도해보세요. (힌트: 1e-4 ~ 3e-4)
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2
        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optimizers.Adam(self.lr).setup(self.pi)

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs = probs[0]
        # TODO: 확률 분포에 따라 action을 샘플링하세요.
        # 힌트: np.random.choice를 사용하고, p에는 probs.data를 넣습니다.
        action = None  # TODO
        return action, probs[action]

    def add(self, reward, prob):
        self.memory.append((reward, prob))

    def update(self):
        self.pi.cleargrads()
        G, loss = 0, 0
        
        # [Simple PG 특징] 에피소드 전체 수익 G를 한 번만 계산
        for reward, prob in reversed(self.memory):
            # TODO: 에피소드 전체 수익 G 계산
            # 힌트: G = reward + gamma * G
            G = None  # TODO

        # 모든 스텝에 동일한 G를 가중치로 적용
        for reward, prob in self.memory:
            # TODO: 각 스텝의 loss 누적
            # 힌트: -log(prob) * G
            loss += None  # TODO

        loss.backward()
        self.optimizer.update()
        self.memory = []

# 3. 학습 루프
episodes = 4000
env = gym.make('CartPole-v1')
agent = Agent()
reward_history = []

for episode in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.add(reward, prob)
        state = next_state
        total_reward += reward

    agent.update()
    reward_history.append(total_reward)
    if episode % 100 == 0:
        print(f"episode :{episode}, total reward : {total_reward:.1f}")

# 결과 시각화
target_reward = 120
plt.plot(reward_history)
plt.axhline(y=target_reward, color='r', linestyle='--', label='Target')
plt.title("Simple Policy Gradient")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.savefig("simple_pg.png", dpi=150, bbox_inches="tight")
plt.show()
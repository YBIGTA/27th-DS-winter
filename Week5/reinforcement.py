import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from dezero import Model, optimizers
import dezero.functions as F
import dezero.layers as L

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

# 2. REINFORCE 에이전트 클래스
class Agent:
    def __init__(self):
        # TODO: 성능이 불안정하면 lr을 더 작게 시도해보세요. (힌트: 1e-4 ~ 3e-4)
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.pi)

    def get_action(self, state):
        state = state[np.newaxis, :]  # 배치 축 추가
        probs = self.pi(state)
        probs = probs[0]
        # TODO: 확률 분포에 따라 action을 샘플링하세요.
        # 힌트: np.random.choice를 사용하고, p에는 probs.data를 넣습니다.
        action = None  # TODO
        return action, probs[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        self.pi.cleargrads()

        G, loss = 0, 0
        # [핵심] 역순으로 보상을 따라가며 각 시점 t의 G_t(Reward-to-go)를 계산
        for reward, prob in reversed(self.memory):
            # TODO: reward-to-go 계산
            # 힌트: G = reward + gamma * G
            G = None  # TODO
            # TODO: 각 시점의 loss 누적
            # 힌트: -log(prob) * G
            loss += None  # TODO

        loss.backward()
        self.optimizer.update()
        self.memory = []

# 3. 학습 루프 및 시각화
episodes = 2000
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
target_reward = 350
plt.plot(reward_history)
plt.title("REINFORCE (Reward-to-go)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.axhline(y=target_reward, color='r', linestyle='--', label='Target')
plt.legend()
plt.savefig("reinforce.png", dpi=150, bbox_inches="tight")
plt.show()

# [과제 답변란]
'''
질문 1: Simple PG와 REINFORCE의 결과를 분석해서 적으시오.
답변: 

질문 2: 왜 REINFORCE 방식이 Simple PG보다 성능이 더 안정적인지 설명하시오.
답변: 
'''
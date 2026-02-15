import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from dezero import Model, optimizers
import dezero.functions as F
import dezero.layers as L

# 1. Actor 네트워크 정의 (정책 네트워크)
class Actor(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x

# 2. Critic 네트워크 정의 (가치 네트워크)
class Critic(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

# 3. A2C 에이전트 클래스
class A2CAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_actor = 0.0002
        self.lr_critic = 0.0005
        self.action_size = 2

        # Actor와 Critic 네트워크 생성
        self.actor = Actor(self.action_size)
        self.critic = Critic()

        # 각각의 optimizer 설정
        self.actor_optimizer = optimizers.Adam(self.lr_actor)
        self.actor_optimizer.setup(self.actor)

        self.critic_optimizer = optimizers.Adam(self.lr_critic)
        self.critic_optimizer.setup(self.critic)

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.actor(state)
        probs = probs[0]
        # TODO: 확률 분포에 따라 action을 샘플링하세요.
        # 힌트: np.random.choice(self.action_size, p=probs.data)
        # 주의: probs는 DeZero Variable이므로 .data로 numpy array를 꺼내야 합니다.
        action = None  # TODO
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        # state와 next_state를 배치 형태로 변환
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        # TODO: Critic으로 현재 상태의 가치 V(s)를 계산하세요.
        # 힌트: self.critic(state)[0, 0]로 스칼라 값을 추출합니다.
        value = None  # TODO

        # TODO: Critic으로 다음 상태의 가치 V(s')를 계산하세요.
        # 힌트: next_value = 0 if done else self.critic(next_state)[0, 0]
        next_value = None  # TODO

        # TODO: TD 타겟을 계산하세요.
        # 힌트: target = reward + self.gamma * next_value
        target = None  # TODO

        # TODO: Advantage를 계산하세요.
        # 힌트: advantage = target - value
        advantage = None  # TODO

        # Critic 업데이트 (가치 함수 학습)
        self.critic.cleargrads()
        # TODO: Critic의 loss를 계산하세요.
        # 힌트: MSE loss = F.mean_squared_error(value, target)
        critic_loss = None  # TODO
        critic_loss.backward()
        self.critic_optimizer.update()

        # Actor 업데이트 (정책 학습)
        self.actor.cleargrads()
        # TODO: Actor의 loss를 계산하세요.
        # 힌트: -log(action_prob) * advantage.data (advantage는 gradient 끊어야 함)
        actor_loss = None  # TODO
        actor_loss.backward()
        self.actor_optimizer.update()

# 4. 학습 루프 및 시각화
episodes = 2000
env = gym.make('CartPole-v1')
agent = A2CAgent()
reward_history = []

for episode in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # A2C는 매 step마다 업데이트
        agent.update(state, prob, reward, next_state, done)

        state = next_state
        total_reward += reward

    reward_history.append(total_reward)
    if episode % 100 == 0:
        print(f"episode: {episode}, total reward: {total_reward:.1f}")

# 결과 시각화
target_reward = 400
plt.plot(reward_history)
plt.title("A2C (Advantage Actor-Critic)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.axhline(y=target_reward, color='r', linestyle='--', label='Target')
plt.legend()
plt.savefig("a2c.png", dpi=150, bbox_inches="tight")
plt.show()

# [과제 답변란]
'''
질문 1: A2C에서 Advantage 함수의 역할과 장점을 설명하시오.
답변:

질문 2: Actor와 Critic이 각각 무엇을 학습하는지 설명하시오.
답변:
'''

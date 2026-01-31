# 강화학습 과제 4주차

발제자: 정진욱

이 과제는 **TODO가 표시된 부분만 구현**하면 됩니다. (그 외 코드는 수정하지 않아도 됩니다.)

### 제출 기한

**2월 9일 (월) 11시 59분**까지

---

## CartPole DQN Agent 만들기

대상 파일: `CartPole.py`

### 구현할 것 (TODO)

#### Train_Q.py (Q-learning)
- `next_state` 배치처리 (30줄)
- `sync_qnet` 구현 (62줄)
- `target` 구현에 필요한 `next_qs` 계산 (83줄)
- `qnet_target` 갱신 (115줄)

### 실행
```bash
# DQN 학습 및 결과확인
python Cartpole.py
```


## 제출 방법

1. `Cartpole.py`의 TODO 부분을 구현하고 학습
2. 출력되는 pyplot 그래프 Week4 폴더에 저장 (후반에 Total reward가 평균 150 이상 나와야합니다)


## 의존성
- `requirements.txt` 이 바뀌었으므로 실행하기전 다시 설치해주세요!

```bash
# on root dir
pip install -r requirements.txt
```


## 참고 자료
- 밑시딥4권 chapter8

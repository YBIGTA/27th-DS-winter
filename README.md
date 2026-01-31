# 27th-DS-winter

27기 DS 겨울 스터디 과제 저장소입니다.

[과제 1 명세](/Week1/README.md)
[과제 3 명세](/Week3/README.md)
[과제 4 명세](/Week4/README.md)

## 준비물

- Miniconda 또는 Anaconda 설치
- 터미널에서 `conda` 명령이 실행 가능해야 합니다 (Windows PowerShell/CMD)

아래 명령들은 **레포지토리 루트(현재 폴더)** 에서 실행한다고 가정합니다.

## Conda 환경 만들기 (Python 3.11)

환경 이름은 원하는 것으로 바꿔도 됩니다. (예: `ds-winter-311`)

```bash
conda create -n ds-winter-311 python=3.11 -y
conda activate ds-winter-311
```

현재 Python 버전 확인:

```bash
python --version
```

## 의존성 설치

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 실행 방법 (Week 1)

예시:

```bash
python Week1/mab_assignment.py
python Week1/frozen_lake_linear_assignment.py
```

## 자주 겪는 문제

### PowerShell에서 `conda activate`가 안 돼요

아래를 **한 번만** 실행한 뒤, 터미널을 재시작하세요.

```bash
conda init powershell
```

### `pip install -r requirements.txt`가 실패해요

- 먼저 `conda activate ds-winter-311` 로 환경이 활성화되어 있는지 확인하세요.
- 회사/학교 네트워크라면 프록시/인증서 문제일 수 있습니다.

# 문의

팀장단이나, 각 과제 발제자에게 문의해주시면 감사하겠습니다!
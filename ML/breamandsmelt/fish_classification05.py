"""
농어 무게 예측 프로그램 - 다항 회귀 모델 비교

이 프로그램은 농어의 길이를 이용해 무게를 예측하는 여러 회귀 모델을 구현하고 비교합니다.
K-최근접 이웃(KNN) 회귀, 선형 회귀, 다항 회귀 모델을 구현하여
각 모델의 특성과 성능을 비교 분석합니다.

주요 특징:
1. 다양한 회귀 모델 구현
   - K-최근접 이웃 회귀
   - 단순 선형 회귀
   - 다항 회귀 (2차 다항식)
2. 특성 공학: 다항식 특성 변환
3. 모델 성능 비교 및 분석

작성일: 2024-12-26
"""

# 필요한 라이브러리 임포트
import numpy as np                              # 배열 및 수치 계산을 위해 사용
from sklearn.model_selection import train_test_split    # 데이터 분할(학습/테스트)에 사용
from sklearn.neighbors import KNeighborsRegressor       # K-최근접 이웃(KNN) 회귀 모델
from sklearn.linear_model import LinearRegression       # 선형 회귀(Linear Regression) 모델
import matplotlib.pyplot as plt                         # 그래프 시각화를 위한 라이브러리

# 데이터 준비
# 농어 56마리의 길이 데이터 (단위: cm)
# 길이 범위: 8.4cm ~ 44.0cm
# 데이터는 크기순으로 정렬되어 있음
perch_length = np.array([
    8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
    21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
    23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
    27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
    39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
    44.0
])

# 농어 56마리의 무게 데이터 (단위: g)
# 무게 범위: 5.9g ~ 1100.0g
# 길이가 증가할수록 무게가 비선형적으로 증가하는 특성을 보임
perch_weight = np.array([
    5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
    115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
    150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
    218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
    556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
    850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
    1000.0
])

# 데이터 분할
# 전체 데이터를 학습용(75%)과 테스트용(25%)으로 나눔
# random_state=42로 시드를 고정하여 실험의 재현성 확보
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42
)

# 데이터 전처리
# scikit-learn의 모델은 2차원 입력을 요구하므로 reshape를 통해 차원 변환
# reshape(-1, 1): 샘플 개수는 자동 계산(-1), 특성 개수는 1개
train_input = train_input.reshape(-1, 1)  # 학습 입력 데이터 변환
test_input = test_input.reshape(-1, 1)    # 테스트 입력 데이터 변환

# 모델 1: K-최근접 이웃 회귀
# 이웃 개수(k)를 3으로 설정하여 더 유연한 모델 생성
knr = KNeighborsRegressor(n_neighbors=3)  # KNN 모델 초기화
knr.fit(train_input, train_target)        # 모델 학습

# 모델 2: 단순 선형 회귀
# 하나의 특성(길이)을 사용한 1차 선형 모델
lr = LinearRegression()  # 선형 회귀 모델 초기화
lr.fit(train_input, train_target)  # 모델 학습

# 특성 공학: 다항식 특성 추가
# 길이의 제곱항을 추가하여 비선형성을 표현
# column_stack: 여러 특성을 열 방향으로 결합
train_poly = np.column_stack((train_input ** 2, train_input))  # [길이², 길이]
test_poly = np.column_stack((test_input ** 2, test_input))    # [길이², 길이]

# 모델 3: 다항 회귀 (2차 다항식)
# 선형 회귀 모델을 다항 특성에 적용
# y = ax² + bx + c 형태의 2차 함수로 학습
lr = LinearRegression()  # 선형 회귀 모델 초기화
lr.fit(train_poly, train_target)  # 다항 특성으로 모델 학습

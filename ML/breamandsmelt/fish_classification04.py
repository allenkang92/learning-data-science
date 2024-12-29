"""
농어 무게 예측 프로그램 - K-최근접 이웃 회귀 모델

이 프로그램은 농어의 길이를 입력으로 받아 무게를 예측하는 회귀 모델을 구현합니다.
이전 버전들과 달리 분류가 아닌 회귀 문제를 다루며, 연속적인 값을 예측합니다.
K-최근접 이웃 회귀(KNN Regression)를 사용하여 농어의 무게를 예측하고,
여러 가지 성능 지표(R² 점수, MAE)를 통해 모델의 성능을 평가합니다.

주요 특징:
1. 회귀 문제: 농어의 길이로 무게를 예측 (연속값 예측)
2. 데이터 시각화: 산점도를 통한 데이터 분포 확인
3. 모델 평가: R² 점수와 평균 절댓값 오차(MAE) 사용
4. 하이퍼파라미터 튜닝: 이웃 개수 조정을 통한 성능 개선

작성일: 2024-12-26
"""

# 필요한 라이브러리 임포트
import numpy as np                  # 수치 계산용 라이브러리 (배열, 난수, 선형대수 등)
import matplotlib.pyplot as plt     # 그래프 시각화를 위한 라이브러리
from sklearn.model_selection import train_test_split    # 데이터 분할(학습/테스트) 함수
from sklearn.neighbors import KNeighborsRegressor       # K-최근접 이웃(KNN) 회귀 모델
from sklearn.metrics import mean_absolute_error         # 평균 절댓값 오차(MAE) 평가 지표

# 데이터 준비
# 농어(perch) 56마리의 길이 데이터 (단위: cm)
# 길이 범위: 8.4cm ~ 44.0cm
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
perch_weight = np.array([
    5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
    115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
    150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
    218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
    556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
    850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
    1000.0
])

# 데이터 탐색을 위한 시각화
# 산점도를 통해 길이와 무게 사이의 관계 확인
plt.scatter(perch_length, perch_weight)
plt.xlabel('length (cm)')  # x축: 농어 길이 (cm)
plt.ylabel('weight (g)')   # y축: 농어 무게 (g)
plt.title('Perch Length vs Weight')  # 그래프 제목
plt.show()                 # 그래프 표시

# 데이터 분할
# 전체 데이터를 학습용(75%)과 테스트용(25%)으로 나눔
train_input, test_input, train_target, test_target = train_test_split(
    perch_length,
    perch_weight,
    random_state=42  # 실험 재현성을 위한 난수 시드 설정
)

# 데이터 전처리
# KNN 모델은 2차원 입력을 요구하므로 1차원 배열을 2차원으로 변환
# reshape(-1, 1): 샘플 개수는 자동 계산(-1), 특성 개수는 1개
train_input = train_input.reshape(-1, 1)  # 학습 입력 데이터 변환
test_input = test_input.reshape(-1, 1)    # 테스트 입력 데이터 변환

# 모델 학습 (기본 설정)
# 기본 이웃 개수(n_neighbors=5)로 KNN 회귀 모델 생성 및 학습
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)

# 모델 평가 1: 결정계수(R²) 점수
# R² 점수는 1에 가까울수록 좋음 (1: 완벽한 예측, 0: 평균으로만 예측)
print("테스트 세트 R² 점수:", knr.score(test_input, test_target))

# 모델 평가 2: 평균 절댓값 오차(MAE)
# 실제 무게와 예측 무게의 차이를 평균한 값
test_prediction = knr.predict(test_input)  # 테스트 세트 예측
mae = mean_absolute_error(test_target, test_prediction)  # MAE 계산
print("평균 절댓값 오차:", mae, "g")  # MAE 출력 (단위: g)

# 과대적합 확인
# 훈련 세트의 점수가 테스트 세트보다 너무 높으면 과대적합
print("훈련 세트 R² 점수:", knr.score(train_input, train_target))

# 모델 개선
# 이웃 개수를 3으로 줄여 더 유연한 모델 생성
knr.n_neighbors = 3  # 이웃 개수 조정
knr.fit(train_input, train_target)  # 모델 재학습

# 개선된 모델 평가
# 훈련 세트와 테스트 세트의 점수를 비교하여 성능 향상 확인
print("\n[이웃 개수 = 3일 때]")
print("훈련 세트 R² 점수:", knr.score(train_input, train_target))
print("테스트 세트 R² 점수:", knr.score(test_input, test_target))

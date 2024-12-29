"""
농어 무게 예측 프로그램 - 고차원 다항 회귀와 규제

이 프로그램은 농어의 길이와 높이, 두께를 이용해 무게를 예측하는 다양한 회귀 모델을 구현하고 비교합니다.
5차 다항식까지 특성을 확장하고, 과대적합을 방지하기 위해 규제(regularization)를 적용합니다.

주요 특징:
1. 다항 특성 확장 (5차)
2. 특성 스케일링
3. 규제가 있는 회귀 모델 비교
   - 라쏘 회귀 (L1 규제)
   - 릿지 회귀 (L2 규제)
   - 기본 선형 회귀 (규제 없음)

작성일: 2024-12-26
"""

# 필요한 라이브러리 임포트
import pandas as pd                   # 데이터 분석용 라이브러리
import numpy as np                    # 수치 계산용 라이브러리
from sklearn.model_selection import train_test_split  # 데이터 분할(학습/테스트)에 사용
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # 다항 특성, 표준 스케일링
from sklearn.linear_model import LinearRegression, Ridge, Lasso       # 선형 회귀, 릿지, 라쏘
import matplotlib.pyplot as plt       # 데이터 시각화를 위한 라이브러리

# 데이터 준비
# CSV 파일에서 농어의 길이, 높이, 두께 데이터를 로드
# bit.ly/perch_csv_data에는 농어의 3가지 측정값이 포함되어 있음
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()           # 판다스 DataFrame을 넘파이 배열로 변환

# 농어 56마리의 무게 데이터 (단위: g)
# 무게 범위: 5.9g ~ 1100.0g
# 길이, 높이, 두께가 증가할수록 무게가 비선형적으로 증가하는 특성을 보임
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
    perch_full, perch_weight, random_state=42
)

# 특성 공학 1: 다항식 특성 확장
# 원본 특성(길이, 높이, 두께)을 5차항까지 확장
# 예: x₁, x₂, x₃ → x₁², x₁x₂, x₁x₃, x₂², x₂x₃, x₃², x₁³, ...
poly = PolynomialFeatures(degree=5, include_bias=False)  # 5차 다항식으로 변환
poly.fit(train_input)                      # 다항 변환 규칙 학습
train_poly = poly.transform(train_input)   # 훈련 세트 변환
test_poly = poly.transform(test_input)     # 테스트 세트 변환

# 특성 공학 2: 표준화 스케일링
# 각 특성의 평균을 0, 표준편차를 1로 변환
# 특성들의 스케일을 통일하여 모델의 성능과 안정성 향상
ss = StandardScaler()
ss.fit(train_poly)                # 스케일 파라미터 계산
train_scaled = ss.transform(train_poly)  # 훈련 세트 스케일 적용
test_scaled = ss.transform(test_poly)    # 테스트 세트 스케일 적용

# 모델 1: 라쏘 회귀 (L1 규제)
# alpha=10으로 강한 규제를 적용
# L1 규제는 불필요한 특성의 계수를 0으로 만들어 특성 선택 효과
lasso = Lasso(alpha=10)  # 라쏘 모델 초기화
lasso.fit(train_scaled, train_target)  # 모델 학습
print("라쏘 회귀 훈련 세트 점수:", lasso.score(train_scaled, train_target))
print("라쏘 회귀 테스트 세트 점수:", lasso.score(test_scaled, test_target))
print("라쏘 회귀 사용한 특성 개수:", np.sum(lasso.coef_ == 0))  # 0이 된 계수의 개수

# 모델 2: 릿지 회귀 (L2 규제)
# alpha=0.1로 약한 규제를 적용
# L2 규제는 모든 특성을 조금씩 사용하면서 과대적합 방지
ridge = Ridge(alpha=0.1)  # 릿지 모델 초기화
ridge.fit(train_scaled, train_target)  # 모델 학습
print("릿지 회귀 훈련 세트 점수:", ridge.score(train_scaled, train_target))
print("릿지 회귀 테스트 세트 점수:", ridge.score(test_scaled, test_target))

# 모델 3: 기본 선형 회귀 (규제 없음)
# 다항 변환만 적용하고 스케일링은 제외
# 규제가 없어 과대적합이 발생할 수 있음
lr = LinearRegression()  # 선형 회귀 모델 초기화
lr.fit(train_poly, train_target)  # 모델 학습
print("선형 회귀 훈련 세트 점수:", lr.score(train_poly, train_target))
print("선형 회귀 테스트 세트 점수:", lr.score(test_poly, test_target))
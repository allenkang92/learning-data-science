"""
물고기 종 분류 프로그램 - 다중 분류와 로지스틱 회귀

이 프로그램은 물고기의 물리적 특성(무게, 길이, 대각선 길이, 높이, 너비)을 이용하여
여러 종의 물고기를 분류하는 모델을 구현하고 비교합니다.
K-최근접 이웃과 로지스틱 회귀를 사용하여 이진 분류와 다중 분류를 수행합니다.

주요 특징:
1. 다양한 물고기 특성을 이용한 분류
2. 이진 분류 (도미/빙어)와 다중 분류 구현
3. 두 가지 분류 모델 비교
   - K-최근접 이웃 분류기
   - 로지스틱 회귀
4. 확률 기반의 예측과 결정 경계 분석

작성일: 2024-12-26
"""

# 필요한 라이브러리 임포트
import pandas as pd                                     # 데이터프레임 처리
import numpy as np                                      # 수치 연산
from sklearn.model_selection import train_test_split    # 데이터 분할
from sklearn.preprocessing import StandardScaler        # 특성 스케일링
from sklearn.neighbors import KNeighborsClassifier      # K-최근접 이웃 분류
from sklearn.linear_model import LogisticRegression     # 로지스틱 회귀
from scipy.special import expit, softmax                # 시그모이드, 소프트맥스 함수
import matplotlib.pyplot as plt                         # 데이터 시각화


# 1. 데이터 로드 및 전처리
# CSV 파일에서 물고기 데이터셋 로드
# 특성: Weight(무게), Length(길이), Diagonal(대각선), Height(높이), Width(너비)
# 타겟: Species(물고기 종)
fish = pd.read_csv('https://bit.ly/fish_csv_data') 
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

# 2. 훈련/테스트 데이터 분할
# 전체 데이터를 학습용(75%)과 테스트용(25%)으로 나눔
# random_state=42로 시드를 고정하여 실험의 재현성 확보
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42
)

# 3. 데이터 스케일링
# 각 특성의 평균을 0, 표준편차를 1로 변환
# 특성들의 스케일을 통일하여 모델의 성능과 안정성 향상
ss = StandardScaler()
ss.fit(train_input)                # 스케일 파라미터 계산
train_scaled = ss.transform(train_input)  # 훈련 세트 스케일 적용
test_scaled = ss.transform(test_input)    # 테스트 세트 스케일 적용

# 4. K-최근접 이웃 모델 훈련 및 평가
# 이웃 개수 k=3으로 설정하여 분류 수행
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
# 아래는 모델 평가 및 예측 관련 코드 (주석 해제하여 사용)
# print(kn.score(train_scaled, train_target))          # 훈련 세트 점수
# print(kn.score(test_scaled, test_target))           # 테스트 세트 점수
# print(kn.classes_)                                  # 타겟 클래스 목록
# print(kn.predict(test_scaled[:5]))                  # 처음 5개 샘플 예측
# proba = kn.predict_proba(test_scaled[:5])          # 예측 확률
# print(np.round(proba, decimals=4))                 # 확률 반올림하여 출력
# distances, indexes = kn.kneighbors(test_scaled[3:4])  # 4번째 샘플의 이웃 찾기
# print(train_target[indexes])                       # 이웃 샘플들의 타겟값


# 5. 로지스틱 회귀 모델 훈련 및 평가 (도미/빙어 이진 분류)
# 도미와 빙어 데이터만 선택하여 이진 분류 수행
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

# 로지스틱 회귀 모델 학습 (이진 분류)
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
# 아래는 이진 분류 모델 평가 및 예측 관련 코드
# print(lr.predict(train_bream_smelt[:5]))           # 처음 5개 샘플 예측
# print(lr.predict_proba(train_bream_smelt[:5]))     # 예측 확률
# print(lr.classes_)                                 # 타겟 클래스 목록
# print(lr.coef_, lr.intercept_)                     # 모델 계수와 절편
# decisions = lr.decision_function(train_bream_smelt[:5])  # 결정 함수 값
# print(decisions)                                   # 결정 함수 출력
# print(expit(decisions))                           # 시그모이드 함수 적용


# 6. 로지스틱 회귀 모델 훈련 및 평가 (다중 분류)
# C=20: 규제 강도를 낮춤 (과대적합 위험 증가, 복잡한 결정 경계 허용)
# max_iter=1000: 최대 반복 횟수 증가 (모델이 수렴하지 않을 경우 대비)
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
# 아래는 다중 분류 모델 평가 및 예측 관련 코드
# print(lr.score(train_scaled, train_target))        # 훈련 세트 점수
# print(lr.score(test_scaled, test_target))         # 테스트 세트 점수
# print(lr.predict(test_scaled[:5]))                # 처음 5개 샘플 예측
# proba = lr.predict_proba(test_scaled[:5])        # 예측 확률
# print(np.round(proba, decimals=3))               # 확률 반올림하여 출력
# print(lr.classes_)                               # 타겟 클래스 목록
# print(lr.coef_.shape, lr.intercept_.shape)       # 모델 파라미터 형태
# decision = lr.decision_function(test_scaled[:5])  # 결정 함수 값
# print(np.round(decision, decimals=2))            # 결정 함수 출력
# proba = softmax(decision, axis=1)                # 소프트맥스 함수 적용
# print(np.round(proba, decimals=3))              # 확률 반올림하여 출력


# 시각화 예제: 시그모이드 함수 그래프
# z = np.arange(-5, 5, 0.1)                        # -5에서 5까지 0.1 간격
# phi = 1 / (1 + np.exp(-z))                      # 시그모이드 함수
# plt.plot(z, phi)                                # 그래프 그리기
# plt.xlabel('z')                                 # x축 레이블
# plt.ylabel('phi')                               # y축 레이블
# plt.show()                                      # 그래프 표시
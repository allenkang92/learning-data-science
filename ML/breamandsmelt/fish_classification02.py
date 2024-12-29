"""
도미와 빙어 분류 프로그램 - 데이터 분할과 모델 평가

이 프로그램은 도미와 빙어의 데이터를 학습용과 테스트용으로 나누어
K-최근접 이웃(KNN) 알고리즘의 성능을 평가합니다.
데이터를 무작위로 섞어 편향되지 않은 평가를 수행합니다.

작성일: 2024-12-26
"""

# 필요한 라이브러리 임포트
import numpy as np  # 과학 계산용 라이브러리. 배열, 난수 생성, 선형대수 등에 사용
from sklearn.neighbors import KNeighborsClassifier  # 머신러닝 알고리즘 중 K-최근접 이웃(KNN)을 사용하기 위한 클래스
import matplotlib.pyplot as plt  # 2D 그래프 시각화를 위한 라이브러리

# 데이터 준비
# 도미 35마리와 빙어 14마리의 길이와 무게 데이터
# 앞쪽 35개는 도미, 뒤쪽 14개는 빙어 데이터입니다.
fish_length = [
    25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
    31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
    35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
    10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0
]
fish_weight = [
    242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
    500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
    700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
    7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9
]

# 데이터 전처리
# 길이와 무게를 쌍으로 묶어 2차원 리스트 생성
fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
# 도미는 1, 빙어는 0으로 라벨링한 타깃 데이터 생성
fish_target = [1]*35 + [0]*14

# NumPy 배열로 변환
# NumPy 배열은 행렬 연산이 가능하고 처리 속도가 빠름
input_arr = np.array(fish_data)    # 입력 데이터를 NumPy 배열로 변환
target_arr = np.array(fish_target)  # 타깃 데이터를 NumPy 배열로 변환

# 데이터 섞기
# 항상 같은 결과를 얻기 위해 난수 시드를 42로 고정
np.random.seed(42)
# 0부터 48까지의 인덱스 배열을 생성하고 무작위로 섞음
index = np.arange(49)
np.random.shuffle(index)

# 데이터 분할
# 전체 49개 데이터 중 앞쪽 35개는 학습용, 뒤쪽 14개는 테스트용으로 분할
train_input = input_arr[index[:35]]    # 학습용 입력 데이터
train_target = target_arr[index[:35]]  # 학습용 타깃 데이터
test_input = input_arr[index[35:]]     # 테스트용 입력 데이터
test_target = target_arr[index[35:]]   # 테스트용 타깃 데이터

# 모델 학습
# K-최근접 이웃 분류기를 생성하고 학습 데이터로 훈련
kn = KNeighborsClassifier()  # 기본 하이퍼파라미터로 모델 초기화
kn.fit(train_input, train_target)  # 모델 학습

# 모델 평가
# 테스트 데이터로 모델의 성능을 평가
score = kn.score(test_input, test_target)  # 모델의 정확도 계산
print("모델 정확도:", score)  # 정확도 출력 (1.0이면 100% 정확)

# 예측 결과 확인
# 테스트 데이터에 대한 예측과 실제 값을 비교
predictions = kn.predict(test_input)  # 테스트 데이터로 예측 수행
print("예측 결과:", predictions)  # 모델이 예측한 값
print("실제 결과:", test_target)  # 실제 정답 값

# 데이터 시각화
# 학습 데이터와 테스트 데이터의 분포를 2차원 평면에 표시
plt.scatter(train_input[:, 0], train_input[:, 1], label='Train')  # 학습 데이터 (파란점)
plt.scatter(test_input[:, 0], test_input[:, 1], label='Test')    # 테스트 데이터 (주황점)
plt.xlabel('length')  # x축: 물고기 길이
plt.ylabel('weight')  # y축: 물고기 무게
plt.grid()           # 격자 표시로 가독성 향상
plt.legend()         # 범례 표시
plt.show()           # 그래프 출력

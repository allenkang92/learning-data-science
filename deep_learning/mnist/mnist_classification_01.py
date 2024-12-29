"""
Fashion MNIST 이미지 분류 - 기본 신경망
이 스크립트는 Fashion MNIST 데이터셋을 사용하여 기본적인 신경망 분류기를 구현합니다.

데이터셋:
    Fashion MNIST: 10개 카테고리의 패션 아이템 흑백 이미지 (28x28 픽셀)
    - 60,000개의 훈련 이미지
    - 10,000개의 테스트 이미지

모델:
    - 단일 Dense 층을 사용한 기본 신경망
    - 입력: 784 (28x28 픽셀)
    - 출력: 10 (클래스 수)

작성일: 2024-12-29
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import SGDClassifier

# 1. 데이터 로드
# keras.datasets.fashion_mnist.load_data(): 
# - 이미지와 레이블을 훈련/테스트 세트로 나누어 반환
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# 데이터 크기 확인
# - train_input: (60000, 28, 28) - 60000개의 28x28 이미지
# - train_target: (60000,) - 60000개의 레이블
print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)

# 2. 데이터 시각화
# 처음 10개 이미지를 시각화하여 데이터 확인
fig, axs = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')  # gray_r: 반전된 흑백
    axs[i].axis('off')
plt.show()

# 레이블 확인
print([train_target[i] for i in range(10)])  # 처음 10개 이미지의 레이블
print(np.unique(train_target, return_counts=True))  # 각 클래스별 이미지 수

# 3. 데이터 전처리
# 픽셀값을 0-1 사이로 정규화
train_scaled = train_input / 255.0
# 2차원 이미지를 1차원 벡터로 변환 (28x28 -> 784)
train_scaled = train_scaled.reshape(-1, 28 * 28)

# 4. SGDClassifier로 성능 비교 (선택 사항)
# 확률적 경사 하강법 기반 분류기로 기준 성능 측정
sc = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))

# 5. 훈련/검증 데이터 분리
# 훈련 데이터의 20%를 검증 세트로 분리
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

print(train_scaled.shape, train_target.shape)
print(val_scaled.shape, val_target.shape)

# 6. Keras 모델 생성
# Dense 층 생성
# - 10개의 뉴런 (출력 클래스 수)
# - softmax 활성화 함수
# - 784개의 입력 특성
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
model = keras.Sequential([dense])

# 모델 컴파일
# - sparse_categorical_crossentropy: 정수 레이블용 크로스 엔트로피
# - accuracy: 정확도 측정
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 7. 모델 훈련 및 평가
print(train_target[:10])  # 처음 10개 샘플의 레이블 확인
# 5 에포크 동안 모델 훈련
model.fit(train_scaled, train_target, epochs=5)
# 검증 세트로 모델 평가
model.evaluate(val_scaled, val_target)
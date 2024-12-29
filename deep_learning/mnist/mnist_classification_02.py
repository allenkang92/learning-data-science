"""
Fashion MNIST 이미지 분류 - 다층 신경망
이 스크립트는 Fashion MNIST 데이터셋을 사용하여 다층 신경망 분류기를 구현합니다.

데이터셋:
    Fashion MNIST: 10개 카테고리의 패션 아이템 흑백 이미지 (28x28 픽셀)
    - 60,000개의 훈련 이미지
    - 10,000개의 테스트 이미지

모델 구조:
    1. Flatten 층: 2D 이미지를 1D 벡터로 변환
    2. Dense 층 (100 뉴런, ReLU 활성화)
    3. Dense 층 (10 뉴런, Softmax 활성화)

작성일: 2024-12-29
"""

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# 1. 데이터 로드 및 전처리
# Fashion MNIST 데이터셋 로드
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# 픽셀값을 0-1 사이로 정규화
train_scaled = train_input / 255.0

# 훈련 데이터의 20%를 검증 세트로 분리
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

# 2. 모델 정의
model = keras.Sequential()
# Flatten: 28x28 이미지를 784 크기의 1차원 벡터로 변환
model.add(keras.layers.Flatten(input_shape=(28, 28)))
# Dense: 100개의 뉴런을 가진 은닉층, ReLU 활성화 함수 사용
model.add(keras.layers.Dense(100, activation='relu'))
# Dense: 10개의 뉴런을 가진 출력층, Softmax 활성화 함수 사용
model.add(keras.layers.Dense(10, activation='softmax'))

# 3. 모델 컴파일
# - optimizer: Adam 최적화 알고리즘
# - loss: 희소 범주형 크로스 엔트로피 (정수 레이블용)
# - metrics: 정확도 측정
model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# 4. 모델 학습
# 5 에포크 동안 훈련 데이터로 모델 학습
model.fit(train_scaled, train_target, epochs=5)

# 5. 모델 평가
# 검증 세트로 모델의 성능 평가
loss, accuracy = model.evaluate(val_scaled, val_target)
print("Loss:", loss)
print("Accuracy:", accuracy)
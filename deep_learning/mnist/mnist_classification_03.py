"""
Fashion MNIST 이미지 분류 - 드롭아웃과 콜백 적용
이 스크립트는 Fashion MNIST 데이터셋에 드롭아웃과 다양한 콜백을 적용한 개선된 신경망을 구현합니다.

데이터셋:
    Fashion MNIST: 10개 카테고리의 패션 아이템 흑백 이미지 (28x28 픽셀)

모델 구조:
    1. Flatten 층: 2D -> 1D 변환
    2. Dense 층 (100 뉴런, ReLU)
    3. Dropout 층 (0.3 비율)
    4. Dense 층 (10 뉴런, Softmax)

주요 기능:
    - 드롭아웃을 통한 과대적합 방지
    - ModelCheckpoint로 최적 모델 저장
    - EarlyStopping으로 조기 종료
    - 학습 과정 시각화

작성일: 2024-12-29
"""

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 로드 및 전처리
# Fashion MNIST 데이터셋 로드
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
# 픽셀값 정규화
train_scaled = train_input / 255.0
# 훈련/검증 데이터 분리
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

# 2. 모델 정의
def model_fn(a_layer=None):
    """
    신경망 모델을 생성하는 함수
    
    Parameters:
        a_layer: 추가할 층 (예: Dropout)
    
    Returns:
        keras.Sequential: 구성된 신경망 모델
    """
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))  # 입력층
    model.add(keras.layers.Dense(100, activation="relu"))  # 은닉층
    if a_layer:
        model.add(a_layer)  # 추가 층 (드롭아웃 등)
    model.add(keras.layers.Dense(10, activation="softmax"))  # 출력층
    return model

# 3. 모델 컴파일 및 훈련
# 드롭아웃 층이 포함된 모델 생성
model = model_fn(keras.layers.Dropout(0.3))
# 모델 컴파일
model.compile(
    optimizer="adam",  # Adam 최적화
    loss="sparse_categorical_crossentropy",  # 손실 함수
    metrics=["accuracy"]  # 평가 지표
)

# 4. 콜백 설정
# ModelCheckpoint: 가장 좋은 모델 저장
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "best-model.keras",  # 저장할 파일명
    save_best_only=True  # 최고 성능 모델만 저장
)
# EarlyStopping: 성능 개선이 없으면 조기 종료
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=2,  # 2번의 에포크 동안 개선이 없으면 종료
    restore_best_weights=True  # 최적 가중치 복원
)

# 5. 모델 훈련
history = model.fit(
    train_scaled, train_target,
    epochs=20,
    verbose=0,  # 출력 레벨
    validation_data=(val_scaled, val_target),
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# 6. 최적 모델 평가
# 저장된 최적 모델 로드
model = keras.models.load_model("best-model.keras")
# 검증 세트로 성능 평가
loss, accuracy = model.evaluate(val_scaled, val_target, verbose=0)
print("Loss:", loss)
print("Accuracy:", accuracy)

# 7. 예측 수행
# 검증 세트에 대한 예측
val_labels = np.argmax(model.predict(val_scaled), axis=-1)
# 예측 정확도 계산
accuracy = np.mean(val_labels == val_target)
print("Prediction Accuracy:", accuracy)

# 8. 학습 과정 시각화
plt.plot(history.history["loss"], label="train")  # 훈련 손실
plt.plot(history.history["val_loss"], label="val")  # 검증 손실
plt.xlabel("epoch")  # x축 레이블
plt.ylabel("loss")  # y축 레이블
plt.legend()  # 범례 표시
plt.show()
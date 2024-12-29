"""
Fashion MNIST 이미지 분류 - CNN(Convolutional Neural Network) 구현
이 스크립트는 Fashion MNIST 데이터셋을 사용하여 CNN 기반 이미지 분류기를 구현합니다.

데이터셋:
    Fashion MNIST: 10개 카테고리의 패션 아이템 흑백 이미지 (28x28 픽셀)

모델 구조:
    1. Conv2D (32필터) + ReLU
    2. MaxPooling2D (2x2)
    3. Conv2D (64필터) + ReLU
    4. MaxPooling2D (2x2)
    5. Flatten
    6. Dense (100) + ReLU
    7. Dropout (0.4)
    8. Dense (10) + Softmax

주요 기능:
    - CNN을 통한 이미지 특성 추출
    - 가중치 시각화
    - 훈련된 모델과 훈련되지 않은 모델 비교
    - 한글 클래스명으로 예측 결과 출력

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
# 이미지를 (샘플 수, 28, 28, 1) 형태로 리셰이프하고 정규화
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
# 훈련/검증 데이터 분리
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

# 2. CNN 모델 생성
model = keras.Sequential()
# 첫 번째 합성곱 층
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', 
                             padding='same', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D(2))  # 특성 맵의 크기를 절반으로 줄임
# 두 번째 합성곱 층
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', 
                             padding='same'))
model.add(keras.layers.MaxPooling2D(2))
# 완전 연결 층으로 전환
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))  # 과대적합 방지
model.add(keras.layers.Dense(10, activation='softmax'))

# 3. 모델 컴파일 및 훈련
model.compile(optimizer='adam', 
             loss='sparse_categorical_crossentropy', 
             metrics=['accuracy'])

# 콜백 설정
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'best-cnn-model.keras',  # 모델 저장 경로
    save_best_only=True      # 최고 성능 모델만 저장
)
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=2,                  # 2에포크 동안 개선 없으면 중단
    restore_best_weights=True    # 최적 가중치 복원
)

# 모델 훈련
history = model.fit(
    train_scaled, train_target,
    epochs=20,
    validation_data=(val_scaled, val_target),
    callbacks=[checkpoint_cb, early_stopping_cb],
)

# 4. 훈련 과정 시각화
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# 5. 모델 평가
model.evaluate(val_scaled, val_target)

# 6. 예측 수행
preds = model.predict(val_scaled[0:1])
# 한글 클래스명 정의
classes = ['티셔츠', '바지', '스웨터', '드레스', '코트', 
           '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']
print(classes[np.argmax(preds)])  # 예측 클래스 출력

# 7. 테스트 데이터 평가
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0
model.evaluate(test_scaled, test_target)

# 8. 가중치 분석 및 시각화
# 저장된 최적 모델 로드
model = keras.models.load_model('best-cnn-model.keras')
conv = model.layers[0]  # 첫 번째 합성곱 층

# 훈련된 가중치의 분포 시각화
conv_weights = conv.weights[0].numpy()
plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

# 필터 가중치 시각화
fig, axs = plt.subplots(2, 16, figsize=(15, 2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[:, :, 0, i * 16 + j], 
                        vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()

# 9. 훈련되지 않은 모델과의 비교
# 새로운 모델 생성 (훈련하지 않음)
no_training_model = keras.Sequential()
no_training_model.add(keras.layers.Conv2D(32, kernel_size=3, 
                                        activation='relu', 
                                        padding='same', 
                                        input_shape=(28, 28, 1)))
no_training_conv = no_training_model.layers[0]
no_training_weights = no_training_conv.weights[0].numpy()

# 훈련되지 않은 가중치의 분포 시각화
plt.hist(no_training_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

# 훈련되지 않은 필터 가중치 시각화
fig, axs = plt.subplots(2, 16, figsize=(15, 2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(no_training_weights[:, :, 0, i * 16 + j], 
                        vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()
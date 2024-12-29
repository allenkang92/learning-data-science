import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import SGDClassifier

# 데이터 로드
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# 데이터 확인
print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)

# 데이터 시각화 (선택 사항)
fig, axs = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()

print([train_target[i] for i in range(10)])
print(np.unique(train_target, return_counts=True))

# 데이터 전처리
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28 * 28)

# SGDClassifier 모델 검증 (선택 사항)
sc = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))

# 훈련/검증 데이터 분리
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

print(train_scaled.shape, train_target.shape)
print(val_scaled.shape, val_target.shape)

# Keras 모델 생성 및 컴파일
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
model = keras.Sequential([dense])
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련 및 평가
print(train_target[:10])
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)
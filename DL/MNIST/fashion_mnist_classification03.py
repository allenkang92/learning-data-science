import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 로드 및 전처리
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

# 2. 모델 정의
def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation="relu"))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model

# 3. 모델 컴파일 및 훈련
model = model_fn(keras.layers.Dropout(0.3))
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# 4. 체크포인트 및 Early Stopping 콜백
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "best-model.keras", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=2, restore_best_weights=True
)

# 5. 모델 훈련
history = model.fit(
    train_scaled,
    train_target,
    epochs=20,
    verbose=0,
    validation_data=(val_scaled, val_target),
    callbacks=[checkpoint_cb, early_stopping_cb],
)

# 6. 모델 평가
model = keras.models.load_model("best-model.keras")
loss, accuracy = model.evaluate(val_scaled, val_target, verbose=0)
print("Loss:", loss)
print("Accuracy:", accuracy)

# 7. 예측
val_labels = np.argmax(model.predict(val_scaled), axis=-1)
accuracy = np.mean(val_labels == val_target)
print("Prediction Accuracy:", accuracy)

# 8. 학습 결과 시각화 (선택 사항)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train", "val"])
plt.show()
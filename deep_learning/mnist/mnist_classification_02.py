import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# 데이터 로드 및 전처리
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

# 모델 정의
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# 모델 컴파일
model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']
)

# 모델 학습
model.fit(train_scaled, train_target, epochs=5)

# 모델 평가
loss, accuracy = model.evaluate(val_scaled, val_target)
print("Loss:", loss)
print("Accuracy:", accuracy)
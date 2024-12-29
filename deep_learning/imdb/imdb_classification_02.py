from tensorflow import keras
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)
test_seq = pad_sequences(test_input, maxlen=100)

# 모델 정의 및 학습 (LSTM, Dropout, 2-RNN, GRU)
def build_and_train_model(model, model_file_name):
    rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
    model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
    checkpoint_cb = keras.callbacks.ModelCheckpoint(model_file_name, save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    history = model.fit(train_seq, train_target, epochs=100, batch_size=64, 
                        validation_data=(val_seq, val_target), callbacks=[checkpoint_cb, early_stopping_cb])
    return history

# LSTM 모델
model = keras.Sequential()
model.add(keras.layers.Embedding(500, 16, input_length=100))
model.add(keras.layers.LSTM(8))
model.add(keras.layers.Dense(1, activation='sigmoid'))
history = build_and_train_model(model, 'best-lstm-model.keras')

# Dropout 적용 LSTM 모델
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.LSTM(8, dropout=0.3))
model2.add(keras.layers.Dense(1, activation='sigmoid'))
history = build_and_train_model(model2, 'best-dropout-model.keras')

# 2-RNN 모델
model3 = keras.Sequential()
model3.add(keras.layers.Embedding(500, 16, input_length=100))
model3.add(keras.layers.LSTM(8, dropout=0.3, return_sequences=True))
model3.add(keras.layers.LSTM(8, dropout=0.3))
model3.add(keras.layers.Dense(1, activation='sigmoid'))
history = build_and_train_model(model3, 'best-2rnn-model.keras')

# GRU 모델
model4 = keras.Sequential()
model4.add(keras.layers.Embedding(500, 16, input_length=100))
model4.add(keras.layers.GRU(8))
model4.add(keras.layers.Dense(1, activation='sigmoid'))
history = build_and_train_model(model4, 'best-gru-model.keras')

# 모델 평가 (2-RNN 모델)
rnn_model = keras.models.load_model('best-2rnn-model.keras')
rnn_model.evaluate(test_seq, test_target)

# 학습 과정 시각화 (각 모델별로 필요에 따라 추가)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
"""
IMDB 영화 리뷰 감성 분석 - 다양한 RNN 아키텍처 비교
이 스크립트는 IMDB 영화 리뷰 데이터셋을 사용하여 여러 RNN 모델의 성능을 비교합니다.

구현된 모델:
    1. LSTM 모델
    2. Dropout이 적용된 LSTM 모델
    3. 2층 LSTM 모델 (with Dropout)
    4. GRU 모델

데이터셋:
    IMDB: 50,000개의 영화 리뷰 텍스트
    - 상위 500개 단어만 사용
    - 시퀀스 길이 100으로 통일
    - 80:20 비율로 훈련/검증 분할

공통 구성:
    - Embedding 층: 500 단어를 16차원 벡터로 변환
    - RMSprop 옵티마이저 (learning_rate=1e-4)
    - 이진 크로스 엔트로피 손실 함수
    - ModelCheckpoint와 EarlyStopping 콜백

작성일: 2024-12-29
"""

from tensorflow import keras
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 전처리
# IMDB 데이터셋 로드 (상위 500개 단어만 사용)
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)

# 훈련/검증 데이터 분할 (80:20)
train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42
)

# 시퀀스 패딩 (길이 100으로 통일)
train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)
test_seq = pad_sequences(test_input, maxlen=100)

# 2. 모델 학습을 위한 유틸리티 함수
def build_and_train_model(model, model_file_name):
    """
    주어진 모델을 컴파일하고 훈련시키는 함수
    
    Parameters:
        model: 훈련시킬 keras 모델
        model_file_name: 최적 모델을 저장할 파일 경로
    
    Returns:
        history: 훈련 히스토리
    """
    # 옵티마이저 설정
    rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
    
    # 모델 컴파일
    model.compile(
        optimizer=rmsprop,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # 콜백 설정
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        model_file_name,
        save_best_only=True  # 최고 성능 모델만 저장
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=3,  # 3에포크 동안 개선이 없으면 중단
        restore_best_weights=True  # 최적 가중치 복원
    )
    
    # 모델 훈련
    history = model.fit(
        train_seq, train_target,
        epochs=100,
        batch_size=64,
        validation_data=(val_seq, val_target),
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    
    return history

# 3. 기본 LSTM 모델
model = keras.Sequential()
model.add(keras.layers.Embedding(500, 16, input_length=100))  # 임베딩 층
model.add(keras.layers.LSTM(8))  # LSTM 층 (8개 유닛)
model.add(keras.layers.Dense(1, activation='sigmoid'))  # 출력층
history = build_and_train_model(model, 'best-lstm-model.keras')

# 4. Dropout이 적용된 LSTM 모델
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.LSTM(8, dropout=0.3))  # 30% 드롭아웃
model2.add(keras.layers.Dense(1, activation='sigmoid'))
history = build_and_train_model(model2, 'best-dropout-model.keras')

# 5. 2층 LSTM 모델 (with Dropout)
model3 = keras.Sequential()
model3.add(keras.layers.Embedding(500, 16, input_length=100))
model3.add(keras.layers.LSTM(8, dropout=0.3, return_sequences=True))  # 첫 번째 LSTM
model3.add(keras.layers.LSTM(8, dropout=0.3))  # 두 번째 LSTM
model3.add(keras.layers.Dense(1, activation='sigmoid'))
history = build_and_train_model(model3, 'best-2rnn-model.keras')

# 6. GRU 모델
model4 = keras.Sequential()
model4.add(keras.layers.Embedding(500, 16, input_length=100))
model4.add(keras.layers.GRU(8))  # GRU 층 (8개 유닛)
model4.add(keras.layers.Dense(1, activation='sigmoid'))
history = build_and_train_model(model4, 'best-gru-model.keras')

# 7. 최종 모델 평가 (2층 LSTM 모델)
rnn_model = keras.models.load_model('best-2rnn-model.keras')
loss, accuracy = rnn_model.evaluate(test_seq, test_target)
print(f"테스트 손실: {loss:.4f}")
print(f"테스트 정확도: {accuracy:.4f}")

# 8. 학습 과정 시각화
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
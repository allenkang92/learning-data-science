"""
IMDB 영화 리뷰 감성 분석 - RNN과 임베딩을 사용한 텍스트 분류
이 스크립트는 IMDB 영화 리뷰 데이터셋을 사용하여 긍정/부정 감성을 분류하는 두 가지 모델을 구현합니다.

데이터셋:
    IMDB: 50,000개의 영화 리뷰 텍스트 (25,000 훈련 + 25,000 테스트)
    - 긍정(1)과 부정(0)으로 레이블링된 이진 분류 문제
    - 가장 빈도가 높은 500개 단어만 사용
    - 각 리뷰는 100개 단어로 패딩/잘라내기

모델 구조:
    1. SimpleRNN 모델:
        - SimpleRNN(8 유닛)
        - Dense(1, sigmoid)
    
    2. Embedding + RNN 모델:
        - Embedding(500 단어, 16 차원)
        - SimpleRNN(8 유닛)
        - Dense(1, sigmoid)

주요 기능:
    - 텍스트 데이터 전처리 (패딩, 원-핫 인코딩)
    - RNN을 사용한 시퀀스 처리
    - 임베딩을 통한 효율적인 단어 표현
    - 조기 종료와 체크포인트를 통한 모델 최적화

작성일: 2024-12-29
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# 1. 데이터 로드 및 전처리
# IMDB 데이터셋 로드 (상위 500개 단어만 사용)
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)

# 훈련 데이터를 훈련 세트와 검증 세트로 분할 (80:20)
train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42
)

# 리뷰 길이 분석
lengths = np.array([len(x) for x in train_input])
print(f"평균 리뷰 길이: {np.mean(lengths):.1f}")
print(f"중간값 리뷰 길이: {np.median(lengths):.1f}")

# 리뷰 길이 분포 시각화
plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()

# 시퀀스 패딩 (최대 길이 100으로 통일)
train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)

# 2. SimpleRNN 모델 구성 및 훈련
model = keras.Sequential()
# SimpleRNN 층: 8개의 뉴런, 입력 shape는 (시퀀스 길이, 단어 수)
model.add(keras.layers.SimpleRNN(8, input_shape=(100, 500)))
# 출력층: 이진 분류를 위한 시그모이드 활성화 함수
model.add(keras.layers.Dense(1, activation='sigmoid'))

# 시퀀스를 원-핫 인코딩으로 변환
train_oh = keras.utils.to_categorical(train_seq)
val_oh = keras.utils.to_categorical(val_seq)

# 모델 구조 출력
model.summary()

# 모델 컴파일
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)  # 학습률 0.0001
model.compile(
    optimizer=rmsprop,
    loss='binary_crossentropy',  # 이진 분류를 위한 손실 함수
    metrics=['accuracy']
)

# 콜백 설정
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'best-simplernn-model.keras',  # 모델 저장 경로
    save_best_only=True  # 최고 성능 모델만 저장
)
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=3,  # 3에포크 동안 개선이 없으면 훈련 중단
    restore_best_weights=True  # 최적 가중치 복원
)

# 모델 훈련
history = model.fit(
    train_oh, train_target,
    epochs=100,
    batch_size=64,
    validation_data=(val_oh, val_target),
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# 훈련 과정 시각화
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# 메모리 사용량 비교 (바이트 단위)
print(f"시퀀스 데이터 크기: {train_seq.nbytes:,} bytes")
print(f"원-핫 인코딩 데이터 크기: {train_oh.nbytes:,} bytes")

# 3. Embedding + RNN 모델 구성 및 훈련
model2 = keras.Sequential()
# Embedding 층: 500개 단어를 16차원 벡터로 변환
model2.add(keras.layers.Embedding(500, 16, input_length=100))
# SimpleRNN 층: 8개의 뉴런
model2.add(keras.layers.SimpleRNN(8))
# 출력층: 이진 분류
model2.add(keras.layers.Dense(1, activation='sigmoid'))

# 모델 구조 출력
model2.summary()

# 모델 컴파일
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(
    optimizer=rmsprop,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 콜백 설정
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'best-embedding-model.keras',
    save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=3,
    restore_best_weights=True
)

# 모델 훈련 (원본 시퀀스 데이터 사용)
history = model2.fit(
    train_seq, train_target,
    epochs=100,
    batch_size=64,
    validation_data=(val_seq, val_target),
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# 훈련 과정 시각화
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
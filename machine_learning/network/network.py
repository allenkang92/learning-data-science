"""
LTE 네트워크 장애 예측 모델
이 스크립트는 LTE 네트워크 데이터를 사용하여 네트워크 장애를 예측하는 분류 모델을 구현합니다.

데이터셋:
    - Kaggle LTE 데이터셋 사용
    - 네트워크 상태, 운영자, 신호 강도 등의 특성 포함
    - 다운로드 속도 3Mbps 미만을 장애로 정의

전처리 단계:
    1. 여러 CSV 파일 통합
    2. LTE 모드 데이터만 선택
    3. 범주형 변수 원-핫 인코딩
    4. 결측치 처리
    5. 불필요한 특성 제거

구현 모델:
    1. 로지스틱 회귀
    2. 그래디언트 부스팅

평가 지표:
    - 정확도 (Accuracy)

작성일: 2024-12-29
"""

import kagglehub
import os
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# 1. 데이터 다운로드 및 로드
# Kaggle 데이터셋 다운로드
path = kagglehub.dataset_download("aeryss/lte-dataset")
base_dir = os.path.join(path, 'Dataset')
# 모든 CSV 파일 경로 수집
file_list = glob(os.path.join(base_dir, '*', "*.csv"))

# 2. 데이터 전처리
# 모든 CSV 파일 통합
df = pd.concat([pd.read_csv(file) for file in file_list], axis=0)

# 장애 레이블 생성 (다운로드 속도 3Mbps 미만을 장애로 정의)
df["Fault"] = 0
df.loc[df['DL_bitrate'] < 3 * 1024, "Fault"] = 1

# LTE 모드 데이터만 선택
df = df.loc[df['NetworkMode'] == "LTE", :].copy()

# 비트레이트 컬럼 제거 (타겟 생성 후 불필요)
df = df.drop(columns=['DL_bitrate', 'UL_bitrate'])

# 범주형 변수 원-핫 인코딩
df = pd.get_dummies(df, columns=['State', 'Operatorname', 'NetworkMode'])

# 3. 특성 엔지니어링
# 불필요하거나 중복되는 특성 제거
df = df.drop(columns=[
    'ServingCell_Lon',    # 위치 정보
    'ServingCell_Lat',    # 위치 정보
    'ServingCell_Distance',  # 거리 정보
    'NRxRSRP',           # 중복 신호 강도
    'NRxRSRQ',           # 중복 신호 품질
    'Timestamp'          # 시간 정보
])

# 수치형 컬럼 처리
for col in ['RSRQ', 'SNR', 'CQI', 'RSSI']:
    # 문자열을 숫자로 변환
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # 결측치를 평균값으로 대체
    df[col].fillna(df[col].mean(), inplace=True)

# 모든 컬럼을 float64 타입으로 통일
df = df.astype('float64')

# 4. 훈련/테스트 데이터 분리
# 미리 정의된 테스트 인덱스 로드
with open('./test_idx.txt', 'r') as f:
    test_idx = f.read()
test_idx_list = [int(idx) for idx in test_idx.split(",")]

# 테스트 데이터 분리
test_df = df.iloc[test_idx_list, :]
train_df = df.drop(index=test_idx_list)

# 5. 모델 학습 및 평가
# 특성과 타겟 분리
X = train_df.drop('Fault', axis=1)
y = train_df['Fault']

# 훈련/검증 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2024
)

# 5.1 로지스틱 회귀 모델
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred):.4f}")

# 5.2 그래디언트 부스팅 모델
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_pred):.4f}")
import kagglehub
import os
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# 데이터 다운로드 및 전처리
path = kagglehub.dataset_download("aeryss/lte-dataset")
base_dir = os.path.join(path, 'Dataset')
file_list = glob(os.path.join(base_dir, '*', "*.csv"))

df = pd.concat([pd.read_csv(file) for file in file_list], axis=0)
df["Fault"] = 0
df.loc[df['DL_bitrate'] < 3 * 1024, "Fault"] = 1
df = df.loc[df['NetworkMode'] == "LTE", :].copy()
df = df.drop(columns=['DL_bitrate', 'UL_bitrate'])
df = pd.get_dummies(df, columns=['State', 'Operatorname', 'NetworkMode'])

# 불필요한 컬럼 제거 및 타입 변환
df = df.drop(columns=['ServingCell_Lon', 'ServingCell_Lat', 'ServingCell_Distance', 'NRxRSRP', 'NRxRSRQ', 'Timestamp'])
for col in ['RSRQ', 'SNR', 'CQI', 'RSSI']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].mean(), inplace=True)
df = df.astype('float64')

# train/test 데이터 분리
with open('./test_idx.txt', 'r') as f:
    test_idx = f.read()
test_idx_list = [int(idx) for idx in test_idx.split(",")]
test_df = df.iloc[test_idx_list, :]
train_df = df.drop(index=test_idx_list)

# 모델 학습 및 평가
X = train_df.drop('Fault', axis=1)
y = train_df['Fault']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred)}")

# Gradient Boosting Classifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_pred)}")
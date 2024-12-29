import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

# 데이터 로드
fish = pd.read_csv('https://bit.ly/fish_csv_data')

# 입력 데이터와 타겟 데이터 분리
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

# 훈련 데이터와 테스트 데이터 분리
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42
)

# 데이터 스케일링
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# SGDClassifier 모델 훈련 및 평가 (log_loss)
sc = SGDClassifier(loss='log_loss', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print('log_loss')
print(f'훈련 데이터 점수: {sc.score(train_scaled, train_target)}')
print(f'테스트 데이터 점수: {sc.score(test_scaled, test_target)}')

# SGDClassifier 모델 훈련 및 평가 (hinge)
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print('hinge')
print(f'훈련 데이터 점수: {sc.score(train_scaled, train_target)}')
print(f'테스트 데이터 점수: {sc.score(test_scaled, test_target)}')

# 에포크에 따른 훈련 데이터와 테스트 데이터 점수 시각화
sc = SGDClassifier(loss='log_loss', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)

for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 데이터 로드
wine = pd.read_csv('https://bit.ly/wine_csv_data')

# 데이터 전처리
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 훈련/테스트 데이터 분리
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42
)

# 데이터 스케일링
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 로지스틱 회귀 모델 훈련 및 평가
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print('Logistic Regression:')
print('Train Score:', lr.score(train_scaled, train_target))
print('Test Score:', lr.score(test_scaled, test_target))
print('Coefficients:', lr.coef_, lr.intercept_)

# 결정 트리 모델 훈련 및 평가
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print('\nDecision Tree (max_depth=None):')
print('Train Score:', dt.score(train_scaled, train_target))
print('Test Score:', dt.score(test_scaled, test_target))

# 결정 트리 시각화 (전체 트리)
plt.figure(figsize=(10, 7))
plot_tree(dt)
plt.show()

# 결정 트리 시각화 (깊이 1)
plt.figure(figsize=(10, 7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# 결정 트리 모델 훈련 및 평가 (깊이 3)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print('\nDecision Tree (max_depth=3):')
print('Train Score:', dt.score(train_scaled, train_target))
print('Test Score:', dt.score(test_scaled, test_target))

# 결정 트리 시각화 (깊이 3)
plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
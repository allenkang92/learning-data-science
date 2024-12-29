"""
와인 품질 분류 - 로지스틱 회귀와 결정 트리 비교
이 스크립트는 와인의 특성을 기반으로 품질을 분류하는 모델을 구현하고,
로지스틱 회귀와 결정 트리의 성능을 비교합니다.

데이터셋:
    - 와인의 화학적 특성 데이터
    - 사용된 특성: 알코올 도수, 당도, pH
    - 타겟: 와인 품질 등급

분석 방법:
    1. 로지스틱 회귀
       - 데이터 스케일링 적용
       - 선형 결정 경계 학습
    
    2. 결정 트리
       - 다양한 깊이(무제한, 1, 3)로 실험
       - 트리 구조 시각화

평가:
    - 훈련 세트와 테스트 세트의 정확도
    - 모델의 해석 가능성 비교

작성일: 2024-12-29
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 1. 데이터 로드
# 온라인 저장소에서 와인 데이터 로드
wine = pd.read_csv('https://bit.ly/wine_csv_data')

# 2. 데이터 전처리
# 필요한 특성만 선택하여 NumPy 배열로 변환
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 3. 데이터 분할
# 80%는 훈련용, 20%는 테스트용
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42
)

# 4. 특성 스케일링
# 평균 0, 표준편차 1로 변환
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 5. 로지스틱 회귀 모델
lr = LogisticRegression()
lr.fit(train_scaled, train_target)

# 로지스틱 회귀 성능 평가
print('Logistic Regression:')
print('Train Score:', f"{lr.score(train_scaled, train_target):.4f}")
print('Test Score:', f"{lr.score(test_scaled, test_target):.4f}")
# 계수와 절편 출력 (특성 중요도 확인)
print('Coefficients:', lr.coef_, lr.intercept_)

# 6. 결정 트리 모델 (깊이 제한 없음)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)

# 결정 트리 성능 평가 (깊이 제한 없음)
print('\nDecision Tree (max_depth=None):')
print('Train Score:', f"{dt.score(train_scaled, train_target):.4f}")
print('Test Score:', f"{dt.score(test_scaled, test_target):.4f}")

# 7. 결정 트리 시각화
# 7.1 전체 트리 시각화
plt.figure(figsize=(10, 7))
plot_tree(dt)
plt.show()

# 7.2 깊이 1의 트리 시각화
plt.figure(figsize=(10, 7))
plot_tree(dt, max_depth=1, filled=True, 
         feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# 8. 깊이 3의 결정 트리
# 과적합 방지를 위해 깊이 제한
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)

# 깊이 3 트리의 성능 평가
print('\nDecision Tree (max_depth=3):')
print('Train Score:', f"{dt.score(train_scaled, train_target):.4f}")
print('Test Score:', f"{dt.score(test_scaled, test_target):.4f}")

# 깊이 3 트리 시각화
plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, 
         feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
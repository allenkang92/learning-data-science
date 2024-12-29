import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import uniform, randint

# 데이터 로드
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 데이터 분할
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)

# 모델 훈련 및 평가 (DecisionTreeClassifier)
dt = DecisionTreeClassifier(random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42))

# GridSearchCV를 사용한 하이퍼파라미터 튜닝
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

# RandomizedSearchCV를 사용한 하이퍼파라미터 튜닝
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25)}
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)

# 최적 모델 평가
dt = gs.best_estimator_
score = dt.score(test_input, test_target)

print(f"최적 모델의 테스트 점수: {score}")
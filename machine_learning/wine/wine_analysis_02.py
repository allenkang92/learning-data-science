"""
와인 품질 분류 - 하이퍼파라미터 튜닝
이 스크립트는 결정 트리 모델의 최적 하이퍼파라미터를 찾기 위해
그리드 서치와 랜덤 서치를 비교 실험합니다.

데이터셋:
    - 와인의 화학적 특성 데이터
    - 사용된 특성: 알코올 도수, 당도, pH
    - 타겟: 와인 품질 등급

분석 방법:
    1. 교차 검증으로 기본 성능 평가
    2. GridSearchCV로 체계적 탐색
       - min_impurity_decrease
       - max_depth
       - min_samples_split
    
    3. RandomizedSearchCV로 무작위 탐색
       - 연속적인 범위에서 샘플링
       - 더 넓은 탐색 공간

평가:
    - 10-폴드 교차 검증
    - 검증 세트와 테스트 세트 성능

작성일: 2024-12-29
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_validate, 
    StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import uniform, randint

# 1. 데이터 로드 및 전처리
# 온라인 저장소에서 와인 데이터 로드
wine = pd.read_csv('https://bit.ly/wine_csv_data')
# 필요한 특성만 선택
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 2. 데이터 분할
# 2.1 테스트 세트 분리 (전체의 20%)
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42
)
# 2.2 검증 세트 분리 (훈련 세트의 20%)
sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42
)

# 3. 기본 모델 평가
dt = DecisionTreeClassifier(random_state=42)
# 층화 K-폴드 교차 검증 (클래스 비율 유지)
scores = cross_validate(
    dt, train_input, train_target,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
)

# 4. GridSearchCV를 사용한 하이퍼파라미터 튜닝
# 탐색할 하이퍼파라미터 범위 지정
params = {
    'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),  # 불순도 감소 최소값
    'max_depth': range(5, 20, 1),  # 트리 최대 깊이
    'min_samples_split': range(2, 100, 10)  # 노드 분할을 위한 최소 샘플 수
}
# 그리드 서치 수행
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                 params, n_jobs=-1)  # 모든 CPU 코어 사용
gs.fit(train_input, train_target)

# 5. RandomizedSearchCV를 사용한 하이퍼파라미터 튜닝
# 연속적인 범위에서 무작위 샘플링
params = {
    'min_impurity_decrease': uniform(0.0001, 0.001),  # 균등 분포에서 샘플링
    'max_depth': randint(20, 50),  # 정수 범위에서 샘플링
    'min_samples_split': randint(2, 25),
    'min_samples_leaf': randint(1, 25)  # 리프 노드의 최소 샘플 수
}
# 랜덤 서치 수행
gs = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    params, 
    n_iter=100,  # 100회 시도
    n_jobs=-1,  # 모든 CPU 코어 사용
    random_state=42
)
gs.fit(train_input, train_target)

# 6. 최종 모델 평가
# 최적의 하이퍼파라미터로 학습된 모델
dt = gs.best_estimator_
# 테스트 세트로 성능 평가
score = dt.score(test_input, test_target)

print(f"최적 모델의 테스트 점수: {score:.4f}")
"""
와인 품질 분류 - 앙상블 학습 모델 비교
이 스크립트는 다양한 앙상블 학습 모델을 사용하여 와인 품질을 분류하고,
각 모델의 성능과 특성 중요도를 비교 분석합니다.

데이터셋:
    - 와인의 화학적 특성 데이터
    - 사용된 특성: 알코올 도수, 당도, pH
    - 타겟: 와인 품질 등급

구현된 모델:
    1. 랜덤 포레스트
       - 배깅 기반 앙상블
       - OOB(Out-of-Bag) 평가
    
    2. 엑스트라 트리
       - 랜덤 포레스트의 변형
       - 더 무작위적인 분할
    
    3. 그래디언트 부스팅
       - 부스팅 기반 앙상블
       - 학습률 조정 실험
    
    4. 히스토그램 기반 그래디언트 부스팅
       - 빠른 학습 속도
       - 순열 중요도 계산

평가:
    - 교차 검증 점수
    - 특성 중요도 분석
    - OOB 점수 (랜덤 포레스트)

작성일: 2024-12-29
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier, 
    GradientBoostingClassifier
)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier

# 1. 데이터 로드 및 전처리
# 온라인 저장소에서 와인 데이터 로드
wine = pd.read_csv('https://bit.ly/wine_csv_data')
# 필요한 특성만 선택
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 2. 데이터 분할
# 훈련 세트 80%, 테스트 세트 20%
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42
)

# 3. 랜덤 포레스트 모델
# 3.1 기본 랜덤 포레스트
rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(
    rf, train_input, train_target, 
    return_train_score=True, n_jobs=-1
)
print('RandomForestClassifier:')
print(f"훈련 점수: {np.mean(scores['train_score']):.4f}, "
      f"테스트 점수: {np.mean(scores['test_score']):.4f}")
rf.fit(train_input, train_target)
print(f"특성 중요도: {rf.feature_importances_}")

# 3.2 OOB 점수 계산
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
rf.fit(train_input, train_target)
print(f"OOB 점수: {rf.oob_score_:.4f}")

# 4. 엑스트라 트리 모델
et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(
    et, train_input, train_target, 
    return_train_score=True, n_jobs=-1
)
print('\nExtraTreesClassifier:')
print(f"훈련 점수: {np.mean(scores['train_score']):.4f}, "
      f"테스트 점수: {np.mean(scores['test_score']):.4f}")
et.fit(train_input, train_target)
print(f"특성 중요도: {et.feature_importances_}")

# 5. 그래디언트 부스팅 모델
# 5.1 기본 그래디언트 부스팅
gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(
    gb, train_input, train_target, 
    return_train_score=True, n_jobs=-1
)
print('\nGradientBoostingClassifier:')
print(f"훈련 점수: {np.mean(scores['train_score']):.4f}, "
      f"테스트 점수: {np.mean(scores['test_score']):.4f}")

# 5.2 하이퍼파라미터 조정된 그래디언트 부스팅
gb = GradientBoostingClassifier(
    n_estimators=500,  # 트리 개수 증가
    learning_rate=0.2,  # 학습률 조정
    random_state=42
)
scores = cross_validate(
    gb, train_input, train_target, 
    return_train_score=True, n_jobs=-1
)
print(f"훈련 점수: {np.mean(scores['train_score']):.4f}, "
      f"테스트 점수: {np.mean(scores['test_score']):.4f}")
gb.fit(train_input, train_target)
print(f"특성 중요도: {gb.feature_importances_}")

# 6. 히스토그램 기반 그래디언트 부스팅 모델
hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(
    hgb, train_input, train_target, 
    return_train_score=True
)
print('\nHistGradientBoostingClassifier:')
print(f"훈련 점수: {np.mean(scores['train_score']):.4f}, "
      f"테스트 점수: {np.mean(scores['test_score']):.4f}")

# 6.1 순열 중요도 계산
hgb.fit(train_input, train_target)
# 훈련 데이터 특성 중요도
result = permutation_importance(
    hgb, train_input, train_target,
    n_repeats=10, random_state=42, n_jobs=-1
)
print(f"훈련 데이터 특성 중요도: {result.importances_mean}")
# 테스트 데이터 특성 중요도
result = permutation_importance(
    hgb, test_input, test_target,
    n_repeats=10, random_state=42, n_jobs=-1
)
print(f"테스트 데이터 특성 중요도: {result.importances_mean}")
print(f"테스트 점수: {hgb.score(test_input, test_target):.4f}")

# 7. 추가 구현 가능한 모델들 (현재 주석 처리됨)
# XGBoost 모델
#xgb = XGBClassifier(tree_method='hist', random_state=42)
#scores = cross_validate(xgb, train_input, train_target, return_train_score=True)
#print('\nXGBClassifier:')
#print(f"훈련 점수: {np.mean(scores['train_score'])}, 테스트 점수: {np.mean(scores['test_score'])}")

# LightGBM 모델
#lgb = LGBMClassifier(random_state=42)
#scores = cross_validate(lgb, train_input, train_target, return_train_score=True, n_jobs=-1)
#print('\nLGBMClassifier:')
#print(f"훈련 점수: {np.mean(scores['train_score'])}, 테스트 점수: {np.mean(scores['test_score'])}")
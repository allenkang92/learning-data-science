import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier

# 데이터 로드
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 훈련/테스트 데이터 분리
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델
rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
print('RandomForestClassifier:')
print(f"훈련 점수: {np.mean(scores['train_score'])}, 테스트 점수: {np.mean(scores['test_score'])}")
rf.fit(train_input, train_target)
print(f"특성 중요도: {rf.feature_importances_}")

rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
rf.fit(train_input, train_target)
print(f"OOB 점수: {rf.oob_score_}")

# 엑스트라 트리 모델
et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)
print('\nExtraTreesClassifier:')
print(f"훈련 점수: {np.mean(scores['train_score'])}, 테스트 점수: {np.mean(scores['test_score'])}")
et.fit(train_input, train_target)
print(f"특성 중요도: {et.feature_importances_}")

# 그래디언트 부스팅 모델
gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print('\nGradientBoostingClassifier:')
print(f"훈련 점수: {np.mean(scores['train_score'])}, 테스트 점수: {np.mean(scores['test_score'])}")

gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(f"훈련 점수: {np.mean(scores['train_score'])}, 테스트 점수: {np.mean(scores['test_score'])}")
gb.fit(train_input, train_target)
print(f"특성 중요도: {gb.feature_importances_}")

# 히스토그램 기반 그래디언트 부스팅 모델
hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, return_train_score=True)
print('\nHistGradientBoostingClassifier:')
print(f"훈련 점수: {np.mean(scores['train_score'])}, 테스트 점수: {np.mean(scores['test_score'])}")

hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target, n_repeats=10, random_state=42, n_jobs=-1)
print(f"훈련 데이터 특성 중요도: {result.importances_mean}")
result = permutation_importance(hgb, test_input, test_target, n_repeats=10, random_state=42, n_jobs=-1)
print(f"테스트 데이터 특성 중요도: {result.importances_mean}")
print(f"테스트 점수: {hgb.score(test_input, test_target)}")

# XGBoost 모델 (주석 처리됨)
#xgb = XGBClassifier(tree_method='hist', random_state=42)
#scores = cross_validate(xgb, train_input, train_target, return_train_score=True)
#print('\nXGBClassifier:')
#print(f"훈련 점수: {np.mean(scores['train_score'])}, 테스트 점수: {np.mean(scores['test_score'])}")

# LightGBM 모델 (주석 처리됨)
#lgb = LGBMClassifier(random_state=42)
#scores = cross_validate(lgb, train_input, train_target, return_train_score=True, n_jobs=-1)
#print('\nLGBMClassifier:')
#print(f"훈련 점수: {np.mean(scores['train_score'])}, 테스트 점수: {np.mean(scores['test_score'])}")
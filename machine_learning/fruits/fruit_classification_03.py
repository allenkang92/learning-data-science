"""
과일 이미지 분류 - PCA와 로지스틱 회귀
이 스크립트는 PCA를 사용하여 과일 이미지의 차원을 축소하고,
로지스틱 회귀를 통해 과일을 분류하는 모델을 구현합니다.

데이터셋:
    - 300개의 흑백 과일 이미지 (100x100 픽셀)
    - 사과(0), 파인애플(1), 바나나(2) 각 100개씩

분석 방법:
    1. PCA로 차원 축소 (10000 -> 50 차원)
    2. 설명된 분산 비율로 PCA 적용
    3. 로지스틱 회귀로 분류
    4. K-평균 군집화로 비교

평가:
    - 교차 검증을 통한 정확도 측정
    - 두 가지 PCA 방법의 성능 비교

작성일: 2024-12-29
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans

# 1. 데이터 로드 및 전처리
# 300개의 100x100 흑백 이미지 로드
fruits = np.load('fruits_300.npy')
# 2차원 배열로 변환 (300, 10000)
fruits_2d = fruits.reshape(-1, 100*100)

# 2. PCA를 활용한 차원 축소
# 2.1 고정된 차원 수로 PCA
pca = PCA(n_components=50)  # 50개의 주성분으로 축소
pca.fit(fruits_2d)
fruits_pca = pca.transform(fruits_2d)

# 3. 로지스틱 회귀 모델 학습 및 평가
# 타겟 레이블 생성 (0: 사과, 1: 파인애플, 2: 바나나)
target = np.array([0]*100 + [1]*100 + [2]*100)
# 교차 검증으로 모델 평가
scores = cross_validate(LogisticRegression(), fruits_pca, target)
print(f"Logistic Regression Accuracy: {np.mean(scores['test_score']):.4f}")

# 4. 설명된 분산 비율로 PCA
# 전체 분산의 50%를 설명하는 주성분 선택
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
fruits_pca = pca.transform(fruits_2d)
# 교차 검증으로 모델 평가
scores = cross_validate(LogisticRegression(), fruits_pca, target)
print(f"Logistic Regression Accuracy (with explained variance ratio): {np.mean(scores['test_score']):.4f}")

# 5. K-평균 군집화
# PCA로 축소된 데이터에 대해 군집화 수행
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)

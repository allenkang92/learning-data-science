import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans

# 데이터 로드
fruits = np.load('fruits_300.npy') 
fruits_2d = fruits.reshape(-1, 100*100)

# PCA를 활용한 차원 축소
pca = PCA(n_components=50) 
pca.fit(fruits_2d)
fruits_pca = pca.transform(fruits_2d)

# Logistic Regression 모델 학습 및 평가
target = np.array([0]*100 + [1]*100 + [2]*100)
scores = cross_validate(LogisticRegression(), fruits_pca, target)
print(f"Logistic Regression Accuracy: {np.mean(scores['test_score'])}")

# PCA with explained variance ratio
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
fruits_pca = pca.transform(fruits_2d)
scores = cross_validate(LogisticRegression(), fruits_pca, target)
print(f"Logistic Regression Accuracy (with explained variance ratio): {np.mean(scores['test_score'])}")

# KMeans 클러스터링
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)


"""
과일 이미지 분류 - K-평균 군집화
이 스크립트는 과일 이미지 데이터셋을 K-평균 군집화 알고리즘을 사용하여 분류합니다.

데이터셋:
    - 300개의 흑백 과일 이미지 (100x100 픽셀)
    - 사과, 파인애플, 바나나 각 100개씩

분석 방법:
    1. 이미지를 1차원 벡터로 변환
    2. K-평균 군집화로 3개의 군집으로 분류
    3. 각 군집의 중심 이미지 계산
    4. 엘보우 방법으로 최적 군집 수 탐색

시각화:
    - 각 군집별 이미지 표시
    - 군집 중심 이미지 표시
    - 엘보우 그래프

작성일: 2024-12-29
"""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 전처리
# 300개의 100x100 흑백 이미지 로드
fruits = np.load('fruits_300.npy')
# 2차원 배열로 변환 (300, 10000)
fruits_2d = fruits.reshape(-1, 100*100)

# 2. K-평균 군집화
# 3개의 군집으로 분류 (사과, 파인애플, 바나나)
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)

# 3. 군집화 결과 확인
# 각 데이터의 군집 레이블 출력
print("군집 레이블:", km.labels_)
# 각 군집의 크기 확인
print("군집별 데이터 수:", np.unique(km.labels_, return_counts=True))

# 4. 이미지 시각화 함수
def draw_fruits(arr, ratio=1):
    """
    과일 이미지를 그리드로 표시하는 함수
    
    Parameters:
        arr: 표시할 이미지 배열
        ratio: 이미지 크기 비율 (기본값: 1)
    """
    n = len(arr)  # 이미지 개수
    # 10개씩 표시할 때 필요한 행 수 계산
    rows = int(np.ceil(n/10))
    # 열 수는 10개로 고정 (단, 이미지가 10개 미만이면 이미지 개수만큼)
    cols = n if rows < 2 else 10
    
    # 서브플롯 생성
    fig, axs = plt.subplots(rows, cols, 
                           figsize=(cols*ratio, rows*ratio), 
                           squeeze=False)

    # 각 서브플롯에 이미지 표시
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:  # 이미지가 있는 경우만 표시
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')  # 축 눈금 제거
    plt.show()

# 5. 군집화 결과 시각화
# 5.1 각 군집별 이미지 출력
draw_fruits(fruits[km.labels_ == 0])  # 군집 0
draw_fruits(fruits[km.labels_ == 1])  # 군집 1
draw_fruits(fruits[km.labels_ == 2])  # 군집 2

# 5.2 군집 중심 이미지 출력
# 군집 중심을 100x100 이미지로 변환하여 표시
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)

# 6. 새로운 이미지 예측
# 100번째 이미지에 대한 각 군집 중심까지의 거리 계산
print("군집 중심까지의 거리:", km.transform(fruits_2d[100:101]))
# 가장 가까운 군집 예측
print("예측된 군집:", km.predict(fruits_2d[100:101]))
# 예측한 이미지 출력
draw_fruits(fruits[100:101])

# 7. 학습 정보
# 수렴할 때까지 반복한 횟수
print("반복 횟수:", km.n_iter_)

# 8. 최적 군집 수 탐색 (엘보우 방법)
inertia = []  # 각 k에 대한 inertia 값 저장
# k=2부터 6까지 테스트
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)

# 엘보우 그래프 그리기
plt.plot(range(2, 7), inertia)
plt.xlabel('k')  # x축: 군집 수
plt.ylabel('inertia')  # y축: 군집 내 거리 제곱합
plt.show()
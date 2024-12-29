import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 데이터 로드
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

# KMeans 모델 생성 및 학습
km = KMeans(n_clusters = 3, random_state = 42)
km.fit(fruits_2d)

# 군집 결과 출력
print(km.labels_)
print(np.unique(km.labels_, return_counts= True))

# 이미지 출력 함수
def draw_fruits(arr, ratio = 1):
    n = len(arr) 
    rows = int(np.ceil(n/10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, figsize = (cols*ratio, rows*ratio), squeeze = False)

    for i in range(rows):
        for j in range(cols):
            if i*10 +j < n: 
                axs[i, j].imshow(arr[i*10 + j], cmap = 'gray_r')
            axs[i, j].axis('off')
    plt.show()

# 군집별 이미지 출력
draw_fruits(fruits[km.labels_ == 0])
draw_fruits(fruits[km.labels_ == 1])
draw_fruits(fruits[km.labels_ == 2])

# 군집 중심 이미지 출력
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio = 3)

# 특정 이미지 예측
print(km.transform(fruits_2d[100:101]))
print(km.predict(fruits_2d[100:101]))
draw_fruits(fruits[100:101])

# 반복 횟수 출력
print(km.n_iter_)

# 최적 군집 개수 찾기 (Inertia)
inertia = []
for k in range(2, 7):
    km = KMeans(n_clusters = k, random_state = 42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)
plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()
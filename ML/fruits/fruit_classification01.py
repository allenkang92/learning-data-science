import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
fruits = np.load('fruits_300.npy')

# 과일 데이터 분류
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)

# 과일별 평균 이미지 계산
apple_mean = np.mean(apple, axis = 0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis = 0).reshape(100, 100)
banana_mean = np.mean(banana, axis = 0).reshape(100, 100)

# 과일 이미지 출력 함수
def plot_fruits(fruit_index, title):
    fig, axs = plt.subplots(10, 10, figsize = (10, 10))
    for i in range(10):
        for j in range(10):
            axs[i, j].imshow(fruits[fruit_index[i*10 + j]], cmap = 'gray_r')
            axs[i, j].axis('off')
    plt.suptitle(title)
    plt.show()

# 과일별 평균값과 가장 유사한 이미지 출력
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis = (1, 2))
apple_index = np.argsort(abs_mean)[:100]
plot_fruits(apple_index, 'Apple')

abs_diff = np.abs(fruits - pineapple_mean)
abs_mean = np.mean(abs_diff, axis = (1, 2))
pineapple_index = np.argsort(abs_mean)[:100]
plot_fruits(pineapple_index, 'Pineapple')

abs_diff = np.abs(fruits - banana_mean)
abs_mean = np.mean(abs_diff, axis = (1, 2))
banana_index = np.argsort(abs_mean)[:100]
plot_fruits(banana_index, 'Banana')
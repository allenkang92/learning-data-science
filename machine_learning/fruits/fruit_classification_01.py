"""
과일 이미지 분류 - 평균 이미지 기반 분석
이 스크립트는 과일 이미지 데이터셋을 사용하여 평균 이미지를 계산하고,
각 이미지와 평균 이미지 간의 차이를 기반으로 과일을 분류합니다.

데이터셋:
    - 300개의 흑백 과일 이미지 (100x100 픽셀)
    - 각 100개씩의 사과, 파인애플, 바나나 이미지

분석 방법:
    1. 각 과일 종류별 평균 이미지 계산
    2. 각 이미지와 평균 이미지 간의 절대 차이 계산
    3. 차이가 가장 작은 이미지들을 해당 과일로 분류

시각화:
    - 10x10 그리드로 각 과일별 100개 이미지 표시
    - 흑백 반전 컬러맵 사용

작성일: 2024-12-29
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로드
# 300개의 100x100 흑백 이미지 로드
fruits = np.load('fruits_300.npy')

# 2. 과일별 데이터 분리
# 각 과일 100개씩 분리하고 1차원으로 펼침 (100x100 -> 10000)
apple = fruits[0:100].reshape(-1, 100*100)      # 0-99: 사과
pineapple = fruits[100:200].reshape(-1, 100*100)  # 100-199: 파인애플
banana = fruits[200:300].reshape(-1, 100*100)     # 200-299: 바나나

# 3. 과일별 평균 이미지 계산
# axis=0: 같은 위치의 픽셀들의 평균을 계산
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)

# 4. 이미지 시각화 함수
def plot_fruits(fruit_index, title):
    """
    과일 이미지를 10x10 그리드로 표시하는 함수
    
    Parameters:
        fruit_index: 표시할 이미지의 인덱스 배열
        title: 그래프 제목
    """
    # 10x10 서브플롯 생성
    fig, axs = plt.subplots(10, 10, figsize=(10, 10))
    
    # 각 서브플롯에 이미지 표시
    for i in range(10):
        for j in range(10):
            # gray_r: 흑백 반전 컬러맵 (0이 흰색, 1이 검은색)
            axs[i, j].imshow(fruits[fruit_index[i*10 + j]], cmap='gray_r')
            axs[i, j].axis('off')  # 축 눈금 제거
            
    plt.suptitle(title)  # 전체 그래프 제목 설정
    plt.show()

# 5. 평균 이미지와의 차이를 기반으로 과일 분류

# 5.1 사과 분류
# 각 이미지와 사과 평균 이미지의 절대 차이 계산
abs_diff = np.abs(fruits - apple_mean)
# 차이의 평균을 계산하여 유사도 측정
abs_mean = np.mean(abs_diff, axis=(1, 2))
# 차이가 가장 작은 100개의 인덱스 선택
apple_index = np.argsort(abs_mean)[:100]
# 선택된 이미지 시각화
plot_fruits(apple_index, 'Apple')

# 5.2 파인애플 분류
abs_diff = np.abs(fruits - pineapple_mean)
abs_mean = np.mean(abs_diff, axis=(1, 2))
pineapple_index = np.argsort(abs_mean)[:100]
plot_fruits(pineapple_index, 'Pineapple')

# 5.3 바나나 분류
abs_diff = np.abs(fruits - banana_mean)
abs_mean = np.mean(abs_diff, axis=(1, 2))
banana_index = np.argsort(abs_mean)[:100]
plot_fruits(banana_index, 'Banana')
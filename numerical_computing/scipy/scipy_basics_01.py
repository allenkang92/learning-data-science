"""
SciPy 기본 기능 실습
이 스크립트는 SciPy의 주요 기능들을 실습하는 예제를 포함합니다.

주요 내용:
1. 통계 (stats): 정규분포 생성 및 샘플링
2. 최적화 (optimize): 함수의 최솟값 찾기
3. 보간 (interpolate): 1차원 데이터 보간

작성일: 2024-12-29
"""

import numpy as np
from scipy import stats, optimize, interpolate

# 1. 통계 (Statistical Functions)
# stats.norm: 정규분포(Normal/Gaussian distribution) 객체 생성
# - loc: 평균 (mean)
# - scale: 표준편차 (standard deviation)
norm_dist = stats.norm(loc=0, scale=1)  # 표준정규분포 N(0,1) 생성
samples = norm_dist.rvs(size=1000)      # 1000개의 랜덤 샘플 추출

# 2. 최적화 (Optimization)
def f(x):
    """
    최적화할 목적 함수
    f(x) = x^2 + 10sin(x)
    
    Parameters:
        x (float): 입력값
    
    Returns:
        float: 계산된 함수값
    """
    return x**2 + 10*np.sin(x)

# optimize.minimize_scalar: 단일 변수 함수의 최솟값을 찾는 함수
result = optimize.minimize_scalar(f)
print("Minimum of f(x) = x^2 + 10sin(x):", result.x)

# 3. 보간 (Interpolation)
# 원본 데이터 포인트 생성
x = np.arange(0, 10)                # x 좌표: 0부터 9까지의 정수
y = np.exp(-x/3.0)                  # y 좌표: 지수 감소 함수

# interpolate.interp1d: 1차원 보간 함수 생성
# - 주어진 점들 사이의 값을 추정하는 함수를 생성
f = interpolate.interp1d(x, y)

# 더 조밀한 x 좌표에서의 y값 계산
x_new = np.arange(0, 9, 0.1)        # 0부터 9까지 0.1 간격
y_new = f(x_new)                    # 보간된 y값 계산

# 결과 확인
print("Interpolated values:", y_new)
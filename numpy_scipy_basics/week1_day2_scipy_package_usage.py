import numpy as np
from scipy import stats, optimize, interpolate

# 통계 : 정규분포에서 난수 생성
norm_dist = stats.norm(loc=0, scale=1)
samples = norm_dist.rvs(size=1000)

# 최적화 : 함수의 최솟값 찾기
def f(x):
    return x**2 +10*np.sin(x)
result = optimize.minimize_scalar(f)
print("Minimum of f(x) = x^2 + 10sin(x):", result.x)

# 보간 : 1D 보간
x = np.arange(0, 10)
y = np.exp(-x/3.0)
f = interpolate.interp1d(x, y)
x_new = np. arange(0, 9, 0.1)
y_new = f(x_new)

# 결과 확인
print("Interpolated values:", y_new)
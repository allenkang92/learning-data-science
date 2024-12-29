import numpy as np
from scipy import linalg

# Numpy로 행렬 생성
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])

# Scipy의 linalg 모듈을 사용해 선형 방정식 풀기
x = linalg.solve(A, b)

print("Solution to Ax = b:")
print(x)

# 결과 확인
print("Verification, Ax:")
print(np.dot(A,x))
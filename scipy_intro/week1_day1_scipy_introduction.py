import numpy as np
from scipy import linalg

# 행렬 생성
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 행렬 곱
C = np.dot(A, B)
print("Matrix multpllication result:")
print(C)

# 역행렬 계산
A_inv = linalg.inv(A)
print("\nInverse of matrix A:")
print(A_inv)
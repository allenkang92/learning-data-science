import numpy as np
from scipy import linalg

# 행렬 생성
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 행렬 덧셈
C = A + B
print("Matrix addition:")
print(C)

# 행렬 곱셈
D = A @ B # or np.dot(A, B)
print("\nMatrix multiplication:")
print(D)

# 역행렬 계산
A_inv = linalg.inv(A)
print("\nInverse of A:")
print(A_inv)

# 행렬식 계산
det_A = linalg.det(A)
print("\nDeterminant of A:")
print(det_A)

# 선형 방정식 해결(Ax = b)
b = np.array([1, 2])
x = linalg.solve(A, b)
print("\nSolution to Ax = b:")
print(x)

# 고유값과 고유벡터 계산
eigenvalues, eigenvectors = linalg.eig(A)
print("\nEigenvalues of A:")
print(eigenvalues)
print("\nEigenvectors of A:")
print(eigenvectors)

# 특이값 분해(SVD)
print("\n7. Singular Value Decomposition:")
U, s, Vt = linalg.svd(A)
print("U:\n", U)
print("특이값:\n", s)
print("V의 전치:\n", Vt)
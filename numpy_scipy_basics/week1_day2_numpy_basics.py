import numpy as np

# 배열 생성
a = np.array([1, 2, 3, 4, 5])
b = np.zeros((3, 3))
c = np.ones((2, 2))
d = np.random.rand(2, 2)

# 배열 연산
print("Array addition:", a + 2)
print("Matrix multiplication:\n", np.dot(c, d))

# 배열 인덱싱과 슬라이싱
print("Slicing:", a[1:4])

# 배열 형태 변경
e = np. arange(10).reshape(2, 5)
print("Reshaped array:\n", e)
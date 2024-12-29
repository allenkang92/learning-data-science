"""
NumPy 기초 기능 실습
이 스크립트는 NumPy의 기본적인 배열 조작 기능을 실습하는 예제입니다.

주요 내용:
1. 다양한 방법으로 배열 생성
2. 기본적인 배열 연산
3. 배열 인덱싱과 슬라이싱
4. 배열 형태 변경

작성일: 2024-12-29
"""

import numpy as np

# 1. 배열 생성
# np.array: 리스트로부터 배열 생성
a = np.array([1, 2, 3, 4, 5])     # 1차원 배열

# np.zeros: 모든 원소가 0인 배열 생성
# - shape: (행, 열)
b = np.zeros((3, 3))              # 3x3 영행렬

# np.ones: 모든 원소가 1인 배열 생성
c = np.ones((2, 2))               # 2x2 일행렬

# np.random.rand: 0과 1 사이의 균일 분포에서 난수 생성
d = np.random.rand(2, 2)          # 2x2 난수 행렬

# 2. 배열 연산
# 스칼라 덧셈: 모든 원소에 2를 더함
print("Array addition:", a + 2)

# 행렬 곱셈: np.dot 사용
# - 두 행렬의 내적(dot product) 계산
print("Matrix multiplication:\n", np.dot(c, d))

# 3. 배열 인덱싱과 슬라이싱
# 슬라이싱: [start:end] 형식
# - 인덱스 1부터 3까지 (4 미포함)
print("Slicing:", a[1:4])

# 4. 배열 형태 변경
# np.arange: 연속된 숫자로 배열 생성
# reshape: 배열의 형태를 변경
e = np.arange(10).reshape(2, 5)   # 0~9를 2x5 행렬로 변경
print("Reshaped array:\n", e)
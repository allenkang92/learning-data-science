"""
SciPy 선형 방정식 풀이 예제
이 스크립트는 SciPy의 linalg 모듈을 사용하여 선형 방정식 Ax = b를 푸는 예제입니다.

문제:
    선형 방정식 Ax = b를 풀이
    여기서:
    A = [[1, 2],
         [3, 4]]
    b = [5, 6]

작성일: 2024-12-29
"""

import numpy as np
from scipy import linalg

# 1. 문제 설정
# 2x2 행렬 A와 벡터 b 생성
A = np.array([[1, 2],   # 2x2 행렬
              [3, 4]])
b = np.array([5, 6])    # 2차원 벡터

# 2. 선형 방정식 풀이
# linalg.solve: Ax = b 형태의 선형 방정식을 푸는 함수
# - A: 정방행렬 (square matrix)
# - b: 상수 벡터
x = linalg.solve(A, b)

# 3. 결과 출력
print("Solution to Ax = b:")
print(x)

# 4. 결과 검증
# np.dot: 행렬 곱셈 수행
# Ax가 b와 같은지 확인
print("Verification, Ax:")
print(np.dot(A,x))  # 이 값이 b와 같아야 함
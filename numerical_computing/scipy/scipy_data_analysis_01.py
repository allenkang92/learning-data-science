"""
SciPy를 이용한 데이터 분석 예제
이 스크립트는 SciPy의 stats 모듈을 사용하여 기본적인 통계 분석을 수행하는 예제입니다.

분석 내용:
1. 가상의 매출 데이터 생성 (정규분포 사용)
2. 기술 통계량 계산 (평균, 표준편차, 중앙값)
3. 95% 신뢰구간 계산
4. 히스토그램을 통한 데이터 시각화

작성일: 2024-12-29
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 1. 데이터 생성
# stats.norm.rvs: 정규분포에서 무작위 표본 추출
# - loc: 평균 (1000)
# - scale: 표준편차 (200)
# - size: 표본 크기 (100)
sales_data = stats.norm.rvs(loc=1000, scale=200, size=100)

# 2. 기술 통계량 계산
# np.mean: 평균값 계산
mean_sales = np.mean(sales_data)
# np.std: 표준편차 계산
std_sales = np.std(sales_data)
# np.median: 중앙값 계산
median_sales = np.median(sales_data)

# 계산된 통계량 출력
print(f"평균 매출: {mean_sales:.2f}")
print(f"매출 표준편차: {std_sales:.2f}")
print(f"중앙값 매출: {median_sales:.2f}")

# 3. 신뢰구간 계산
# stats.t.interval: t-분포를 사용한 신뢰구간 계산
# - confidence: 신뢰수준 (95%)
# - df: 자유도 (n-1)
# - loc: 표본평균
# - scale: 표준오차 (stats.sem 사용)
conf_int = stats.t.interval(confidence=0.95, 
                          df=len(sales_data)-1,
                          loc=mean_sales,
                          scale=stats.sem(sales_data))
print(f"95% 신뢰구간: {conf_int}")

# 4. 데이터 시각화
# plt.hist: 히스토그램 생성
# - bins: 구간 수
# - edgecolor: 막대 테두리 색상
plt.hist(sales_data, bins=20, edgecolor='black')
plt.title("매출 분포")
plt.xlabel("일일 매출")
plt.ylabel("빈도")
plt.show()

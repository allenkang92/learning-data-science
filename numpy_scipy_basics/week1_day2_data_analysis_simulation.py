import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 가상의 매출 데이터 생성(정규분포 가정)
sales_data = stats.norm.rvs(loc=1000, scale=200, size=100)

# 기본 통계량 계산
mean_sales = np.mean(sales_data)
std_sales = np.std(sales_data)
median_sales = np.median(sales_data)

print(f"평균 매출: {mean_sales:.2f}")
print(f"매출 표준편차 : {std_sales:.2f}")
print(f"중앙값 매출: {median_sales:.2f}")

# 신뢰구간 계산
conf_int = stats.t.interval(confidence=0.95, df=len(sales_data)-1,
                            loc=mean_sales,
                            scale=stats.sem(sales_data))
print(f"95% 신뢰구간: {conf_int}")

# 데이터 시각화
plt.hist(sales_data, bins=20, edgecolor= 'black')
plt.title("매출 분포")
plt.xlabel("일일 매출")
plt.ylabel("빈도")
plt.show()

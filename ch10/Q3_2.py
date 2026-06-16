#      heating_load  wall   roof  glazing  height
# 0            84.7  23.6  170.2     39.1     6.2
# 1            98.4  27.9  159.3     40.6     6.0
# 2            69.0  22.6  100.0     20.7     8.3
# 3            74.5  21.5  116.8     29.8     7.1
# 4            85.5  36.3  160.3     11.5     5.1
# ..            ...   ...    ...      ...     ...
# 545          79.5  35.8  100.2     47.2    12.2
# 546          87.3  23.9  177.6     21.1     5.4
# 547          78.8  31.8  145.8     23.0    10.2
# 548          71.2  27.3  102.5     43.4    13.5
# 549          79.9  33.9  177.9     46.7     4.7

# [550 rows x 5 columns]
# (550, 5)
# <class 'pandas.DataFrame'>
# RangeIndex: 550 entries, 0 to 549
# Data columns (total 5 columns):
#  #   Column        Non-Null Count  Dtype  
# ---  ------        --------------  -----  
#  0   heating_load  550 non-null    float64
#  1   wall          550 non-null    float64
#  2   roof          550 non-null    float64
#  3   glazing       550 non-null    float64
#  4   height        550 non-null    float64
# dtypes: float64(5)
# memory usage: 21.6 KB
# None

import pandas as pd

df = pd.read_csv("data/heating.csv")

# print(df)
# print(df.shape)
# print(df.info())

# 1 모든 독립변수를 포함한 회귀모형을 적합. 절편을 제외한 회귀계수의 합 구하기


# 2 유의한 변수(0.05 이하)만을 사용하여 다중회귀분석 다시 수행. R^2 값 구하기

# 3 새로운 건물(wall=20, roof=150, glazing=20, height=5)
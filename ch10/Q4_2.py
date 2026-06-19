import pandas as pd

df = pd.read_csv("data/cafe_sales.csv")

# print(df)
# print(df.shape)
# print(df.info())

#            order_date  category        item  price
# 0    2025.03.08 23:30       tea    GreenTea   6861
# 1    2025.10.01 13:05    coffee       Latte   6080
# 2    2025.04.30 17:32  smoothie       Mango   6116
# 3    2025.12.07 06:45       tea    BlackTea   7807
# 4    2024.04.10 00:40   dessert      Cookie   7964
# ..                ...       ...         ...    ...
# 995  2025.08.31 20:59   dessert     Brownie   3501
# 996  2025.10.04 04:56  smoothie       Mango   5190
# 997  2025.10.23 02:42  smoothie  Strawberry   4810
# 998  2025.07.03 20:42    coffee  Cappuccino   5896
# 999  2025.04.04 20:31  smoothie       Mango   4549

# [1000 rows x 4 columns]
# (1000, 4)
# <class 'pandas.DataFrame'>
# RangeIndex: 1000 entries, 0 to 999
# Data columns (total 4 columns):
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   order_date  1000 non-null   str  
#  1   category    1000 non-null   str  
#  2   item        1000 non-null   str  
#  3   price       1000 non-null   int64
# dtypes: int64(1), str(3)
# memory usage: 31.4 KB
# None

# 2-1 연-월별 총 매출액을 계산하여 큰 순서대로 정렬했을 때, 2번째로 큰 매출액을 입력하시오.
df["order_rate"]
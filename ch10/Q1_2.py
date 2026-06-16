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

import pandas as pd

df = pd.read_csv("data/cafe_sales.csv")

# print(df)
# print(df.shape)
# print(df.info())

# 1
df["order_date"] = pd.to_datetime(df["order_date"])
# print(df["order_date"])
# 0     2025-03-08 23:30:00
# 1     2025-10-01 13:05:00
# 2     2025-04-30 17:32:00
# 3     2025-12-07 06:45:00
# 4     2024-04-10 00:40:00
#               ...        
# 995   2025-08-31 20:59:00
# 996   2025-10-04 04:56:00
# 997   2025-10-23 02:42:00
# 998   2025-07-03 20:42:00
# 999   2025-04-04 20:31:00
df["year-month"] = df["order_date"].dt.to_period("M")
# print(df["year-month"])
# 0      2025-03
# 1      2025-10
# 2      2025-04
# 3      2025-12
# 4      2024-04
#         ...   
# 995    2025-08
# 996    2025-10
# 997    2025-10
# 998    2025-07
# 999    2025-04

df_1 = df[["year-month", "price"]]
# print(df_1)
re1 = df_1.groupby("year-month")["price"].sum().sort_values(ascending=False).iloc[1]
# print(re1) # 328741

# 2 -> 2024-10
monthly = df_1.groupby("year-month")["price"].sum().sort_values(ascending=False)
# print(monthly)
target_month = monthly.index[3]    # 4번째로 큰 연-월 (라벨)
# print(target_month)                # 예: 2024-10

cond = (df["year-month"] == target_month)
target_df = df[cond]
cat_sales = target_df.groupby("category")["price"].sum().sort_values(ascending=False).iloc[0]
print(cat_sales) # 104660
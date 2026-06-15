#     order_no   cus_id   product  quantity   price
# 0     536342  CUS0084   Monitor         2  110.02
# 1    C536505  CUS0058   Headset         2   52.11
# 2     536522  CUS0027   Headset         3   93.67
# 3     536544  CUS0077   Headset         3   74.05
# 4     536682  CUS0089   Headset         2  115.39
# ..       ...      ...       ...       ...     ...
# 951   536106  CUS0099    Laptop         2  107.95
# 952   536270  CUS0092   Headset         1   65.74
# 953  C536467  CUS0002   Monitor         1   33.90
# 954   536435  CUS0051    Laptop         1   15.22
# 955   536102  CUS0051  Keyboard         2   73.58

# [956 rows x 5 columns]
# (956, 5)
# <class 'pandas.DataFrame'>
# RangeIndex: 956 entries, 0 to 955
# Data columns (total 5 columns):
#  #   Column    Non-Null Count  Dtype  
# ---  ------    --------------  -----  
#  0   order_no  956 non-null    str    
#  1   cus_id    956 non-null    str    
#  2   product   956 non-null    str    
#  3   quantity  956 non-null    int64  
#  4   price     956 non-null    float64
# dtypes: float64(1), int64(1), str(3)
# memory usage: 37.5 KB
# None

import pandas as pd

df = pd.read_csv("data/order_data.csv")

# print(df)
# print(df.shape)
# print(df.info())
# 새로운 컬럼 order_ant 생성
df["order_ant"] = df["quantity"] * df["price"]
df["cancel_TF"] = df["order_no"].str.contains('C')
cond = (df["cancel_TF"] == True)
# print(cond)
df.loc[cond, "order_ant"] = df.loc[cond, "order_ant"] * (-1)
# print(df.loc[cond, "order_ant"])

# 3-1
# print(cond)
# print(df[cond])
cancel_df = df[cond]
# print(cancel_df["order_ant"].abs().idxmax()) # 185

# 3-2
cus = df.groupby("cus_id")["order_ant"].sum()
print(round(cus.max(), 3)) # 2877.21

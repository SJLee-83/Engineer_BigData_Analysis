import pandas as pd

df = pd.read_csv("data/chem.csv")

# print(df)
# print(df.shape)
# print(df.info())

#    sample   co  nmhc  etc
# 0     샘플1   79    54   31
# 1     샘플2   84    57   58
# 2     샘플3  109    74  113
# 3     샘플4   15    77   21
# 4     샘플5   65    77  115
# ..    ...  ...   ...  ...
# 95   샘플96   56    58   86
# 96   샘플97   33    59  111
# 97   샘플98   55    79  105
# 98   샘플99   69    51   59
# 99  샘플100   94    45   59

# [100 rows x 4 columns]
# (100, 4)
# <class 'pandas.DataFrame'>
# RangeIndex: 100 entries, 0 to 99
# Data columns (total 4 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   sample  100 non-null    str  
#  1   co      100 non-null    int64
#  2   nmhc    100 non-null    int64
#  3   etc     100 non-null    int64
# dtypes: int64(3), str(1)
# memory usage: 3.3 KB
# None

# co와 nmhc 컬럼을 Min-Max Scaling
from sklearn.preprocessing import MinMaxScaler
df[["co", "nmhc"]] = MinMaxScaler().fit_transform(df[["co", "nmhc"]])

# 스케일링된 co, nmhc 컬럼의 표준편차
co_std = df["co"].std()
nmhc_std = df["nmhc"].std()

# co의 표준편차와 nmhc의 표준편차를 뺀 값을 소수전 3자리로 반올림
answer = co_std - nmhc_std
print(round(answer, 3)) # -0.017
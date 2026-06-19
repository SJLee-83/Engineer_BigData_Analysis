import pandas as pd

df = pd.read_csv("data/tourist.csv")

# print(df)
# print(df.shape)
# print(df.info())

#        나라    관광   공무   사업   기타
# 0     국가1  1184  270  380   55
# 1     국가2  1059  184  267   86
# 2     국가3  1129  168  261   50
# 3     국가4   692  106  214  125
# 4     국가5  1335  296  296   84
# ..    ...   ...  ...  ...  ...
# 95   국가96   898  119  244  132
# 96   국가97  1111  195  234  118
# 97   국가98  1065  172  362   72
# 98   국가99  1408  254  288  149
# 99  국가100  1133  294  233  133

# [100 rows x 5 columns]
# (100, 5)
# <class 'pandas.DataFrame'>
# RangeIndex: 100 entries, 0 to 99
# Data columns (total 5 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   나라      100 non-null    str
#  1   관광      100 non-null    int64
#  2   공무      100 non-null    int64
#  3   사업      100 non-null    int64
#  4   기타      100 non-null    int64
# dtypes: int64(4), str(1)
# memory usage: 4.0 KB
# None

# 관광객비율이 두 번째로 높은 국가의 사업 방문객 수를 a
df["총합"] = (df["관광"]+df["공무"]+df["사업"]+df["기타"])
# df["총합2"] = df[["관광","공무","사업","기타"]].sum(axis=1)
# print(df)
df["관광객비율"] = df["관광"]/df["총합"]
a = df.sort_values("관광객비율", ascending=False).iloc[1]
# print(a)
a = 203
# 나라           국가85
# 관광           1499
# 공무            130
# 사업            203
# 기타            114
# 총합           1946
# 관광객비율    0.770298
# Name: 84, dtype: object

# 관관이 두 번째로 높은 국가읙 공무 방문객 수를 b
b = df.sort_values("관광", ascending=False).iloc[1]
# print(b)
b = 238
# 나라           국가42
# 관광           1484
# 공무            238
# 사업            366
# 기타            120
# 총합           2208
# 관광객비율    0.672101
# Name: 41, dtype: object

# a + b = ?
print(a + b) # 441
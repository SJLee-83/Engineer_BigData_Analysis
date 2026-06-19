import pandas as pd

train=pd.read_csv("data/gas_train.csv")
test=pd.read_csv("data/gas_test.csv")

# print(train)
# print("---")
# print(train.shape)
# print("---")
# print(train.info())
# print("---")
# print("---")
# print("---")
# print(test)
# print("---")
# print(test.shape)
# print("---")
# print(test.info())

#      시군구명  생활및판매  공공문화  복지의료  업무오락체육   총가스사용량
# 0     구로구      2     0     0       0   9077.8
# 1     구로구      6     0     1       2  10105.5
# 2     구로구     27     0     0       0   8603.6
# 3     구로구      2     0     0       0  11076.8
# 4     구로구     83     0     1      19  10781.4
# ...   ...    ...   ...   ...     ...      ...
# 3191   중구     13     1     0       2  12294.7
# 3192  성동구     27     0     1       3  10410.7
# 3193  성동구     62     0     0       0  10473.8
# 3194  성동구     38     0     0       0   9657.9
# 3195  성동구     29     1     0      14   9961.5

# [3196 rows x 6 columns]
# ---
# (3196, 6)
# ---
# <class 'pandas.DataFrame'>
# RangeIndex: 3196 entries, 0 to 3195
# Data columns (total 6 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   시군구명    3196 non-null   str    
#  1   생활및판매   3196 non-null   int64  
#  2   공공문화    3196 non-null   int64  
#  3   복지의료    3196 non-null   int64  
#  4   업무오락체육  3196 non-null   int64  
#  5   총가스사용량  3196 non-null   float64
# dtypes: float64(1), int64(4), str(1)
# memory usage: 149.9 KB
# None
# ---
# ---
# ---
#      시군구명  생활및판매  공공문화  복지의료  업무오락체육
# 0     구로구      2     0     0       0
# 1     구로구      6     0     1       2
# 2     구로구     27     0     0       0
# 3     구로구      2     0     0       0
# 4     구로구     80     0     1      19
# ...   ...    ...   ...   ...     ...
# 1471   중구     13     1     0       2
# 1472  성동구     24     0     1       1
# 1473  성동구     34     0     0       0
# 1474  성동구     30     0     0       0
# 1475  성동구     22     1     0      11

# [1476 rows x 5 columns]
# ---
# (1476, 5)
# ---
# <class 'pandas.DataFrame'>
# RangeIndex: 1476 entries, 0 to 1475
# Data columns (total 5 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   시군구명    1476 non-null   str  
#  1   생활및판매   1476 non-null   int64
#  2   공공문화    1476 non-null   int64
#  3   복지의료    1476 non-null   int64
#  4   업무오락체육  1476 non-null   int64
# dtypes: int64(4), str(1)
# memory usage: 57.8 KB
# None

X_train = train.drop(["시군구명", "총가스사용량"], axis=1)
y = train["총가스사용량"]

X_test = test.drop(["시군구명"], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

model1 = RandomForestRegressor(random_state=42)
model2 = RandomForestRegressor(n_estimators=200, random_state=42)
model3 = GradientBoostingRegressor(random_state=42)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

y_val_pred1 = model1.predict(X_val)
y_val_pred2 = model2.predict(X_val)
y_val_pred3 = model3.predict(X_val)

from sklearn.metrics import root_mean_squared_error
re1 = root_mean_squared_error(y_val, y_val_pred1)
re2 = root_mean_squared_error(y_val, y_val_pred2)
re3 = root_mean_squared_error(y_val, y_val_pred3)

# print(re1)
# print(re2)
# print(re3)

# 1131.2288529502387
# 1133.8452272452778
# 1243.3562541100478

y_pred = model1.predict(X_test)
result = pd.DataFrame(y_pred, columns=["pred"])
result.to_csv("result.csv", index=False)

print(pd.read_csv("result.csv"))
import pandas as pd

train = pd.read_csv("data/farm_train.csv")
test = pd.read_csv("data/farm_test.csv")

# print(train)
# print("---")
# print(train.shape)
# print("---")
# print(train.info())
# print("---")
# print("---")
# print(test)
# print("---")
# print(test.shape)
# print("---")
# print(test.info())

#               농업면적    연도  지역       비료사용량         비료잔여량 작물종류 토양유형  농약검출여부 등급
# 0     20079.652837  2004  대구  407.985516    146.290507   보리   점토       2  C
# 1     73858.643204  2012  울산  221.229692   1967.333638    밀   점토       0  B
# 2     65718.150861  2012  강원  370.967205   2253.522610    쌀   점토       0  B
# 3     37366.182902  2005  광주  274.128236   1487.535265    쌀   양토       0  B
# 4     81515.151289  2007  충남  213.410655    683.306745    쌀   양토       1  B
# ...            ...   ...  ..         ...           ...  ...  ...     ... ..
# 3995  44694.802107  2003  대전  393.291895    244.165391   보리   양토       2  C
# 3996  33128.645424  2017  울산  320.015330  30696.978909   보리   양토       0  A
# 3997   1471.231928  2010  울산  288.900758   5209.895639    밀   모래       0  A
# 3998  98663.569901  2020  광주  506.057309   1321.613918   보리   점토       2  B
# 3999   7774.163552  2013  대구  212.869117    359.236453    밀   점토       2  C

# [4000 rows x 9 columns]
# ---
# (4000, 9)
# ---
# <class 'pandas.DataFrame'>
# RangeIndex: 4000 entries, 0 to 3999
# Data columns (total 9 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   농업면적    4000 non-null   float64
#  1   연도      4000 non-null   int64  
#  2   지역      4000 non-null   str    
#  3   비료사용량   4000 non-null   float64
#  4   비료잔여량   4000 non-null   float64
#  5   작물종류    4000 non-null   str    
#  6   토양유형    4000 non-null   str    
#  7   농약검출여부  4000 non-null   int64  
#  8   등급      4000 non-null   str    
# dtypes: float64(3), int64(2), str(4)
# memory usage: 281.4 KB
# None
# ---
# ---
#              농업면적    연도  지역       비료사용량         비료잔여량 작물종류 토양유형 등급
# 0    43284.730127  2008  울산  138.767165    415.437269   보리   점토  B
# 1    97624.943779  2014  충북  373.560368   2809.120053   보리   양토  A
# 2    47115.257674  2023  대전  249.082661   1586.452521    쌀   점토  B
# 3    82817.237255  2012  충북  242.284660    142.494412    쌀   점토  C
# 4    73866.403857  2012  광주  166.180358   4365.543337    밀   점토  A
# ..            ...   ...  ..         ...           ...  ...  ... ..
# 995  97391.943065  2019  경기  235.737496    187.648314    쌀   점토  C
# 996  45914.535781  2015  세종  330.264101    136.146895    쌀   모래  C
# 997  11116.966854  2022  대전  311.614389      9.165499   보리   양토  C
# 998  51511.091047  2002  충남  302.713729   1064.587497    밀   점토  B
# 999  42508.529809  2018  서울  236.351129  10560.654573    밀   모래  A

# [1000 rows x 8 columns]
# ---
# (1000, 8)
# ---
# <class 'pandas.DataFrame'>
# RangeIndex: 1000 entries, 0 to 999
# Data columns (total 8 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   농업면적    1000 non-null   float64
#  1   연도      1000 non-null   int64  
#  2   지역      1000 non-null   str    
#  3   비료사용량   1000 non-null   float64
#  4   비료잔여량   1000 non-null   float64
#  5   작물종류    1000 non-null   str    
#  6   토양유형    1000 non-null   str    
#  7   등급      1000 non-null   str    
# dtypes: float64(3), int64(1), str(4)
# memory usage: 62.6 KB
# None

# 농작물에서 농약 검출 여부 예측
X = train.drop(["농약검출여부"], axis=1)
y = train["농약검출여부"]

X_full = pd.concat([X, test], axis=0)
# print(X_full.shape) # (5000, 8)

X_full = pd.get_dummies(X_full)

X_train = X_full[:4000]
X_test = X_full[4000:]

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
model1 = RandomForestClassifier(random_state=42)
model2 = RandomForestClassifier(n_estimators=200, random_state=42)
model3 = GradientBoostingClassifier(random_state=42)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

y_pred_val1 = model1.predict(X_val)
y_pred_val2 = model2.predict(X_val)
y_pred_val3 = model3.predict(X_val)

from sklearn.metrics import f1_score
result1 = f1_score(y_val, y_pred_val1, average="macro")
result2 = f1_score(y_val, y_pred_val2, average="macro")
result3 = f1_score(y_val, y_pred_val3, average="macro")

print(result1)
print("---")
print(result2) 
print("---")
print(result3)
# 0.8653752934125158
# ---
# 0.8683086814602956
# ---
# 0.912018966053628

y_pred = model3.predict(X_test)
result = pd.DataFrame(y_pred, columns=["pred"])
result.to_csv("result.csv", index=False)

print(pd.read_csv("result.csv"))
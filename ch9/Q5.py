import pandas as pd

train = pd.read_csv("data/farm_train.csv")
test = pd.read_csv("data/farm_test.csv")

# print(train.shape)
# print(test.shape)
# print(train.info())
# print(test.info())

# (4000, 9)
# (1000, 8)
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

# print(train.head())
# print(test.head())
#            농업면적    연도  지역       비료사용량        비료잔여량 작물종류 토양유형  농약검출여부 등급
# 0  20079.652837  2004  대구  407.985516   146.290507   보리   점토       2  C
# 1  73858.643204  2012  울산  221.229692  1967.333638    밀   점토       0  B
# 2  65718.150861  2012  강원  370.967205  2253.522610    쌀   점토       0  B
# 3  37366.182902  2005  광주  274.128236  1487.535265    쌀   양토       0  B
# 4  81515.151289  2007  충남  213.410655   683.306745    쌀   양토       1  B
#            농업면적    연도  지역       비료사용량        비료잔여량 작물종류 토양유형 등급
# 0  43284.730127  2008  울산  138.767165   415.437269   보리   점토  B
# 1  97624.943779  2014  충북  373.560368  2809.120053   보리   양토  A
# 2  47115.257674  2023  대전  249.082661  1586.452521    쌀   점토  B
# 3  82817.237255  2012  충북  242.284660   142.494412    쌀   점토  C
# 4  73866.403857  2012  광주  166.180358  4365.543337    밀   점토  A

X_train = train.drop(["연도", "지역", "농약검출여부"], axis=1)
y = train["농약검출여부"]

X_test = test.drop(["연도", "지역"], axis=1)

X_full = pd.concat([X_train, X_test], axis=0)
# print(X_full.shape)

X_full = pd.get_dummies(X_full)
# print(X_full)

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

y_val_pred1 = model1.predict(X_val)
y_val_pred2 = model2.predict(X_val)
y_val_pred3 = model3.predict(X_val)

from sklearn.metrics import f1_score
re1 = f1_score(y_val, y_val_pred1, average="macro")
re2 = f1_score(y_val, y_val_pred2, average="macro")
re3 = f1_score(y_val, y_val_pred3, average="macro")

# print(re1)
# print(re2)
# print(re3)

# 0.9071997236954562
# 0.9053276958318071
# 0.9227350255542627

y_pred = model3.predict(X_test)
result = pd.DataFrame(y_pred, columns=["pred"])
result.to_csv("result.csv", index=False)

print(pd.read_csv("result.csv"))
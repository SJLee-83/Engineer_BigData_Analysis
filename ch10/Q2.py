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
# (3196, 6)                                                                                                                                                                                                 
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
# (1476, 5)

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

import pandas as pd

train=pd.read_csv("data/gas_train.csv")
test=pd.read_csv("data/gas_test.csv")

# print(train)
# print(train.info())
# print(train.shape)
# print(test.info())
# print(test.shape)

X = train.drop(["총가스사용량", "시군구명"], axis=1)
y = train["총가스사용량"]

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)
# 1127.7654774015512

# from sklearn.ensemble import GradientBoostingRegressor
# model = GradientBoostingRegressor()
# model.fit(X_train, y_train)
# y_val_pred = model.predict(X_val)
# 1411.3551519951927

# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(n_estimators=200)
# model.fit(X_train, y_train)
# y_val_pred = model.predict(X_val)
# 1359.7366444882714

from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(y_val, y_val_pred)
# print(rmse)

test = test.drop(["시군구명"], axis=1)

y_pred = model.predict(test)
result = pd.DataFrame(y_pred, columns = ["pred"])
result.to_csv("result.csv", index=False)
print(result)

#               pred
# 0      8360.344365
# 1     10600.037402
# 2      9060.159031
# 3      8360.344365
# 4     11480.003786
# ...            ...
# 1471  11766.260414
# 1472  11298.696830
# 1473   9466.928923
# 1474   9645.906966
# 1475   9903.818937

# [1476 rows x 1 columns]
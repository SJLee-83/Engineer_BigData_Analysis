# (venv) C:\Users\great\Desktop\Engineer_BigData_Analysis\ch11>python Q2.py                                                                                                                                 
# <class 'pandas.DataFrame'>                                                                                                                                                                                
# RangeIndex: 2614 entries, 0 to 2613                                                                                                                                                                       
# Data columns (total 10 columns):                                                                                                                                                                          
#  #   Column              Non-Null Count  Dtype                                                                                                                                                            
# ---  ------              --------------  -----                                                                                                                                                            
#  0   Guild_Contribution  2614 non-null   float64                                                                                                                                                          
#  1   Character_Level     2614 non-null   int64                                                                                                                                                            
#  2   Weekly_Playtime     2614 non-null   float64                                                                                                                                                          
#  3   Chat_Activity       2614 non-null   int64                                                                                                                                                            
#  4   Days_Inactive       2614 non-null   int64                                                                                                                                                            
#  5   Guild_Rank          2614 non-null   int64                                                                                                                                                            
#  6   PVP_Score           2614 non-null   float64                                                                                                                                                          
#  7   Quest_Completed     2614 non-null   int64                                                                                                                                                            
#  8   Item_Level          2614 non-null   float64                                                                                                                                                          
#  9   User_Level          2614 non-null   int64                                                                                                                                                            
# dtypes: float64(4), int64(6)                                                                                                                                                                              
# memory usage: 204.3 KB                                                                                                                                                                                    
# None                                                                                                                                                                                                      
#       Guild_Contribution  Character_Level  Weekly_Playtime  Chat_Activity  Days_Inactive  Guild_Rank  PVP_Score  Quest_Completed  Item_Level  User_Level                                                  
# 0                 184.76               20            46.11             97             14           0     397.70               93      230.45           0                                                  
# 1                 987.68              120            58.79              0              0           5     999.00              471      250.54           2
# 2                 274.90               40            41.66            136              0           8     633.58              245      253.44           1
# 3                 581.71               83            15.56            186              1           3     144.42                0      200.18           1
# 4                 129.14               19            28.74            135              4           3     240.33              209      176.01           0
# ...                  ...              ...              ...            ...            ...         ...        ...              ...         ...         ...
# 2609              407.43               24            28.74            200              6           3     385.58              270      185.88           0
# 2610              570.01               95            30.72            159              6           5     365.22              223      460.43           1
# 2611              166.38               42            44.11              9             12           0     305.10              171      268.56           0
# 2612              357.05                1            26.92             11              9           0     548.67              347      286.64           0
# 2613              652.53               94            19.52             97             13           8     645.24              482      175.05           1

# [2614 rows x 10 columns]
# <class 'pandas.DataFrame'>
# RangeIndex: 903 entries, 0 to 902
# Data columns (total 9 columns):
#  #   Column              Non-Null Count  Dtype  
# ---  ------              --------------  -----  
#  0   Guild_Contribution  903 non-null    float64
#  1   Character_Level     903 non-null    int64  
#  2   Weekly_Playtime     903 non-null    float64
#  3   Chat_Activity       903 non-null    int64  
#  4   Days_Inactive       903 non-null    int64  
#  5   Guild_Rank          903 non-null    int64  
#  6   PVP_Score           903 non-null    float64
#  7   Quest_Completed     903 non-null    int64  
#  8   Item_Level          903 non-null    float64
# dtypes: float64(4), int64(5)
# memory usage: 63.6 KB
# None
#      Guild_Contribution  Character_Level  Weekly_Playtime  Chat_Activity  Days_Inactive  Guild_Rank  PVP_Score  Quest_Completed  Item_Level
# 0                634.80               47            29.47            103             18           1     193.46              240      143.62
# 1                318.91               45            24.57            176             12           0     173.30                0      205.63
# 2                182.23                1            18.98            159              6           1     227.91              152      100.00
# 3                590.06               13            13.23             67             16           0     294.10              400      196.88
# 4                276.95               56            34.14            277              2           0     401.53              434      402.83
# ..                  ...              ...              ...            ...            ...         ...        ...              ...         ...
# 898               11.16               34             1.27              0              1           5     250.75              233      157.50
# 899              184.47               40             3.19            165             21           3      75.35              358      353.18
# 900              384.69               11             7.02             76             18           6     373.18              167      142.70
# 901              196.60               24            24.92              0              0           1     298.99              355      136.54
# 902              721.66               34            70.00            216             11           3     286.05              420      456.01

# [903 rows x 9 columns]

import pandas as pd

train = pd.read_csv("data/game_train.csv")
test = pd.read_csv("data/game_test.csv")

# print(train.info())
# print(train)
# print(test.info())
# print(test)

X = train.drop(["User_Level"], axis=1)
y = train["User_Level"]

X_full = pd.concat([X, test], axis=0)

X_train = X_full[:train.shape[0]]
X_test = X_full[train.shape[0]:]

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2)
# print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
# (2091, 9) (523, 9) (2091,) (523,)

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# y_val_pred = model.predict(X_val)

# (venv) C:\Users\great\Desktop\Engineer_BigData_Analysis\ch11>python Q2.py
#      pred
# 0       0
# 1       0
# 2       0
# 3       0
# 4       1
# ..    ...
# 898     0
# 899     0
# 900     0
# 901     0
# 902     1

# [903 rows x 1 columns]

# from sklearn.ensemble import GradientBoostingClassifier
# model = GradientBoostingClassifier(random_state=42)
# model.fit(X_train, y_train)
# y_val_pred = model.predict(X_val)

#      pred
# 0       0
# 1       0
# 2       0
# 3       0
# 4       1
# ..    ...
# 898     0
# 899     0
# 900     0
# 901     0
# 902     1

# [903 rows x 1 columns]

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)

from sklearn.metrics import f1_score
f1 = f1_score(y_val, y_val_pred, average = 'macro')
print(f1)

# 0.7520642927381557 # RF
# 0.7964032629515904 # GB
# 0.7958878657977504 # RF + 트리수 증가

y_pred = model.predict(X_test)
result = pd.DataFrame(y_pred, columns=["pred"])
result.to_csv('result.csv', index=False)

result = pd.read_csv('result.csv')
print(result)
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

import pandas as pd

train = pd.read_csv("data/game_train.csv")
test = pd.read_csv("data/game_train.csv")

print(train.info())
print(train)
print(test.info())
print(test)

# X_train = train.drop(["User"])
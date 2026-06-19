import pandas as pd

df=pd.read_csv("data/elderly_health.csv")

train = df[:2000]
test = df[2000:]

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

#         id  age diabetic activity  glus_fast   bmi  blood_pressure  predicted
# 0        1   70       no       no        154  26.8             136          1
# 1        2   58       no      yes        113  23.7             136          0
# 2        3   55       no       no        156  25.5             141          0
# 3        4   50       no      yes        183  29.8             100          0
# 4        5   40       no       no        122  29.9             125          1
# ...    ...  ...      ...      ...        ...   ...             ...        ...
# 1995  1996   63       no      yes        105  24.3             129          0
# 1996  1997   41      pre      yes         94  27.7             114          0
# 1997  1998   56       no      yes        160  28.2             111          0
# 1998  1999   60       no      yes         95  31.5             131          0
# 1999  2000   47      pre      yes        116  20.4             104          0

# [2000 rows x 8 columns]
# ---
# (2000, 8)
# ---
# <class 'pandas.DataFrame'>
# RangeIndex: 2000 entries, 0 to 1999
# Data columns (total 8 columns):
#  #   Column          Non-Null Count  Dtype  
# ---  ------          --------------  -----  
#  0   id              2000 non-null   int64  
#  1   age             2000 non-null   int64  
#  2   diabetic        2000 non-null   str    
#  3   activity        2000 non-null   str    
#  4   glus_fast       2000 non-null   int64  
#  5   bmi             2000 non-null   float64
#  6   blood_pressure  2000 non-null   int64  
#  7   predicted       2000 non-null   int64  
# dtypes: float64(1), int64(5), str(2)
# memory usage: 125.1 KB
# None
# ---
# ---
# ---
#         id  age diabetic activity  glus_fast   bmi  blood_pressure  predicted
# 2000  2001   42       no       no        159  23.5             106          1
# 2001  2002   63       no       no        143  28.3             139          1
# 2002  2003   52      yes      yes        173  32.6             146          1
# 2003  2004   44       no       no         95  27.9             104          1
# 2004  2005   72      yes      yes        121  21.4             100          1
# ...    ...  ...      ...      ...        ...   ...             ...        ...
# 2222  2223   49      yes       no        185  29.7             149          1
# 2223  2224   68       no       no         94  30.2             132          1
# 2224  2225   49       no      yes        108  32.9             147          0
# 2225  2226   76       no       no        170  23.6             135          1
# 2226  2227   65      pre      yes         92  27.0             130          0

# [227 rows x 8 columns]
# ---
# (227, 8)
# ---
# <class 'pandas.DataFrame'>
# RangeIndex: 227 entries, 2000 to 2226
# Data columns (total 8 columns):
#  #   Column          Non-Null Count  Dtype  
# ---  ------          --------------  -----  
#  0   id              227 non-null    int64  
#  1   age             227 non-null    int64  
#  2   diabetic        227 non-null    str    
#  3   activity        227 non-null    str    
#  4   glus_fast       227 non-null    int64  
#  5   bmi             227 non-null    float64
#  6   blood_pressure  227 non-null    int64  
#  7   predicted       227 non-null    int64  
# dtypes: float64(1), int64(5), str(2)
# memory usage: 14.3 KB
# None

# 1-1 train 데이터에서 id 변수를 제외한 모든 변수를 활용하여 로지스틱 회귀분석을 수행하시오. diabetic이 'no'인 사람 대비 'yes'인 사람의 오즈비를 구하시오. (소수점 둘째자리까지) -> 20.36
from statsmodels.formula.api import logit
model = logit("predicted ~ age + diabetic + activity + glus_fast + bmi + blood_pressure", data=train).fit()
# print(model.summary())
# Optimization terminated successfully.
#          Current function value: 0.486192
#          Iterations 7
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:              predicted   No. Observations:                 2000
# Model:                          Logit   Df Residuals:                     1992
# Method:                           MLE   Df Model:                            7
# Date:                Fri, 19 Jun 2026   Pseudo R-squ.:                  0.2883
# Time:                        20:18:41   Log-Likelihood:                -972.38
# converged:                       True   LL-Null:                       -1366.3
# Covariance Type:            nonrobust   LLR p-value:                7.486e-166
# ===================================================================================
#                       coef    std err          z      P>|z|      [0.025      0.975]
# -----------------------------------------------------------------------------------
# Intercept          -6.7577      0.754     -8.960      0.000      -8.236      -5.279
# diabetic[T.pre]     1.3780      0.133     10.376      0.000       1.118       1.638
# diabetic[T.yes]     3.0138      0.231     13.055      0.000       2.561       3.466
# activity[T.yes]    -1.4809      0.121    -12.246      0.000      -1.718      -1.244
# age                 0.0869      0.005     15.991      0.000       0.076       0.098
# glus_fast           0.0211      0.002     11.304      0.000       0.017       0.025
# bmi                -0.0171      0.015     -1.160      0.246      -0.046       0.012
# blood_pressure     -0.0017      0.004     -0.450      0.653      -0.009       0.006
# ===================================================================================

import numpy as np
coef = np.exp(model.params).drop(["Intercept"])
# print(round(coef.iloc[1], 2))

# print(help(np.exp))

# 1-2 위 모델에서 test 데이터 중 노령층일 확률이 가장 높은 사람의 glus_fast 값을 구하시오. (정수로 작성) -> 140
y_pred = model.predict(test)
# print(y_pred.idxmax()) 

# 2152,60,no,no,140,27.7,105,1

# 1-3 위 모델로 test 데이터를 예측한 후, 확률이 0.2 초과인 사람을 노령층으로 분류했을 때 민감도를 구하시오. (반올림하여 소수 둘째 자리까지 작성)
# print(y_pred)
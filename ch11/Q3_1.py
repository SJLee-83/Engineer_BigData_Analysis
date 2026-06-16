#         id  age diabetic activity  glus_fast   bmi  blood_pressure  predicted
# 0        1   70       no       no        154  26.8             136          1
# 1        2   58       no      yes        113  23.7             136          0
# 2        3   55       no       no        156  25.5             141          0
# 3        4   50       no      yes        183  29.8             100          0
# 4        5   40       no       no        122  29.9             125          1
# ...    ...  ...      ...      ...        ...   ...             ...        ...
# 2222  2223   49      yes       no        185  29.7             149          1
# 2223  2224   68       no       no         94  30.2             132          1
# 2224  2225   49       no      yes        108  32.9             147          0
# 2225  2226   76       no       no        170  23.6             135          1
# 2226  2227   65      pre      yes         92  27.0             130          0

# [2227 rows x 8 columns]
# <class 'pandas.DataFrame'>
# RangeIndex: 2227 entries, 0 to 2226
# Data columns (total 8 columns):
#  #   Column          Non-Null Count  Dtype  
# ---  ------          --------------  -----  
#  0   id              2227 non-null   int64  
#  1   age             2227 non-null   int64  
#  2   diabetic        2227 non-null   str    
#  3   activity        2227 non-null   str    
#  4   glus_fast       2227 non-null   int64  
#  5   bmi             2227 non-null   float64
#  6   blood_pressure  2227 non-null   int64  
#  7   predicted       2227 non-null   int64  
# dtypes: float64(1), int64(5), str(2)
# memory usage: 139.3 KB
# None
# (2227, 8)

import pandas as pd
import numpy as np

df=pd.read_csv("data/elderly_health.csv")

# print(df)
# print(df.info())
# print(df.shape)

# 1-1 id 변수를 제외한 모든 변수를 활용하여 로지스틱 회귀분석 수행 & diabetic이 'no'인 사람 대비 'yes'인 사람의 오즈비 구하기
df_1 = df.drop(["id"], axis=1)
# print(df_1)

train = df_1[:2000]
test = df_1[2000:]

import statsmodels.api as sm
from statsmodels.formula.api import logit

model = logit("predicted ~ age + diabetic + activity + glus_fast + bmi + blood_pressure", data=train).fit()
# print(model.summary())

odds_ratio = np.exp(model.params)
# print(round(odds_ratio, 2))

# Optimization terminated successfully.
#          Current function value: 0.486192
#          Iterations 7
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:              predicted   No. Observations:                 2000
# Model:                          Logit   Df Residuals:                     1992
# Method:                           MLE   Df Model:                            7
# Date:                Tue, 16 Jun 2026   Pseudo R-squ.:                  0.2883
# Time:                        18:59:06   Log-Likelihood:                -972.38
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
# Intercept           0.00
# diabetic[T.pre]     3.97
# diabetic[T.yes]    20.36 -> 정답
# activity[T.yes]     0.23
# age                 1.09
# glus_fast           1.02
# bmi                 0.98
# blood_pressure      1.00
# dtype: float64

# 1-2 test 데이터 중 노령층일 확률이 가장 높은 사람
pred = model.predict(test)
# print(pred)

max_idx = pred.idxmax()
print(test.loc[max_idx]) # 2152
# Optimization terminated successfully.
#          Current function value: 0.486192
#          Iterations 7
# age                 78
# diabetic           pre
# activity            no
# glus_fast          181 <-
# bmi               26.3
# blood_pressure     115
# predicted            1
# Name: 2152, dtype: object

# 1-3
from sklearn.metrics import recall_score

pred = (pred > 0.2).astype(int)
sensitivity = recall_score(test['predicted'], pred)
print(round(sensitivity, 2)) # 0.94
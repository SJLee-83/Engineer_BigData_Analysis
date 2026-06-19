import pandas as pd

df = pd.read_csv("data/churn.csv")

# print(df)
# print(df.shape)
# print(df.info())

#      Churn  AccountWeeks  ContractRenewal  DataPlan  DataUsage  CustServCalls  DayMins  DayCalls  MonthlyCharge  OverageFee  RoamMins
# 0        0            51                1         0        1.0              0    303.7        83           72.5         9.3       3.8
# 1        1            61                1         0       -0.2              2    194.8        89           66.3        10.9      13.3
# 2        0            84                1         1       -1.3              2    157.6       109           35.7        10.0      12.5
# 3        0            65                1         0        1.0              1    223.1       129           64.2         9.1       5.1
# 4        0           154                1         0        1.0              0    137.0       115           40.0         7.7       4.6
# ..     ...           ...              ...       ...        ...            ...      ...       ...            ...         ...       ...
# 995      0            32                1         0        0.6              1    241.2        94           64.1         8.5      13.4
# 996      0            24                0         0       -1.1              1    155.2        97           48.5         9.8       8.3
# 997      0            67                1         0        1.9              3    184.6        69           45.8         7.3       8.5
# 998      0            72                1         0        1.0              0    153.2        65           34.8         9.6      14.5
# 999      0           116                1         0        0.8              0    207.6       110           46.4        10.9      11.8

# [1000 rows x 11 columns]
# (1000, 11)
# <class 'pandas.DataFrame'>
# RangeIndex: 1000 entries, 0 to 999
# Data columns (total 11 columns):
#  #   Column           Non-Null Count  Dtype  
# ---  ------           --------------  -----  
#  0   Churn            1000 non-null   int64  
#  1   AccountWeeks     1000 non-null   int64  
#  2   ContractRenewal  1000 non-null   int64  
#  3   DataPlan         1000 non-null   int64  
#  4   DataUsage        1000 non-null   float64
#  5   CustServCalls    1000 non-null   int64  
#  6   DayMins          1000 non-null   float64
#  7   DayCalls         1000 non-null   int64  
#  8   MonthlyCharge    1000 non-null   float64
#  9   OverageFee       1000 non-null   float64
#  10  RoamMins         1000 non-null   float64
# dtypes: float64(5), int64(6)
# memory usage: 86.1 KB
# None

# 로지스틱 획귀 분석을 수행해 p-value가 0.05 이상인 유의하지 않은 독립변수 개수
from statsmodels.formula.api import logit

model = logit("Churn ~ AccountWeeks + ContractRenewal + DataPlan + DataUsage + CustServCalls + DayMins + MonthlyCharge + OverageFee + RoamMins", data = df).fit()
# print(model.summary()) # 7개
# Optimization terminated successfully.
#          Current function value: 0.393742
#          Iterations 6
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:                  Churn   No. Observations:                 1000
# Model:                          Logit   Df Residuals:                      990
# Method:                           MLE   Df Model:                            9
# Date:                Fri, 19 Jun 2026   Pseudo R-squ.:                 0.02332
# Time:                        11:51:18   Log-Likelihood:                -393.74
# converged:                       True   LL-Null:                       -403.14
# Covariance Type:            nonrobust   LLR p-value:                   0.02690
# ===================================================================================
#                       coef    std err          z      P>|z|      [0.025      0.975]
# -----------------------------------------------------------------------------------
# Intercept          -1.9726      0.788     -2.505      0.012      -3.516      -0.429
# AccountWeeks        0.0026      0.002      1.097      0.273      -0.002       0.007 v
# ContractRenewal     0.1649      0.325      0.507      0.612      -0.472       0.802 v
# DataPlan            0.2881      0.198      1.458      0.145      -0.099       0.675 v
# DataUsage          -0.1728      0.072     -2.391      0.017      -0.314      -0.031
# CustServCalls       0.1350      0.074      1.829      0.067      -0.010       0.280 v
# DayMins            -0.0036      0.002     -2.091      0.036      -0.007      -0.000
# MonthlyCharge       0.0041      0.005      0.771      0.441      -0.006       0.015 v
# OverageFee         -0.0128      0.036     -0.354      0.724      -0.084       0.058 v
# RoamMins            0.0100      0.034      0.298      0.766      -0.056       0.076 v
# ===================================================================================

# p-value가 0.05 미만인 유의한 변수만을 사용해 다시 로지스틱 회귀 분석 수행. 유의한 회귀 계수의 합
model = logit("Churn ~ DataUsage + DayMins", data = df).fit()
# print(model.summary())
# Optimization terminated successfully.
#          Current function value: 0.393742
#          Iterations 6
# Optimization terminated successfully.
#          Current function value: 0.397599
#          Iterations 6
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:                  Churn   No. Observations:                 1000
# Model:                          Logit   Df Residuals:                      997
# Method:                           MLE   Df Model:                            2
# Date:                Fri, 19 Jun 2026   Pseudo R-squ.:                 0.01375
# Time:                        12:00:21   Log-Likelihood:                -397.60
# converged:                       True   LL-Null:                       -403.14
# Covariance Type:            nonrobust   LLR p-value:                  0.003908
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     -1.0395      0.303     -3.434      0.001      -1.633      -0.446
# DataUsage     -0.1697      0.071     -2.376      0.017      -0.310      -0.030
# DayMins       -0.0039      0.002     -2.264      0.024      -0.007      -0.001
# ==============================================================================
answer = model.params["DataUsage"] + model.params["DayMins"]
# print(round(answer,3)) # -0.174

# DataUsage 변수가 5만큼 증가할 때 오즈비 
import numpy as np
coef = model.params['DataUsage']

print(round(np.exp(coef*5), 3)) # 0.428
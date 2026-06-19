import pandas as pd

df = pd.read_csv("data/attrition.csv")

# print(df)
# print(df.shape)
# print(df.info())

#       attrition  age  income  overtime
# 0             0   44    6467         1
# 1             0   41    6201         0
# 2             1   25    4017         1
# 3             0   28    6360         1
# 4             1   58    2369         0
# ...         ...  ...     ...       ...
# 5995          1   25    2950         1
# 5996          1   56    5786         0
# 5997          1   53    5884         0
# 5998          1   38    3897         0
# 5999          1   49    2556         0

# [6000 rows x 4 columns]
# (6000, 4)
# <class 'pandas.DataFrame'>
# RangeIndex: 6000 entries, 0 to 5999
# Data columns (total 4 columns):
#  #   Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   attrition  6000 non-null   int64
#  1   age        6000 non-null   int64
#  2   income     6000 non-null   int64
#  3   overtime   6000 non-null   int64
# dtypes: int64(4)
# memory usage: 187.6 KB
# None

# 1-1 로지스틱 회귀모델을 적합한 후, p-value가 0.05보다 작은 변수의 회귀계수를 구하시오. 단 절편항은 제외.(반올림하여 소수 셋째 자리까지 작성) -> -0.002
from statsmodels.formula.api import logit
model = logit("attrition ~ age + income + overtime", data=df).fit()
# print(model.summary())

# Optimization terminated successfully.
#          Current function value: 0.387298
#          Iterations 7
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:              attrition   No. Observations:                 6000
# Model:                          Logit   Df Residuals:                     5996
# Method:                           MLE   Df Model:                            3
# Date:                Fri, 19 Jun 2026   Pseudo R-squ.:                  0.4215
# Time:                        22:22:09   Log-Likelihood:                -2323.8
# converged:                       True   LL-Null:                       -4016.9
# Covariance Type:            nonrobust   LLR p-value:                     0.000
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      7.7724      0.246     31.650      0.000       7.291       8.254
# age           -0.0044      0.004     -1.208      0.227      -0.012       0.003
# income        -0.0015   3.66e-05    -41.263      0.000      -0.002      -0.001
# overtime       0.0397      0.074      0.537      0.592      -0.105       0.185
# ==============================================================================
# print(round(model.params["income"],3))

# 1-2 나이가 1 증가할 때 이직할 오즈비를 구하시오.(반올림하여 소수 셋째 자리까지 작성) -> 0.996
import numpy as np

odds_ratio = np.exp(model.params["age"])
print(round(odds_ratio, 3))

# 1-3 새로운 직원(age=40, income=4500, overtime=1)에 대해, 모델을 이용하여 이직 확률을 예측하시오.(반올림하여 소수 셋째 자리까지 작성) -> 0.697
new = pd.DataFrame({"age": [40], "income": [4500], "overtime": [1]})
pred = model.predict(new)
print(round(pred, 3))
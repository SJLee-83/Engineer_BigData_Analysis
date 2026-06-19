import pandas as pd

df = pd.read_csv("data/retention.csv")

# print(df)
# print(df.shape)
# print(df.info())

#     CustomerID  MonthlyCharges  CustomerTenure  HasPhoneService  HasTechInsurance  Churn
# 0            1       77.450712              47                1                 1      1
# 1            2       67.926035              68                1                 1      0
# 2            3       79.715328              45                0                 1      1
# 3            4       92.845448               2                1                 0      1
# 4            5       66.487699              27                1                 1      0
# ..         ...             ...             ...              ...               ...    ...
# 75          76       82.328538              29                1                 1      0
# 76          77       71.305706              63                1                 0      1
# 77          78       65.514890              22                1                 0      0
# 78          79       71.376412              26                0                 0      1
# 79          80       40.186466              28                1                 0      0

# [80 rows x 6 columns]
# (80, 6)
# <class 'pandas.DataFrame'>
# RangeIndex: 80 entries, 0 to 79
# Data columns (total 6 columns):
#  #   Column            Non-Null Count  Dtype  
# ---  ------            --------------  -----  
#  0   CustomerID        80 non-null     int64  
#  1   MonthlyCharges    80 non-null     float64
#  2   CustomerTenure    80 non-null     int64  
#  3   HasPhoneService   80 non-null     int64  
#  4   HasTechInsurance  80 non-null     int64  
#  5   Churn             80 non-null     int64  
# dtypes: float64(1), int64(5)
# memory usage: 3.9 KB
# None

# 2-1 -> 0.008
from statsmodels.formula.api import logit
model = logit("Churn ~ MonthlyCharges + CustomerTenure + HasPhoneService + HasTechInsurance", data=df).fit()
# print(model.summary())
# Optimization terminated successfully.
#          Current function value: 0.582234
#          Iterations 6
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:                  Churn   No. Observations:                   80
# Model:                          Logit   Df Residuals:                       75
# Method:                           MLE   Df Model:                            4
# Date:                Sat, 20 Jun 2026   Pseudo R-squ.:                  0.1585
# Time:                        06:43:44   Log-Likelihood:                -46.579
# converged:                       True   LL-Null:                       -55.352
# Covariance Type:            nonrobust   LLR p-value:                  0.001513
# ====================================================================================
#                        coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------------
# Intercept           -4.4731      1.437     -3.114      0.002      -7.289      -1.657
# MonthlyCharges       0.0503      0.019      2.640      0.008       0.013       0.088
# CustomerTenure       0.0428      0.014      3.016      0.003       0.015       0.071
# HasPhoneService     -0.3558      0.525     -0.677      0.498      -1.386       0.674
# HasTechInsurance    -0.4868      0.518     -0.940      0.347      -1.502       0.529
# ====================================================================================
# print(round(model.pvalues["MonthlyCharges"], 3))

# 2-2 -> 0.701
import numpy as np

odds_ratio = np.exp(model.params["HasPhoneService"])
# print(round(odds_ratio,3))

# 2-3 -> 80
pred = model.predict(df)
# print(pred)
print(sum(pred > 0.3)) # 65
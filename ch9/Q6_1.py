import pandas as pd

df = pd.read_csv("data/design.csv")

train = df[:140]
test = df[140:]

# print(train.head())
# print(train.shape)
# print(train.info())
# print("======")
# print("======")
# print(test.head())
# print(test.shape)
# print(test.info())

#    id     design        c1        c2        c3        c4        c5
# 0   1  58.962471  0.374540  0.642032  0.103124  0.168935  0.707239
# 1   2  42.006534  0.950714  0.084140  0.902553  0.278590  0.152539
# 2   3  55.831980  0.731994  0.161629  0.505252  0.177010  0.576288
# 3   4  64.458592  0.598658  0.898554  0.826457  0.088703  0.606715
# 4   5  61.342792  0.156019  0.606429  0.320050  0.120636  0.424131
# (140, 7)
# <class 'pandas.DataFrame'>
# RangeIndex: 140 entries, 0 to 139
# Data columns (total 7 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   id      140 non-null    int64  
#  1   design  140 non-null    float64
#  2   c1      140 non-null    float64
#  3   c2      140 non-null    float64
#  4   c3      140 non-null    float64
#  5   c4      140 non-null    float64
#  6   c5      140 non-null    float64
# dtypes: float64(6), int64(1)
# memory usage: 7.8 KB
# None
# ======
# ======
#       id     design        c1        c2        c3        c4        c5
# 140  141  66.574168  0.962447  0.491616  0.954051  0.462680  0.274055
# 141  142  48.521635  0.251782  0.473472  0.606175  0.301378  0.554178
# 142  143  57.754669  0.497249  0.173202  0.228643  0.747609  0.651420
# 143  144  52.072094  0.300878  0.433852  0.671701  0.502720  0.829742
# 144  145  65.085687  0.284840  0.398505  0.618128  0.232213  0.206421
# (60, 7)
# <class 'pandas.DataFrame'>
# RangeIndex: 60 entries, 140 to 199
# Data columns (total 7 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   id      60 non-null     int64  
#  1   design  60 non-null     float64
#  2   c1      60 non-null     float64
#  3   c2      60 non-null     float64
#  4   c3      60 non-null     float64
#  5   c4      60 non-null     float64
#  6   c5      60 non-null     float64
# dtypes: float64(6), int64(1)
# memory usage: 3.4 KB
# None

# 1-1 -> 3개
from statsmodels.formula.api import ols
model = ols("design ~ c1 + c2 + c3 + c4 + c5", data=train).fit()
# print(model.summary())
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                 design   R-squared:                       0.266
# Model:                            OLS   Adj. R-squared:                  0.238
# Method:                 Least Squares   F-statistic:                     9.697
# Date:                Sat, 20 Jun 2026   Prob (F-statistic):           6.37e-08
# Time:                        06:27:12   Log-Likelihood:                -468.72
# No. Observations:                 140   AIC:                             949.4
# Df Residuals:                     134   BIC:                             967.1
# Df Model:                           5                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     53.0220      2.294     23.112      0.000      48.485      57.559
# c1            -4.9403      2.008     -2.460      0.015      -8.912      -0.969
# c2            11.3795      2.045      5.564      0.000       7.335      15.424
# c3             2.6960      1.920      1.404      0.163      -1.101       6.493
# c4             5.7978      2.176      2.664      0.009       1.494      10.102
# c5            -1.5018      2.067     -0.726      0.469      -5.590       2.587
# ==============================================================================
# Omnibus:                        3.969   Durbin-Watson:                   2.052
# Prob(Omnibus):                  0.137   Jarque-Bera (JB):                3.019
# Skew:                           0.227   Prob(JB):                        0.221
# Kurtosis:                       2.441   Cond. No.                         7.68
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# 1-2 -> 0.501
from statsmodels.formula.api import ols
model = ols("design ~ c1 + c2 + c4", data=train).fit()
# print(model.summary())
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                 design   R-squared:                       0.251
# Model:                            OLS   Adj. R-squared:                  0.234
# Method:                 Least Squares   F-statistic:                     15.17
# Date:                Sat, 20 Jun 2026   Prob (F-statistic):           1.43e-08
# Time:                        06:29:00   Log-Likelihood:                -470.14
# No. Observations:                 140   AIC:                             948.3
# Df Residuals:                     136   BIC:                             960.0
# Df Model:                           3                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     53.6637      1.808     29.686      0.000      50.089      57.239
# c1            -4.8767      2.010     -2.426      0.017      -8.852      -0.901
# c2            11.1863      2.039      5.487      0.000       7.154      15.218
# c4             6.2272      2.159      2.885      0.005       1.958      10.496
# ==============================================================================
# Omnibus:                        3.056   Durbin-Watson:                   1.972
# Prob(Omnibus):                  0.217   Jarque-Bera (JB):                2.947
# Skew:                           0.296   Prob(JB):                        0.229
# Kurtosis:                       2.606   Cond. No.                         5.96
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

train_pred = model.predict(train)
# print(train_pred)
print(round(train_pred.corr(train["design"]), 3))

# 1-3 -> 8.488
pred = model.predict(test)
# print(pred)

from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(test["design"], pred)
# print(round(rmse, 3))
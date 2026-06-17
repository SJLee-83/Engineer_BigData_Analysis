import pandas as pd

df = pd.read_csv("data/design.csv")

# print(df)
# print(df.shape)
# print(df.info())

#       id     design        c1        c2        c3        c4        c5
# 0      1  58.962471  0.374540  0.642032  0.103124  0.168935  0.707239
# 1      2  42.006534  0.950714  0.084140  0.902553  0.278590  0.152539
# 2      3  55.831980  0.731994  0.161629  0.505252  0.177010  0.576288
# 3      4  64.458592  0.598658  0.898554  0.826457  0.088703  0.606715
# 4      5  61.342792  0.156019  0.606429  0.320050  0.120636  0.424131
# ..   ...        ...       ...       ...       ...       ...       ...
# 195  196  49.976045  0.349210  0.930757  0.473962  0.872124  0.091582
# 196  197  61.148418  0.725956  0.858413  0.667558  0.932118  0.917314
# 197  198  38.647434  0.897110  0.428994  0.172320  0.565133  0.136819
# 198  199  49.458373  0.887086  0.750871  0.192289  0.696651  0.950237
# 199  200  69.747684  0.779876  0.754543  0.040869  0.922499  0.446006

# [200 rows x 7 columns]
# (200, 7)
# <class 'pandas.DataFrame'>
# RangeIndex: 200 entries, 0 to 199
# Data columns (total 7 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   id      200 non-null    int64  
#  1   design  200 non-null    float64
#  2   c1      200 non-null    float64
#  3   c2      200 non-null    float64
#  4   c3      200 non-null    float64
#  5   c4      200 non-null    float64
#  6   c5      200 non-null    float64
# dtypes: float64(6), int64(1)
# memory usage: 11.1 KB
# None

train = df[:140]
test = df[140:]

# 1 train 데이터에서 c1, c2, c3, c4, c5를 독립변수로, design을 종속 변수로 하는 다중 회구모형을 적합한 후, 절편을 제외한 독립변수 중 p-value가 0.05보다 작은 변수의 개수를 구하시오.
from statsmodels.formula.api import ols
model = ols("design ~ c1 + c2 + c3 + c4 + c5", data=train).fit()
# print(model.summary()) # 3
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                 design   R-squared:                       0.266
# Model:                            OLS   Adj. R-squared:                  0.238
# Method:                 Least Squares   F-statistic:                     9.697
# Date:                Wed, 17 Jun 2026   Prob (F-statistic):           6.37e-08
# Time:                        22:34:28   Log-Likelihood:                -468.72
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

# 2 1에서 적합한 회귀모형의 결과를 바탕으로, p-value가 0.05 이하인 유의한 독립변수만을 선택하여 새로운 다중 회귀모형을 적합한 후, train 데이터에서 design의 예측값을 산출하고 예측값과 실제 design 값 
# 사이의 피어슨 상관계수를 반올림하여 소수 셋째 자리까지 구하시오.
model = ols("design ~ c1 + c2 + c4", data=train).fit()
# print(model.summary())
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                 design   R-squared:                       0.251
# Model:                            OLS   Adj. R-squared:                  0.234
# Method:                 Least Squares   F-statistic:                     15.17
# Date:                Wed, 17 Jun 2026   Prob (F-statistic):           1.43e-08
# Time:                        22:35:14   Log-Likelihood:                -470.14
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
pred = model.predict(train)
corr = df["design"].corr(pred)
# print(round(corr, 3)) # 0.501

# 3 2에서 적합한 회귀모형을 이용하여 test 데이터에서 design의 예측값을 산출한 후, 예측값과 실제 design 값 사이의 RMSE를 반올림하여 소수 셋째 자리까지 계산하시오.
pred_test = model.predict(test)
# print(pred_test)

from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(test["design"], pred_test)
print(round(rmse, 3)) # 8.488


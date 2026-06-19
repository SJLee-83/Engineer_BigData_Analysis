import pandas as pd

df = pd.read_csv("data/heating.csv")

# print(df)
# print(df.shape)
# print(df.info())

#      heating_load  wall   roof  glazing  height
# 0            84.7  23.6  170.2     39.1     6.2
# 1            98.4  27.9  159.3     40.6     6.0
# 2            69.0  22.6  100.0     20.7     8.3
# 3            74.5  21.5  116.8     29.8     7.1
# 4            85.5  36.3  160.3     11.5     5.1
# ..            ...   ...    ...      ...     ...
# 545          79.5  35.8  100.2     47.2    12.2
# 546          87.3  23.9  177.6     21.1     5.4
# 547          78.8  31.8  145.8     23.0    10.2
# 548          71.2  27.3  102.5     43.4    13.5
# 549          79.9  33.9  177.9     46.7     4.7

# [550 rows x 5 columns]
# (550, 5)
# <class 'pandas.DataFrame'>
# RangeIndex: 550 entries, 0 to 549
# Data columns (total 5 columns):
#  #   Column        Non-Null Count  Dtype  
# ---  ------        --------------  -----  
#  0   heating_load  550 non-null    float64
#  1   wall          550 non-null    float64
#  2   roof          550 non-null    float64
#  3   glazing       550 non-null    float64
#  4   height        550 non-null    float64
# dtypes: float64(5)
# memory usage: 21.6 KB
# None

# 2-1 모든 독립변수를 포함한 회귀모형을 적합하시오. 이때 절편을 제외한 회귀계수의 합을 구하시오. (반올림하여 소수 셋째 자리까지 작성) -> 0.254
from statsmodels.formula.api import ols
model = ols("heating_load ~ wall + roof + glazing + height", data=df).fit()
# print(model.summary())
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:           heating_load   R-squared:                       0.754
# Model:                            OLS   Adj. R-squared:                  0.752
# Method:                 Least Squares   F-statistic:                     417.8
# Date:                Fri, 19 Jun 2026   Prob (F-statistic):          2.02e-164
# Time:                        22:33:38   Log-Likelihood:                -1772.0
# No. Observations:                 550   AIC:                             3554.
# Df Residuals:                     545   BIC:                             3576.
# Df Model:                           4                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     38.3821      1.504     25.517      0.000      35.427      41.337
# wall           0.0304      0.030      1.012      0.312      -0.029       0.090
# roof           0.2483      0.006     39.564      0.000       0.236       0.261
# glazing        0.2217      0.023      9.693      0.000       0.177       0.267
# height        -0.2469      0.077     -3.212      0.001      -0.398      -0.096
# ==============================================================================
# Omnibus:                        0.698   Durbin-Watson:                   1.896
# Prob(Omnibus):                  0.705   Jarque-Bera (JB):                0.785
# Skew:                          -0.077   Prob(JB):                        0.675
# Kurtosis:                       2.899   Cond. No.                         805.
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
re_sum = model.params["wall"] + model.params["roof"] + model.params["glazing"] + model.params["height"]
# print(round(re_sum, 3))

# 2-2 유의한 변수(0.05 이하)만을 사용하여 다중회귀분석을 다시 수행하고, 결정계수 값을 구하시오. (반올림하여 소수 셋째 자리까지 작성) -> 0.754
from statsmodels.formula.api import ols
model = ols("heating_load ~ roof + glazing + height", data=df).fit()
# print(model.summary())
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:           heating_load   R-squared:                       0.754
# Model:                            OLS   Adj. R-squared:                  0.752
# Method:                 Least Squares   F-statistic:                     556.7
# Date:                Fri, 19 Jun 2026   Prob (F-statistic):          1.31e-165
# Time:                        22:36:34   Log-Likelihood:                -1772.5
# No. Observations:                 550   AIC:                             3553.
# Df Residuals:                     546   BIC:                             3570.
# Df Model:                           3                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     39.2337      1.247     31.467      0.000      36.784      41.683
# roof           0.2480      0.006     39.559      0.000       0.236       0.260
# glazing        0.2210      0.023      9.668      0.000       0.176       0.266
# height        -0.2500      0.077     -3.255      0.001      -0.401      -0.099
# ==============================================================================
# Omnibus:                        0.592   Durbin-Watson:                   1.896
# Prob(Omnibus):                  0.744   Jarque-Bera (JB):                0.661
# Skew:                          -0.075   Prob(JB):                        0.718
# Kurtosis:                       2.920   Cond. No.                         658.
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# print(round(model.rsquared,3))

# 2-3 새로운 건물(wall=20, roof=150, glazing=20, height=5)이 주어졌을 때, 2-2문제에서 사용한 모델을 이용하여 난방 부하를 예측하시오. (반올림하여 소수 셋째 자리까지 작성) -> 79.612
new = pd.DataFrame({"wall":[20], "roof":[150], "glazing":[20], "height":[5]})
pred = model.predict(new)
print(round(pred, 3))
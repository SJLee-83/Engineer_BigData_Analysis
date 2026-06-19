import pandas as pd

df = pd.read_csv("data/promotion_data.csv")

# print(df)
# print(df.shape)
# print(df.info())

#             sales  visit_count  page_time  ad_clicks  promotion_budget
# 0    35341.611727           36        168         18             11340
# 1    64757.849132           32        214         21             11224
# 2    40615.684241           56        172          6              5468
# 3    52009.480757           70        234          5              7486
# 4    32462.259065           27         88          8              7547
# ..            ...          ...        ...        ...               ...
# 195  46537.182799           27        131          8              4121
# 196  31829.997843           21        142          5              5155
# 197  24390.270081           35         86         29             11870
# 198  43025.977501           30        244         28             11433
# 199  25265.513559           53        142         22              7384

# [200 rows x 5 columns]
# (200, 5)
# <class 'pandas.DataFrame'>
# RangeIndex: 200 entries, 0 to 199
# Data columns (total 5 columns):
#  #   Column            Non-Null Count  Dtype  
# ---  ------            --------------  -----  
#  0   sales             200 non-null    float64
#  1   visit_count       200 non-null    int64  
#  2   page_time         200 non-null    int64  
#  3   ad_clicks         200 non-null    int64  
#  4   promotion_budget  200 non-null    int64  
# dtypes: float64(1), int64(4)
# memory usage: 7.9 KB
# None

# 2-1 홍보 전후 매출 평균이 35000원과 같은지 단일 t 검정을 수행하고, p-value를 구하시오. (반올림하여 소수 셋째 자리까지 작성)

# 2-2 매출액과 다른 모든 변수들 간의 피어슨 상관계수를 구하고, 그 중 상관계수를 구하고, 그 중 상관관계가 가장 강한(절댓값이 가장 큰) 상관계수 값을 구하시오. (반올림하여 소수 셋째 자리까지 작성)
# # corr = df.corr["sales"].drop("sales")
# print(help(df.corr))

# 2-3 매출액을 종속변수로, 나머지 모든 변수를 독립 변수로 하여 회귀분석을 수행하시오. 절편을 포함한 모든 변수 중에서 p-value가 가장 작은 변수의 회귀계수를 구하시오. (반올림하여 소수 셋째 자리까지 작성) -> 130.7392
from statsmodels.formula.api import ols
model = ols("sales ~ visit_count + page_time + ad_clicks + promotion_budget", data=df).fit()
# print(model.summary())

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                  sales   R-squared:                       0.343
# Model:                            OLS   Adj. R-squared:                  0.330
# Method:                 Least Squares   F-statistic:                     25.45
# Date:                Fri, 19 Jun 2026   Prob (F-statistic):           5.60e-17
# Time:                        20:41:21   Log-Likelihood:                -2160.2
# No. Observations:                 200   AIC:                             4330.
# Df Residuals:                     195   BIC:                             4347.
# Df Model:                           4                                         
# Covariance Type:            nonrobust                                         
# ====================================================================================
#                        coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------------
# Intercept         2.234e+04   4136.622      5.400      0.000    1.42e+04    3.05e+04
# visit_count         28.7893     43.854      0.656      0.512     -57.701     115.279
# page_time          130.7392     13.501      9.684      0.000     104.113     157.365
# ad_clicks         -116.0846     85.288     -1.361      0.175    -284.290      52.121
# promotion_budget    -0.3601      0.336     -1.072      0.285      -1.023       0.303
# ==============================================================================
# Omnibus:                        0.318   Durbin-Watson:                   1.941
# Prob(Omnibus):                  0.853   Jarque-Bera (JB):                0.286
# Skew:                          -0.091   Prob(JB):                        0.867
# Kurtosis:                       2.970   Cond. No.                     3.63e+04
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# [2] The condition number is large, 3.63e+04. This might indicate that there are
# strong multicollinearity or other numerical problems.
import pandas as pd

df = pd.read_csv("data/system_cpu.csv")

# print(df)
# print(df.shape)
# print(df.info())

#        ERP  Feature1  Feature2  Feature3    CPU
# 0     30.6     235.1      44.5      44.0  112.3
# 1     40.3      36.6      46.4      36.1   58.6
# 2     57.7      52.2      66.5       2.0   55.3
# 3    128.3     196.2      59.8      57.4  103.2
# 4     80.3      75.2      59.6      58.2  104.1
# ..     ...       ...       ...       ...    ...
# 110  207.9     178.6      32.4     118.7  140.5
# 111  150.7     215.0       8.3      80.3   59.2
# 112  186.3      68.0      -7.2      92.9   82.0
# 113  141.0      73.0     158.5      82.4  145.0
# 114  155.8      35.9     249.6      73.7  145.1

# [115 rows x 5 columns]
# (115, 5)
# <class 'pandas.DataFrame'>
# RangeIndex: 115 entries, 0 to 114
# Data columns (total 5 columns):
#  #   Column    Non-Null Count  Dtype  
# ---  ------    --------------  -----  
#  0   ERP       115 non-null    float64
#  1   Feature1  115 non-null    float64
#  2   Feature2  115 non-null    float64
#  3   Feature3  115 non-null    float64
#  4   CPU       115 non-null    float64
# dtypes: float64(5)
# memory usage: 4.6 KB
# None

# ERP와 가장 상관 관계가 높은 값을 구하시오 (반올림 소수 셋째자리까지)
corr = df.corr()["ERP"].drop("ERP")
# print(round(corr.max(), 3)) # 0.882

# CPU와 컬럼이 100 미만인 것만 찾아 ERP를 종속변수, 나머지 변수들을 독립변수로 설정해 선형회귀 모델을 만들고 적합한 결정 계수를 구하시오. (반올림 소수 셋째자리까지)
filtered_df = df[df["CPU"] < 100]

from statsmodels.formula.api import ols
model = ols("ERP ~ Feature1 + Feature2 + Feature3 + CPU", data=filtered_df).fit()
# print(model.summary()) 
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                    ERP   R-squared:                       0.755
# Model:                            OLS   Adj. R-squared:                  0.736
# Method:                 Least Squares   F-statistic:                     39.30
# Date:                Fri, 19 Jun 2026   Prob (F-statistic):           5.36e-15
# Time:                        18:05:07   Log-Likelihood:                -260.40
# No. Observations:                  56   AIC:                             530.8
# Df Residuals:                      51   BIC:                             540.9
# Df Model:                           4                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     51.4133     19.112      2.690      0.010      13.045      89.782
# Feature1      -0.0242      0.059     -0.409      0.684      -0.143       0.094
# Feature2       0.0602      0.106      0.569      0.572      -0.152       0.273
# Feature3       1.4126      0.113     12.458      0.000       1.185       1.640
# CPU           -0.4651      0.234     -1.985      0.053      -0.936       0.005
# ==============================================================================
# Omnibus:                        3.758   Durbin-Watson:                   1.762
# Prob(Omnibus):                  0.153   Jarque-Bera (JB):                2.757
# Skew:                           0.436   Prob(JB):                        0.252
# Kurtosis:                       3.648   Cond. No.                         780.
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# print(round(model.rsquared, 3)) # 0.755 -> R-Squared = 결정계수

# 2에서 만든 모델에서 독립변수 중 p-value가 가장 높은 값을 구하시오.(반올림 소수 셋째자리까지)
pval = model.pvalues.drop("Intercept")
print(pval)
print("---")
print(round(pval.max(), 3))

# Feature1    6.841646e-01
# Feature2    5.717204e-01
# Feature3    4.300038e-17
# CPU         5.255548e-02
# dtype: float64
# ---
# 0.684

# print(model.pvalues)
# Intercept    9.630266e-03
# Feature1     6.841646e-01
# Feature2     5.717204e-01
# Feature3     4.300038e-17
# CPU          5.255548e-02
# dtype: float64
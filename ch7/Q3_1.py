import pandas as pd

df = pd.read_csv("data/clam.csv")

train = df[:210]
test = df[210:]

# print(df)
# print("---")
# print(train.shape)
# print(test.shape)

#      age    length  diameter    height     weight  gender
# 0      6  0.474627  0.211352  0.178189  78.971766       1
# 1      1  0.465847  0.339388  0.170522  98.781960       1
# 2      4  0.122807  0.238691  0.106924  88.792625       0
# 3      4  0.204579  0.360543  0.034261   1.028847       0
# 4      8  0.243458  0.358037  0.128080   6.503367       0
# ..   ...       ...       ...       ...        ...     ...
# 295    3  0.559766  0.390519  0.079062  57.877344       0
# 296    8  0.371229  0.355306  0.040285  31.909257       1
# 297    2  0.595571  0.104829  0.132500   4.924550       1
# 298    7  0.291551  0.289444  0.073238  51.474851       0
# 299    6  0.503335  0.061925  0.187127   6.850624       1

# [300 rows x 6 columns]
# ---
# (210, 6)
# (90, 6)

# weight을 독립변수로 gender를 종속변수로 사용해 로지스틱 회귀모델을 만들고, weight 변수가 무게가 한 단위 증가할 때 수컷일 오즈비?
from statsmodels.formula.api import logit
model = logit("gender ~ weight", data=train).fit()
# print(model.summary())
# Optimization terminated successfully.
#          Current function value: 0.690045
#          Iterations 4
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:                 gender   No. Observations:                  210
# Model:                          Logit   Df Residuals:                      208
# Method:                           MLE   Df Model:                            1
# Date:                Fri, 19 Jun 2026   Pseudo R-squ.:                0.003431
# Time:                        17:10:04   Log-Likelihood:                -144.91
# converged:                       True   LL-Null:                       -145.41
# Covariance Type:            nonrobust   LLR p-value:                    0.3178
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     -0.3140      0.276     -1.137      0.256      -0.855       0.227
# weight         0.0047      0.005      0.997      0.319      -0.005       0.014
# ==============================================================================

import numpy as np
coef = model.params["weight"]
# print(round(np.exp(coef), 4)) # 1.0047

# gender를 종속변수로 나머지를 독립변수로 하는 로지스틱 회귀 모델 적합 후 잔차 이탈도 계산
model = logit("gender ~ age + length + diameter + height + weight", data=train).fit()
# print(model.summary())
# Optimization terminated successfully.
#          Current function value: 0.690045
#          Iterations 4
# Optimization terminated successfully.
#          Current function value: 0.683173
#          Iterations 4
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:                 gender   No. Observations:                  210
# Model:                          Logit   Df Residuals:                      204
# Method:                           MLE   Df Model:                            5
# Date:                Fri, 19 Jun 2026   Pseudo R-squ.:                 0.01336
# Time:                        17:31:33   Log-Likelihood:                -143.47
# converged:                       True   LL-Null:                       -145.41
# Covariance Type:            nonrobust   LLR p-value:                    0.5662
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      0.6701      0.651      1.030      0.303      -0.605       1.946
# age           -0.0482      0.055     -0.881      0.378      -0.156       0.059
# length        -0.7832      0.998     -0.785      0.433      -2.739       1.173
# diameter      -1.1573      1.389     -0.833      0.405      -3.880       1.565
# height        -2.2977      2.640     -0.870      0.384      -7.472       2.877
# weight         0.0054      0.005      1.098      0.272      -0.004       0.015
# ==============================================================================

# print(model.llf)
deviance = -2 * model.llf
# print(round(deviance, 2))
# -143.46633759253683
# 286.93

# 독립변수 weight만 사용해 학습한 모델에서 test 데이터의 gender를 예측하고, 오류율 구하기(소수 셋째 자리까지)
from statsmodels.formula.api import logit
model = logit("gender ~ weight", data=train).fit()
pred = model.predict(test["weight"])
# print(pred)

# Optimization terminated successfully.
#          Current function value: 0.690045
#          Iterations 4
# Optimization terminated successfully.
#          Current function value: 0.683173
#          Iterations 4
# Optimization terminated successfully.
#          Current function value: 0.690045
#          Iterations 4
# 210    0.523147
# 211    0.512870
# 212    0.515427
# 213    0.513579
# 214    0.488791
#          ...   
# 295    0.489605
# 296    0.459142
# 297    0.427810
# 298    0.482078
# 299    0.430030
# Length: 90, dtype: float64

pred_class = (pred >= 0.5).astype(int)
# print(pred_class)

# Optimization terminated successfully.
#          Current function value: 0.690045
#          Iterations 4
# Optimization terminated successfully.
#          Current function value: 0.683173
#          Iterations 4
# Optimization terminated successfully.
#          Current function value: 0.690045
#          Iterations 4
# 210    1
# 211    1
# 212    1
# 213    1
# 214    0
#       ..
# 295    0
# 296    0
# 297    0
# 298    0
# 299    0
# Length: 90, dtype: int64

error_rate = (pred_class != test["gender"]).mean()
print(round(error_rate, 3)) # 0.478
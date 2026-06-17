import pandas as pd

df = pd.read_csv("data/hr.csv")

# print(df)
# print(df.shape)
# print(df.info())

#       사원번호         부서 성과등급         연봉  근속연수  교육참가횟수  만족도
# 0    E0001  Marketing    B   73200000  14.0       5  6.0
# 1    E0002         IR    A   95100000   3.0       4  9.0
# 2    E0003    Manager    B   44700000   5.0       2  8.0
# 3    E0004         IR    B  122100000  16.0       3  8.0
# 4    E0005         IR    C   66000000   NaN       4  8.0
# ..     ...        ...  ...        ...   ...     ...  ...
# 995  E0996    Finance    B   90400000  15.0       3  2.0
# 996  E0997         IR    C  147600000  12.0       5  5.0
# 997  E0998      Sales    C   95600000   4.0       4  9.0
# 998  E0999      Sales    C   94400000   8.0       6  1.0
# 999  E1000         IR    A   92400000   9.0       2  5.0

# [1000 rows x 7 columns]
# (1000, 7)
# <class 'pandas.DataFrame'>
# RangeIndex: 1000 entries, 0 to 999
# Data columns (total 7 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   사원번호    1000 non-null   str    
#  1   부서      1000 non-null   str    
#  2   성과등급    1000 non-null   str    
#  3   연봉      1000 non-null   int64  
#  4   근속연수    952 non-null    float64
#  5   교육참가횟수  1000 non-null   int64  
#  6   만족도     880 non-null    float64
# dtypes: float64(2), int64(2), str(3)
# memory usage: 54.8 KB
# None

# 만족도 결측치는 만족도 평균값으로 대체
satisfied_avg = df["만족도"].mean()
df["만족도"] = df["만족도"].fillna(satisfied_avg)

# 근속연수 결측치는 같은 부서 내에서 동일한 성과등급을 가진 직원들의 평균 근속연수로 대체. 평균의 소수점 이하는 절사하여 정수로 변환
worked_year = df.groupby(["부서", "성과등급"])["근속연수"].mean()
# print(worked_year)
# print("---")
worked_year = df.groupby(["부서", "성과등급"])["근속연수"].transform("mean")
# print(worked_year)
# 부서         성과등급
# Finance    A       11.333333
#            B       10.166667
#            C        9.893617
# HR         A        9.941176
#            B        9.769231
#            C       10.720000
# IR         A       11.814815
#            B       10.622642
#            C       10.525424
# Manager    A       11.065217
#            B       10.852459
#            C       10.744186
# Marketing  A       11.036364
#            B       11.517857
#            C        9.442308
# Sales      A        8.196429
#            B        9.433962
#            C       11.032258
# Name: 근속연수, dtype: float64
# ---
# 0      11.517857   ← 0번 행(Marketing,B)의 그룹 평균
# 1      11.814815   ← 1번 행(IR,A)의 그룹 평균
# 2      10.852459   ← 2번 행(Manager,B)의 그룹 평균
# 3      10.622642
# 4      10.525424
#          ...    
# 995    10.166667
# 996    10.525424
# 997    11.032258
# 998    11.032258
# 999    11.814815
# Name: 근속연수, Length: 1000, dtype: float64
import numpy as np

df["근속연수"] = df["근속연수"].fillna(np.floor(worked_year))

# 연봉 / 근속연수 계산 후 세 번째로 높은 사람의 근속 연수
df["연근"] = df["연봉"] / df["근속연수"]
# print(df.sort_values("연근", ascending=False).iloc[2]["근속연수"]) # 1.0

# 연봉 / 만족도 계산 후 두 번째로 높은 사람의 교육 참가 횟수
df["연만"] = df["연봉"] / df["만족도"]
# print(df.sort_values("연만", ascending=False).iloc[1]["교육참가횟수"]) # 6

# 3, 4의 합 -> # 7
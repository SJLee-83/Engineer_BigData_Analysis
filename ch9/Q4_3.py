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

# 1 만족도의 결측치는 만족도의 평균값으로 대체
satisfied_mean = df["만족도"].mean()
# print(satisfied_mean)
df["만족도"] = df["만족도"].fillna(satisfied_mean)
# print(df.info())

# 2 근속연수의 결측치는 같은부서 내에서 동일한 성과등급을 가진 직원들의 평균 근속연수로 대체한다. 이때, 평균 근속연수는 소수점 이하를 절사하여 정수로 변환하여 사용한다.
year_mean = df.groupby(["부서", "성과등급"])["근속연수"].mean()
print(year_mean)
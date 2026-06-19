import pandas as pd

df = pd.read_csv("data/student_assessment.csv")

# print(df)
# print(df.shape)
# print(df.info())

#       id_assessment  student_id  study_period_days  score
# 0               103           0                 22   87.0
# 1                93           1                 70   92.0
# 2                15           2                 22    2.0
# 3               107           3                 22   51.0
# 4                72           4                 22   73.0
# ...             ...         ...                ...    ...
# 2560             96        2695                 35    3.0
# 2561             38        2696                 22   77.0
# 2562             87        2697                 22   88.0
# 2563             88        2698                 70   67.0
# 2564             50        2699                 22   24.0

# [2565 rows x 4 columns]
# (2565, 4)
# <class 'pandas.DataFrame'>
# RangeIndex: 2565 entries, 0 to 2564
# Data columns (total 4 columns):
#  #   Column             Non-Null Count  Dtype  
# ---  ------             --------------  -----  
#  0   id_assessment      2565 non-null   int64  
#  1   student_id         2565 non-null   int64  
#  2   study_period_days  2565 non-null   int64  
#  3   score              2544 non-null   float64
# dtypes: float64(1), int64(3)
# memory usage: 80.3 KB
# None

# 결측치가 있는 행을 제거하고 학생이 가장 많이 수강한 과목 찾기 -> 해당 과목 점수를 표준화 -> 가장 큰 표준화된 값 구하기
df = df.dropna()
# print(df.info())

id = df["id_assessment"].value_counts().idxmax()
# print(id)
cond = df["id_assessment"] == id
df = df[cond]
# print(df)

from sklearn.preprocessing import StandardScaler
df["score"] = StandardScaler().fit_transform(df[["score"]])

print(round(df["score"].max(),3)) # 2.183


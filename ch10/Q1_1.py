#            major_topic             sub_topic  student_id  is_correct
# 0          Mathematics  Discrete Mathematics         359           1
# 1          Mathematics  Discrete Mathematics         621           1
# 2     Computer Science             Databases        1366           1
# 3          Mathematics  Discrete Mathematics        1377           0
# 4              Physics             Mechanics         535           0
# ...                ...                   ...         ...         ...
# 2659       Mathematics        Linear Algebra        1037           0
# 2660       Mathematics        Linear Algebra        1476           0
# 2661       Mathematics        Linear Algebra         663           0
# 2662       Mathematics        Linear Algebra         542           0
# 2663       Mathematics        Linear Algebra         966           0

# [2664 rows x 4 columns]
# (2664, 4)
# <class 'pandas.DataFrame'>
# RangeIndex: 2664 entries, 0 to 2663
# Data columns (total 4 columns):
#  #   Column       Non-Null Count  Dtype
# ---  ------       --------------  -----
#  0   major_topic  2664 non-null   str  
#  1   sub_topic    2664 non-null   str  
#  2   student_id   2664 non-null   int64
#  3   is_correct   2664 non-null   int64
# dtypes: int64(2), str(2)
# memory usage: 83.4 KB
# None

import pandas as pd

df = pd.read_csv("data/subject_performance.csv")

# print(df)
# print(df.shape)
# print(df.info())

df_1 = df.groupby("sub_topic")["is_correct"].mean()
# print(df_1)
# sub_topic
# Algorithms              0.787162
# Calculus                0.673913
# Databases               0.640523
# Discrete Mathematics    0.643836
# Electromagnetism        0.658621
# Linear Algebra          0.787162
# Mechanics               0.693727
# Operating Systems       0.658621
# Quantum                 0.667774
# Name: is_correct, dtype: float64

result = df_1.sort_values(ascending=False).drop_duplicates()
print(round(result, 3)) #0.674

# sub_topic
# Algorithms              0.787
# Mechanics               0.694
# Calculus                0.674
# Quantum                 0.668
# Electromagnetism        0.659
# Discrete Mathematics    0.644
# Databases               0.641
# Name: is_correct, dtype: float64
import pandas as pd

df = pd.read_csv("data/hamspam.csv")

# print(df)
# print(df.shape)
# print(df.info())

#      label                                               text
# 0      ham  Incididunt tempor elit sit do aliqua eiusmod a...
# 1      ham       Adipiscing tempor lorem ipsum dolore tempor.
# 2     spam    Tempor tempor magna sed incididunt consectetur.
# 3      ham  Consectetur incididunt amet adipiscing amet lo...
# 4      ham                  Sit ipsum labore sit do sit elit.
# ...    ...                                                ...
# 1995   ham  Lorem tempor aliqua dolor dolore tempor do lab...
# 1996  spam  Sit ut aliqua adipiscing incididunt sit consec...
# 1997  spam  Ipsum aliqua incididunt et dolore tempor eiusm...
# 1998   ham          Eiusmod do elit consectetur labore do do.
# 1999   ham  Sed sed lorem aliqua dolore labore do incididu...

# [2000 rows x 2 columns]
# (2000, 2)
# <class 'pandas.DataFrame'>
# RangeIndex: 2000 entries, 0 to 1999
# Data columns (total 2 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   label   2000 non-null   str  
#  1   text    2000 non-null   str  
# dtypes: str(2)
# memory usage: 31.4 KB
# None

# 1. 각 메시지의 단어 개수를 공백 기준으로 계산

df["text"] = df["text"].split(" ")
print(df["text"])

# df["count"] = len(df["text"].plot(" "))
# print(df["count"])

# 2. 스팸 메시지의 평균 단어 개수와 정상 메시지의 평균 단어 개수를 각각 구하시오.

# 3. 두 평균의 차이의 절댓값을 구하여 입력하시오.
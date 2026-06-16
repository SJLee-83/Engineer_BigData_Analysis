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

import pandas as pd

df = pd.read_csv("data/hamspam.csv")

# print(df)
# print(df.shape)
# print(df.info())

# 1 - 각 메시지의 단어 개수를 띄어쓰기 기준으로 계산
df["word-cnt"] = df["text"].str.count(" ") + 1
# print(df)

# 2 - 스팸 메시지의 평균 단어 개수와 정상 메시지의 평균 단어 개수를 구하시오.
cond1 = (df["label"] == "spam")
cond2 = (df["label"] == "ham")

spam_msg = df[cond1]
ham_msg = df[cond2]

# print(spam_msg)

spam_wd_cnt = spam_msg["word-cnt"].mean()
# print(spam_wd_cnt)
ham_wd_cnt = ham_msg["word-cnt"].mean()

# 3 - 두 평균의 차이의 절댓값을 구하여 입력하시오.
result = abs(spam_wd_cnt-ham_wd_cnt)
print(round(result, 3)) # 0.047
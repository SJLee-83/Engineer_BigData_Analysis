import pandas as pd

df = pd.read_csv("data/greenhouse_gas.csv")

# print(df.head())        # 앞 5행 — 데이터가 어떻게 생겼나
# print(df.shape)         # (행 수, 열 수) — 크기 파악
# print(df.info())        # 컬럼명, 결측치, 데이터 타입 한눈에
# print(df.describe())    # 수치형 컬럼 통계 요약 (평균, 분위수 등)

new_df = df.columns[1:] # Country 제외, 2001~2021까지의 컬럼명들 저장
# Index(['2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
#        '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018',
#        '2019', '2020', '2021'],
#       dtype='str')

max_idx = df[new_df].idxmax() # 각 컬럼별 최대값의 인덱스 값 가져오기
# print(df[new_df])
# print(max_idx)

df2 = df.loc[max_idx]
# print(df2)


# print(df2['Country'].value_counts())
result = df.set_index('Country').max()   # 연도별 최댓값 숫자
# print(result)

# 1-1
# Country                                                                                                                                                                                                   
# China     11                                                                                                                                                                                              
# Brazil    10                                                                                                                                                                                              
# Name: count, dtype: int64  

# 1-2

# 먼저 국가만 뽑아

# 각 국가들의 처음, 마지막 열을 뽑은 다음 새로운 테이블 구성
df["CAGR"] = (df['2021'] / df['2001']) ** (1/20) - 1
# print(df)
# print(round(df["CAGR"].max(), 3))
# 0.047


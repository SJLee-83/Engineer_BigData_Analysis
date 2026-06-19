import pandas as pd

df = pd.read_csv("data/drinks.csv")

# print(df)
# print(df.shape)
# print(df.info())

#          country  beer_servings  spirit_servings  wine_servings  total_litres_of_pure_alcohol      continent                                                                                              
# 0    Afghanistan              0                0              0                           0.0           Asia                                                                                              
# 1        Albania             89              132             54                           4.9         Europe                                                                                              
# 2        Algeria             25                0             14                           0.7         Africa                                                                                              
# 3        Andorra            245              138            312                          12.4         Europe                                                                                              
# 4         Angola            217               57             45                           5.9         Africa                                                                                              
# ..           ...            ...              ...            ...                           ...            ...                                                                                              
# 188    Venezuela            333              100              3                           7.7  South America                                                                                              
# 189      Vietnam            111                2              1                           2.0           Asia                                                                                              
# 190        Yemen              6                0              0                           0.1           Asia                                                                                              
# 191       Zambia             32               19              4                           2.5         Africa                                                                                              
# 192     Zimbabwe             64               18              4                           4.7         Africa                                                                                              
                                                                                                                                                                                                          
# [193 rows x 6 columns]                                                                                                                                                                                    
# (193, 6)                                                                                                                                                                                                  
# <class 'pandas.DataFrame'>                                                                                                                                                                                
# RangeIndex: 193 entries, 0 to 192                                                                                                                                                                         
# Data columns (total 6 columns):                                                                                                                                                                           
#  #   Column                        Non-Null Count  Dtype                                                                                                                                                  
# ---  ------                        --------------  -----  
#  0   country                       193 non-null    str    
#  1   beer_servings                 193 non-null    int64  
#  2   spirit_servings               193 non-null    int64  
#  3   wine_servings                 193 non-null    int64  
#  4   total_litres_of_pure_alcohol  193 non-null    float64
#  5   continent                     193 non-null    str    
# dtypes: float64(1), int64(3), str(2)
# memory usage: 9.2 KB
# None

# 대륙별 맥주 소비량 평균 계산 -> 평균이 가장 큰 대륙 찾기
r1 = df.groupby("continent")["beer_servings"].mean().idxmax()
print(r1) # Europe
# continent
# Europe           193.777778
# South America    175.083333
# North America    145.434783
# Oceania           89.687500
# Africa            61.471698
# Asia              37.045455
# Name: beer_servings, dtype: float64

# 1번에서 찾은 대륙에서 맥주 소비량이 5번쩨로 많은 국가의 맥주 소비량
cond = (df["continent"] == "Europe")
df1 = df[cond].sort_values("beer_servings", ascending=False).iloc[4]
print(df1) # 313
# country                         Ireland
# beer_servings                       313
# spirit_servings                     118
# wine_servings                       165
# total_litres_of_pure_alcohol       11.4
# continent                        Europe
# Name: 81, dtype: object
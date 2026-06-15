#                 DateTime  S1_Temp  S2_Temp  S3_Temp  S1_Humidity  S2_Humidity  Illuminance  Noise    CO2  Pressure
# 0    2024-01-01 00:00:00     26.7     23.5     24.7         63.5         72.1        320.8   47.3    NaN    1008.2
# 1    2024-01-01 01:00:00     27.0      NaN     23.4         63.9         50.5        251.0   48.9  904.0    1008.1
# 2    2024-01-01 02:00:00     25.6     27.5     26.2         68.5         53.1          NaN   57.5  752.7    1017.5
# 3    2024-01-01 03:00:00     24.7     24.3     26.0         56.1         66.9        329.5   55.0  873.7    1005.6
# 4    2024-01-01 04:00:00     22.6     25.7     23.0         63.2         59.2        312.4   34.5  959.0    1003.8
# ..                   ...      ...      ...      ...          ...          ...          ...    ...    ...       ...
# 495  2024-01-21 15:00:00     25.6     23.2     23.5         56.4         52.6        309.6   34.1  766.2    1016.6
# 496  2024-01-21 16:00:00     27.0     24.9      NaN         61.8         52.5        272.9   31.3  787.7    1004.9
# 497  2024-01-21 17:00:00     27.3      NaN     26.2         58.2         54.5        293.9   58.3  896.4    1010.2
# 498  2024-01-21 18:00:00     26.8     24.0     26.6         65.9         64.0        284.0   44.5  705.3    1009.6
# 499  2024-01-21 19:00:00     24.2     26.5      NaN         65.0         68.7        366.2   52.8  798.6    1009.9

# [500 rows x 10 columns]
# (500, 10)
# <class 'pandas.DataFrame'>
# RangeIndex: 500 entries, 0 to 499
# Data columns (total 10 columns):
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   DateTime     500 non-null    str    
#  1   S1_Temp      450 non-null    float64
#  2   S2_Temp      350 non-null    float64
#  3   S3_Temp      420 non-null    float64
#  4   S1_Humidity  500 non-null    float64
#  5   S2_Humidity  500 non-null    float64
#  6   Illuminance  470 non-null    float64
#  7   Noise        500 non-null    float64
#  8   CO2          480 non-null    float64
#  9   Pressure     500 non-null    float64
# dtypes: float64(9), str(1)
# memory usage: 39.2 KB
# None

import pandas as pd

df = pd.read_csv("data/sensor_data.csv")

# print(df)
# print(df.shape)
# print(df.info())

# 결측치가 가장 많은 컬럼: S2_Temp
# 해당 컬럼의 결측치를 중앙값으로 대체
med = df["S2_Temp"].median()
df["S2_Temp"] = df["S2_Temp"].fillna(med)
# 결측치를 대체한 후 해당 컬럼의 평균을 소수점 셋째 자리까지 구하시오.
print(round(df["S2_Temp"].mean(), 3))
# 25.055
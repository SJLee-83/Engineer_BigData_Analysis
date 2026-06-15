import pandas as pd

df = pd.read_csv("data/order_data.csv")

print(df)
print(df.shape)
print(df.info())
# 새로운 컬럼 order_ant 생성

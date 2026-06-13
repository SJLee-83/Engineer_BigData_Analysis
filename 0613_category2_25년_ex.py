# 출력을 원하실 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

import pandas as pd

train = pd.read_csv("data/customer_train.csv")
test = pd.read_csv("data/customer_test.csv")

# 사용자 코딩
# print(train.info())
# print(test.info())

# train, test set 분리
X = train.drop(['총구매액'], axis=1)
y = train['총구매액']

X_full = pd.concat([X, test], axis=0)
X_full = X_full.drop(['회원ID'], axis=1)

# print(X_full.shape)

# 결측치 제거
X_full['환불금액'] = X_full['환불금액'].fillna(0)

# 수치형 변수 스케일링 skip -> 랜덤포레스트는 스케일링 유무에 결과가 크게 영향을 안 받음

# 범주형 변수 인코딩
X_full = pd.get_dummies(X_full) # get_dummies -> 범주형 데이터만 원핫 인코딩
# print(X_full.shape)
# print(X_full)

# 데이터 분리
X_train = X_full[:train.shape[0]] # train 데이터 row 갯수만큼 train 데이터로 분리
X_test = X_full[train.shape[0]:] # 나머지 뒤에 test 데이터로 분리
# print(X_train.shape, X_test.shape)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2)
# print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)

from sklearn.metrics import root_mean_squared_error, r2_score
rmse = root_mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)
# print(rmse, r2)

y_pred = model.predict(X_test)
result = pd.DataFrame(y_pred, columns=['pred'])
result.to_csv('result.csv', index=False)

result = pd.read_csv('result.csv')
print(result)
# 답안 제출 참고
# 아래 코드는 예시이며 변수명 등 개인별로 변경하여 활용
# pd.DataFrame변수.to_csv("result.csv", index=False)
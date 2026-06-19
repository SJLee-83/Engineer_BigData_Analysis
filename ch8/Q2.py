import pandas as pd

train = pd.read_csv("data/churn_train.csv")
test = pd.read_csv("data/churn_test.csv")

# print(train)
# print("---")
# print(test)
# print("---")
# print(train.shape)
# print("---")
# print(test.shape)
# print("---")
# print(train.info())
# print("---")
# print(test.info())

#      customerID  gender  SeniorCitizen Partner Dependents  tenure  ...          StreamingTV      StreamingMovies        Contract PaperlessBilling     PaymentMethod TotalCharges
# 0      CUST0454    Male              0      No         No       7  ...  No internet service  No internet service        Two year               No  Electronic check       163.45
# 1      CUST1145  Female              1      No        Yes      53  ...                  Yes                  Yes  Month-to-month               No       Credit card      4140.89
# 2      CUST1138  Female              1      No        Yes      68  ...                   No                  Yes  Month-to-month               No      Mailed check      5152.36
# 3      CUST2645    Male              1      No        Yes      44  ...  No internet service  No internet service  Month-to-month              Yes       Credit card      5251.84
# 4      CUST2632    Male              0     Yes         No       7  ...                  Yes                   No        One year               No  Electronic check       725.69
# ...         ...     ...            ...     ...        ...     ...  ...                  ...                  ...             ...              ...               ...          ...
# 4111   CUST5809  Female              0     Yes        Yes      28  ...                   No                  Yes  Month-to-month              Yes       Credit card      3201.24
# 4112   CUST4720    Male              1     Yes        Yes      33  ...  No internet service  No internet service        Two year              Yes      Mailed check      1124.31
# 4113   CUST0173  Female              0     Yes        Yes      46  ...  No internet service  No internet service  Month-to-month              Yes       Credit card      1492.70
# 4114   CUST1244    Male              1      No         No      28  ...                  Yes                  Yes        One year              Yes     Bank transfer      1296.96
# 4115   CUST4989  Female              0      No        Yes      17  ...  No internet service  No internet service        Two year               No      Mailed check       743.92

# [4116 rows x 19 columns]
# ---
#      customerID  gender  SeniorCitizen Partner Dependents  tenure  ...          TechSupport          StreamingTV      StreamingMovies        Contract PaperlessBilling     PaymentMethod
# 0      CUST0769    Male              1      No        Yes      47  ...                  Yes                   No                  Yes  Month-to-month              Yes       Credit card
# 1      CUST0675    Male              1     Yes        Yes      12  ...                  Yes                  Yes                   No        One year               No      Mailed check
# 2      CUST0210    Male              1      No        Yes      59  ...  No internet service  No internet service  No internet service        Two year              Yes       Credit card
# 3      CUST3795  Female              0      No         No      13  ...  No internet service  No internet service  No internet service        One year              Yes      Mailed check
# 4      CUST1829    Male              0     Yes        Yes      20  ...                  Yes                  Yes                   No        Two year              Yes  Electronic check
# ...         ...     ...            ...     ...        ...     ...  ...                  ...                  ...                  ...             ...              ...               ...
# 1759   CUST4159    Male              1      No        Yes       4  ...                   No                  Yes                   No        One year               No     Bank transfer
# 1760   CUST1431    Male              1     Yes        Yes      58  ...                  Yes                   No                   No        Two year              Yes      Mailed check
# 1761   CUST1309  Female              1     Yes        Yes      50  ...  No internet service  No internet service  No internet service  Month-to-month               No      Mailed check
# 1762   CUST5334    Male              0      No         No      15  ...                  Yes                   No                   No        One year              Yes  Electronic check
# 1763   CUST0783  Female              0     Yes        Yes      56  ...                   No                  Yes                  Yes        Two year               No       Credit card

# [1764 rows x 18 columns]
# ---
# (4116, 19)
# ---
# (1764, 18)
# ---
# <class 'pandas.DataFrame'>
# RangeIndex: 4116 entries, 0 to 4115
# Data columns (total 19 columns):
#  #   Column            Non-Null Count  Dtype  
# ---  ------            --------------  -----  
#  0   customerID        4116 non-null   str    
#  1   gender            4116 non-null   str    
#  2   SeniorCitizen     4116 non-null   int64  
#  3   Partner           4116 non-null   str    
#  4   Dependents        4116 non-null   str    
#  5   tenure            4116 non-null   int64  
#  6   PhoneService      4116 non-null   str    
#  7   MultipleLines     4116 non-null   str    
#  8   InternetService   4116 non-null   str    
#  9   OnlineSecurity    4116 non-null   str    
#  10  OnlineBackup      4116 non-null   str    
#  11  DeviceProtection  4116 non-null   str    
#  12  TechSupport       4116 non-null   str    
#  13  StreamingTV       4116 non-null   str    
#  14  StreamingMovies   4116 non-null   str    
#  15  Contract          4116 non-null   str    
#  16  PaperlessBilling  4116 non-null   str    
#  17  PaymentMethod     4116 non-null   str    
#  18  TotalCharges      4116 non-null   float64
# dtypes: float64(1), int64(2), str(16)
# memory usage: 611.1 KB
# None
# ---
# <class 'pandas.DataFrame'>
# RangeIndex: 1764 entries, 0 to 1763
# Data columns (total 18 columns):
#  #   Column            Non-Null Count  Dtype
# ---  ------            --------------  -----
#  0   customerID        1764 non-null   str  
#  1   gender            1764 non-null   str  
#  2   SeniorCitizen     1764 non-null   int64
#  3   Partner           1764 non-null   str  
#  4   Dependents        1764 non-null   str  
#  5   tenure            1764 non-null   int64
#  6   PhoneService      1764 non-null   str  
#  7   MultipleLines     1764 non-null   str  
#  8   InternetService   1764 non-null   str  
#  9   OnlineSecurity    1764 non-null   str  
#  10  OnlineBackup      1764 non-null   str  
#  11  DeviceProtection  1764 non-null   str  
#  12  TechSupport       1764 non-null   str  
#  13  StreamingTV       1764 non-null   str  
#  14  StreamingMovies   1764 non-null   str  
#  15  Contract          1764 non-null   str  
#  16  PaperlessBilling  1764 non-null   str  
#  17  PaymentMethod     1764 non-null   str  
# dtypes: int64(2), str(16)
# memory usage: 248.2 KB
# None

X = train.drop(["TotalCharges", "customerID"], axis=1)
X_test = test.drop(["customerID"], axis=1)
y = train["TotalCharges"]

X_full = pd.concat([X, X_test], axis=0)
# print(X_full.shape)
# print(X_full)

X_full = pd.get_dummies(X_full)
# print(X_full.shape)
# print(X_full)

X_train = X_full[:4116]
X_test = X_full[4116:] 

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

model1 = RandomForestRegressor(random_state=42)
model2 = RandomForestRegressor(n_estimators=200, random_state=42)
model3 = GradientBoostingRegressor(random_state=42)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

y_val_pred1 = model1.predict(X_val)
y_val_pred2 = model2.predict(X_val)
y_val_pred3 = model3.predict(X_val)

from sklearn.metrics import mean_absolute_error

re1 = mean_absolute_error(y_val, y_val_pred1)
re2 = mean_absolute_error(y_val, y_val_pred2)
re3 = mean_absolute_error(y_val, y_val_pred3)

# print(re1)
# print(re2)
# print(re3)

# 940.1045995732201
# 936.6720546106391
# 915.7246622449933

y_pred = model3.predict(X_test)
result = pd.DataFrame(y_pred, columns=["pred"])
result.to_csv("result.csv", index=False)

print(pd.read_csv("result.csv"))
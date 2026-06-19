import pandas as pd

train = pd.read_csv("data/mart_train.csv")
test = pd.read_csv("data/mart_test.csv")

# print(train)
# print("---")
# print(train.shape)
# print("---")
# print(train.info())
# print("---")
# print("---")
# print("---")
# print(test)
# print("---")
# print(test.shape)
# print("---")
# print(test.info())

#     branch       city customer_type  gender            product_line       total payment_method  rating time_of_day  day_name
# 0        A     Yangon        Member  Female       Health and beauty   823457.25        Ewallet     9.1   afternoon  Saturday
# 1        C  Naypyitaw        Normal  Female  Electronic accessories   120330.00           Cash     9.6     morning    Friday
# 2        A     Yangon        Normal    Male      Home and lifestyle   510788.25    Credit card     7.4   afternoon    Sunday
# 3        A     Yangon        Member    Male       Health and beauty   733572.00        Ewallet     8.4     evening    Sunday
# 4        A     Yangon        Normal    Male       Sports and travel   951567.75        Ewallet     5.3     morning    Friday
# ..     ...        ...           ...     ...                     ...         ...            ...     ...         ...       ...
# 695      A     Yangon        Member  Female      Home and lifestyle   688038.75           Cash     6.6     evening   Tuesday
# 696      A     Yangon        Member  Female       Sports and travel   170352.00        Ewallet     6.9     evening   Tuesday
# 697      B   Mandalay        Normal    Male      Home and lifestyle   391797.00        Ewallet     4.3     evening    Sunday
# 698      A     Yangon        Member    Male  Electronic accessories   986296.50    Credit card     7.8     evening   Tuesday
# 699      C  Naypyitaw        Normal    Male      Home and lifestyle  1535625.00        Ewallet     8.0     evening  Saturday

# [700 rows x 10 columns]
# ---
# (700, 10)
# ---
# <class 'pandas.DataFrame'>
# RangeIndex: 700 entries, 0 to 699
# Data columns (total 10 columns):
#  #   Column          Non-Null Count  Dtype  
# ---  ------          --------------  -----  
#  0   branch          700 non-null    str    
#  1   city            700 non-null    str    
#  2   customer_type   700 non-null    str    
#  3   gender          700 non-null    str    
#  4   product_line    700 non-null    str    
#  5   total           700 non-null    float64
#  6   payment_method  700 non-null    str    
#  7   rating          700 non-null    float64
#  8   time_of_day     700 non-null    str    
#  9   day_name        700 non-null    str    
# dtypes: float64(2), str(8)
# memory usage: 54.8 KB
# None
# ---
# ---
# ---
#     branch       city customer_type  gender         product_line payment_method  rating time_of_day   day_name
# 0        C  Naypyitaw        Normal  Female  Fashion accessories        Ewallet     9.6   afternoon   Thursday
# 1        B   Mandalay        Normal    Male   Food and beverages    Credit card     4.3     evening  Wednesday
# 2        B   Mandalay        Member  Female  Fashion accessories    Credit card     5.0     evening  Wednesday
# 3        B   Mandalay        Member    Male    Health and beauty           Cash     9.2     morning     Sunday
# 4        B   Mandalay        Member  Female   Home and lifestyle           Cash     6.3   afternoon   Saturday
# ..     ...        ...           ...     ...                  ...            ...     ...         ...        ...
# 295      C  Naypyitaw        Normal    Male    Health and beauty        Ewallet     6.2   afternoon    Tuesday
# 296      B   Mandalay        Normal  Female   Home and lifestyle        Ewallet     4.4     evening   Saturday
# 297      A     Yangon        Member    Male   Food and beverages           Cash     7.7   afternoon   Saturday
# 298      A     Yangon        Normal    Male   Home and lifestyle           Cash     4.1   afternoon     Friday
# 299      A     Yangon        Member  Female  Fashion accessories           Cash     6.6   afternoon     Monday

# [300 rows x 9 columns]
# ---
# (300, 9)
# ---
# <class 'pandas.DataFrame'>
# RangeIndex: 300 entries, 0 to 299
# Data columns (total 9 columns):
#  #   Column          Non-Null Count  Dtype  
# ---  ------          --------------  -----  
#  0   branch          300 non-null    str    
#  1   city            300 non-null    str    
#  2   customer_type   300 non-null    str    
#  3   gender          300 non-null    str    
#  4   product_line    300 non-null    str    
#  5   payment_method  300 non-null    str    
#  6   rating          300 non-null    float64
#  7   time_of_day     300 non-null    str    
#  8   day_name        300 non-null    str    
# dtypes: float64(1), str(8)
# memory usage: 21.2 KB
# None

X = train.drop(["total"], axis=1)
y = train["total"]

X_full = pd.concat([X, test], axis=0)
# print(X_full.info())
# print(X_full.shape)

X_full = pd.get_dummies(X_full)
# print(X_full)

X_train = X_full[:700]
X_test = X_full[700:]

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

from sklearn.metrics import root_mean_squared_error

re1 = root_mean_squared_error(y_val, y_val_pred1)
re2 = root_mean_squared_error(y_val, y_val_pred2)
re3 = root_mean_squared_error(y_val, y_val_pred3)

# print(re1)
# print(re2)
# print(re3)

# 371748.72138504375
# 371172.1141227833
# 384846.4657914099

y_pred = model2.predict(X_test)
result = pd.DataFrame(y_pred, columns=["pred"])
result.to_csv("result.csv", index=False)

print(pd.read_csv("result.csv"))
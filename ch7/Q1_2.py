import pandas as pd

df = pd.read_csv("data/stock_market.csv")

# print(df)
# print(df.shape)
# print(df.info())

#           DE1       DE2       DE3       DE4       DE5       DE6       DE7       DE8       DE9      DE10  ...      DE71      DE72      DE73      DE74      DE75      DE76      DE77        open       close    volume
# 0    1.764052  0.400157  0.978738  2.240893  1.867558 -0.977278  0.950088 -0.151357 -0.103219  0.410599  ...  0.729091  0.128983  1.139401 -1.234826  0.402342 -0.684810 -0.870797  129.549102  126.502056  981466.0
# 1   -0.578850 -0.311553  0.056165 -1.165150  0.900826  0.465662 -1.536244  1.488252  1.895889  1.178780  ...  1.117016 -1.315907 -0.461585 -0.068242  1.713343 -0.744755 -0.826439  126.502056  128.041811  473416.0
# 2   -0.098453 -0.663478  1.126636 -1.079932 -1.147469 -0.437820 -0.498032  1.929532  0.949421  0.087551  ... -1.540797  0.063262  0.156507  0.232181 -0.597316 -0.237922 -1.424061  128.041811  128.957651   46597.0
# 3   -0.493320 -0.542861  0.416050 -1.156182  0.781198  1.494485 -2.069985  0.426259  0.676908 -0.637437  ...  1.658131 -0.118164 -0.680178  0.666383 -0.460720 -1.334258 -1.346718  128.957651  122.965246  455747.0
# 4    0.693773 -0.159573 -0.133702  1.077744 -1.126826 -0.730678 -0.384880  0.094352 -0.042171 -0.286887  ... -0.206904  0.880179 -1.698106  0.387280 -2.255564 -1.022507  0.038631  122.965246  114.922790  242198.0
# ..        ...       ...       ...       ...       ...       ...       ...       ...       ...       ...  ...       ...       ...       ...       ...       ...       ...       ...         ...         ...       ...
# 995  1.020937 -0.942656 -0.437589  0.400197  0.280305 -0.239677 -0.523317 -0.070756  0.813117 -0.394868  ... -0.658888  0.129819  0.343603  0.240547  0.869118 -0.200800 -2.477692   50.254255   31.155158  955899.0
# 996  0.116310 -0.387522 -0.135163  0.427945  0.020041 -1.281488 -0.003546  0.517104 -0.798964 -0.590362  ... -0.297669  0.652641  0.807874 -2.239079 -1.378841 -0.020303  1.058002   31.155158   35.253943  646770.0
# 997 -0.167591 -0.304002  0.407705  1.625603 -0.480631  1.546496 -0.288261 -0.625352  1.186237  0.270277  ...  0.200219  0.242471  2.453099  0.111067 -1.904594  0.643397  0.416017   35.253943   44.833041  887211.0
# 998 -1.780806 -0.133918  0.825544 -0.223826  0.426823  1.024924  0.847111  1.370670  1.187785 -0.938389  ...  0.946004 -0.418479 -1.790980 -1.013184 -0.316554 -2.145770 -0.987003   44.833041   56.070234  174999.0
# 999 -0.917195 -0.191702 -0.795384  0.676718 -0.321088  1.302167 -0.407516  1.520332  0.355029 -1.480904  ... -0.229409 -0.014024  1.347451  0.966457 -0.553087  0.502263 -0.417586   56.070234   56.017586  640705.0

# [1000 rows x 80 columns]
# (1000, 80)
# <class 'pandas.DataFrame'>
# RangeIndex: 1000 entries, 0 to 999
# Data columns (total 80 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   DE1     1000 non-null   float64
#  1   DE2     1000 non-null   float64
#  2   DE3     1000 non-null   float64
#  3   DE4     1000 non-null   float64
#  4   DE5     1000 non-null   float64
#  5   DE6     1000 non-null   float64
#  6   DE7     1000 non-null   float64
#  7   DE8     1000 non-null   float64
#  8   DE9     1000 non-null   float64
#  9   DE10    1000 non-null   float64
#  10  DE11    1000 non-null   float64
#  11  DE12    1000 non-null   float64
#  12  DE13    1000 non-null   float64
#  13  DE14    1000 non-null   float64
#  14  DE15    1000 non-null   float64
#  15  DE16    1000 non-null   float64
#  16  DE17    1000 non-null   float64
#  17  DE18    1000 non-null   float64
#  18  DE19    1000 non-null   float64
#  19  DE20    1000 non-null   float64
#  20  DE21    1000 non-null   float64
#  21  DE22    1000 non-null   float64
#  22  DE23    1000 non-null   float64
#  23  DE24    1000 non-null   float64
#  24  DE25    1000 non-null   float64
#  25  DE26    1000 non-null   float64
#  26  DE27    1000 non-null   float64
#  27  DE28    1000 non-null   float64
#  28  DE29    1000 non-null   float64
#  29  DE30    1000 non-null   float64
#  30  DE31    1000 non-null   float64
#  31  DE32    1000 non-null   float64
#  32  DE33    1000 non-null   float64
#  33  DE34    1000 non-null   float64
#  34  DE35    1000 non-null   float64
#  35  DE36    1000 non-null   float64
#  36  DE37    1000 non-null   float64
#  37  DE38    1000 non-null   float64
#  38  DE39    1000 non-null   float64
#  39  DE40    1000 non-null   float64
#  40  DE41    1000 non-null   float64
#  41  DE42    1000 non-null   float64
#  42  DE43    1000 non-null   float64
#  43  DE44    1000 non-null   float64
#  44  DE45    1000 non-null   float64
#  45  DE46    1000 non-null   float64
#  46  DE47    1000 non-null   float64
#  47  DE48    1000 non-null   float64
#  48  DE49    1000 non-null   float64
#  49  DE50    1000 non-null   float64
#  50  DE51    1000 non-null   float64
#  51  DE52    1000 non-null   float64
#  52  DE53    1000 non-null   float64
#  53  DE54    1000 non-null   float64
#  54  DE55    1000 non-null   float64
#  55  DE56    1000 non-null   float64
#  56  DE57    1000 non-null   float64
#  57  DE58    1000 non-null   float64
#  58  DE59    1000 non-null   float64
#  59  DE60    1000 non-null   float64
#  60  DE61    1000 non-null   float64
#  61  DE62    1000 non-null   float64
#  62  DE63    1000 non-null   float64
#  63  DE64    1000 non-null   float64
#  64  DE65    1000 non-null   float64
#  65  DE66    1000 non-null   float64
#  66  DE67    1000 non-null   float64
#  67  DE68    1000 non-null   float64
#  68  DE69    1000 non-null   float64
#  69  DE70    1000 non-null   float64
#  70  DE71    1000 non-null   float64
#  71  DE72    1000 non-null   float64
#  72  DE73    1000 non-null   float64
#  73  DE74    1000 non-null   float64
#  74  DE75    1000 non-null   float64
#  75  DE76    1000 non-null   float64
#  76  DE77    1000 non-null   float64
#  77  open    1000 non-null   float64
#  78  close   1000 non-null   float64
#  79  volume  1000 non-null   float64
# dtypes: float64(80)
# memory usage: 625.1 KB
# None

# DE1 ~ DE77 컬럼 중 주가지수의 종가 'close'와 상관 관계가 높은 변수를 찾아 해당 변수의 평균값 구하기(소수 넷째자리까지 계산)
df_corr = df.corr()["close"].abs()
print(df_corr)

col = df_corr.loc["DE1":"DE77"].idxmax()
print(col)

print(round(df[col].mean(),4)) # -0.0004
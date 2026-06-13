# 출력을 원할 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("data/bcc.csv")

# 사용자 코딩
# 1번
df['log_resistin'] = np.log(df['Resistin'])

group1 = df[df['Classification'] == 1]['log_resistin']
group2 = df[df['Classification'] == 2]['log_resistin']

var1 = group1.var()
var2 = group2.var()

dof_1 = len(group1) - 1
dof_2 = len(group2) - 1

# print(dof_1) # 51
# print(dof_2) # 63 환자가 분자로

f_stat = var2/var1
print(round(f_stat,3))

# 2번
n1 = len(group1)
n2 = len(group2)
pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
print(round(pooled_var,3))

# 3번
mean1 = group1.mean()
mean2 = group2.mean()
t_stat = (mean1-mean2) / np.sqrt(pooled_var*(1/n1+1/n2))
p_value=2*(1-stats.t.cdf(abs(t_stat), df=n1+n2-2))
print(round(p_value,3))

ttest_result = stats.ttest_ind(group1, group2, equal_var=True)
print(ttest_result)
# 문제에서는 단계별로 풀이를 진행하는 걸 유도하지만, 만약 공식이 기억이 안 나고, 조금이라도 추가 점수를 획득하고 싶다면 이 함수를 활용해라.

# 해당 화면에서는 제출하지 않으며, 문제 풀이 후 답안제출에서 결괏값 제출

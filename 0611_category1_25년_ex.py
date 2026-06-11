# 출력을 원할 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

import pandas as pd

df = pd.read_csv("data/employee_performance.csv")

# 사용자 코딩
mean_satisfaction = df['고객만족도'].mean() # '고객만족도' 컬럼의 평균 구하기
df['고객만족도'] = df['고객만족도'].fillna(mean_satisfaction) # fillna -> 결측치를 mean_satisfaction 값으로 채우기

df = df.dropna(subset=['근속연수']) # dropna -> '근속연수' 컬럼 값이 결측치인 row 제거

quantile_3 = df['고객만족도'].quantile(0.75) # quantile(0.75) -> 고객만족도의 4분위 중 3사분위수
print(int(quantile_3))

# quantile(q)는 "데이터를 정렬했을 때 하위 q 비율 지점의 값"을 돌려줘.

# quantile(0.6) → 하위 60% 지점 값 (= 상위 40% 지점)
# quantile(0.9) → 하위 90% 지점 값 (상위 10%)
# quantile(0.33) → 하위 33% 지점 값
# quantile(0.5) → 중앙값 (median과 동일)

vvalue = df.groupby('부서')['연봉'].mean().sort_values(ascending=False).iloc[1] # iloc -> 오름차순 앞에서 ? 번째
print(int(vvalue))

# 해당 화면에서는 제출하지 않으며, 문제 풀이 후 답안제출에서 결괏값 제출
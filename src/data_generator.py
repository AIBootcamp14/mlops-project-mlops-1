import pandas as pd
import random
import os

# 원본 데이터 로드
df = pd.read_csv('../data/spam.csv', sep=',', header=None, names=['label', 'text'])

# 새로운 데이터 생성
new_data_count = 1000
new_data_df = df.sample(n=new_data_count, replace=True, random_state=random.randint(1, 1000))

# 일부 데이터를 'ham'에서 'spam'으로, 또는 그 반대로 변경하여 드리프트를 시뮬레이션
for i in range(200):
    idx = random.choice(new_data_df.index)
    if new_data_df.loc[idx, 'label'] == 'ham':
        new_data_df.loc[idx, 'label'] = 'spam'
    else:
        new_data_df.loc[idx, 'label'] = 'ham'

# 새로운 데이터 파일로 저장
new_data_df.to_csv('new_data.csv', index=False)

print(f"✅ {len(new_data_df)}개의 새로운 데이터가 new_data.csv에 추가되었습니다.")


import pandas as pd
import numpy as np

#독립변수 Feature cores, threads, launch_date, lithography, base_frequency, turbo_frequency, tdp, cache_size
#종속변수 Feature Price

#데이터셋 불러오기
data_df = pd.read_csv("TeamProject/intel_processors.csv")
#사용될 데이터셋 구성 확인
print(data_df.info())

#독립변수 이외의 열 삭제
data_df = data_df.drop(['id', 'name', 'processor_number', 'bus_speed', 'configurable_tdp_up_frequency', 'configurable_tdp_up', 'product_line', 'socket', 'memory_type', 'url', 'vertical_segment', 'max_memory_size', 'status', 'max_temp', 'sku', 'package_size', 'fullname'], axis=1, inplace=False)
print(data_df.info())

#종속변수 price가 null인 데이터셋은 삭제
data_df = data_df[data_df['price'].notnull()]
print(data_df.info())
print(data_df.isnull().sum())

#lithography, tdp가 null값 삭제
data_df = data_df.dropna(subset=['lithography', 'tdp'])
print(data_df.info())
print(data_df.isnull().sum())


#base_frequency가 null값 중간값 삽입
data_df['base_frequency'] = data_df['base_frequency'].replace(np.nan, data_df['base_frequency'].median())
print(data_df.info())
print(data_df.isnull().sum())

#turbo_frequency가 null값 base_frequency값 삽입
data_df['turbo_frequency'] = np.where(pd.notnull(data_df['turbo_frequency'])==True, data_df['turbo_frequency'], data_df['base_frequency'])
print(data_df.info())
print(data_df.isnull().sum())

#threads가 null값 core값 삽입
data_df['threads'] = np.where(pd.notnull(data_df['threads'])==True, data_df['threads'], data_df['cores'])
print(data_df.info())
print(data_df.isnull().sum())

#launch_date의 -값을 뺀 숫자로만 생성(최근의 날짜면 큰숫자, 오래된 날짜면 작은 숫자가 나옴)
print(data_df['launch_date'].head())
data_df['launch_date'] = data_df['launch_date'].str.replace('-', '')
data_df['launch_date'] = data_df['launch_date'].str.extract(r'(\d\d\d\d\d\d)')
data_df['launch_date'] = data_df['launch_date'].astype('float64')
print(data_df.launch_date)
print(data_df.info())
print(data_df.isnull().sum())

#csv파일 저장
data_df.to_csv("TeamProject/intel_processor_set.csv", index=False)

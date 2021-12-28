from os import sep
import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import re

from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

nsmc_train_df = pd.read_csv('13/ratings_train.txt', encoding='utf8', sep='\t', engine='python')
#print(nsmc_train_df.head())
#print(nsmc_train_df.info())

nsmc_train_df = nsmc_train_df[nsmc_train_df['document'].notnull()]
#print(nsmc_train_df.info())
#print(nsmc_train_df['label'].value_counts())

nsmc_train_df['document'] = nsmc_train_df['document'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))
#print(nsmc_train_df.head())

nsmc_test_df = pd.read_csv('13/ratings_test.txt', encoding = 'utf8', sep = '\t', engine = 'python')
#print(nsmc_test_df.head())
#print(nsmc_test_df.info())

nsmc_test_df = nsmc_test_df[nsmc_test_df['document'].notnull()]
#print(nsmc_test_df['label'].value_counts())

nsmc_test_df['document'] = nsmc_test_df['document'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))
#print(nsmc_test_df.head())
#print(nsmc_test_df.info())

okt = Okt()
def okt_tokenizer(text):
    tokens = okt.morphs(text)
    return tokens

tfidf = TfidfVectorizer(tokenizer=okt_tokenizer, ngram_range=(1,2), min_df=3, max_df=0.9)
tfidf.fit(nsmc_train_df['document'])
nsmc_train_tfidf = tfidf.transform(nsmc_train_df['document'])

SA_lr = LogisticRegression(random_state=0)
SA_lr.fit(nsmc_train_tfidf, nsmc_train_df['label'])

params = {'C':[1, 3, 3.5, 4, 4.5, 5]}
SA_lr_grid_cv = GridSearchCV(SA_lr, param_grid=params, cv=3, scoring='accuracy', verbose=1)

SA_lr_grid_cv.fit(nsmc_train_df, nsmc_train_df['label'])
print(SA_lr_grid_cv.best_params_, round(SA_lr_grid_cv.best_score_, 4))
SA_lr_best = SA_lr_grid_cv.best_estimator_

#평가용 데이터의 피처 벡터화
nsmc_test_tfidf = tfidf.transform(nsmc_test_df['document'])
test_predict = SA_lr_best.predict(nsmc_test_tfidf)
print('감성 분석 정확도 : ', round(accuracy_score(nsmc_test_df['label'], test_predict), 3))

st = input('감성 분석할 문장 입력 >> ')

#입력데이터 전처리 수행
st = re.compile(r'[^ ㄱ-ㅣ가-힣]+').findall(st)
print(st)
st = [" ".join(st)]
print(st)

#1) 입력 텍스트의 피처 벡터화
st_tfidf = tfidf.transform(st)
#2) 최적 감성 분석 모델에 적용하여 감성 분석 평가
st_predict = SA_lr_best.predict(st_tfidf)
#3) 예측값 출력하기
if(st_predict == 0):
    print(st , "->> 부정 감성")
else:
    print(st , "->> 긍정 감성")
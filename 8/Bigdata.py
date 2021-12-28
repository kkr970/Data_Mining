import pandas as pd
import glob
import re
from functools import reduce
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud

all_files = glob.glob('8_data/myCabinetExcelData*.xls')
#print(all_files)

all_files_data = []
for file in all_files:
    data_frame = pd.read_excel(file)
    all_files_data.append(data_frame)

#print(all_files_data[0])

all_files_data_concat = pd.concat(all_files_data, axis=0, ignore_index=True)
#print(all_files_data_concat)

#all_files_data_concat.to_csv('C:/Users/user/Desktop/My_Python/8_data/riss_bigdata.csv', encoding='utf-8', index=False);

all_title = all_files_data_concat['제목']
#print(all_title)

stopWords = set(stopwords.words("english"))
lemma = WordNetLemmatizer()

words=[]

for title in all_title:
    EnWords = re.sub(r"[^a-zA-Z]+", " ", str(title)) #특수문자 제거
    EnWordsToken = word_tokenize(EnWords.lower()) #소문자로 정규화
    EnwordsTokenStop = [w for w in EnWordsToken if w not in stopWords] #단어 토근화
    EnwordsTokenStopLemma = [lemma.lemmatize(w) for w in EnwordsTokenStop] #불용어 제거
    words.append(EnwordsTokenStopLemma) #표제어 추출
#print(words)

words2 = list(reduce(lambda x, y:x+y, words)) #2차원리스트 -> 1차원 리스트
#print(words2)

count = Counter(words2)
#print(count)

word_count = dict()
for tag, counts in count.most_common(50):
    if(len(str(tag)) > 1): #표제어가 1개초과인 문자어만 선택
        word_count[tag] = counts
        #print("%s : %d"%(tag, counts))

del word_count['big']
del word_count['data']

sorted_Keys = sorted(word_count, key=word_count.get, reverse=True)
sorted_Values = sorted(word_count.values(), reverse=True)
#plt.bar(range(len(word_count)), sorted_Values, align = 'center')
#plt.xticks(range(len(word_count)), list(sorted_Keys), rotation = '85')
#plt.show()

all_files_data_concat['doc_count'] = 0
summary_year = all_files_data_concat.groupby('출판일', as_index=False)['doc_count'].count()
#print(summary_year)

#plt.figure(figsize=(12, 5))
#plt.xlabel("year")
#plt.ylabel("doc_count")
#plt.grid(True)
#plt.plot(range(len(summary_year)), summary_year['doc_count'])
#plt.xticks(range(len(summary_year)), [text for text in summary_year['출판일']])
#plt.show()

stopwords = set(STOPWORDS)
wc = WordCloud(background_color='ivory', stopwords=stopwords, width=800, height=600)
cloud = wc.generate_from_frequencies(word_count)
plt.figure(figsize=(8,8))
plt.imshow(cloud)
plt.axis("off")
#plt.show()

cloud.to_file('C:/Users/user/Desktop/My_Python/8/riss_bigdata_wordCloud.jpg')

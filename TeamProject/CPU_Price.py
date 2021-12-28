import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')
data_df = pd.read_csv('TeamProject/intel_processor_set.csv')

#Y, X데이터 분할
Y = data_df['price']
X = data_df.drop(['price'], axis=1, inplace=False)

#훈련용, 평가용 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 412)

#모델 생성
lr = LinearRegression()

#모델 훈련
lr.fit(X_train, Y_train)

#예측 수행
Y_predict = lr.predict(X_test)

#모델 구축하기
mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)
print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))
print('Y 절편 값: ', np.round(lr.intercept_, 2))
print('회귀 계수 값: ', np.round(lr.coef_, 2))

#모델 시각화
coef = pd.Series(data = np.round(lr.coef_, 2), index = X.columns)
print(coef.sort_values(ascending = False))

fig, axs = plt.subplots(ncols = 3, nrows = 3)
x_features = ['cores', 'threads', 'launch_date', 'lithography', 'base_frequency', 'turbo_frequency', 'cache_size', 'tdp']
plot_color = ['r', 'b', 'y', 'g', 'c', 'm', 'r', 'b']

for i, feature in enumerate(x_features):
    row = int(i/3)
    col = i%3
    sns.regplot(x = feature, y = 'price', data = data_df, ax = axs[row][col], color=plot_color[i])
plt.show()

print("가격측정을 원하는 CPU 정보를 입력하세요.")
cores = int(input("cores : "))
threads = int(input("threads : "))
tdp = int(input("tdp : "))
base_frequency = int(input("base_frequency : "))
turbo_frequency = int(input("turbo_frequency : "))
cache_size = int(input("cache_size : "))
lithography = int(input("lithography : "))
launch_date = int(input("launch_date(ex: 2021년2월=202102) : "))

price_predict = lr.predict([[cores, threads, launch_date, lithography, base_frequency, turbo_frequency, cache_size, tdp]])
print("이 CPU의 예상 가격은 %.2f$입니다." %price_predict)

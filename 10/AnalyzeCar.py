import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data_df = pd.read_csv('10/auto-mpg.csv', header=0, engine='python')

#print("데이터셋 크기 : ", data_df.shape)
#print(data_df.head())

data_df = data_df.drop(['origin'], axis = 1, inplace = False)
#horsePower를 안없애는 방법?
data_df['horsepower'] = data_df['horsepower'].replace('?', 0)
data_df['horsepower'] = data_df['horsepower'].replace('0', data_df['horsepower'].median())
data_df['horsepower2'] = data_df['horsepower'].astype('float64')
data_df = data_df.drop(['horsepower'], axis=1, inplace=False)
print(data_df.head())

#print(data_df.info())

#X, Y 분할하기
Y = data_df['mpg']
X = data_df.drop(['mpg'], axis = 1, inplace = False)



#훈련용 데이터와 평가용 데이터 분할하기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

lr = LinearRegression()

lr.fit(X_train, Y_train)

Y_predict = lr.predict(X_test)

mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)
#print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
#print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))

#print('Y 절편 값: ', np.round(lr.intercept_, 2))
#print('회귀 계수 값: ', np.round(lr.coef_, 2))

coef = pd.Series(data = np.round(lr.coef_, 2), index = X.columns)
coef.sort_values(ascending = False)

fig, axs = plt.subplots(figsize = (16, 16), ncols = 3, nrows = 2)
x_features = ['model_year', 'acceleration', 'displacement','horsepower2', 'weight', 'cylinders']
plot_color = ['r', 'b', 'y', 'g', 'r', 'b']
for i, feature in enumerate(x_features):
    row = int(i/3)
    col = i%3
    sns.regplot(x = feature, y = 'mpg', data = data_df, ax = axs[row][col], color = plot_color[i])
plt.show()

print("연비를 예측하고 싶은 차의 정보를 입력해주세요.")
cylinders_1 = int(input("cylinders : "))
displacement_1 = int(input("displacement : "))
weight_1 = int(input("weight : "))
acceleration_1 = int(input("acceleration : "))
model_year_1 = int(input("model_year : "))
horsepower_1 = int(input("horsepower : "))

mpg_predict = lr.predict([[cylinders_1, displacement_1, weight_1, acceleration_1 , model_year_1, horsepower_1]])

print("이 자동차의 예상 연비(MPG)는 %.2f입니다." %mpg_predict)


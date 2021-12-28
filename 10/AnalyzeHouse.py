import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


from sklearn.datasets import load_boston
boston = load_boston()

#print(boston.DESCR)
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
#print(boston_df.head())

boston_df['PRICE'] = boston.target
#print(boston_df.head())
#print("보스톤 주택 가격 데이터셋 크기 : ",boston_df.shape)
#print(boston_df.info())

Y = boston_df['PRICE']
X = boston_df.drop(['PRICE'], axis=1, inplace=False)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=156)

lr = LinearRegression()

lr.fit(X_train, Y_train)
Y_predict = lr.predict(X_test)
#print(Y_predict)

mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)
#print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
#print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))
#print('Y 절편 값: ', lr.intercept_)
#print('회귀 계수 값: ', np.round(lr.coef_, 1))

coef = pd.Series(data = np.round(lr.coef_, 2), index = X.columns)
coef.sort_values(ascending= False)

fig, axs = plt.subplots(figsize = (16, 16), ncols=3, nrows=5)

x_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

for i, feature in enumerate(x_features):
    row = int(i/3)
    col = i%3
    sns.regplot(x=feature, y='PRICE', data=boston_df, ax=axs[row][col])


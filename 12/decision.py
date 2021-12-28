import numpy as np
from numpy import lib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import graphviz

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import joblib

feature_name_df = pd.read_csv("C:/Users/user/Desktop/My_Python/12/data/features.txt", sep='\s+', header = None, names=['index', 'feature_name'], engine='python')
#print(feature_name_df.head())
#print(feature_name_df.shape)

feature_name = feature_name_df.iloc[:,1].values.tolist()
#print(feature_name[:5])

X_train = pd.read_csv("C:/Users/user/Desktop/My_Python/12/data/train/X_train.txt", sep='\s+', names=feature_name, engine='python')
X_test = pd.read_csv("C:/Users/user/Desktop/My_Python/12/data/test/X_test.txt", sep='\s+', names=feature_name, engine='python')
Y_train = pd.read_csv("C:/Users/user/Desktop/My_Python/12/data/train/Y_train.txt", sep='\s+', names=['action'], engine='python')
Y_test =  pd.read_csv("C:/Users/user/Desktop/My_Python/12/data/test/Y_test.txt", sep='\s+', names=['action'], engine='python')

#print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

#print(X_train.info())
#print(X_train.head())
#print(Y_train['action'].value_counts())

label_name_df = pd.read_csv('C:/Users/user/Desktop/My_Python/12/data/activity_labels.txt', sep = '\s+', header = None, names = ['index', 'label'], engine = 'python')
label_name = label_name_df.iloc[:, 1].values.tolist()
#print(label_name)

#dt_HAR = DecisionTreeClassifier(random_state=156)
#dt_HAR.fit(X_train, Y_train)
#joblib.dump(dt_HAR, './dt_HAR.pkl')
dt_HAR = joblib.load('./dt_HAR.pkl')

Y_predict = dt_HAR.predict(X_test)
#print(Y_predict)

accuracy = accuracy_score(Y_test, Y_predict)
#print('Decision tree Predict Accuracy: {0:.4f}'.format(accuracy))
#print('Decision tree HYPER params: \n', dt_HAR.get_params())

params = {
    'max_depth' : [6, 8, 10, 12, 16, 20, 24]
}
#grid_cv = GridSearchCV(dt_HAR, param_grid = params, scoring ='accuracy', cv = 5, return_train_score = True)
#grid_cv.fit(X_train, Y_train)
#cv_results_df = pd.DataFrame(grid_cv.cv_results_)
#cv_results_df[['param_max_depth', 'mean_test_score', 'mean_train_score']]

params = {
    'max_depth' : [8, 16, 20],
    'min_samples_split' : [8, 16, 24]
}

#grid_cv = GridSearchCV(dt_HAR, param_grid = params, scoring ='accuracy', cv = 5, return_train_score = True)
#grid_cv.fit(X_train, Y_train)
#joblib.dump(grid_cv, './grid_cv.pkl')
grid_cv = joblib.load('./grid_cv.pkl')

cv_results_df = pd.DataFrame(grid_cv.cv_results_)
#print(cv_results_df[['param_max_depth', 'param_min_samples_split', 'mean_test_score', 'mean_train_score']])
#print('best : {0:.4f}, best_hyper_params : {1}'.format(grid_cv.best_score_, grid_cv.best_params_))

best_dt_HAR = grid_cv.best_estimator_
best_Y_predict = best_dt_HAR.predict(X_test)
best_accuracy = accuracy_score(Y_test,best_Y_predict)
print("best decision tree predict accuracy: {0:.4f}".format(best_accuracy))

feature_importance_values = best_dt_HAR.feature_importances_
feature_importance_values_s = pd.Series(feature_importance_values, index=X_train.columns)

#feature_top10 = feature_importance_values_s.sort_values(ascending=False)[:10]
#plt.figure(figsize=(10, 5))
#plt.title('Feature TOP 10')
#sns.barplot(x=feature_top10, y=feature_top10.index)
#plt.show()

#os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz.bin'

export_graphviz(best_dt_HAR, out_file="tree.dot", class_names=label_name, feature_names=feature_name, impurity= True, filled=True)
with open("tree.dot")as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
#graphviz.Source(dot_graph).render(filename='tree.png')


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

titanic = pd.read_csv('7/titanic2.csv')

#print(titanic.isnull().sum())
#titanic['age'] = titanic['age'].fillna(titanic['age'].median())

#print(titanic['embarked'].value_counts())
#titanic['embarked'] = titanic['embarked'].fillna('S')

#print(titanic['embark_town'].value_counts())
#titanic['embark_town'] = titanic['embark_town'].fillna('Southampton')

#print(titanic['deck'].value_counts())
#titanic['deck'] = titanic['deck'].fillna('C')

#print(titanic.isnull().sum())

#print(titanic.info())

#print(titanic.survived.value_counts())

#f, ax = plt.subplots(1, 2, figsize = (10, 5))
#titanic['survived'][titanic['sex'] == 'male'].value_counts().plot.pie(explode = [0,0.1], autopct = '%1.1f%%', ax = ax[0], shadow = False)
#titanic['survived'][titanic['sex'] == 'female'].value_counts().plot.pie(explode = [0,0.1], autopct = '%1.1f%%', ax = ax[1], shadow = False)

#ax[0].set_title('Survived (Male)')
#ax[1].set_title('Survived (Female)')
#plt.show()

#sns.countplot(x='pclass', hue='survived', data = titanic)
#plt.title('Pclass - Survived')
#plt.show()

titanic_corr = titanic.corr(method='pearson')
#print(titanic_corr)
#titanic_corr.to_csv('C:/Users/user/Desktop/My_Python/8/titanic_corr.csv')

#print(titanic['survived'].corr(titanic['adult_male']))
#print(titanic['survived'].corr(titanic['fare']))

#이거 오류 왜생기냐 대체
#sns.pairplot(titanic, hue='survived')
#plt.show()

#sns.catplot(x = 'pclass', y = 'survived', hue = 'sex', data = titanic, kind = 'point')
#plt.show()

def category_age(x):
    if x<10:
        return 0
    if x<20:
        return 1
    if x<30:
        return 2
    if x<40:
        return 3
    if x<50:
        return 4
    if x<60:
        return 5
    if x<70:
        return 6
    else:
        return 7

titanic['age2'] = titanic['age'].apply(category_age)
titanic['sex'] = titanic['sex'].map({'male':1, 'female':0})
titanic['family'] = titanic['sibsp'] + titanic['parch']+1
titanic.to_csv('C:/Users/user/Desktop/My_Python/8/titanic3.csv', index=False)
heatmap_data = titanic[['survived', 'sex', 'age2', 'family', 'pclass', 'fare']]
colormap = plt.cm.RdBu
sns.heatmap(heatmap_data.astype(float).corr(), linewidths = 0.1, vmax= 1.0, square = True, cmap = colormap, linecolor = 'white', annot = True,annot_kws = {"size": 10})

plt.show()
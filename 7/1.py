import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols, glm
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

wine = pd.read_csv('6/wine.csv')
wine.columns = wine.columns.str.replace(' ','_')

red_wine_quality = wine.loc[wine['type'] == 'red', 'quality']
white_wine_quality = wine.loc[wine['type'] == 'white', 'quality']

#print(stats.ttest_ind(red_wine_quality, white_wine_quality, equal_var = False))
Rformula = 'quality ~ fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol'
regression_result = ols(Rformula, data = wine).fit()
#print(regression_result.summary())

sample1 = wine[wine.columns.difference(['quality', 'type'])]
sample1 = sample1[0:5][:]
sample1_predict = regression_result.predict(sample1)
#print(sample1_predict)
#print(wine[0:5]['quality'])

data = {"fixed_acidity" : [8.5, 8.1], "volatile_acidity":[0.8, 0.5],"citric_acid":[0.3, 0.4], "residual_sugar":[6.1, 5.8], "chlorides":[0.055,0.04], "free_sulfur_dioxide":[30.0, 31.0], "total_sulfur_dioxide":[98.0,99], "density":[0.996, 0.91], "pH":[3.25, 3.01], "sulphates":[0.4, 0.35],"alcohol":[9.0, 0.88]}
sample2 = pd.DataFrame(data, columns=sample1.columns)
#print(sample2)
sample2_predict = regression_result.predict(sample2)
#print(sample2_predict)

sns.set_style('dark')
sns.histplot(red_wine_quality, kde = True, color = "red", label = 'red wine')
sns.histplot(white_wine_quality, kde = True, label = 'white wine')
plt.title('Quality of Wine Type')
plt.legend()
#plt.show()

others = list(set(wine.columns).difference(set(['quality', 'fixed_acidity'])))
p1, resids1 = sm.graphics.plot_partregress('quality', 'fixed_acidity', others, data = wine, ret_coords=True)
#plt.show()
p2, resids2 = sm.graphics.plot_partregress('quality', 'volatile_acidity', others, data = wine, ret_coords=True)
#plt.show()


import pandas as pd
import math
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

retail_df = pd.read_excel('12/Online Retail.xlsx')
#print(retail_df.head())
#print(retail_df.info())

retail_df = retail_df[retail_df['Quantity']>0]
retail_df = retail_df[retail_df['UnitPrice']>0]
retail_df = retail_df[retail_df['CustomerID'].notnull()]
retail_df['CustomerID'] = retail_df['CustomerID'].astype(int)
#print(retail_df.info())
#print(retail_df.isnull().sum())
#print(retail_df.shape)

retail_df.drop_duplicates(inplace = True)
#print(retail_df.shape)

pd.DataFrame([{'Product':len(retail_df['StockCode'].value_counts()), 'Transaction':len(retail_df['InvoiceNo'].value_counts()), 'Customer':len(retail_df['CustomerID'].value_counts())}], columns = ['Product', 'Transaction', 'Customer'], index = ['counts'])
#print(retail_df['Country'].value_counts())

retail_df['SaleAmount'] = retail_df['UnitPrice']*retail_df['Quantity']
#print(retail_df.head())

aggregations = {
    'InvoiceNo':'count',
    'SaleAmount':'sum',
    'InvoiceDate':'max'
}
customer_df = retail_df.groupby('CustomerID').agg(aggregations)
customer_df = customer_df.reset_index()
#print(customer_df.head())

customer_df = customer_df.rename(columns = {'InvoiceNo':'Freq', 'InvoiceDate':'ElapsedDays'})
#print(customer_df.head())


customer_df['ElapsedDays'] = datetime.datetime(2011,12,10) - customer_df['ElapsedDays']
#print(customer_df.head())

customer_df['ElapsedDays'] = customer_df['ElapsedDays'].apply(lambda x: x.days+1)
#print(customer_df.head())

#fig, ax = plt.subplots()
#ax.boxplot([customer_df['Freq'], customer_df['SaleAmount'], customer_df['ElapsedDays']], sym = 'bo')
#plt.xticks([1, 2, 3], ['Freq', 'SaleAmount', 'ElapsedDays'])
#plt.show()

customer_df['Freq_log'] = np.log1p(customer_df['Freq'])
customer_df['SaleAmount_log'] = np.log1p(customer_df['SaleAmount'])
customer_df['ElapsedDays_log'] = np.log1p(customer_df['ElapsedDays'])
#print(customer_df.head())

#fig, ax = plt.subplots()
#ax.boxplot([customer_df['Freq_log'], customer_df['SaleAmount_log'], customer_df['ElapsedDays_log']], sym = 'bo')
#plt.xticks([1, 2, 3], ['Freq_log', 'SaleAmount_log', 'ElapsedDays_log'])
#plt.show()

X_features = customer_df[['Freq_log', 'SaleAmount_log', 'ElapsedDays_log']].values
X_features_scaled = StandardScaler().fit_transform(X_features)

distortions = []
for i in range(1, 11):
    kmeans_i = KMeans(n_clusters = i, random_state = 0)
    kmeans_i.fit(X_features_scaled)
    distortions.append(kmeans_i.inertia_)

#plt.plot(range(1,11), distortions, marker = 'o')
#plt.xlabel('Number of clusters')
#plt.ylabel('Distortion')
#plt.show()

kmeans = KMeans(n_clusters=3, random_state=0)
Y_labels = kmeans.fit(X_features_scaled)
customer_df['ClusterLabel'] = Y_labels
#print(customer_df.head())

def silhouetteViz(n_cluster, X_features):
    kmeans = KMeans(n_clusters = n_cluster, random_state = 0)
    Y_labels = kmeans.fit_predict(X_features)

    silhouette_values = silhouette_samples(X_features, Y_labels, metric = 'euclidean')
   
    y_ax_lower, y_ax_upper = 0, 0
    y_ticks = []

    for c in range(n_cluster):
        c_silhouettes = silhouette_values[Y_labels == c]
        c_silhouettes.sort()
        y_ax_upper += len(c_silhouettes)
        color = cm.jet(float(c) / n_cluster)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouettes, height = 1.0, edgecolor = 'none', color = color)
        y_ticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouettes)

    silhouette_avg = np.mean(silhouette_values)
    plt.axvline(silhouette_avg, color = 'red', linestyle = '--')
    plt.title('Number of Cluster : ' + str(n_cluster) + '\n' + 'Silhouette Score : '+ str(round(silhouette_avg, 3)))
    plt.yticks(y_ticks, range(n_cluster))
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.tight_layout()
    plt.show()

def clusterScatter(n_cluster, X_features):
    c_colors = []
    kmeans = KMeans(n_clusters = n_cluster, random_state = 0)
    Y_labels = kmeans.fit_predict(X_features)

    for i in range(n_cluster):
        c_color = cm.jet(float(i) / n_cluster) #클러스터의 색상 설정
        c_colors.append(c_color)
        #클러스터의 데이터 분포를 동그라미로 시각화
        plt.scatter(X_features[Y_labels == i,0], X_features[Y_labels == i,1], marker = 'o', color = c_color, edgecolor = 'black', s = 50, label = 'cluster '+ str(i))

    for i in range(n_cluster):
        plt.scatter(kmeans.cluster_centers_[i,0], kmeans.cluster_centers_[i,1], marker = '^', color = c_colors[i], edgecolor = 'w', s = 200)
    
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

#silhouetteViz(3, X_features_scaled)
#silhouetteViz(4, X_features_scaled)
#silhouetteViz(5, X_features_scaled)
#silhouetteViz(6, X_features_scaled)

#clusterScatter(3, X_features_scaled)
#clusterScatter(4, X_features_scaled)
#clusterScatter(5, X_features_scaled)
#clusterScatter(6, X_features_scaled)

best_cluster = 4
kmeans = KMeans(n_clusters = best_cluster, random_state = 0)
Y_labels = kmeans.fit_predict(X_features_scaled)

customer_df['ClusterLabel'] = Y_labels
#print(customer_df.head())
customer_df.to_csv('C:/Users/user/Desktop/My_Python/12/Online_Retail_Customer_Cluster.csv')

#print(customer_df.groupby('ClusterLabel')['CustomerID'].count())

customer_cluster_df = customer_df.drop(['Freq_log', 'SaleAmount_log', 'ElapsedDays_log'],axis = 1, inplace = False)
#print(customer_cluster_df.head())

#주문 1회당 평균 구매금액: SaleAmountAvg
customer_cluster_df['SaleAmountAvg'] = customer_cluster_df['SaleAmount']/customer_cluster_df['Freq']
#print(customer_cluster_df.head())

customer_cluster_df.drop(['CustomerID'], axis = 1, inplace = False).groupby('ClusterLabel').mean()
#print(customer_cluster_df.head())

def clusterScatter3D(n_cluster, X_features):
    c_colors = []
    kmeans = KMeans(n_clusters = n_cluster, random_state = 0)
    Y_labels = kmeans.fit_predict(X_features)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')

    for i in range(n_cluster):
        c_color = cm.jet(float(i) / n_cluster) #클러스터의 색상 설정
        c_colors.append(c_color)
        #클러스터의 데이터 분포를 동그라미로 시각화
        ax.scatter(X_features[Y_labels == i,0], X_features[Y_labels == i,1], X_features[Y_labels == i,2], marker = 'o', color = c_color, edgecolor = 'black', s = 50, label = 'cluster '+ str(i))

    #각 클러스터의 중심점을 삼각형으로 표시
    for i in range(n_cluster):
        ax.scatter(kmeans.cluster_centers_[i,0], kmeans.cluster_centers_[i,1], kmeans.cluster_centers_[i,2], marker = '^', color = c_colors[i], edgecolor = 'w', s = 200)

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
#clusterScatter3D(3, X_features_scaled)

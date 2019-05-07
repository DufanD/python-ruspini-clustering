#%%all
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = np.array(pd.read_csv('datasets/ruspini.csv'))

kmeans = KMeans(n_clusters=4)

new_cluster = kmeans.fit_predict(data[:, :2])

new_data = pd.DataFrame({
    'x': data[:, 0],
    'y': data[:, 1],
    'cluster': new_cluster[:],
})

plt.figure('Ruspini Clustering with K-Means')
plt.scatter(new_data['x'].values,
            new_data['y'].values,
            s=100,
            c=new_data['cluster'].values)
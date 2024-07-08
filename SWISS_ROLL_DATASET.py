# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:51:04 2024

@author: Dragana
"""

from sklearn.datasets import make_swiss_roll
import time as time
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
#%%
import os
os.chdir(os.path.dirname(__file__))
#%%
from TSNE import *
from PCA import *
from KernelPCA import *
#%%

X, color= make_swiss_roll(n_samples=1500, hole = True,random_state = 0)
#%%
# Plot the Swiss roll dataset
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.viridis) #cmap='tab10'
ax.set_title('Скуп података швајцарска ролница')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(azim=-66, elev=12)
plt.show()
#%%
pca_algorithm = PCA()
x_pca, explained_variance= pca_algorithm.pca(X, 30)
x_pca = x_pca.real
print(explained_variance)
#%%
plt.figure(figsize=(8, 6))
plt.stem(np.cumsum(explained_variance))
plt.title('Кумулативна објашњена варијанса за првих 30 компоненти')
plt.xlabel('Кумулативна објашњена варијанса')
plt.ylabel('Број компоненти')
plt.show()
#%%
plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=color, cmap=plt.cm.viridis)
plt.colorbar()
plt.title('PCA визуaлизација швајцарске ролнице ')
plt.xlabel('компонента 1')
plt.ylabel('компонента 2')
plt.show()
#%%
kernel_pca = KernelPCA(0.05)
x_kpca, explained_variance= kernel_pca.kpca(X, 30)
x_kpca = x_kpca.real
print(np.cumsum(explained_variance))
#%%
plt.figure(figsize=(8, 6))
plt.stem(np.cumsum(explained_variance))
plt.title('Кумулативна објашњена варијанса за првих 30 компоненти')
plt.xlabel('Кумулативна објашњена варијанса')
plt.ylabel('Број компоненти')
plt.show()
#%%
plt.figure(figsize=(8, 6))
plt.scatter(x_kpca[:, 0], x_kpca[:, 1], c=color, cmap=plt.cm.viridis)
plt.colorbar()
plt.title('Kernel PCA визуaлизација С-криве ')
plt.xlabel('компонента 1')
plt.ylabel('компонента 2')
plt.show()
#%%
tsne_algorithm = TSNE(X)
Y,C_array = tsne_algorithm.tsne( 2, 30.0)
#%%
plt.figure(figsize=(8, 6))
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.viridis)
plt.colorbar()
plt.title('t-SNE визуализација швајцарске ролнице')
plt.xlabel('Компонента 1')
plt.ylabel('Компонента 2')
plt.show()
#%%
plt.figure(figsize=(8, 6))
plt.plot(np.arange(0,1000,10),C_array)
plt.title('t-SNE губитак у току тренирања швајцарске ролнице')
plt.xlabel('Итерација')
plt.ylabel('Губитак')
plt.show()

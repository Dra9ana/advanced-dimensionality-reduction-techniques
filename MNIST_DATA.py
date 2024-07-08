# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 15:56:31 2024

@author: Dragana
"""
from __future__ import print_function
import time

import numpy as np
import pandas as pd

from sklearn.datasets import load_digits

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from sklearn.datasets import fetch_openml
#%%
from TSNE import *
#%%
import os
os.chdir(os.path.dirname(__file__))
#%%
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
labels= y.apply(lambda i: int(i)).to_numpy()
X = X/255
#%%

def balanced_sampling(data, labels, samples_per_class):
    balanced_data = []
    balanced_labels = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        sampled_indices = np.random.choice(label_indices, samples_per_class, replace=False)
        balanced_data.extend(data[sampled_indices])
        balanced_labels.extend(labels[sampled_indices])
    balanced_data = np.array(balanced_data)
    balanced_labels = np.array(balanced_labels)
    return balanced_data, balanced_labels
#%%
x_in, y_in = balanced_sampling(X.values,labels,600)

#%%

np.random.seed(42)

rndperm = np.random.permutation(x_in.shape[0])
x_in = x_in[rndperm,:]
y_in = y_in[rndperm]
print("y_in", y_in)
print("x_in", x_in)
#%%
pca_algorithm = PCA()
x_pca, explained_variance= pca_algorithm.pca(x_in, 30)
x_pca = x_pca.real
print(np.cumsum(explained_variance))
#%%
plt.figure(figsize=(8, 6))
plt.stem(np.cumsum(explained_variance)[0:30])
plt.title('Кумулативна објашњена варијанса за првих 30 компоненти')
plt.xlabel('Кумулативна објашњена варијанса')
plt.ylabel('Број компоненти')
plt.show()
#%%
plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y_in, cmap='tab10')
plt.colorbar()
plt.title('PCA визуaлизација MNIST скупа података')
plt.xlabel('компонента 1')
plt.ylabel('компонента 2')
plt.show()
#%%
kernel_pca = KernelPCA(0.1)
x_kpca, explained_variance= kernel_pca.kpca(x_in, 30)
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
plt.scatter(x_kpca[:, 0], x_kpca[:, 1], c=y_in, cmap=plt.cm.viridis)
plt.colorbar()
plt.title('Kernel PCA визуaлизација С-криве ')
plt.xlabel('компонента 1')
plt.ylabel('компонента 2')
plt.show()
#%%
tsne_algorithm = TSNE(x_pca)
Y, C_array= tsne_algorithm.tsne( 2,40.0)
#%%
plt.figure(figsize=(8, 6))
plt.scatter(Y[:, 0], Y[:, 1],c = y_in, cmap='tab10')
plt.colorbar()
plt.title('t-SNE визуализација MNIST скупа података')
plt.xlabel('Компонента 1')
plt.ylabel('Компонента 2')
plt.show()
#%%
plt.figure(figsize=(8, 6))
plt.plot(np.arange(0,500,10),C_array)
plt.title('t-SNE губитак у току тренирања  MNIST скупа података')
plt.xlabel('Итерација')
plt.ylabel('Губитак')
plt.show()
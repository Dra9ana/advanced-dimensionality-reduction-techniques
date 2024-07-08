# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 17:59:13 2024

@author: Dragana
"""
#%%
import os
os.chdir(os.path.dirname(__file__))
#%%
import numpy as np
from sklearn.datasets import fetch_openml
#%%
from TSNE import *
from KernelPCA import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#%%
from KernelPCA import *
#%%
# Fetch CIFAR-10 dataset
cifar10 = fetch_openml(name="CIFAR_10", version=1)
# Access the data and labels
X, y = cifar10.data, cifar10.target
#%%
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
N_samples = 600
N_classes = 10
x_in, y_in = balanced_sampling(X.values,labels,N_samples)

#%%

np.random.seed(42)

rndperm = np.random.permutation(x_in.shape[0])
x_in = x_in[rndperm,:]
y_in = y_in[rndperm]
print("y_in", y_in)
print("x_in", x_in)
#%%
from keras.applications import ResNet50
H,W,C = (32,32,3)
N = N_samples*N_classes
resnet_model = ResNet50(weights='imagenet', include_top=False,input_shape = (H,W,C))
#%%
features = resnet_model.predict(x_in.reshape(N,H,W,C))
#%%
kernel_pca = KernelPCA(0.05)
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
# kernel_pca = KernelPCA(
#     n_components=50, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1
# )
#x_kernel_pca = kernel_pca.fit_transform(x_in)

#%%
# pca_algorithm = PCA()
# x_pca, explained_variance= pca_algorithm.pca(x_in, 30)
# x_pca = x_pca.real
# print(np.cumsum(explained_variance))
#%%
# plt.figure(figsize=(8, 6))
# plt.stem(np.cumsum(explained_variance)[0:30])
# plt.title('Кумулативна објашњена варијанса за првих 30 компоненти')
# plt.xlabel('Кумулативна објашњена варијанса')
# plt.ylabel('Број компоненти')
# plt.show()
# #%%
plt.figure(figsize=(8, 6))
plt.scatter(x_kernel_pca[:, 0], x_kernel_pca[:, 1], c=y_in, cmap='tab10')
plt.colorbar()
plt.title('PCA визуaлизација MNIST скупа података')
plt.xlabel('компонента 1')
plt.ylabel('компонента 2')
plt.show()
#%%
tsne_algorithm = TSNE(x_kpca)
Y, C_array= tsne_algorithm.tsne( 2, 30.0)
#%%
y_out = []
for l in y_in:
    if l in [1,2,3,4,6]:
        y_out += [0]
    else:
        y_out +=[1]
#%%
plt.figure(figsize=(8, 6))
plt.scatter(Y[:, 0], Y[:, 1],c = y_in, cmap='tab10')
plt.colorbar()
plt.title('t-SNE визуализација CIFAR скупа података')
plt.xlabel('Компонента 1')
plt.ylabel('Компонента 2')
plt.show()
#%%
plt.figure(figsize=(8, 6))
plt.plot(np.arange(0,500,10),C_array)
plt.title('t-SNE губитак у току тренирања  CIFAR скупа података')
plt.xlabel('Итерација')
plt.ylabel('Губитак')
plt.show()
#%%
#%%
plt.figure(figsize=(8, 6))
plt.scatter(Y[:, 0], Y[:, 1],c = y_out, cmap='tab10')
plt.colorbar()
plt.title('t-SNE визуализација CIFAR скупа података')
plt.xlabel('Компонента 1')
plt.ylabel('Компонента 2')
plt.show()
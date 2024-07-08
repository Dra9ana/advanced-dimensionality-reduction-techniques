# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 13:15:20 2024

@author: Dragana
"""
import numpy as np

class PCA:
   def __init__(self):
      pass
   def pca(self,X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y, l / np.sum(l)
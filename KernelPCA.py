# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:09:09 2024

@author: Dragana
"""
import numpy as np 

class KernelPCA:
    def __init__(self, gamma=1.0):
        self.gamma = gamma
    
    def _kernel(self, X):
        """ Compute the Gaussian (RBF) kernel matrix """
        X_square = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
        return np.exp(-self.gamma * X_square)
    
    def kpca(self, X=np.array([]), no_dims=50):
        """
        Runs Kernel PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions using a Gaussian kernel.
        """
        print("Preprocessing the data using Kernel PCA...")
        
        # Compute the kernel matrix
        K = self._kernel(X)
        
        # Center the kernel matrix
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        
        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(K_centered)
        
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]
        
        # Select the top no_dims eigenvectors (principal components)
        alphas = eigvecs[:, :no_dims]
        lambdas = eigvals[:no_dims]
        
        # Project the data onto the new space
        Y = np.dot(K_centered, alphas / np.sqrt(lambdas))
        
        return Y, lambdas / np.sum(lambdas)

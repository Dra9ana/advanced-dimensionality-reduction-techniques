# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:03:20 2024

@author: Dragana
"""
import os
os.chdir(os.path.dirname(__file__))
#%%
import numpy as np


class TSNE:
    def __init__(self, x: np.array):
        '''
        Constructor.
        Parameters
        ----------
        x : float np.array - input data with dimensions n_sample*n_features
        y : float np.array - input labels

        Returns
        -------
        None.

        '''
        self.x = x
        self.n_sample, self.n_feature = x.shape

    def compute_entropy(self, distances_i, i, precision=1.0):
        '''
        Compute the perobabilities and entropy of row i, for a specific value of the
        precision of a Gaussian distribution.
        precision = 1/sigma^2
        p_{i,j} = exp(-distance_{i,j} * precision)/sum(exp(-distances_i .* precision))
        P = exp(-distances_i * precision) -> pij = P/sum(P)
        entropy = sum(pij * log(pij)) = sum(P/sum(P)*log(P/sum(P)))
        (1) sum(P) is not dependent on the array position -> it is a constant
        (2) entropy = -sum(P*(log P - log(sum(P))))/sum(P) =
                = -sum(P*log(P))/sum(P) + sum(P)*log(sum(P))/sum(P) =
                = -sum(P*log(P))/sum(P) + log(sum(P))
        (3) log(P) = log(exp(-distances_i* precision)) = -distances_i*precision
        (2) and (3) -> entropy = sum(P)*distances*precision)/sum(P)+ log(sum(P))
        Parameters
        -----------
        distances_i : np.array of len n_sample
        precision: equal to 1/sigma^2 where sigma is standard deviation of Gaussian pdf

        Returns
        -------
        tuple (entropy, pji)
        entropy - entropy of the i-th row
        pij - conditional probability that jth element is neighbour of i
        '''

        exp_distances_i= np.exp(-distances_i.copy() * precision)
        #exp_distances_i[i] = 0
        exp_distances_i_sum = sum(exp_distances_i)
        entropy = precision * np.sum(distances_i * exp_distances_i) / exp_distances_i_sum + np.log(exp_distances_i_sum)
        pi = exp_distances_i / exp_distances_i_sum
        return entropy, pi

    def compute_distances(self):
        '''
        Compute distance between i-th and j-th sample i.e. between i-th and j-th row
        Because of computational optimization it is calculated as:
        distances_{i,j}= ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i \cdot x_j).

        Parameters
        ----------
        None

        Returns
        -------
        distances : matrix which element with indexes i, j is distance between samples

        '''
        x_i_norm = np.sum(np.square(self.x), 1)
        x_dot_prod = np.dot(self.x, self.x.T)
        x_j_norm_minus_dot_prod = np.add(-2 * x_dot_prod, x_i_norm).T
        distances = np.add(x_j_norm_minus_dot_prod, x_i_norm)
        # exclude point i from calculation
        distances[range(0, self.n_sample), range(0, self.n_sample)] = 0
        return distances

    def binary_search(self, i, distance_i, tolerance=1e-5, logK=np.log(30), max_iter=50):
        '''
        Binary search of parameter precision so that entropy equals to logK
        precision - 1/sigma^2
        Parameters
        ----------
        i : integer
            index of sample
        distance_i : np.array of length n_features
            row of distances between ith sample and other samples
        logK : float number
            log ofvirtual number of neighbours
        max_iter : int
            maximum number of iterations allowed
        tolerance: float
            tolerance of making an error
        Returns
        -------
        pi : probability for being a neighbour of ith sample

        '''

        precision_min = -np.inf
        precision_max = np.inf
        precision = np.ones((self.n_sample, 1))

        (entropy, pi) = self.compute_entropy(distance_i, i, precision[i])
        d_entropy = entropy - logK  # difference between current entropy and wanted entropy

        current_iter = 0
        while np.abs(d_entropy) > tolerance and current_iter < max_iter:
            # if difference between entropies is too big we should decrease entropy
            # -> too heavy tails -> decrease sigma -> increase precision
            if d_entropy > 0:
                precision_min = precision[i].copy()
                # if it's not bounded, cointinue searching for upper bound
                if precision_max == np.inf or precision_max == -np.inf:
                    precision[i] = precision[i] * 2.
                else:  # otherwise, try a middle
                    precision[i] = (precision_min + precision_max) / 2.
            else:  # increase entropy -> not enough heavy tails -> increase sigma -> decrease precision
                precision_max = precision[i].copy()
                # if it's not bounded continue searching for lower bound
                if precision_min == np.inf or precision_min == -np.inf:
                    precision[i] = precision[i] / 2.
                else:  # otherwise try a middle
                    precision[i] = (precision_max + precision_min) / 2.

            # Recompute the values
            (entropy, pi) = self.compute_entropy(distance_i, i, precision[i])
            d_entropy = entropy - logK
            current_iter += 1
        return pi, precision

    def x2p(self, tol=1e-5, K=30.0, max_iter=50):
        '''
        Parameters
        ----------
        tol : TYPE, optional
            DESCRIPTION. The default is 1e-5.
        K : TYPE, optional
            DESCRIPTION. The default is 30.0.

        Returns
        -------
        P : TYPE
            DESCRIPTION.

        '''

        print("Computing pairwise distances...")
        distances = self.compute_distances()
        P = np.zeros((self.n_sample, self.n_sample))
        logK = np.log(K)

        # Loop over all datapoints
        for i in range(self.n_sample):
            if i % 500 == 0:
                print("Computing P-values for point %d of %d..." %
                      (i, self.n_sample))
            # setting distance_i[i] to 0 is not enough because exp(0) = 1
            distance_i = np.concatenate( (distances[i, 0:i], distances[i,i+1:self.n_sample]))
            #distance_i = distances[i,:]
            # Set the row of P for
            PP, last_precision = self.binary_search(i, distance_i, tol, logK, max_iter)
            P[i, :i] = PP[:i]
            P[i, i+1:] = PP[i:]

        # Return final P-matrix
        print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / last_precision)))
        return P

    def compute_p_values(self, P):
        P = P + np.transpose(P)  # pij + pji
        P = P / np.sum(P)  # normalize
        P = P * 4.  # early exaggeration
        P = np.maximum(P, 1e-12)  # remove small ones
        return P

    def compute_pairwise_afinities(self, Y):
        y_squared_sum = np.sum(np.square(Y), 1)
        y_dot_prod = np.dot(Y, Y.T)
        y_i_norm = np.add(-2.*y_dot_prod, y_squared_sum)
        y_j_norm = y_i_norm.T
        Q = 1. / (1. + np.add(y_j_norm, y_squared_sum))
        Q[range(self.n_sample), range(self.n_sample)] = 0.  # remove Q[i,i]
        Q_norm = Q / np.sum(Q)  # normalize
        Q_norm = np.maximum(Q_norm, 1e-12)# TODO
        return (Q_norm, Q)

    def tsne(self, n_dims=2, K=40.0):
        """
            Runs t-SNE on the dataset in the NxD array X to reduce its
            dimensionality to n_dims dimensions. The syntaxis of the function is
            `Y = tsne.tsne(X, n_dims, K), where X is an NxD NumPy array.
        """
        # Initialize variables
        max_iter = 500# u radu
        initial_momentum = 0.5# u radu
        final_momentum = 0.8# u radu
        eta = 100# TODO lose u radu 
        min_gain = 0.01# ne znam sta je ovo

        Y = np.random.randn(self.n_sample, n_dims)
        dC_t = np.zeros((self.n_sample, n_dims))
        dY = np.zeros((self.n_sample, n_dims))
        gains = np.ones((self.n_sample, n_dims))

        # Compute P-values
        P = self.x2p(1e-5, K)
        P = self.compute_p_values(P)
        C_array = []
        # Run iterations
        for iter in range(max_iter):
            Q, t_distribution_i_j = self.compute_pairwise_afinities(Y)
            # Compute gradient
            diff_pij_qij = P - Q
            product_1 = np.multiply(diff_pij_qij, t_distribution_i_j)
            for i in range(self.n_sample):
                diff_yi_yj = (Y[i, :] - Y)
                dC_t[i, :] = np.matmul(product_1[i,:], diff_yi_yj)
            #    dC_t[i, :] = np.sum(np.tile(diff_pij_qij[:, i] * t_distribution_i_j[:, i], (n_dims, 1)).T * diff_yi_yj, 0)
            
            

            # Perform the update
            if iter < 250:# TODO nije kao u radu 
                momentum = initial_momentum
            else:
                momentum = final_momentum

            gains = (gains + 0.2) * ((dC_t > 0.) != (dY > 0.)) + \
                (gains * 0.8) * ((dC_t > 0.) == (dY > 0.))
            gains[gains < min_gain] = min_gain

            dY = momentum * dY - eta * (gains * dC_t)
            Y = Y + dY
            Y = Y - np.tile(np.mean(Y, 0), (self.n_sample, 1))

            # Compute current value of cost function
            if (iter + 1) % 10 == 0:
                C = np.sum(P * np.log(P / Q))
                C_array.append(C)
                print("Iteration %d: error is %f" % (iter + 1, C))

            # Stop lying about P-values
            if iter == 50:
                P = P / 4.
        print(C_array)
        # Return solution
        return Y, C_array
# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/22 13:58
@Function:
"""

import numpy as np

class TCA:

    def __init__(self, dim=5, kernel='rbf', kernel_param=1, mu=1):
        """

        :param dim:
        :param kernel: 'rbf' | 'linear' | 'poly'
        :param kernel_param:
        :param mu:
        """
        self.dim = dim
        self.kernel = kernel
        self.kernel_param = kernel_param
        self.mu = mu
        self.X = None
        self.V = None

    def _get_L(self, n_source, n_targetget):
        """
        Get MMD matrix
        :param n_source: number of source samples
        :param n_targetget: number of target samples
        :return:
        """
        L_ss = (1. / (n_source * n_source)) * np.full((n_source, n_source), 1)
        L_st = (-1. / (n_source * n_targetget)) * np.full((n_source, n_targetget), 1)
        L_ts = (-1. / (n_targetget * n_source)) * np.full((n_targetget, n_source), 1)
        L_tt = (1. / (n_targetget * n_targetget)) * np.full((n_targetget, n_targetget), 1)

        L_up = np.hstack((L_ss, L_st))
        L_down = np.hstack((L_ts, L_tt))
        L = np.vstack((L_up, L_down))

        return L

    def _kernel_func(self, x1, x2=None):
        """
        Calculate kernel for TCA
        :param x1:
        :param x2:
        :return:
        """
        n1, dim = x1.shape
        K = None
        if x2 is not None:
            n2 = x2.shape[0]
        if self.kernel == 'linear':
            if x2 is not None:
                K = np.dot(x2, x1.T)
            else:
                K = np.dot(x1, x1.T)
        elif self.kernel == 'poly':
            if x2 is not None:
                K = np.power(np.dot(x1, x2.T), self.kernel_param)
            else:
                K = np.power(np.dot(x1, x1.T), self.kernel_param)
        elif self.kernel == 'rbf':
            if x2 is not None:
                sum_x2 = np.sum(np.multiply(x2, x2), axis=1)
                sum_x2 = sum_x2.reshape((len(sum_x2), 1))
                K = np.exp(-1 * (np.tile(np.sum(np.multiply(x1, x1), axis=1).T, (n2, 1)) + np.tile(sum_x2, (1, n1))
                                 - 2 * np.dot(x2, x1.T)) / (dim * 2 * self.kernel_param))
            else:
                P = np.sum(np.multiply(x1, x1), axis=1)
                P = P.reshape((len(P), 1))
                K = np.exp(-1 * (np.tile(P.T, (n1, 1)) + np.tile(P, (1, n1)) - 2 * np.dot(x1, x1.T)) / (dim * 2 * self.kernel_param))
        # more kernels can be added
        return K
    
    def fit_transform(self, x_source, x_target):
        """
        TCA main method. Wrapped from Sinno J. Pan and Qiang Yang's "Domain adaptation via 
        transfer component ayalysis. IEEE TNN 2011" 
        :param x_source: Source domain data feature matrix. 
        :param x_target: Target domain data feature matrix. 
        :return: 
        """
        n_source = x_source.shape[0]
        n_target = x_target.shape[0]
        self.X = np.vstack((x_source, x_target))
        L = self._get_L(n_source, n_target)
        L[np.isnan(L)] = 0
        K = self._kernel_func(self.X)
        K[np.isnan(K)] = 0
        H = np.identity(n_source + n_target) - 1. / (n_source + n_target) * np.ones(shape=(n_source + n_target, 1)) * np.ones(
            shape=(n_source + n_target, 1)).T
        forPinv = self.mu * np.identity(n_source + n_target) + np.dot(np.dot(K, L), K)
        forPinv[np.isnan(forPinv)] = 0
        Kc = np.dot(np.dot(np.dot(np.linalg.pinv(forPinv), K), H), K)
        Kc[np.isnan(Kc)] = 0
        D, V = np.linalg.eig(Kc)
        eig_values = D.reshape(len(D), 1)
        eig_values_sorted = np.sort(eig_values[::-1], axis=0)
        index_sorted = np.argsort(-eig_values, axis=0)
        V = V[:, index_sorted]
        self.V = V.reshape((V.shape[0], V.shape[1]))
        x_source_tca = np.dot(K[:n_source, :], V)
        x_target_tca = np.dot(K[n_source:, :], V)

        x_source_tca = np.asarray(x_source_tca[:, :self.dim], dtype=float)
        x_target_tca = np.asarray(x_target_tca[:, :self.dim], dtype=float)

        return x_source_tca, x_target_tca

    def fit(self, x_target):
        """
        change the other target data
        :param x_target: Out-of-sample target data feature matrix.
        :return:
        """
        K_target = self._kernel_func(self.X, x_target)
        x_target_tca = np.dot(K_target, self.V)
        x_target_tca = x_target_tca[:, :self.dim]

        return x_target_tca


if __name__ == '__main__':
    pass
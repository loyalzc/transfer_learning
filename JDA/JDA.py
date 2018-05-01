# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/3/17 20:40
@Function:
"""
from Demos.dde.ddeclient import sl
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class JDA:

    def __init__(self, base_classifier=DecisionTreeClassifier(), dim=5, N=10, k=200, lmbda=1.0, kernel='liner', kernel_param=1.0):
        self.base_classifier = base_classifier
        self.dim = dim
        self.N = N
        self.k = k
        self.lmbda = lmbda
        self.kernel = kernel
        self.kernel_param = kernel_param
        self.Zs = None
        self.Zt = None

    def fit_transform(self, x_source, x_target, y_source):
        self.base_classifier.fit(x_source, y_source)
        # 伪target加入伪标签
        y_target0 = self.base_classifier.predict(x_target)
        ns = x_source.shape[-1]
        nt = x_target.shape[-1]
        for i in range(self.N):

            Z, A = self._get_JDA_matrix(x_source, x_target, y_source, y_target0)
            Z = np.matmul(Z, np.diag(1 / np.sqrt(np.sum(np.square(Z), 0))))
            self.Zs = Z[:, :ns]
            self.Zt = Z[:, ns:]

            self.base_classifier.fit(self.Zs.T, y_source)
            y_target0 = self.base_classifier.predict(self.Zt.T)
        return self.Zs.T, self.Zt.T

    def transform(self):
        pass

    def _kernel_func(self, x1, x2=None):
        n1, dim = x1.shape
        K = None
        if self.kernel == 'linear':
            if x2 is not None:
                K = np.dot(x2, x1.T)
            else:
                K = np.dot(x1, x1.T)
        elif self.kernel == 'rbf':
            if x2 is not None:
                n2 = np.shape(x2)[0]
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

    def _get_JDA_matrix(self, x_source, x_target, y_source, y_target0):
        X = np.vstack((x_source, x_target))
        X = np.matmul(X, np.diag(1 / np.sqrt(np.sum(np.square(X), 0))))

        m, n = np.shape(X)
        n_source = np.shape(x_source)[0]
        n_target = np.shape(x_target)[0]

        C = len(np.unique(y_source))

        a = 1 / n_source * np.ones([n_source, 1])
        b = -1 / n_target * np.ones([n_target, 1])

        e = np.hstack((a, b))
        M = np.matmul(e, e.T) * C
        N = 0
        if len(y_target0) != 0 and len(y_target0) == n_target:
            for c in np.reshape(np.unique(y_target0), 1):
                e = np.zeros([n, -1])
                idx = [i for i, y in enumerate(y_source) if y == c]
                e[idx] = 1 / len(idx)

                e[np.where(np.isinf(e))[0]] = 0
                N = N + np.matmul(e, e.T)
        M = M + N
        M = M / np.sqrt(np.sum(np.diag(np.matmul(M.T, M))))
        H = np.eye(n) - 1 / n * np.ones([n, n])

        if "primal" == self.kernel:
            a = np.matmul(np.matmul(X, M), X.T) + self.lmbda * np.eye(m)
            b = np.matmul(np.matmul(X, H), X.T)
            eigenvalue, eigenvector = sl.eig(a, b)
            av = np.array(list(map(lambda item: np.abs(item), eigenvalue)))
            idx = np.argsort(av)[:self]
            _ = eigenvalue[idx]

            A = eigenvector[:, idx]
            Z = np.matmul(A.T, X)

        else:
            K = self._kernel_func(X)
            a = np.matmul(np.matmul(K, M), K.T) + self.lmbda * np.eye(n)
            b = np.matmul(np.matmul(K, H), K.T)
            eigenvalue, eigenvector = sl.eig(a, b)
            av = np.array(list(map(lambda item: np.abs(item), eigenvalue)))
            idx = np.argsort(av)[:self]
            _ = eigenvalue[idx]

            A = eigenvector[:, idx]
            Z = np.matmul(A.T, K)

        return Z, A



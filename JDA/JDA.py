# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/3/17 20:40
@Function:
"""
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import scipy.linalg as sl


class JDA:

    def __init__(self, base_classifier=DecisionTreeClassifier(), N=10, k=200, lmbda=1.0, kernel='liner', gamma=1.0):
        self.base_classifier = base_classifier
        self.N = N
        self.k = k
        self.lmbda = lmbda
        self.kernel = kernel
        self.gamma = gamma
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

    def _kernel_func(self, x_train, x_other=None):
        if self.kernel == "linear":
            if x_other is None:
                return np.matmul(x_train.T, x_train)
            else:
                return np.matmul(x_train.T, x_other)

    def _get_JDA_matrix(self, x_source, x_target, y_source, y_target0):
        x_trian = np.hstack((x_source, x_target))
        x_trian = np.matmul(x_trian, np.diag(1 / np.sqrt(np.sum(np.square(x_trian), 0))))

        m, n = x_trian.shape
        ns = x_source.shape[-1]
        nt = x_target.shape[-1]

        C = len(np.unique(y_source))

        a = 1 / (ns * np.ones([ns, 1]))
        b = -1 / (nt * np.ones([nt, 1]))

        e = np.vstack((a, b))
        M = np.matmul(e, e.T) * C

        if len(y_target0) != 0 and len(y_target0) == nt:
            for c in np.unique(y_target0):
                e = np.zeros([n, -1])
                idx = [i for i, y in enumerate(y_source) if y == c]
                e[idx] = 1 / len(idx)

                e[np.where(np.isinf(e))[0]] = 0
                M = M + np.matmul(e, e.T)
        divider = np.sqrt(np.sum(np.diag(np.matmul(M.T, M))))
        M = M / divider

        a = np.eye(n)
        b = 1 / (n * np.ones([n, n]))
        H = a - b

        if "primal" == self.kernel:
            Z = None
            A = None

        else:
            K = self._kernel_func(self.kernel, x_trian)
            a = np.matmul(np.matmul(K, M), K.T) + self.lmbda * np.eye(n)
            b = np.matmul(np.matmul(K, H), K.T)
            eigenvalue, eigenvector = sl.eig(a, b)
            av = np.array(list(map(lambda item: np.abs(item), eigenvalue)))
            idx = np.argsort(av)[:self]
            _ = eigenvalue[idx]

            A = eigenvector[:, idx]
            Z = np.matmul(A.T, K)

        return Z, A



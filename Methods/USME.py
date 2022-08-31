import xlwt
import xlrd
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
from time import time
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv

class KELMOR(ClassifierMixin, BaseEstimator):

    def __init__(self, C=100, method="full", S=None, eps=1e-5, kernel="linear", gamma=0.1, degree=3, coef0=1, kernel_params=None):
        self.C = C
        self.kernel = kernel
        self.method = method
        self.S = S
        self.eps = eps
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X, self.y = X, y
        n, d = X.shape
        #  ---------------规范化类别标签：0,1,2,3,4,5-----------------
        self.le_ = preprocessing.LabelEncoder()
        self.le_.fit(y)
        y = self.le_.transform(y)
        #  --------------------------------------------------------
        classes = np.unique(y)
        nclasses = len(classes)

        self.M = np.array([[(i - j) ** 2 for i in range(nclasses)] for j in range(nclasses)])
        T = self.M[y, :]
        K = self._get_kernel(X)
        if self.method == "full":
            self.beta = np.linalg.inv((1 / self.C) * np.eye(n) + K).dot(T)
        else:
            raise ValueError("Invalid value for argument 'method'.")
        return self

    def predict(self, X):
        K = self._get_kernel(X, self.X)
        coded_preds = K.dot(self.beta)
        # print("coded_preds::",coded_preds.shape)
        predictions = np.argmin(np.linalg.norm(coded_preds[:, None] - self.M, axis=2, ord=1), axis=1)
        predictions = self.le_.inverse_transform(predictions)
        return predictions

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {'gamma': self.gamma,
                      'degree': self.degree,
                      'coef0': self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def predict_proba(self,X):
        K = self._get_kernel(X, self.X)
        coded_tmp = K.dot(self.beta)
        predictions = np.linalg.norm(coded_tmp[:, None] - self.M, axis=2, ord=2)
        predictions = -predictions
        predictions = np.exp(predictions)
        predictions_sum = np.sum(predictions, axis=1, keepdims=True)
        proba_matrix = predictions / predictions_sum
        return proba_matrix

class USME():
    def __init__(self, X, y, labeled, budget, X_test, y_test):
        self.X = X
        self.y = y
        self.nSample, self.nDim = X.shape
        self.labels = sorted(np.unique(self.y))
        self.nClass = len(self.labels)
        self.X_test = X_test
        self.y_test = y_test
        self.labeled = list(deepcopy(labeled))
        self.model = KELMOR(C=100, kernel='rbf', gamma=0.1)
        self.n_theta = [i for i in range(self.nClass - 1)]
        self.theta = None
        self.unlabeled = self.initialization()
        self.budget = deepcopy(budget)
        self.budgetLeft = deepcopy(budget)
        self.pdist = pdist(self.X, metric="euclidean")
        self.dist_matrix = squareform(self.pdist)

        '''Evaluation criteria'''
        self.AccList = []
        self.MAEList = []
        self.F1List = []
        self.CostList = []
        self.ALC_Acc = 0.0
        self.ALC_MAE = 0.0
        self.ALC_F1 = 0.0
        self.ALC_Cost = 0.0

    def initialization(self):
        unlabeled = [i for i in range(self.nSample)]
        for idx in self.labeled:
            unlabeled.remove(idx)
        self.model.fit(self.X[self.labeled], self.y[self.labeled])
        return unlabeled

    def evaluation(self):
        self.model.fit(self.X[self.labeled], self.y[self.labeled])


    def select(self):
        while self.budgetLeft > 0:
            prob_matrix = self.model.predict_proba(self.X[self.unlabeled])
            metric = OrderedDict()
            for i, idx in enumerate(self.unlabeled):
                metric[idx] = 0.0
                for prob in prob_matrix[i]:
                    if prob > 0.0:
                        metric[idx] -= prob * np.log2(prob)
            tar_idx = max(metric, key=metric.get)
            self.unlabeled.remove(tar_idx)
            self.labeled.append(tar_idx)
            self.budgetLeft -= 1
            self.evaluation()

if __name__ == '__main__':
    names_list= ["Newthyroid"]
    for name in names_list:
        print("########################{}".format(name))
        data_path = Path(r"D:\OCdata")
        partition_path = Path(r"E:\AOCOI\Partitions")
        """--------------read the whole data--------------------"""
        read_data_path = data_path.joinpath(name + ".csv")
        data = np.array(pd.read_csv(read_data_path, header=None))
        X = np.asarray(data[:, :-1], np.float64)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = data[:, -1]
        y -= y.min()
        nClass = len(np.unique(y))
        Budget = 20 * nClass

        """--------read the partitions--------"""
        read_partition_path = str(partition_path.joinpath(name + ".xls"))
        book_partition = xlrd.open_workbook(read_partition_path)

        """-----read the kmeans results according to the partition-----"""

        workbook = xlwt.Workbook()
        count = 0
        for SN in book_partition.sheet_names():
            S_Time = time()
            train_idx = []
            test_idx = []
            labeled = []
            table_partition = book_partition.sheet_by_name(SN)
            for idx in table_partition.col_values(0):
                if isinstance(idx,float):
                    train_idx.append(int(idx))
            for idx in table_partition.col_values(1):
                if isinstance(idx,float):
                    test_idx.append(int(idx))
            for idx in table_partition.col_values(2):
                if isinstance(idx,float):
                    labeled.append(int(idx))

            X_train = X[train_idx]
            y_train = y[train_idx].astype(np.int32)
            X_test = X[test_idx]
            y_test = y[test_idx]

            if Budget > len(X_train)-nClass:
                Budget = len(X_train)-nClass

            model = USME(X=X_train, y=y_train, labeled=labeled, budget=Budget, X_test=X_test, y_test=y_test)
            model.select()
            # SheetNames = "{}".format(count)
            sheet = workbook.add_sheet(SN)
            for i, idx in enumerate(train_idx):
                sheet.write(i, 0,  int(idx))
            for i, idx in enumerate(test_idx):
                sheet.write(i, 1, int(idx))
            for i, idx in enumerate(labeled):
                sheet.write(i, 2, int(idx))
            for i, idx in enumerate(model.labeled):
                sheet.write(i, 3, int(idx))

        save_path = Path(r"E:\AOCOI\SelectedResult\USME")
        save_path = str(save_path.joinpath(name + ".xls"))
        workbook.save(save_path)


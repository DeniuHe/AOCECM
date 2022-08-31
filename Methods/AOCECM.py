import xlwt
import xlrd
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y


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


class AOCECM():
    def __init__(self, X, y, labeled, budget, X_test, y_test, cluster_label,cluster_center):
        self.X = X
        self.y = y
        self.nSample, self.nDim = X.shape
        self.labels = sorted(np.unique(self.y))
        self.nClass = len(self.labels)
        self.X_test = X_test
        self.y_test = y_test
        self.labeled = list(deepcopy(labeled))
        self.gamma = 0.1
        self.model = KELMOR(C=100, kernel='rbf', gamma=0.1)
        self.n_theta = [i for i in range(self.nClass - 1)]
        self.cost_matrix = self.cal_cost_matrix()
        self.theta = None
        self.unlabeled = self.initialization()
        self.budget = deepcopy(budget)
        self.budgetLeft = deepcopy(budget)
        # self.pdist = pdist(self.X, metric="euclidean")
        # self.dist_matrix = squareform(self.pdist)
        self.cluster_label = cluster_label
        self.cluster_center = cluster_center

    def cal_cost_matrix(self):
        cost_matrix = np.zeros((len(self.labels),len(self.labels)))
        for i, y1 in enumerate(self.labels):
            for j, y2 in enumerate(self.labels):
                cost_matrix[i,j] = abs(y1-y2)
        return cost_matrix

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
            candidate = []
            num_labeled = len(self.labeled)
            labeled_set = set(self.labeled)
            for lab in range(num_labeled+1):
                # print("lab::",lab)
                lab_cluster_set = set(np.where(np.asarray(self.cluster_label[num_labeled+1])==lab)[0])
                flag_set = labeled_set & lab_cluster_set
                if not flag_set:
                    candidate.append(self.cluster_center[num_labeled+1][lab])

            if len(candidate) == 1:
                tar_idx = candidate[0]
                self.unlabeled.remove(tar_idx)
                self.labeled.append(tar_idx)
                self.budgetLeft -= 1
                self.evaluation()
            elif len(candidate) > 1:
                prob_matrix = self.model.predict_proba(self.X[candidate])
                MS = OrderedDict()
                for i, idx in enumerate(candidate):
                    ordjdx = np.argsort(prob_matrix[i])
                    MS[idx] = prob_matrix[i][ordjdx[-1]] - prob_matrix[i][ordjdx[-2]]

                EC = OrderedDict()
                for i, idx in enumerate(candidate):
                    # print("idx的概率估计::",prob_matrix[i])
                    tmp_model = KELMOR(C=100, kernel='rbf', gamma=0.1)
                    tmp_labeled = deepcopy(self.labeled)
                    tmp_labeled.append(idx)
                    tmp_X = self.X[tmp_labeled]
                    tmp_y = self.y[tmp_labeled]

                    Expected_Cost=0.0
                    for l, lab in enumerate(self.labels):
                        tmp_y[-1] = lab
                        tmp_model.fit(X=tmp_X, y=tmp_y)
                        tmp_prob_matrix = tmp_model.predict_proba(X=self.X[self.unlabeled])
                        tmp_prob_max = np.argmax(tmp_prob_matrix,axis=1)
                        tmp_cost_matrix = np.zeros_like(tmp_prob_matrix)
                        for t, tm in enumerate(tmp_prob_max):
                            tmp_cost_matrix[t] = self.cost_matrix[tm]
                        tmp_cost_total = tmp_cost_matrix * tmp_prob_matrix
                        tmp_cost_mean = np.mean(np.sum(tmp_cost_total,axis=1))
                        Expected_Cost += tmp_cost_mean * prob_matrix[i][l]
                    EC[idx] = Expected_Cost

                metric = OrderedDict()

                alpha = 0.9
                for idx in candidate:
                    metric[idx] = alpha * EC[idx] + (1-alpha) * MS[idx]

                tar_idx = min(metric, key=metric.get)
                self.unlabeled.remove(tar_idx)
                self.labeled.append(tar_idx)
                self.budgetLeft -= 1
                self.evaluation()


if __name__ == '__main__':

    names_list= ["newthyroid"]
    for name in names_list:
        print("########################{}".format(name))

        data_path = Path(r"D:\OCdata")

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
        partition_path = Path(r"E:\AOCOI\Partitions")
        read_partition_path = str(partition_path.joinpath(name + ".xls"))
        book_partition = xlrd.open_workbook(read_partition_path)

        """-----read the kmeans_label -----"""
        kmeans_path = Path(r"E:\AOCOI\KmeansResult")
        read_kmeans_label_path = str(kmeans_path.joinpath(name + "-label.xls"))
        book_kmeans_label = xlrd.open_workbook(read_kmeans_label_path)

        kmeans_path = Path(r"E:\AOCOI\KmeansResult")
        read_kmeans_center_path = str(kmeans_path.joinpath(name + "-center.xls"))
        book_kmeans_center = xlrd.open_workbook(read_kmeans_center_path)

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

            # ----------------Get Clustering Result-------------------------
            # table_cluster = book_kmeans.sheet_by_name(SN)

            table_cluster_label = book_kmeans_label.sheet_by_name(SN)
            table_cluster_center = book_kmeans_center.sheet_by_name(SN)

            cluster_label = OrderedDict()
            cluster_center = OrderedDict()
            k_list = list(range(nClass+1, 21*nClass+2))
            for k in k_list:
                cluster_label[k] = []
                cluster_center[k] = []
            for c, k in enumerate(k_list):
                for label in table_cluster_label.col_values(c):
                    if isinstance(label,float):
                        cluster_label[k].append(int(label))
                for idx in table_cluster_center.col_values(c):
                    if isinstance(idx,float):
                        cluster_center[k].append(int(idx))

            model = AOCECM(X=X_train, y=y_train, labeled=labeled, budget=Budget, X_test=X_test, y_test=y_test,cluster_label=cluster_label,cluster_center=cluster_center)
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
        save_path = Path(r"E:\AOCOI\SelectedResult\AOCECM")
        save_path = str(save_path.joinpath(name + ".xls"))
        workbook.save(save_path)





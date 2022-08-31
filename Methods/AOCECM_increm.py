'''
AOCECM based on Incremental Update KELMOR
The KELMOR model only trained once at the initial moment.
In the subsequent iterations, the KELMOR model is incrementally updated.
'''

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
from numpy.linalg import inv




class AOCECMIn():
    def __init__(self, X, y, labeled, budget, X_test, y_test, cluster_label,cluster_center):
        self.X = X
        self.y = y
        self.nSample, self.nDim = X.shape
        self.labels = sorted(np.unique(self.y))
        self.nClass = len(self.labels)
        self.M = np.array([[(i - j) ** 2 for i in range(self.nClass)] for j in range(self.nClass)])
        self.X_test = X_test
        self.y_test = y_test
        self.labeled = list(deepcopy(labeled))
        self.gamma = 0.1
        self.degree = 3
        self.coef0 = 1

        self.n_theta = [i for i in range(self.nClass - 1)]
        self.cost_matrix = self.cal_cost_matrix()
        self.theta = None
        params = {'gamma': self.gamma,
                  'degree': self.degree,
                  'coef0': self.coef0}
        self.K = pairwise_kernels(self.X, metric="rbf",filter_params=True, **params)
        self.unlabeled = self.initialization()
        self.budget = deepcopy(budget)
        self.budgetLeft = deepcopy(budget)
        self.cluster_label = cluster_label
        self.cluster_center = cluster_center

# -------------------Tool function------------------------
    def cal_cost_matrix(self):
        cost_matrix = np.zeros((len(self.labels),len(self.labels)))
        for i, y1 in enumerate(self.labels):
            for j, y2 in enumerate(self.labels):
                cost_matrix[i,j] = abs(y1-y2)
        return cost_matrix

    def Get_Inverse(self,n, Ai, B, C, D):
        M = np.zeros((n+1,n+1))
        DCAB = inv(D-C@Ai@B)
        M[:n,:n] = Ai+ Ai@B@DCAB@C@Ai
        M[:n,n] = -Ai@B@DCAB.reshape(1)
        M[n,:n] = -DCAB@C@Ai
        M[n,n] = DCAB
        return M
#---------------------------------------------------------

    def initialization(self):
        unlabeled = [i for i in range(self.nSample)]
        for idx in self.labeled:
            unlabeled.remove(idx)
        self.model_initial_train()
        return unlabeled

    def model_initial_train(self):
        n = len(self.labeled)
        self.T_lab = self.M[self.y[self.labeled],:]
        self.K_lab = self.K[np.ix_(self.labeled,self.labeled)]
        self.K_lab_inv = np.linalg.inv(0.01 * np.eye(n) + self.K_lab)
        self.beta = self.K_lab_inv.dot(self.T_lab)


    def model_incremental_train(self, new_idx):
        # 一定要先运行该函数然后再执行 新样本的添加
        num_labeled = len(self.labeled)
        Ai = self.K_lab_inv
        B = self.K[np.ix_(self.labeled,[new_idx])]
        C = B.T
        D = self.K[new_idx,new_idx] + 0.01   # 1/C=0.01
        K_bar_inverse = self.Get_Inverse(n=num_labeled,Ai=Ai, B=B, C=C, D=D)
        T_bar = np.vstack((self.T_lab,self.M[self.y[new_idx]]))
        beta_bar = K_bar_inverse @ T_bar
        # ---------------------------
        self.K_lab_inv = K_bar_inverse
        self.T_lab = T_bar
        self.beta = beta_bar

    def tmp_incremental_train_predict(self, tmp_idx, tmp_label):
        num_labeled = len(self.labeled)
        Ai = self.K_lab_inv
        B = self.K[np.ix_(self.labeled,[tmp_idx])]
        C = B.T
        D = self.K[tmp_idx,tmp_idx] + 0.01   # C=100, 1/C=0.01
        K_bar_inverse = self.Get_Inverse(n=num_labeled,Ai=Ai, B=B, C=C, D=D)
        T_bar = np.vstack((self.T_lab,self.M[self.y[tmp_label]]))
        beta_bar = K_bar_inverse @ T_bar
        # --------------obtain tmp probability---------------------
        params = {'gamma': self.gamma,
                  'degree': self.degree,
                  'coef0': self.coef0}
        tmp_labeled = deepcopy(self.labeled)
        tmp_labeled.append(tmp_idx)
        K = pairwise_kernels(self.X[self.unlabeled],self.X[tmp_labeled], metric="rbf",filter_params=True, **params)
        coded_tmp = K.dot(beta_bar)
        predictions = np.linalg.norm(coded_tmp[:, None] - self.M, axis=2, ord=2)
        predictions = -predictions
        predictions = np.exp(predictions)
        predictions_sum = np.sum(predictions, axis=1, keepdims=True)
        proba_matrix = predictions / predictions_sum
        return proba_matrix


    def predict_proba(self,X):
        params = {'gamma': self.gamma,
                  'degree': self.degree,
                  'coef0': self.coef0}
        K = pairwise_kernels(X,self.X[self.labeled], metric="rbf",filter_params=True, **params)
        coded_tmp = K.dot(self.beta)
        predictions = np.linalg.norm(coded_tmp[:, None] - self.M, axis=2, ord=2)
        predictions = -predictions
        predictions = np.exp(predictions)
        predictions_sum = np.sum(predictions, axis=1, keepdims=True)
        proba_matrix = predictions / predictions_sum
        return proba_matrix



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
                self.model_incremental_train(new_idx=tar_idx)
                self.unlabeled.remove(tar_idx)
                self.labeled.append(tar_idx)
                self.budgetLeft -= 1

            elif len(candidate) > 1:
                prob_matrix = self.predict_proba(self.X[candidate])

                MS = OrderedDict()
                for i, idx in enumerate(candidate):
                    ordjdx = np.argsort(prob_matrix[i])
                    MS[idx] = prob_matrix[i][ordjdx[-1]] - prob_matrix[i][ordjdx[-2]]

                EC = OrderedDict()
                for i, idx in enumerate(candidate):
                    Expected_Cost=0.0
                    for l, lab in enumerate(self.labels):
                        tmp_prob_matrix = self.tmp_incremental_train_predict(tmp_idx=idx,tmp_label=lab)
                        tmp_prob_max = np.argmax(tmp_prob_matrix,axis=1)
                        tmp_cost_matrix = np.zeros_like(tmp_prob_matrix)
                        for t, tm in enumerate(tmp_prob_max):
                            tmp_cost_matrix[t] = self.cost_matrix[tm]
                        tmp_cost_total = tmp_cost_matrix * tmp_prob_matrix
                        tmp_cost_mean = np.mean(np.sum(tmp_cost_total,axis=1))
                        Expected_Cost += tmp_cost_mean * prob_matrix[i][l]
                    EC[idx] = Expected_Cost

                metric = OrderedDict()

                alpha = 0.9  # alpha is is the trade-off parameter lambda
                for idx in candidate:
                    metric[idx] = alpha * EC[idx] + (1-alpha) * MS[idx]

                tar_idx = min(metric, key=metric.get)
                self.model_incremental_train(new_idx=tar_idx)
                self.unlabeled.remove(tar_idx)
                self.labeled.append(tar_idx)
                self.budgetLeft -= 1



if __name__ == '__main__':

    names_list= ["Newthyroid"]
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


            model = AOCECMIn(X=X_train, y=y_train, labeled=labeled, budget=Budget, X_test=X_test, y_test=y_test,cluster_label=cluster_label,cluster_center=cluster_center)
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





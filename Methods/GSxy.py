import xlwt
import xlrd
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import pairwise_distances



class iGSxy():
    def __init__(self, X, y, labeled, budget, X_test, y_test):
        self.X = X
        self.y = y
        self.nSample, self.nDim = X.shape
        self.labels = sorted(np.unique(self.y))
        self.nClass = len(self.labels)
        self.X_test = X_test
        self.y_test = y_test
        self.labeled = list(deepcopy(labeled))

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
        return unlabeled



    def select(self):
        while self.budgetLeft > 0:
            # -----------------------GSx_dist
            UL_x_dist = pairwise_distances(self.X[self.unlabeled],self.X[self.labeled])
            # -----------------------GSy_dist
            lr_model = LinearRegression()
            lr_model.fit(self.X[self.labeled],self.y[self.labeled])
            unlabeled_hat = lr_model.predict(self.X[self.unlabeled])
            unlabeled_hat = unlabeled_hat.reshape(-1,1)
            labeled_y = self.y[self.labeled]
            labeled_y = labeled_y.reshape(-1,1)
            UL_y_dist = pairwise_distances(unlabeled_hat,labeled_y)
            # ----------------------GSxy
            UL_xy_dist = UL_x_dist * UL_y_dist
            UL_xy_dist_min = np.min(UL_xy_dist,axis=1)
            max_idx = np.argmax(UL_xy_dist_min)
            # --------------------- max-min-idx
            tar_idx = self.unlabeled[max_idx]
            self.unlabeled.remove(tar_idx)
            self.labeled.append(tar_idx)
            self.budgetLeft -= 1



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

            model = iGSxy(X=X_train, y=y_train, labeled=labeled, budget=Budget, X_test=X_test, y_test=y_test)
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
        save_path = Path(r"E:\AOCOI\SelectedResult\iGS")
        save_path = str(save_path.joinpath(name + ".xls"))
        workbook.save(save_path)


import pandas as pd
import numpy as np
import xlrd
import xlwt
from time import time
from collections import OrderedDict
from sklearn.svm import SVC
from scipy.special import expit
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score, mutual_info_score
from sklearn.metrics.pairwise import rbf_kernel


class REDSVM():
    def __init__(self):
        self.gamma = 0.1
        self.C = 100
        self.eX = self.ey = None

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.int32)
        self.nSample, self.nDim = X.shape
        self.labels = list(np.sort(np.unique(y)))
        self.nClass = len(self.labels)
        self.nTheta = self.nClass - 1
        self.extend_part = np.eye(self.nClass-1)
        self.label_dict = self.get_label_dict()
        self.eX, self.ey = self.train_set_construct(X=self.X, y=self.y)
        self.gram_train = self.get_gram_train()
        self.model = SVC(kernel='precomputed', C=100, probability=True)
        self.model.fit(self.gram_train, y=self.ey)
        return self

    def get_gram_train(self):
        gram_train_1 = rbf_kernel(X=self.eX[:,:self.nDim],gamma=0.1)
        gram_train_2 = self.eX[:,self.nDim:] @ self.eX[:,self.nDim:].T
        gram_train = gram_train_1 + gram_train_2
        return gram_train

    def get_label_dict(self):
        label_dict = OrderedDict()
        for i, lab in enumerate(self.labels):
            tmp_label = np.ones(self.nTheta)
            for k, pad in enumerate(self.labels[:-1]):
                if lab <= pad:
                    tmp_label[k] = 1
                else:
                    tmp_label[k] = -1
            label_dict[lab] = tmp_label
        return label_dict

    def train_set_construct(self, X, y):
        eX = np.zeros((self.nSample * self.nTheta, self.nDim + self.nTheta))
        ey = np.zeros(self.nSample * self.nTheta)
        for i in range(self.nSample):
            eXi = np.hstack((np.tile(X[i], (self.nTheta, 1)), self.extend_part))
            eX[self.nTheta * i: self.nTheta * i + self.nTheta] = eXi
            ey[self.nTheta * i: self.nTheta * i + self.nTheta] = self.label_dict[y[i]]
        return eX, ey

    def test_set_construct(self, X_test):
        nTest = X_test.shape[0]
        eX = np.zeros((nTest * self.nTheta, self.nDim + self.nTheta))
        for i in range(nTest):
            eXi = np.hstack((np.tile(X_test[i],(self.nTheta,1)), self.extend_part))
            eX[self.nTheta * i: self.nTheta * i + self.nTheta] = eXi
        return eX

    def get_gram_test(self, eX_test):
        gram_test_1 = rbf_kernel(X=eX_test[:,:self.nDim], Y=self.eX[:,:self.nDim], gamma=0.1)
        gram_test_2 = eX_test[:,self.nDim:] @ self.eX[:,self.nDim:].T
        gram_test = gram_test_1 + gram_test_2
        return gram_test

    def predict(self, X_test):
        nTest = X_test.shape[0]
        eX_test = self.test_set_construct(X_test=X_test)
        gram_test = self.get_gram_test(eX_test)
        y_extend = self.model.predict(gram_test)
        y_tmp = y_extend.reshape(nTest,self.nTheta)
        y_pred = np.sum(y_tmp < 0, axis=1).astype(np.int32)
        return y_pred

    def predict_proba(self, X_test):
        nTest = X_test.shape[0]
        eX_test = self.test_set_construct(X_test=X_test)
        gram_test = self.get_gram_test(eX_test)
        dist_tmp = self.model.decision_function(gram_test)
        dist_matrix = dist_tmp.reshape(nTest, self.nTheta)
        accumulative_proba = expit(dist_matrix)
        prob = np.pad(
            accumulative_proba,
            pad_width=((0, 0), (1, 1)),
            mode='constant',
            constant_values=(0, 1))
        prob = np.diff(prob)
        return prob

    def distant_to_theta(self, X_test):
        nTest = X_test.shape[0]
        eX_test = self.test_set_construct(X_test=X_test)
        gram_test = self.get_gram_test(eX_test)
        dist_tmp = self.model.decision_function(gram_test)
        dist_matrix = dist_tmp.reshape(nTest, self.nTheta)
        return dist_matrix





# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class results():
    def __init__(self):
        self.MZEList = []
        self.MAEList = []
        self.F1List = []
        self.MIList = []
        self.ALC_MZE = []
        self.ALC_MAE = []
        self.ALC_F1 = []
        self.ALC_MI = []

class stores():
    def __init__(self):
        self.MZEList_mean = []
        self.MZEList_std = []
        # -----------------
        self.MAEList_mean = []
        self.MAEList_std = []
        # -----------------
        self.F1List_mean = []
        self.F1List_std = []
        # -----------------
        self.MIList_mean = []
        self.MIList_std = []
        # -----------------
        self.ALC_MZE_mean = []
        self.ALC_MZE_std = []
        # -----------------
        self.ALC_MAE_mean = []
        self.ALC_MAE_std = []
        # -----------------
        self.ALC_F1_mean = []
        self.ALC_F1_std = []
        # -----------------
        self.ALC_MI_mean = []
        self.ALC_MI_std = []
        # -----------------
        self.ALC_MZE_list = []
        self.ALC_MAE_list = []
        self.ALC_F1_list = []
        self.ALC_MI_list = []

# --------------------------------------

def get_train_test_init_selected_ids(name,method,result_path,data_path, save_path):
    read_data_path = data_path.joinpath(name + ".csv")
    data = np.array(pd.read_csv(read_data_path, header=None))
    X = np.asarray(data[:, :-1], np.float64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = data[:, -1]
    y -= y.min()
    read_path = str(result_path.joinpath(name + ".xls"))
    book = xlrd.open_workbook(read_path)
    workbook = xlwt.Workbook()

    for SN in book.sheet_names():
        """Store"""
        sheet = workbook.add_sheet(SN)

        """Read"""
        S_time = time()   # -------record the time consumption---------
        table = book.sheet_by_name(SN)
        train_idx = []
        test_idx = []
        init_idx = []
        selected_idx = []
        for idx in table.col_values(0):
            if isinstance(idx,float):
                train_idx.append(int(idx))
        for idx in table.col_values(1):
            if isinstance(idx,float):
                test_idx.append(int(idx))
            else:
                break
        for idx in table.col_values(2):
            if isinstance(idx,float):
                init_idx.append(int(idx))
            else:
                break
        for idx in table.col_values(3):
            if isinstance(idx,float):
                selected_idx.append(int(idx))
            else:
                break
        # ---------------------------------
        X_pool = X[train_idx]
        y_pool = y[train_idx].astype(int)
        X_test = X[test_idx]
        y_test = y[test_idx].astype(int)
        # ---------------------------------
        MZE_list = []
        MAE_list = []
        F1_list = []
        MI_list = []

        # ----------------------------------------------------------- #
        # ---------------------Run Classifier------------------------ #
        # ----------------------------------------------------------- #
        init_len = len(init_idx)
        whole_len = len(selected_idx)
        # print(init_len)
        # print(whole_len)
        for i in range(init_len,whole_len+1):
            current_ids = selected_idx[:i]
            # --------------------------------------
            model = REDSVM()
            model.fit(X=X_pool[current_ids], y=y_pool[current_ids])
            y_hat = model.predict(X_test=X_test)
            # --------------------------------------
            # --------------Metrics-----------------
            MZE = 1 - accuracy_score(y_hat, y_test)
            MAE = mean_absolute_error(y_hat, y_test)
            F1 = f1_score(y_pred=y_hat, y_true=y_test,average="macro")
            MI = mutual_info_score(labels_true=y_test, labels_pred=y_hat)

            # -------------------------------------
            MZE_list.append(MZE)
            MAE_list.append(MAE)
            F1_list.append(F1)
            MI_list.append(MI)

        for j in range(len(MZE_list)):
            sheet.write(j,0,MZE_list[j])
            sheet.write(j,1,MAE_list[j])
            sheet.write(j,2,MI_list[j])
            sheet.write(j,3,F1_list[j])

    save_path = str(save_path.joinpath(method))
    save_path = Path(save_path)
    save_path = str(save_path.joinpath(name + ".xls"))
    workbook.save(save_path)




if __name__ == '__main__':

    methods_list = ["USME","USLC","USMS","Random","MCSVMA","McPAL","iGSx","FISTA","ALOR","ALCE","LogitA",
                    "AOCECM"]
    names_list = ["Newthyroid","Balance-scale","Thyroid","Knowledge","Machine","Housing","Computer","Obesity","Stock"]


    data_path = Path(r"D:\OCdata")
    for method in methods_list:
        print("=================================={}".format(method))
        result_path = Path(r"E:\AOCOI\SelectedResult")
        save_path = Path(r"E:\AOCOI\ALresult_REDSVM")
        result_path = str(result_path.joinpath(method))
        result_path = Path(result_path)
        for name in names_list:
            print("@@@@@@@@@@@@@@@@@@@@@@{}".format(name))
            get_train_test_init_selected_ids(name,method,result_path,data_path,save_path)




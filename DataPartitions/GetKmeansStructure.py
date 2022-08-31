import xlwt
import xlrd
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler





data_path = Path("D:\OCdata")
part_path = Path("E:\AOCOI\Partitions")
names_list = ["Newthyroid","Balance-scale","Thyroid","Knowledge","Machine","Housing","Computer","Obesity","Stock"]

for name in names_list:
    path = data_path.joinpath(name + ".csv")
    print("########################{}".format(path))
    data = np.array(pd.read_csv(path, header=None))
    scaler = StandardScaler()
    X = scaler.fit_transform(np.asarray(data[:, :-1], np.float64))
    y = data[:, -1]
    y -= y.min()
    nClass = len(np.unique(y))
    workbook_1 = xlwt.Workbook()
    workbook_2 = xlwt.Workbook()
    # -----------------------------
    read_path = str(part_path.joinpath(name + ".xls"))
    book = xlrd.open_workbook(read_path)
    for SN in book.sheet_names():

        table = book.sheet_by_name(SN)
        pool_idx = []
        for idx in table.col_values(0):
            if isinstance(idx,float):
                pool_idx.append(int(idx))

        pool_idx = np.array(pool_idx)

        SheetNames = "{}".format(SN)
        sheet_1 = workbook_1.add_sheet(SheetNames)
        sheet_2 = workbook_2.add_sheet(SheetNames)
        column = 0
        for n_clusters in range(nClass+1, 21*nClass+2):
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(X=X[pool_idx])
            # --------------store the clustering labels-----------------
            for j, jdx in enumerate(kmeans.labels_):
                sheet_1.write(j,column,int(jdx))

            # -----------------------------------------------------------
            center = np.zeros(n_clusters)
            for lab in range(n_clusters):
                tmp_center = kmeans.cluster_centers_[lab]
                lab_ids = np.where(kmeans.labels_==lab)[0]
                min_dist = np.inf
                for idx in lab_ids:
                    dist = np.linalg.norm(X[idx] - tmp_center)
                    if dist <= min_dist:
                        min_dist = dist
                        center[lab] = idx
            # ---------------Store the clustering centers idx----------------
            for j, jdx in enumerate(center):
                sheet_2.write(j,column,int(jdx))
            column += 1
            # ---------------------------------------------------------------

    save_path = Path(r"E:\AOCOI\KmeansResult")
    save_path = str(save_path.joinpath(name + "-label.xls"))
    workbook_1.save(save_path)

    save_path = Path(r"E:\AOCOI\KmeansResult")
    save_path = str(save_path.joinpath(name + "-center.xls"))
    workbook_2.save(save_path)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from shir2_functions import *
import pandas as pd
import numpy as np


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    "crop"
    dataset = pd.read_csv('crop.csv')
    # dataset = dataset.sample(frac=1)
    "balanced classes"
    g = dataset.groupby('label', group_keys=False)
    balanced_df = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()))).reset_index(drop=True)
    dataset = balanced_df.groupby('label').apply(pd.DataFrame.sample, frac=0.6).reset_index(level='label', drop=True)
    X = dataset.iloc[:, 1:]
    y = dataset.iloc[:, 0:1]
    y = np.asarray(y)

    radar_first = X.iloc[:, 0:49]
    radar_second = X.iloc[:, 49:98]
    optic_first = X.iloc[:, 98:136]
    optic_second = X.iloc[:, 136:]
    all = np.concatenate((radar_first, optic_first, y), axis=1)
    y = all[:, -1]
    y = np.asarray(y)
    labels = np.unique(y)
    X = all[:, :-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train = np.concatenate((X_train, y_train[:, None]), axis=1)
    test = np.concatenate((X_test, y_test[:, None]), axis=1)
    ind_selected, jm, coor_jm, jm_mean=eliminate_features_kmedoids(labels, train)


    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    train1 = X_train[:, ind_selected]
    test1 = X_test[:, ind_selected]
    train1 = np.concatenate((train1,y_train[:, None]), axis=1)
    test1 = np.concatenate((test1, y_test[:, None]), axis=1)
    acc_svm = float(pred_svm(train1, test1))
    acc_knn = float(knn_pred(train1, test1))
    acc_rand_forest = float(pred_randomforest(train1, test1))
    print("eliminate features:", "num_of_fetures", np.shape(train1)[1] - 1, "\n", "svm", acc_svm, "\n",
          "knn", acc_knn, "\n", "random_forest",
          acc_rand_forest)

    inx_mmrm=mmrm(pd.DataFrame(X_train), pd.DataFrame(y_train), 3)
    print(inx_mmrm)
    train1 = X_train[:, inx_mmrm]
    test1 = X_test[:, inx_mmrm]
    train1 = np.concatenate((train1, y_train[:, None]), axis=1)
    test1 = np.concatenate((test1, y_test[:, None]), axis=1)
    acc_svm = float(pred_svm(train1, test1))
    acc_knn = float(knn_pred(train1, test1))
    acc_rand_forest = float(pred_randomforest(train1, test1))
    print("eliminate features:", "num_of_fetures", np.shape(train1)[1] - 1, "\n", "svm", acc_svm, "\n",
          "knn", acc_knn, "\n", "random_forest",
          acc_rand_forest)
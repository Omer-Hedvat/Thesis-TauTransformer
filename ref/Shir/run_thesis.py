import utils as ob
import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy.random as nr
import numpy as np

warnings.filterwarnings('ignore')

"Obesity"
dataset = pd.read_csv('ref/Shir/Data/Obesity.csv')
X = dataset.iloc[:, 0:dataset.shape[1]-1]
y = dataset.iloc[:, -1]
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
y = np.reshape(y, (np.shape(y)[0],))
num_cols = dataset.select_dtypes(np.number)
n_cols = num_cols.columns
cat_cols = [x for x in X.columns if x not in n_cols]
for col in cat_cols:
    X[col] = labelencoder.fit_transform(X[col])

###########################################
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)
X=np.asarray(X)
all_data=np.concatenate((X, y.reshape((np.shape(y)[0], 1))), axis=1)


"crop"
# dataset = pd.read_csv('crop.csv')
# dataset = dataset.sample(frac=1)
# "balanced classes"
# g = dataset.groupby('label', group_keys=False)
#
# balanced_df = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()))).reset_index(drop=True)
# "sample 10% from dataset(after balance"
# # test= dataset[~dataset.isin(balanced_df)]
# # test=test.groupby('label').apply(pd.DataFrame.sample, frac=0.001).reset_index(level='label',drop=True)
# dataset = balanced_df.groupby('label').apply(pd.DataFrame.sample, frac=0.6).reset_index(level='label', drop=True)
# X = dataset.iloc[:, 1:]
# y = dataset.iloc[:, 0:1]
# # y=np.asarray(y)
# ###########################################
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)
# X = pd.DataFrame(X)
# ##########################################
# radar_first = X.iloc[:, 0:49]
# radar_second = X.iloc[:, 49:98]
# optic_first = X.iloc[:, 98:136]
# optic_second = X.iloc[:, 136:]
#
# all_data=np.concatenate((radar_first,optic_first,radar_second,optic_second,y),axis=1)

"isolet"
# # #
# dataset = pd.read_csv('ref/Shir/Data/isolet.csv')
# X = dataset.iloc[:, 0:dataset.shape[1] - 1]
# y = dataset.iloc[:, -1]
# y = np.reshape(y, (np.shape(y)[0],))
# y = np.asarray(y)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X = pd.DataFrame(X)
# X = np.asarray(X)
# all_data = np.concatenate((X, y.reshape((np.shape(y)[0], 1))), axis=1)



y = all_data[:, -1]
y = np.asarray(y)

labels = np.unique(y)
idx = np.random.choice(range(np.shape(all_data)[0]), size=int(0.7 * np.shape(all_data)[0]), replace=False)
valid_idx = np.random.choice(idx, size=int(0.3 * len(idx)), replace=False)
validation = all_data[valid_idx]
train_idx = [x for x in idx if x not in set(valid_idx)]
train = all_data[train_idx]
test_idx = [x for x in range(0, np.shape(all_data)[0]) if x not in set(idx)]
test = all_data[test_idx]
train = np.asarray(train)
best_comb = ob.hyper_parms_tuning(train, validation)
train = np.concatenate((train, validation))

option_for_jm = 0
ind_selected, jm, coor_jm, jm_mean = ob.eliminate_features(labels, train, option_for_jm, *best_comb)

print("eliminate features", ind_selected)

y_train = train[:, np.shape(train)[1] - 1:]
train1 = train[:, ind_selected]
y_test = test[:, np.shape(test)[1] - 1:]
test1 = test[:, ind_selected]
n = np.shape(train1)[1]
train1 = np.concatenate((train1, y_train), axis=1)
test1 = np.concatenate((test1, y_test), axis=1)
acc_svm = float(ob.pred_svm(train1, test1))
acc_knn = float(ob.knn_pred(train1, test1))
acc_rand_forest = float(ob.pred_randomforest(train1, test1))
print("eliminate features:", "num_of_fetures", np.shape(train1)[1] - 1, "\n", "svm", acc_svm, "\n", "knn", acc_knn, "\n", "random_forest",
      acc_rand_forest)

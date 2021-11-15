
import scipy
from scipy.spatial import distance
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC



def calc_p(dataset, option=0, rand_=False):
    dataset = np.asarray(dataset)
    X = range(np.shape(dataset)[0])
    dist = np.zeros((np.shape(dataset)[0], np.shape(dataset)[0]))
    if not rand_:
        for x in X:
            for y in X:
                if option == 0:
                    _x = np.array(dataset[x])
                    _y = np.array(dataset[y])

                    dist[x][y] = distance.euclidean(_x, _y)

                else:
                    _x = dataset[x, 0]
                    _y = dataset[y, 0]

                    dist[x][y] = 1 - (np.trace(np.matmul(_x, _y)) / (
                                np.sqrt(np.trace(np.matmul(_x, _x))) * np.sqrt(np.trace(np.matmul(_y, _y)))))
    else:
        for x in X:
            for y in X:
                if option == 0:
                    _x = np.array(dataset[x])
                    _y = np.array(dataset[y])
                    k = _x.size


                    i=0
                    while (k>0):
                        if _x[i] == 0 or _y[i] == 0:
                            _x = np.delete(_x, i)
                            _y = np.delete(_y, i)
                        else:
                            i=i+1
                        k = k - 1

                    dist[x][y] = distance.euclidean(_x, _y)

    dist = np.asarray(dist)
    # eps=0.5*((np.median(dist)) ** 2)
    eps = 0.4*((np.median(dist)) ** 2) #obesity
    # eps = 0.25 * ((np.median(dist)) ** 2)  # crop
    # eps = 0.1 * ((np.median(dist)) ** 2)  # isolet


    len = np.shape(dist)[0]
    w = np.zeros((len, len))
    for i in range(0, len):
        for j in range(0, len):
            w[i][j] = np.exp(-1 * ((dist[i][j] ** 2) / (2 * eps ** 2)))

    row_sum = []
    for row in range(len):
        sum = 0
        for col in range(len):
            sum = sum + w[row, col]
        row_sum.append(sum)

    col_sum = []
    for col in range(len):
        sum = 0
        for row in range(len):
            sum = sum + w[row, col]
        col_sum.append(sum)

    p = np.zeros((len, len))
    for i in range(len):
        for j in range(len):
            p[i][j] = w[i][j] / (row_sum[i] * col_sum[j])

    row_sum = []
    for row in range(len):
        sum = 0
        for col in range(len):
            sum = sum + p[row, col]
        row_sum.append(sum)
    for i in range(len):
        for j in range(len):
            p[i][j] = p[i][j] / (row_sum[i])
    return (p)


def calc_mu_sigma(labels, data, y):
    mu = np.zeros((np.shape(data)[1], np.shape(labels)[0]))
    sigma = np.zeros((np.shape(data)[1], np.shape(labels)[0]))
    i = 1
    for feature in range(np.shape(data)[1]):
        for label in labels:

            mu[feature][i - 1] = np.mean(data[np.where(y == label)[0], feature])
            sigma[feature][i - 1] = np.std(data[np.where(y == label)[0], feature])
            if sigma[feature][i - 1] == 0:
                sigma[feature][i - 1] = 0.0000001
            i += 1
        i = 1

    return mu, sigma


def calc_B_JM(labels, mu, sigma, b_index):
    B = np.zeros((np.shape(labels)[0], np.shape(labels)[0]))
    JM = np.zeros((np.shape(labels)[0], np.shape(labels)[0]))
    for i in range(np.shape(B)[0]):
        for j in range(np.shape(B)[0]):
            B[i][j] = (1 / 8) * ((mu[b_index][i] - mu[b_index][j]) ** 2) * (
                    2 / (sigma[b_index][i] ** 2 + sigma[b_index][j] ** 2)) + 0.5 * np.log(
                ((sigma[b_index][i] ** 2 + sigma[b_index][j] ** 2) / (2 * (sigma[b_index][i] * sigma[b_index][j]))))
    JM = 2 * (1 - np.exp(-B))

    return JM



def build_jm(labels, data, y, option=0, rand=False):
    run = 1
    while (run):
        if option == 1:
            run = 0
        mu, sigma = calc_mu_sigma(labels, data, y)

        new_jm = np.zeros((np.shape(data)[1], (np.shape(labels)[0]) ** 2))
        jm = np.array(np.zeros((np.shape(data)[1], 1)), dtype=object)
        jm_mean = np.zeros((1, np.shape(data)[1]))
        for i in range(0, np.shape(data)[1]):
            if not rand:
                jm[i, 0] = calc_B_JM(labels, mu, sigma, i)

                jm_mean[0][i] = np.mean(jm[i, 0])
                if option == 0:

                    new_jm[i][:] = np.reshape(jm[i, 0], ((1, (np.shape(labels)[0]) ** 2)))


        if option == 0:
            if np.any(np.isnan(new_jm)) == True:
                run = 1
            else:
                run = 0
    if option == 0:
        return new_jm, jm_mean
    else:
        return jm, jm_mean



def compute_eigenvectors(m, dim, score_points=False):
    eigval, eigvec = LA.eigh(np.array(m))
    eigvec = np.flip(eigvec, axis=1)
    eigvec = np.asarray(eigvec)
    U, S, _ = scipy.sparse.linalg.svds(A=m, k=dim + 1)
    sort_eig_vecs_idx = S.argsort()[::-1]
    U = U[:, sort_eig_vecs_idx]
    S[::-1].sort()
    S = S[1:]
    S = np.diag(S)
    U = U[:, 1:]
    coors = np.matmul(U, S)
    cor = coors[:, :dim]
    if score_points:

        return cor, S
    return cor


def Score_points(coor_jm, eigen_values):
    scores = np.zeros((1, np.shape(coor_jm)[0]))
    for i in range(0, np.shape(coor_jm)[0]):
        x = 0
        y = 0
        for j in range(0, np.shape(coor_jm)[1]):
            scores[0, i] +=coor_jm[i, j] / np.sqrt(eigen_values[x, y])
            x +=1
            y +=1

        scores[0, i] = scores[0, i] / np.max(np.abs(coor_jm[i, :]))
    return scores


def calc_coor_jm(jm, jm_mean, option=0, dim=3, score_points=False, rand_=False):
    p = calc_p(jm, option, rand_)
    coor_jm = compute_eigenvectors(p, dim, score_points)

    return coor_jm


def eliminate_features(labels, train, option, dim=3, k=1, dist_method=0,score_points=True,rand=False):
    jm, jm_mean = build_jm(labels, train[:, :np.shape(train)[1] - 1], train[:, np.shape(train)[1] - 1:], option,rand)
    coor_jm, eigen_val = calc_coor_jm(jm, jm_mean, option, dim, score_points,rand)
    scores = Score_points(coor_jm, eigen_val)
    ind_selected = find_features(coor_jm, jm_mean, k, dist_method, scores)

    return ind_selected, jm, coor_jm, jm_mean


def test_find_features(mat, jm_mean, dist_method, k=1):
    dist = np.zeros(((np.shape(mat)[0]), (np.shape(mat)[0])))
    for x in range(np.shape(mat)[0]):
        for y in range(np.shape(mat)[0]):
            if x == y:
                dist[x, y] = np.inf
            else:
                if dist_method == 0:
                    dist[x, y] = distance.euclidean(mat[x], mat[y])
                elif dist_method == 1:
                    dist[x, y] = distance.cityblock(mat[x], mat[y])
                elif dist_method == 2:
                    dist[x, y] = distance.cosine(mat[x], mat[y])

    threshold = k * np.mean(np.min(dist, axis=0))

    for x in range(np.shape(dist)[0]):
        for y in range(np.shape(dist)[0]):
            if dist[x, y] <= threshold and y > x:

                if jm_mean[0][x] < jm_mean[0][y]:
                    dist[:, x] = np.inf
                    dist[x, :] = np.inf

                else:

                    dist[:, y] = np.inf
                    dist[y, :] = np.inf



    ind_selected = []
    for x in range(np.shape(dist)[0]):
        if np.all(dist[x, :] == np.inf) == False:
            ind_selected.append(x)
    return ind_selected


def hyper_parms_tuning(train, valid, option_for_jm=0):
    y = train[:, -1]
    y = np.asarray(y)
    labels = np.unique(y)
    ind_selected, jm, coor_jm, jm_mean = test_eliminate_features(labels, train, option_for_jm, k=1, dim=8)
    max_acc = 0
    best_comb = [2, 0.5, 0]
    for dim in range(2, 3):
        for dist_method in range(0, 3):
            for k in np.arange(0.5, 2, 0.1):
                ind_selected = test_find_features(coor_jm[:, 0:dim], jm_mean, dist_method, k)
                if ind_selected == []:
                    break

                y_train = train[:, np.shape(train)[1] - 1:]
                train1 = train[:, ind_selected]
                y_test = valid[:, np.shape(valid)[1] - 1:]
                test1 = valid[:, ind_selected]
                train1 = np.concatenate((train1, y_train), axis=1)
                test1 = np.concatenate((test1, y_test), axis=1)
                acc_svm = float(pred_svm(train1, test1))
                acc_knn = float(knn_pred(train1, test1))
                acc_rand_forest = float(pred_randomforest(train1, test1))

                if ((acc_svm + acc_knn + acc_rand_forest) / 3) > max_acc:
                    max_acc = (acc_svm + acc_knn + acc_rand_forest) / 3
                    best_comb = [dim, k, dist_method]

    print("best_comb", best_comb)
    return best_comb




def find_features(mat, jm_mean, k=1, dist_method=0, score=0):
    dist = np.zeros(((np.shape(mat)[0]), (np.shape(mat)[0])))

    for x in range(np.shape(mat)[0]):
        for y in range(np.shape(mat)[0]):
            if x == y:
                dist[x, y] = np.inf
            else:
                if dist_method == 0:
                    dist[x, y] = distance.euclidean(mat[x], mat[y])
                elif dist_method == 1:
                    dist[x, y] = distance.cityblock(mat[x], mat[y])
                elif dist_method == 2:
                    dist[x, y] = distance.cosine(mat[x], mat[y])

    threshold = k * np.mean(np.min(dist, axis=0))

    for x in range(np.shape(dist)[0]):
        for y in range(np.shape(dist)[0]):
            if dist[x, y] <= threshold and y > x:

                if np.abs(score[0, x]) >= np.abs(score[0, y]):
                    dist[:, y] = np.inf
                    dist[y, :] = np.inf
                else:
                    dist[:, x] = np.inf
                    dist[x, :] = np.inf

    ind_selected = []
    for x in range(np.shape(dist)[0]):
        if np.all(dist[x, :] == np.inf) == False:
            ind_selected.append(x)
    return ind_selected




def test_eliminate_features(labels, train, option, dim=3, k=1, dist_method=0):
    jm, jm_mean = build_jm(labels, train[:, :np.shape(train)[1] - 1], train[:, np.shape(train)[1] - 1:], option)
    coor_jm, eigen_val = calc_coor_jm(jm, jm_mean, option, dim=dim, score_points=True)
    ind_selected = test_find_features(coor_jm, jm_mean, k, dist_method)
    return ind_selected, jm, coor_jm, jm_mean

def knn_pred(train, test):
    knn = KNeighborsClassifier()
    X_train, y_train = train[:, :np.shape(train)[1] - 1], train[:, np.shape(train)[1] - 1:np.shape(train)[1]]
    X_test, y_test = test[:, :np.shape(test)[1] - 1], test[:, np.shape(test)[1] - 1:np.shape(test)[1]]
    knn.fit(X_train, y_train)
    return knn.score(X_test, y_test)


def pred_randomforest(train, test):
    X_train, y_train = train[:, :np.shape(train)[1] - 1], (train[:, np.shape(train)[1] - 1])
    X_test, y_test = test[:, :np.shape(test)[1] - 1], (test[:, np.shape(test)[1] - 1])
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def pred_svm(train, test):
    svm = SVC()
    X_train, y_train = train[:, :np.shape(train)[1] - 1], train[:, np.shape(train)[1] - 1:np.shape(train)[1]]
    X_test, y_test = test[:, :np.shape(test)[1] - 1], test[:, np.shape(test)[1] - 1:np.shape(test)[1]]
    model = svm.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return svm.score(X_test, y_test)



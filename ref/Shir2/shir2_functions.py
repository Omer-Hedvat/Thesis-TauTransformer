from sklearn_extra.cluster import KMedoids
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from mrmr import mrmr_classif

def calc_epsilon(data_list, eps_type, epsilon_factor=4):
    distance_np = squareform(pdist(data_list))  # read about squareform
    if eps_type == 'maxmin':
        # option #1 - epsilon= max min(distance)
        idist = distance_np + np.identity(len(distance_np))
        eps_maxmin = np.max(np.min(idist, axis=0))
        eps = eps_maxmin * epsilon_factor

    elif eps_type == 'mean':
        # option #2 - epsilon= mean(distance)
        eps_mean = np.mean(distance_np)
        eps = eps_mean * epsilon_factor

    else:
        raise KeyError('eps_type should be either maxmin or mean')

    return distance_np, eps


def kernel_calc(datalist, eps_type, epsilon_factor):
    distance_np, eps = calc_epsilon(datalist, eps_type, epsilon_factor)
    kernel = np.exp(-(distance_np ** 2) / (2 * eps))
    return kernel, eps




def diffusion_mapping(data_list, alpha, eps_type, epsilon_factor, **kwargs):

    assert 'dim' in kwargs.keys()

    w, epsilon = kernel_calc(data_list, eps_type, epsilon_factor)
    v = np.sum(w, axis=0)
    # v_col=kernel.sum( axis=1)

    # 1st normalization - treats uneven density
    v = v ** alpha
    V_x_y = v * v[:, None]
    a = w / V_x_y

    sum_row = np.sum(a, axis=0)
    p = a / sum_row[:, None]
    #
    # # compute eigenvectors of (a_ij)
    singular_vectors, singular_values, _ = LA.svd(p, full_matrices=False)
    #
    # # Compute embedding coordinates
    diffusion_coordinates = singular_vectors[:, 1:kwargs['dim'] + 1].T * (singular_values[1:kwargs['dim'] + 1][:, None])

    return diffusion_coordinates


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

def build_jm(labels, data, y):
    run = 1
    while (run):
        mu, sigma = calc_mu_sigma(labels, data, y)
        new_jm = np.zeros((np.shape(data)[1], (np.shape(labels)[0]) ** 2))
        jm = np.array(np.zeros((np.shape(data)[1], 1)), dtype=object)
        jm_mean = np.zeros((1, np.shape(data)[1]))
        for i in range(0, np.shape(data)[1]):
                jm[i, 0] = calc_B_JM(labels, mu, sigma, i)
                jm_mean[0][i] = np.mean(jm[i, 0])
                new_jm[i][:] = np.reshape(jm[i, 0], ((1, (np.shape(labels)[0]) ** 2)))

        if np.any(np.isnan(new_jm)) == True:
            run = 1
        else:
            run = 0

    return new_jm, jm_mean

def find_features_kmedoids(train,jm_mean,coor_jm,factor):

    n_features =np.shape(train)[1]
    n_clusters=int(n_features/factor)
    coor_jm=coor_jm[(np.where(jm_mean>np.quantile(jm_mean,q=0.2))[1])]
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=0).fit(coor_jm)
    inx=[]
    for i in range(0,n_clusters):
        cluster_val=np.where([kmedoids.labels_==i])[1]
        if len(cluster_val)>1:
            max_jm_mean=0
            for j in range(0, len(cluster_val)):
                if jm_mean[0][cluster_val[j]]>max_jm_mean:
                    max_jm_mean=jm_mean[0][cluster_val[j]]
                    index=cluster_val[j]
            inx.append(index)
        else:
            inx.append(cluster_val[0])


    return [inx][0]
def plot_clusters_by_jm_mean(coor_jm, jm_mean):
    n = range(0, np.shape(coor_jm[:, 0])[0])
    fig, ax1 = plt.subplots()
    vmax = np.max(jm_mean)
    vmin = np.min(jm_mean)

    b = ax1.scatter(coor_jm[:, 0], coor_jm[:, 1], c=np.transpose(jm_mean)[:, 0], vmin=vmin, vmax=vmax,
                        cmap='plasma')


    plt.colorbar(b)

    plt.show()

def eliminate_features_kmedoids(labels, train):
    jm, jm_mean = build_jm(labels, train[:, :np.shape(train)[1] - 1], train[:, np.shape(train)[1] - 1:])
    coor_jm =diffusion_mapping(jm, 1,"maxmin",epsilon_factor=200,dim=3)
    #coor_jm = diffusion_mapping(jm, 1, "mean", epsilon_factor=2, dim=2)
    coor_jm=coor_jm.T
    plot_clusters_by_jm_mean(coor_jm, jm_mean)
    ind_selected = find_features_kmedoids(train,jm_mean,coor_jm,25 )
    print(ind_selected)
    return ind_selected, jm, coor_jm, jm_mean



def mmrm(X, y, num_features):
        selected_features = mrmr_classif(X=X, y=y, K=num_features)
        return selected_features
def pred_svm(train, test):
    svm = SVC()
    X_train, y_train = train[:, :np.shape(train)[1] - 1], train[:, np.shape(train)[1] - 1:np.shape(train)[1]]
    X_test, y_test = test[:, :np.shape(test)[1] - 1], test[:, np.shape(test)[1] - 1:np.shape(test)[1]]
    model = svm.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # compute and print accuracy score

    return accuracy_score(y_test, y_pred)



def knn_pred(train, test):
    knn = KNeighborsClassifier()
    X_train, y_train = train[:, :np.shape(train)[1] - 1], train[:, np.shape(train)[1] - 1:np.shape(train)[1]]
    X_test, y_test = test[:, :np.shape(test)[1] - 1], test[:, np.shape(test)[1] - 1:np.shape(test)[1]]
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    return accuracy_score(y_test, y_pred)


def pred_randomforest(train, test):
    X_train, y_train = train[:, :np.shape(train)[1] - 1], (train[:, np.shape(train)[1] - 1])
    X_test, y_test = test[:, :np.shape(test)[1] - 1], (test[:, np.shape(test)[1] - 1])
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return accuracy_score(y_test, y_pred)

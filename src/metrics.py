from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.svm import SVC


def knn_acc(x, y, need_std=False):
    neigh = KNeighborsClassifier(n_neighbors=1)
    output = cross_validate(neigh, x, y, cv=7, scoring='accuracy')
    re = output['test_score']
    if need_std:
        return round(np.mean(re), 4), round(np.var(re), 4)
    else:
        return round(np.mean(re), 4)


def knn_acc_my(x, y, need_std=False):
    nbrs = NearestNeighbors(n_neighbors=2).fit(x)
    distances, indices = nbrs.kneighbors(x)
    result = []
    for ind_i, i in enumerate(indices):
        for j in i:
            if j != ind_i:
                break
        result.append(y[j])
    result = np.array(result)
    return round(np.sum(result == y)/len(y), 4), 0


def svm_acc(x, y, need_std=False):
    clf = SVC(kernel='rbf')
    output = cross_validate(clf, x, y, cv=7, scoring='accuracy')
    re = output['test_score']
    if need_std:
        return round(np.mean(re), 4), round(np.var(re), 4)
    else:
        return round(np.mean(re), 4)

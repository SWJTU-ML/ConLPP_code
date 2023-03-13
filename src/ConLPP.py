from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise
import networkx as nx
import numpy as np


class ConLPP:
    def __init__(self, n_components=3, purning=True):
        self.n_dims = n_components
        self.parameter_c = 0.05
        self.parameter_p_1 = 1
        self.parameter_p_2 = 1
        self.is_purn = purning

    def __kernal(self, i, j):
        return np.exp(-np.linalg.norm(self.x[i]-self.x[j])/self.gama)

    def __get_K_NN_Rho(self, k):
        distances = self.max_k_distances[:, :k]
        distances /= np.max(distances)
        return np.sum(np.exp(-distances**2/self.gama), axis=1)

    def __rbf(self, dist, t=1.0):
        return np.exp(-(dist/t))  

    def __cal_rbf_dist(self, n_neighbors=10, t=1):
        dist = self.distMatrix**2
        n = dist.shape[0]
        rbf_dist = self.__rbf(dist, t)

        W = np.zeros((n, n))  
        for i in range(n):
            index_ = np.argsort(dist[i])[1:1 + n_neighbors]
            W[i, index_] = rbf_dist[i, index_]
            W[index_, i] = rbf_dist[index_, i]

        return W

    def __lpp(self, data, s_1, s_2, n_dims=2, n_neighbors=30, t=1.0):
        N = data.shape[0]
        W = self.__cal_rbf_dist(n_neighbors, t)  
        W_2 = s_1
        D = np.zeros_like(W)
        D_2 = np.zeros_like(W)

        for i in range(N):
            D_2[i, i] = np.sum(W_2[i])
            D[i, i] = np.sum(W[i])  

        L = D - W  
        L_2 = D_2 - W_2  
        L += L_2
        if np.max(abs(s_1)) > 0:
            XDXT = np.dot(np.dot(data.T, D_2), data)+s_2
        else:
            XDXT = np.dot(np.dot(data.T, D), data)+s_2
        XLXT = np.dot(np.dot(data.T, L), data)

        eig_val, eig_vec = np.linalg.eig(
            np.dot(np.linalg.pinv(XDXT), XLXT))  
        sort_index_ = np.argsort(np.abs(eig_val))
        eig_val = eig_val[sort_index_]  

        j = 0
        while eig_val[j] < 1e-6:
            j += 1

        sort_index_ = sort_index_[j:j+n_dims]
        eig_val_picked = eig_val[j:j+n_dims]
        eig_vec_picked = eig_vec[:, sort_index_]  
        self.tt = XDXT
        WEIGHT = eig_vec_picked

        T = np.dot(WEIGHT.T, self.tt).dot(WEIGHT)
        t_l = WEIGHT.copy()
        for i in range(len(T)):
            t_l[:, i] = t_l[:, i]*(T[i, i]**-0.5)
        return t_l.real


    def __get_pre_cluster(self, x, rho, distances, indices, prun=False):
        def pruning(node, target_pre_center):
            node_ind = indices[node]
            if len(np.where(target_pre_center == node)[0]) <= 0:
                return
            for i in np.where(target_pre_center == node)[0]:
                if i in node_ind:
                    pruning(i, target_pre_center)
                else:
                    target_pre_center[i] = -1
                    pruning(i, target_pre_center)
        n = len(x)
        pre_center = np.ones(n, dtype=np.int0)*-1
        sort_rho = np.flipud(np.argsort(rho))
        dist_pre = np.ones(n)
        dist_pre *= np.inf
        for i in range(n):
            min_dis = 0
            dis = np.inf
            find = False
            for j, index in enumerate(indices[i]):
                if index == i:
                    continue
                if rho[index] > rho[i]:
                    if distances[i, j] < dis:
                        find = True
                        min_dis = j
                        dis = distances[i, min_dis]
            if find:
                pre_center[i] = indices[i, min_dis]
        if prun:
            for i in np.where(pre_center == -1)[0]:
                pruning(i, pre_center)
        zero_index = []
        for i in range(len(pre_center)):
            if pre_center[i] == -1:
                zero_index.append(i)
        for i in sort_rho:
            if pre_center[i] == -1 or pre_center[i] in zero_index:
                continue
            pre_center[i] = pre_center[pre_center[i]]
        return pre_center

    def __compute_s_nx(self, k, x, is_purn):

        rho = self.__get_K_NN_Rho(k)
        distances = self.max_k_distances[:, :k]
        indices = self.max_k_indices[:, :k]
        pre_center = self.__get_pre_cluster(
            x, rho, distances, indices, prun=is_purn)
        pre_center = pre_center.astype(np.int0)
        t = []
        for i in range(len(pre_center)):
            if pre_center[i] == -1:
                t.append(i)
                pre_center[i] = i
        t_2 = []
        for i in t:
            if len(np.where(pre_center == i)[0]) > 1:
                t_2.append(i)
        t = t_2

        if len(t) < 1:
            return 0, 0, 0
        G = nx.Graph()
        G.add_nodes_from(t)
        W = np.zeros([len(x), len(x)])
        for i in t:
            set_cluster = np.where(pre_center == i)[0]
            for j in set_cluster:
                for k in set_cluster:
                    if j >= k:
                        continue
                    W[j, k] = self.__kernal(j, k)
                    W[k, j] = W[j, k]
        cluster_nbr = []


        for i in t:
            a = set()
            for j in np.where(pre_center == i)[0]:
                a.update(indices[j])
            cluster_nbr.append(a)

        for t_i in range(len(t)):
            for t_j in range(len(t)):
                if t_i >= t_j:
                    continue
                len_t_i = len(cluster_nbr[t_i])
                len_t_j = len(cluster_nbr[t_j])
                min_len = len_t_i if len_t_i < len_t_j else len_t_j
                intersection = cluster_nbr[t_i] & cluster_nbr[t_j]
                if len(intersection) > min_len*self.parameter_c:
                    G.add_edge(t[t_i], t[t_j])
        # compute s_2
        discon_set = []
        for i in nx.connected_components(G):
            discon_set.append(i)
        len_set = len(discon_set)
        discon_conunt = 0
        if len_set > 1:
            s_2 = 0
            for set_i in range(len_set):
                for set_j in range(set_i):
                    discon_dist = []
                    index_disconnect = []
                    for i in discon_set[set_i]:
                        for j in discon_set[set_j]:
                            discon_dist.append(self.distMatrix[i, j])
                            index_disconnect.append([i, j])
                    len_disconnect = int(
                        len(discon_dist)*self.parameter_p_1)  
                    discon_dist = np.array(discon_dist)

                    disc_sort = np.argsort(discon_dist)

                    index_disconnect = np.array(index_disconnect)
                    index = index_disconnect[disc_sort[:len_disconnect]]

                    temp = x[index[:, 0]]-x[index[:, 1]]
                    discon_conunt += len_disconnect
                    s_2 += np.dot(temp.T, temp)
            s_2 /= discon_conunt

        else:
            s_2 = 0
            list_discon = list(discon_set[0])
            len_list = len(list_discon)
            discon_dist = []
            index_disconnect = []
            for i in range(len_list):
                for j in range(i):
                    discon_dist.append(
                        self.distMatrix[list_discon[i], list_discon[j]])
                    index_disconnect.append([i, j])
            len_disconnect = int(len(discon_dist)*self.parameter_p_2)  
            discon_dist = np.array(discon_dist)

            disc_sort = np.argsort(-discon_dist)

            index_disconnect = np.array(index_disconnect)
            index = index_disconnect[disc_sort[:len_disconnect]]
            temp = x[index[:, 0]]-x[index[:, 1]]
            s_2 = np.dot(temp.T, temp)
            discon_conunt = len_disconnect if len_disconnect > 1 else 1
            s_2 /= discon_conunt

        return W, s_2, len_set

    def __init_start(self, x, k_end=None, gamma_percent=0.01):
        self.distMatrix = pairwise.euclidean_distances(x)
        self.gama = gamma_percent*np.max(self.distMatrix)**2
        if k_end != None:
            nbrs = NearestNeighbors(n_neighbors=k_end).fit(x)
            self.max_k_distances, self.max_k_indices = nbrs.kneighbors(x)

    def fit(self, x, k_start=5, k_end=15, weight=None, gamma_percent=0.01):
        x = x-np.mean(x, axis=0)
        s_r_1 = 0
        s_r_2 = 0
        t_con = 0
        t_discon = 0
        self.x = x
        self.__init_start(x, k_end-1, gamma_percent)

        for i in range(k_start, k_end, 1):
            s_1, s_2, self.len_set = self.__compute_s_nx(
                i, x, is_purn=self.is_purn)
            s_r_1 += s_1*np.exp(1/i)
            t_con += np.exp(1/i)
            t_discon += np.exp(i)
            s_r_2 += s_2*np.exp(i)
            t_con = 1/t_con*(k_end-k_start)
            t_discon = 1/t_discon*(k_end-k_start)
            s_r_1 *= t_con
            s_r_2 *= t_discon

        self.s_1 = s_r_1
        self.s_2 = s_r_2

        self.change_weight(x, weight)
        return self

    def change_weight(self, x, w=None):
        if w == None:
            w = 1
        self.weight = self.__lpp(
            x, self.s_1*w, self.s_2*w, self.n_dims, n_neighbors=10, t=self.gama)

    def transform(self, x):
        return np.dot(x, self.weight).real

    def fit_transform(self, x, k_start=5, k_end=15, w=None, gamma_percent=0.01):
        self.fit(x, k_start, k_end, w, gamma_percent)
        return self.transform(x)

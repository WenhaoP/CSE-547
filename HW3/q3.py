import numpy as np
import pandas as pd
import scipy

def main():

    songs = pd.read_csv("hw3-bundle/hw3-bundle/spectral_clustering/data/songs.csv")
    n, d = songs.shape

    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = int(np.linalg.norm(songs.iloc[i] - songs.iloc[j]) < 1)
            D = np.diag(np.sum(A, axis = 1))
    D_inv_sqrt = np.diag(1 / np.sqrt(np.sum(A, axis = 1)))
    L = D - A
    cal_L = D_inv_sqrt @ L @ D_inv_sqrt
    eig_vals, eig_vecs = np.linalg.eig(cal_L)
    v = eig_vecs[:, 1]
    x = D_inv_sqrt @ v
    is_cluster_1 = (x >= 0)
    print(is_cluster_1)

    cluster_1_feature_mean = np.mean(songs[is_cluster_1], axis = 0)
    cluster_2_feature_mean = np.mean(songs[np.logical_not(is_cluster_1)], axis = 0)
    
    print((cluster_1_feature_mean - cluster_2_feature_mean).abs())

    scipy.stats.ttest_ind(songs[is_cluster_1], songs[np.logical_not(is_cluster_1)], equal_var=True)

if __name__ == "__main__":
    main()
from datetime import datetime
from numpy.linalg import lstsq, norm
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize
import scipy.sparse as ss
import pandas as pd
import numpy as np
import pickle
import sys
import os

## GDM algorithms
def get_beta(cent, centers, m):
    betas = np.array([cent + m[x]*(centers[x,:] - cent) for x in range(centers.shape[0])])
    betas[betas<0] = 0
    betas = normalize(betas, 'l1')
    return betas

def gdm(wdfn, K, ncores=-1):
    glob_cent = np.mean(wdfn, axis=0)
    kmeans = KMeans(n_clusters=K, n_jobs=ncores, max_iter=1000).fit(wdfn)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    m = []
    for k in range(K):
        k_dist = euclidean(glob_cent, centers[k])
        Rk = max(np.apply_along_axis(lambda x: euclidean(glob_cent, x), 1, wdfn[labels==k,:]))
        m.append(Rk/k_dist)

    beta_means = get_beta(glob_cent, centers, m)

    return beta_means

## Geometric Theta
def proj_on_s(beta, doc, K, ind_remain=[], first=True, distance=False):
    if first:
        ind_remain = np.arange(K)
    s_0 = beta[0,:]
    if beta.shape[0]==1:
        if distance:
            return norm(doc-s_0)
        else:
            theta = np.zeros(K)
            theta[ind_remain] = 1.
            return theta
    beta_0 = beta[1:,:]
    alpha = lstsq((beta_0-s_0).T, doc-s_0)[0]
    if np.all(alpha>=0) and alpha.sum()<=1:
        if distance:
            p_prime = (alpha*(beta_0-s_0).T).sum(axis=1)
            return norm(doc-s_0-p_prime)
        else:
            theta = np.zeros(K)
            theta[ind_remain] = np.append(1-alpha.sum(), alpha)
            return theta
    elif np.any(alpha<0):
        ind_remain = np.append(ind_remain[0], ind_remain[1:][alpha>0])
        return proj_on_s(np.vstack([s_0, beta_0[alpha>0,:]]), doc, K, ind_remain, False, distance)
    else:
        return proj_on_s(beta_0, doc, K, ind_remain[1:], False, distance)

model = sys.argv[1]
project_dir = "/mnt/dropbox/Dropbox/BK_LB_Projects/topic_model_experiments"
data_dir = os.path.join(project_dir, model, "DTM")
res_dir = os.path.join(project_dir, model, "alt_topic_models")

# load DTM
corpus_files = [f for f in os.listdir(data_dir)
                if "corpus_" in f]
corpus = pd.concat([pd.read_csv(os.path.join(data_dir, f), header=None)
                    for f in corpus_files])
doc_id = corpus[0]
term_id = corpus[1]
value = corpus[2]

dtm = ss.csr_matrix((value, (doc_id, term_id)),
                    shape=(max(doc_id) + 1, max(term_id) + 1))
#dtm = dtm[np.squeeze(np.asarray(dtm.sum(axis=1) != 0)),:]
norm_dtm = normalize(dtm, "l1").toarray()
t0 = datetime.now()
phi = gdm(norm_dtm, 50)
phi_time = datetime.now() - t0
phi_res = {"time": phi_time, "data": phi}
with open(os.path.join(res_dir, "phi.pkl"), "wb") as ofile:
    pickle.dump(phi_res, ofile)
np.savetxt(os.path.join(res_dir, "phi.txt"), phi)
print "phi", phi_time
t0 = datetime.now()

K = phi.shape[0]

def proj(x):
    return proj_on_s(phi, x, K)

theta = np.apply_along_axis(proj, 1, norm_dtm)
theta_time = datetime.now() - t0
theta_res = {"time": theta_time, "data": theta}
with open(os.path.join(res_dir, "theta.pkl"), "wb") as ofile:
    pickle.dump(theta_res, ofile)
np.savetxt(os.path.join(res_dir, "theta.txt"), theta)
print "theta", theta_time

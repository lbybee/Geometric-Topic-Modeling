from numpy.linalg import lstsq, norm
from datetime import datetime
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import scipy.optimize as so
import numpy as np


def gen_phi(dtm, K, KMeans_kwds=None):
    """estimates phi for the given DTM

    Parameters
    ----------
    dtm : array-like
        sparse matrix of normalized word counts
    K : scalar
        number of topics
    KMeans_kwds : dict-like or None
        additional key-words to pass to KMeans

    Returns
    -------
    numpy array corresponding to phi
    """

    if KMeans_kwds is None:
        KMeans_kwds = {}

    # global center of data (C)
    # C = 1/D \sum_d \bar{w}_d
    C =  np.mean(dtm, axis=0)

    # generate K-means estimates (\mu_k)
    t0 = datetime.now()
    kmeans_mod = KMeans(n_clusters=K, **KMeans_kwds)
    kmeans_fit = kmeans_mod.fit(dtm)
    print datetime.now() - t0

    mu = kmeans_fit.cluster_centers_
    labels = kmeans_fit.labels_

    # now we generate each \phi_k
    phi = []
    for k in range(K):
        # ||C - \mu_k||_2
        dist_k = euclidean(C, mu[k])
        # R_k = max_d ||C - \bar{w}_d||_2
        R_k = np.max(np.linalg.norm(C - dtm[labels==k,:], axis=1))
        # m_k = R_k / ||C - \mu_k||_2
        m_k = R_k / dist_k
        # \phi_k = C + m_k (\mu_k - C)
        tphi = C + m_k * (mu[k] - C)
        phi.append(np.squeeze(np.array(tphi)))

    phi = np.array(phi)
    # rescale phi thresholding at 0
    # \xi_{k,v} = \phi_{k,v} \mathbb{I}(\phi_{k,v} > 0)
    # \phi_{k,v} = \xi_{k,v} / \sum_q \xi_{k,q}
    phi = phi * (phi > 0)
    phi = (phi.T / phi.sum(axis=1)).T
    return phi


def gen_theta(dtm, phi):
    """generates the estimate for theta based on phi
    and data

    Parameters
    ----------
    dtm : array-like
        sparse matrix of normalized word counts
    phi : array-like
        matrix of phi values returned by gen_phi

    Returns
    -------
    numpy array corresponding to theta

    Note that here \hat{\theta_d} is defined as

    \tilde{\phi} = \hat{\phi} st. \bar{w_{d,v}} > 0
    \tilde{w_d} = \bar{w_d} st. \bar{w_{d,v}} > 0
    P = (\tilde{\phi} \tilde{\phi}^T){-1}
    B_d = P \tilde{\phi} \tilde{w}_d
    I = [[1,...,1]], I : 1 x K
    \hat{\theta} = B_d + P I'(IPI)^{-1}(1 - I B_d)
    """

    # get shape
    D, V = dtm.shape
    K = phi.shape[0]

    # first we build a 3d matrix consisting of V outer-products
    # for each \hat{\phi}^T_v.  The reason for this is that
    # we need a way to build P (on the cheap) for each document
    # and this is dependent on the non-zero w_d counts.  By
    # noting that X'X = \sum_n X_i X_i', we can get P
    # efficiently.
    # TODO there has got to be a better way to do this
    phi_outer = []
    for v in range(V):
        tmp = np.asmatrix(phi[:,v])
        phi_outer.append(tmp.T.dot(tmp))
    phi_outer = np.array(phi_outer)

    I = np.ones(shape=(1, K))
    theta = []
    for d in range(D):
#        nz_ind = dtm[d,:] > 0
        nz_ind = dtm[d,:].nonzero()[1]
#        print d, len(nz_ind)
        # TODO don't invert that matrix!!!
        tilde_phi = phi[:,nz_ind]
        tilde_w = dtm[d,nz_ind].toarray()
#        try:
#            P = np.linalg.inv(phi_outer[nz_ind].sum(axis=0))
#        except Exception as e:
#            print d, len(nz_ind), e
#            raise e
#        B_d = P.dot(tilde_phi).dot(tilde_w.T)
#        tmp = np.linalg.inv(I.dot(P).dot(I.T))
#        theta_d = B_d + P.dot(I.T).dot(tmp).dot(1 - I.dot(B_d))
#        def func(x):
#
#            res = np.linalg.norm(tilde_phi.T.dot(x) - tilde_w.T) ** 2
#            return res
#
#        def eqfunc(x):
#
#            return np.sum(x) - 1
#
#        theta_d = so.fmin_slsqp(func, np.zeros(K), eqcons=[eqfunc],
#                                bounds=[(0, 1) for i in range(K)])
        theta_d = so.lsq_linear(tilde_phi.T, tilde_w[0],
                                bounds=(-np.zeros(K), np.ones(K))).x
        print theta_d
        theta.append(theta_d)

    theta = np.array(theta)
    return theta


# alt theta (from original code)
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


def proj_wrapper(dtm, phi):
    """wraps the projection"""

    D, V = dtm.shape
    K = phi.shape[0]

    theta = []
    for d in range(D):
        theta_d = proj_on_s(phi, np.squeeze(np.array(dtm[d,:].toarray())), K)
        theta.append(theta_d)
        print theta_d

    return np.array(theta)

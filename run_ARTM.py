from sklearn.preprocessing import normalize
from datetime import datetime
import scipy.sparse as ss
import pandas as pd
import numpy as np
import pickle
import ARTM
import sys
import os


model = sys.argv[1]
K = int(sys.argv[2])

project_dir = "/mnt/dropbox/Dropbox/BK_LB_Projects/topic_model_experiments"
data_dir = os.path.join(project_dir, model, "DTM")
res_dir = os.path.join(project_dir, model, "alt_topic_models")
if not os.path.isdir(res_dir):
    os.mkdir(res_dir)

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
norm_dtm = normalize(dtm, "l1")

t0 = datetime.now()
kwds = {"n_jobs": -1,
        "max_iter": 1000,
        "verbose": 1,
        "tol": 1e-3,
        "precompute_distances": True}
phi = ARTM.gen_phi(norm_dtm, K, KMeans_kwds=kwds)
phi_time = datetime.now() - t0
phi_res = {"time": phi_time, "data": phi}
with open(os.path.join(res_dir, "phi.%d.pkl" % K), "wb") as ofile:
    pickle.dump(phi_res, ofile)
np.savetxt(os.path.join(res_dir, "phi.%d.txt" % K), phi)
print "phi", phi_time

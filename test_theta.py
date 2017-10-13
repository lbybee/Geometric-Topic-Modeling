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

# load phi
phi = np.loadtxt(os.path.join(res_dir, "phi.%d.txt" % K))

t0 = datetime.now()
#theta = ARTM.gen_theta(norm_dtm, phi)
theta = ARTM.proj_wrapper(norm_dtm, phi)
theta_time = datetime.now() - t0
theta_res = {"time": theta_time, "data": theta}
with open(os.path.join(res_dir, "theta.%d.pkl" % K), "wb") as ofile:
    pickle.dump(theta_res, ofile)
np.savetxt(os.path.join(res_dir, "theta.%d.txt" % K), theta)
print "theta", theta_time

import numpy as np
from load_mnist_cifar import load_mnist
from joblib import Parallel, delayed
import pickle

def rff_features(X, num_features, seed, sigma=5.0):
  d = X.shape[1]
  np.random.seed(seed)
  W = np.random.normal(size=(d, num_features)) / sigma
  b = 2.0 * np.pi * np.random.uniform(size=(num_features,))
  Z = np.sqrt(2.0 / num_features) * np.cos(X @ W + b)
  return Z


def get_mse(Z, W, Y):
  preds = Z @ W
  return np.mean((preds - Y) ** 2)

def get_gac(K):
  n=K.shape[0]
  Kr=np.diag(np.sqrt(1/(1e-8+np.diag(K))))@K@np.diag(np.sqrt(1/(1e-8+np.diag(K))))
  return 1-np.sum(np.square(np.eye(n)-Kr))/(n**2-n)

def run_for_feat_count(feat_count, X_tr, Y_tr, X_te, Y_te, seed):
  Ph_tr = rff_features(X_tr, feat_count, seed)
  Ph_te = rff_features(X_te, feat_count, seed)
  Wh, *_ = np.linalg.lstsq(Ph_tr, Y_tr, rcond=None)
  K_tr=Ph_tr@Ph_tr.T
  return get_mse(Ph_tr, Wh, Y_tr), get_mse(Ph_te, Wh, Y_te), get_gac(K_tr)

feat_counts=np.unique(np.geomspace(1,10000,30).astype(int))
XLAB='Number of RFFs'

mse_tr_seeds = []
mse_te_seeds = []
gac_seeds=[]

for seed in range(100):
  X_tr, Y_tr, X_te, Y_te = load_mnist(n_tr=1000, n_te=1000, seed=seed, one_hot=True)
  results = Parallel(n_jobs=-1)(delayed(run_for_feat_count)(feat_count, X_tr, Y_tr, X_te, Y_te, seed) for feat_count in feat_counts)
  
  mse_trs, mse_tes, gacs = zip(*results)
  
  for mse_tr, mse_te, gac in zip(mse_trs, mse_tes, gacs):
    print( f'seed={seed:<2} | train loss={mse_tr:.4f} | test loss={mse_te:.4f} | gac={gac:.4f} | ')
  
  mse_tr_seeds.append(mse_trs)
  mse_te_seeds.append(mse_tes)
  gac_seeds.append(gacs)

  with open('mnist_rff.pkl','wb') as f:
    pickle.dump((mse_tr_seeds, mse_te_seeds, gac_seeds, gac_seeds, feat_counts, XLAB, 'log', None), f)



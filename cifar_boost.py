import numpy as np
from sklearn.tree import DecisionTreeRegressor
import sys
from load_mnist_cifar import load_cifar
from joblib import Parallel, delayed
import pickle

N_TR=1000
N_TE=1000

EPOCHS=100
LR=.01
GAMMA=0.95

MAX_DEPTHS= np.arange(1,26,1).astype(int)

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

def get_mse(y,fh):
  return np.mean((y-fh)**2)


def get_gac(tree, X):
  n = X.shape[0]
  P = tree.decision_path(X)[:, 1:]
  K1 = (P @ P.T).toarray()
  col_max = K1.max(axis=0)
  Kb = (K1 == col_max)
  return 1-np.sum(np.square(np.eye(n)-Kb))/(n**2-n)

def run_for_depth(max_depth, X_tr, y_tr, X_te, y_te, seed):
  y_tr_mean=np.mean(y_tr)
  fh_tr = y_tr_mean * np.ones((N_TR, 1))
  fh_te = y_tr_mean * np.ones((N_TE, 1))
  fh_tr_old = y_tr_mean * np.ones((N_TR, 1))
  fh_te_old = y_tr_mean * np.ones((N_TE, 1))
  gacs = []
  loss_diffs = []
  loss_old= y_tr_mean
  for m in range(EPOCHS):
    res = y_tr - fh_tr
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=seed)
    tree.fit(X_tr, res)
    h_tr = tree.predict(X_tr).reshape(-1, 1)
    h_te = tree.predict(X_te).reshape(-1, 1)
    fh_tr_diff=fh_tr-fh_tr_old
    fh_te_diff=fh_te-fh_te_old
    fh_tr_old=np.copy(fh_tr)
    fh_te_old=np.copy(fh_te)
    fh_tr += (LR * h_tr + GAMMA*fh_tr_diff)
    fh_te += (LR * h_te + GAMMA*fh_te_diff)
    gacs.append(get_gac(tree, X_tr))
    loss=np.sum(np.square(y_tr-fh_tr))
    loss_diffs.append(np.maximum(0,loss_old - loss))
    loss_old=loss
  
  mse_tr = get_mse(y_tr, fh_tr)
  mse_te = get_mse(y_te, fh_te)
  tot_gac = np.mean(np.array(gacs) * np.array(loss_diffs)) / np.mean(loss_diffs)
  max_gac = np.max(gacs)
  return mse_tr, mse_te, tot_gac, max_gac

mse_tr_seeds=[]
mse_te_seeds=[]
tot_gac_seeds=[]
max_gac_seeds=[]
for seed in range(10):
  X_tr, y_tr, X_te, y_te = load_cifar(N_TR, N_TE, seed)
  results = Parallel(n_jobs=-1)(delayed(run_for_depth)(max_depth, X_tr, y_tr, X_te, y_te, seed) for max_depth in MAX_DEPTHS)
  mse_trs, mse_tes, tot_gacs, max_gacs = zip(*results)
  for max_depth, mse_tr, mse_te, tot_gac, max_gac in zip(MAX_DEPTHS, mse_trs, mse_tes, tot_gacs, max_gacs):
    print(f'seed={seed} | max_depth={max_depth:<2} | train loss={mse_tr:.4f} | test loss={mse_te:.4f} | tot gac={tot_gac:.4f} | max gac={max_gac:.4f} | ')
  
  mse_tr_seeds.append(mse_trs)
  mse_te_seeds.append(mse_tes)
  tot_gac_seeds.append(tot_gacs)
  max_gac_seeds.append(max_gacs)
  
  with open('cifar_boost.pkl','wb') as f:
    pickle.dump((mse_tr_seeds, mse_te_seeds, tot_gac_seeds, max_gac_seeds, MAX_DEPTHS, 'Max Depth', None, None), f)


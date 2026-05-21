import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestClassifier
from load_mnist_cifar import load_mnist, np_one_hot
import pickle
plt.rcParams.update({'pdf.fonttype': 42, 'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman']})
plt.rcParams.update({ "text.latex.preamble": r""" \usepackage[T1]{fontenc} \usepackage{lmodern} \usepackage{amsmath} """ })

lines=[]
labs=[]
for c,txt in zip([0,1,2], ['Training MSE', 'Test MSE', 'GAC (right y-axis)']):
  lines.append(Line2D([0],[0],color='C'+str(c),lw=2))
  labs.append(txt)
def forward(p):
  return 1/(1 - np.minimum(1,p)+1e-10)

def inverse(y):
  return 1 - 1/(np.maximum(0,y)+1e-10)

def get_mse(y, fh):
  y_oh=np_one_hot(y,10)
  return np.mean((fh - y_oh) ** 2)



def get_gac(rf, X_tr):
  n = X_tr.shape[0]
  Ks = []
  for tree, es in zip(rf.estimators_, rf.estimators_samples_):
    leaves = tree.apply(X_tr)
    K = np.zeros((n, n), dtype=np.int32)
    # group samples by leaf
    leaf_to_samples = {}
    for i, l in enumerate(leaves):
      leaf_to_samples.setdefault(l, []).append(i)
    es = np.asarray(es)
    for samples in leaf_to_samples.values():
      samples = np.array(samples)
      # intersection with bootstrap samples
      mask = np.isin(samples, es)
      bootstrap_members = samples[mask]
      if bootstrap_members.size == 0:
        continue
      # each column chooses those bootstrap samples
      K[np.ix_(samples, bootstrap_members)] = 1
    Ks.append(K)
  K = np.mean(Ks, axis=0)
  mask = np.diag(K) != 0
  Kr = K[mask][:, mask]
  d = np.diag(Kr)
  scale = np.sqrt(1/(1e-8 + d))
  Krn = scale[:, None] * Kr * scale[None, :]
  nr=Krn.shape[0]
  return 1 - np.sum((np.eye(nr) - Krn)**2) / (nr**2 - nr)

BOOT=False
XLAB='$N^{\\text{max}}_{\\text{leaf}}$ / $N_{\\text{tree}}$'

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])



max_leafs=np.hstack((np.geomspace(10,2000,11),2000*np.ones(10))).astype(int)
n_trees=np.hstack((np.ones(11),np.linspace(2,20,10))).astype(int)
compl_proxy = np.arange(len(max_leafs))
tick_indices = np.linspace(0, len(compl_proxy) - 1, 5, dtype=int)
x_ticks=compl_proxy[tick_indices]
x_ticklabels= np.array([f"{l}/{t}" for l, t in zip(max_leafs, n_trees)])[tick_indices]


mse_tr_seeds=[]
mse_te_seeds=[]
gac_seeds=[]

for seed in range(10):
  X_tr, y_tr, X_te, y_te = load_mnist(n_tr=10000,n_te=10000, seed=seed)
  mse_trs=[]
  mse_tes=[]
  gacs=[]
  for max_leaf, n_tree in zip(max_leafs, n_trees):
    rf = RandomForestClassifier(n_estimators=n_tree, max_leaf_nodes=max_leaf, n_jobs=-1, bootstrap=BOOT, random_state=seed)
    _=rf.fit(X_tr, y_tr)
    fh_tr= rf.predict_proba(X_tr)
    fh_te= rf.predict_proba(X_te)
    mse_trs.append(get_mse(y_tr, fh_tr))
    mse_tes.append(get_mse(y_te, fh_te))
    gacs.append(get_gac(rf, X_tr))
    print( f'seed={seed} | max_leaf={max_leaf:<4} | n_tree={n_tree:<2} | train loss={mse_trs[-1]:.4f} | test loss={mse_tes[-1]:.4f} | gac={gacs[-1]:.4f} | ')
  
  mse_tr_seeds.append(mse_trs)
  mse_te_seeds.append(mse_tes)
  gac_seeds.append(gacs)
  
  mse_tr_mean=np.mean(mse_tr_seeds,0)
  mse_te_mean=np.mean(mse_te_seeds,0)
  gac_mean=np.mean(gac_seeds,0)
  mse_tr_d1=np.quantile(np.array(mse_tr_seeds),q=0.1,axis=0)
  mse_tr_d9=np.quantile(np.array(mse_tr_seeds),q=0.9,axis=0)
  mse_te_d1=np.quantile(np.array(mse_te_seeds),q=0.1,axis=0)
  mse_te_d9=np.quantile(np.array(mse_te_seeds),q=0.9,axis=0)
  gac_d1=np.quantile(np.array(gac_seeds),q=0.1,axis=0)
  gac_d9=np.quantile(np.array(gac_seeds),q=0.9,axis=0)
  
  fig, axs = plt.subplots(1, 2, figsize=(8, 2.5))
  
  _=axs[0].plot(compl_proxy, mse_tr_mean, 'C0', marker="o", markersize=3)
  _=axs[0].plot(compl_proxy, mse_te_mean, 'C1', marker="o", markersize=3)
  _=axs[0].plot(compl_proxy, mse_tr_d1, 'C0:')
  _=axs[0].plot(compl_proxy, mse_tr_d9, 'C0:')
  _=axs[0].plot(compl_proxy, mse_te_d1, 'C1:')
  _=axs[0].plot(compl_proxy, mse_te_d9, 'C1:')
  _=axs[0].set_xlabel(XLAB)
  _=axs[0].set_xticks(x_ticks)
  _=axs[0].set_xticklabels(x_ticklabels, fontsize=9)
  
  ax0t=axs[0].twinx()
  _=ax0t.plot(compl_proxy, gac_mean, 'C2', marker="o", markersize=3)
  _=ax0t.plot(compl_proxy, gac_d1, 'C2:')
  _=ax0t.plot(compl_proxy, gac_d9, 'C2:')
  _=ax0t.set_yscale('function', functions=(forward, inverse))
  y_ticks=np.linspace(forward(np.min(gac_mean)), forward(np.max(gac_mean)),5)
  p_ticks = inverse(y_ticks)
  _=ax0t.set_yticks(p_ticks)
  _=ax0t.set_yticklabels([f"{p:.3f}" for p in p_ticks])
  
  _=axs[1].plot(gac_mean, mse_tr_mean, 'C0', marker="o", markersize=3)
  _=axs[1].plot(gac_mean, mse_te_mean, 'C1', marker="o", markersize=3)
  _=axs[1].plot(gac_mean, mse_tr_d1, 'C0:')
  _=axs[1].plot(gac_mean, mse_tr_d9, 'C0:')
  _=axs[1].plot(gac_mean, mse_te_d1, 'C1:')
  _=axs[1].plot(gac_mean, mse_te_d9, 'C1:')
  _=axs[1].set_xlabel('GAC')
  _=axs[1].set_xscale('function', functions=(forward, inverse))
  _=axs[1].set_xticks(p_ticks)
  _=axs[1].set_xticklabels([f"{p:.3f}" for p in p_ticks])
  
  fig.legend(lines, labs, loc='lower center', ncol=len(lines))
  fig.tight_layout()
  fig.subplots_adjust(bottom=.32)
  if BOOT:
    fig.savefig('figures/mnist_rf_boot.pdf')
    with open('mnist_rf_boot.pkl','wb') as f:
      pickle.dump((mse_tr_seeds, mse_te_seeds, gac_seeds, compl_proxy, XLAB, x_ticks, x_ticklabels), f)
  else:
    fig.savefig('figures/mnist_rf.pdf')
    with open('mnist_rf.pkl','wb') as f:
      pickle.dump((mse_tr_seeds, mse_te_seeds, gac_seeds, gac_seeds, compl_proxy, XLAB, x_ticks, x_ticklabels), f)

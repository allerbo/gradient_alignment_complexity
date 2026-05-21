import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import sys
from matplotlib.lines import Line2D
plt.rcParams.update({'pdf.fonttype': 42, 'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman']})
plt.rcParams.update({ "text.latex.preamble": r""" \usepackage[T1]{fontenc} \usepackage{lmodern} \usepackage{amsmath} """ })

def forward(p):
  return 1/(1 - np.minimum(1,p)+1e-10)

def inverse(y):
  return 1 - 1/(np.maximum(0,y)+1e-10)

def get_S_knn(k, X_tr, X_te, y_tr, seed):
  n_tr=X_tr.shape[0]
  n_te=X_te.shape[0]
  knn=KNeighborsRegressor(n_neighbors=k)
  _=knn.fit(X_tr, np.squeeze(y_tr))
  K_tr=np.zeros((n_tr,n_tr), dtype=int)
  K_tr[np.arange(n_tr)[:, None], knn.kneighbors(X_tr)[1]]=1
  K_te=np.zeros((n_te,n_tr), dtype=int)
  K_te[np.arange(n_te)[:, None], knn.kneighbors(X_te)[1]]=1
  S_tr=K_tr/(K_tr@np.ones((n_tr,1)))
  S_te=K_te/(K_te@np.ones((n_tr,1)))
  return S_tr, S_te

def get_S_dt1(dt, X_tr, X_te):
  P_tr = dt.decision_path(X_tr)[:, 1:]
  P_te = dt.decision_path(X_te)[:, 1:]
  K1 = (P_te @ P_tr.T).toarray()
  row_max = (K1.max(axis=1)).reshape(-1,1)
  Kb = (K1 == row_max)
  n_tr=Kb.shape[1]
  S=Kb/(Kb@np.ones((n_tr,1)))
  return S

def get_S_dt(max_leaf, X_tr, X_te, y_tr, seed):
  dt = DecisionTreeRegressor(max_leaf_nodes=max_leaf, random_state=seed)
  _=dt.fit(X_tr, np.squeeze(y_tr))
  S_tr=get_S_dt1(dt, X_tr, X_tr)
  S_te=get_S_dt1(dt, X_tr, X_te)
  return S_tr, S_te


def get_S_rf(n_tree, X_tr, X_te, y_tr, seed):
  rf=RandomForestRegressor(n_estimators=n_tree, random_state=seed)
  _=rf.fit(X_tr,np.squeeze(y_tr))
  S_tr=get_S_rf1(rf, X_tr, X_tr)
  S_te=get_S_rf1(rf, X_tr, X_te)
  return S_tr, S_te

def get_S_rf1(rf, X_tr, X_te):
  P_te_all=rf.decision_path(X_te)[0].todense()
  P_tr_all=rf.decision_path(X_tr)[0].todense()
  starts_te=rf.decision_path(X_te)[1]
  starts_tr=rf.decision_path(X_tr)[1]
  n=X_tr.shape[0]
  assert np.all(starts_te==starts_tr)
  Ks=[]
  for ii,es in zip(range(len(starts_te)-1),rf.estimators_samples_):
    P_te=P_te_all[:,starts_te[ii]:(starts_te[ii+1])]
    P_tr=P_tr_all[:,starts_te[ii]:(starts_te[ii+1])]
    S=np.zeros((n,n))
    S[np.arange(n),es]=1
    K1=P_te@P_tr.T@S.T
    row_max = (K1.max(axis=1)).reshape(-1,1)
    K=np.asarray((K1==row_max)@S).astype(np.int32)
    Ks.append(K/(K@np.ones((n,1))))
  return np.mean(Ks,0)




def get_gac(S_tr, S_te, y_tr):
  K=S_tr
  if np.any(np.sum(K,axis=0)==0):
    return np.nan
  n=K.shape[0]
  Kr=np.diag(np.sqrt(1/(1e-8+np.diag(K))))@K@np.diag(np.sqrt(1/(1e-8+np.diag(K))))
  return 1 - np.sum((np.eye(n) - Kr)**2) / (n**2 - n)

def get_vne(S_tr, S_te, y_tr):
  K=S_tr
  if np.any(np.sum(K,axis=0)==0):
    return np.nan
  n=K.shape[0]
  Kr=np.diag(np.sqrt(1/(1e-8+np.diag(K))))@K@np.diag(np.sqrt(1/(1e-8+np.diag(K))))
  s=np.linalg.svd(Kr)[1]
  s/=np.sum(s)
  return -np.sum(s*np.log(s+1e-8))/np.log(n)

def get_enp(S_tr, S_te, y_tr):
  n_tr=S_tr.shape[0]
  return np.trace(S_tr)/n_tr

def get_genpv(S_tr, S_te, y_tr):
  n_te,n_tr=S_te.shape
  return np.trace(S_te.T@S_te)/n_te

def get_genprx(S_tr, S_te, y_tr):
  n_te,n_tr=S_te.shape
  y_te=np.random.normal(0,1,(n_te,1))
  err_te=np.mean((y_te-S_te@y_tr)**2)
  err_tr=np.mean((y_tr-S_tr@y_tr)**2)
  opt=max(err_te-err_tr,0)
  return 1/n_tr*(2*n_tr-1+n_tr*opt-np.sqrt((2*n_tr-1+n_tr*opt)**2-4*(n_tr-1)*n_tr*opt))/2






ks=range(1,21)
max_leafs=range(2,51)
n_trees=range(3,51)


COMPL_DICT={'GAC': 'compl_gacs', 'ENP': 'compl_enps', 'GENP-V': 'compl_genps', 'vNE': 'compl_vnes', 'GENP-RX': 'compl_rxs', '$\\|\\hat{\\theta}\\|^2_2$': 'compl_norms'}
LABS_DICT={'gac': 'GAC', 'vne': 'vNE', 'enp': 'ENP', 'genpv': 'GENP-V', 'genprx': 'GENP-RX', 'norm': '$\\|\\hat{\\theta}\\|^2_2$'}
ALGS_DICT={'gac': get_gac, 'vne': get_vne, 'enp': get_enp, 'genpv': get_genpv, 'genprx': get_genprx}
ALGS1_DICT={'knn': get_S_knn, 'dt': get_S_dt, 'rf': get_S_rf}

COMPL_ALGS=['gac', 'enp', 'genpv', 'genprx']
FIG_APDX=''
FIG2=False

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

if FIG2:
  COMPL_ALGS=['gac', 'enp', 'genpv', 'genprx', 'vne']
  FIG_APDX='2'


lines=[]
labs=[]
for c, compl_alg in enumerate(COMPL_ALGS):
  labs.append(LABS_DICT[compl_alg])
  lines.append(Line2D([0],[0],color='C'+str(c),lw=2))

#fig, ax_mat=plt.subplots(2,3,figsize=(11,4))
fig, axs=plt.subplots(1,3,figsize=(11,2.45))

d=1
n=20

for alg, title, xlab, sweeps, ax in zip(['knn','dt','rf'], ['k-Nearest Neighbors', 'Decision Trees', 'Random Forests'], ['$\\kappa$', '$N^{\\text{max}}_{\\text{leaf}}$', '$N_{\\text{tree}}$'], [ks,max_leafs,n_trees], axs):
  compls_sweep={}
  for compl_alg in COMPL_ALGS:
    compls_sweep[compl_alg]=[]
  for sweep in sweeps:
    compls_seed={}
    for compl_alg in COMPL_ALGS:
      compls_seed[compl_alg]=[]
    for seed in range(100):
      np.random.seed(seed)
      X_tr=np.random.normal(0,1,(n,d))
      y_tr=np.random.normal(0,1,(n,1))
      X_te=np.random.normal(0,1,(n,d))
      S_tr, S_te = ALGS1_DICT[alg](sweep, X_tr, X_te, y_tr, seed)
      for compl_alg in COMPL_ALGS:
        compls_seed[compl_alg].append(ALGS_DICT[compl_alg](S_tr, S_te, y_tr))
    for compl_alg in COMPL_ALGS:
      compls_sweep[compl_alg].append(compls_seed[compl_alg])
  if alg=='rf':
    axts=[ax, ax.twinx(), ax.twinx(), ax.twinx()]
    if FIG2:
      axts.append(ax)
  for c, (compl_alg, yticks) in enumerate(zip(COMPL_ALGS, [[0.98, 0.99],[0.640, 0.643],[0.5,0.6],[0.42,0.43], [None, None]])):
    if alg=='rf':
      if FIG2:
        axts[c].plot(sweeps, np.nanmedian(np.array(compls_sweep[compl_alg]),axis=1), 'C'+str(c), zorder=5-c)
        axts[c].plot(sweeps, np.nanquantile(np.array(compls_sweep[compl_alg]),q=0.25,axis=1), 'C'+str(c)+':', zorder=5-c, lw=1)
        axts[c].plot(sweeps, np.nanquantile(np.array(compls_sweep[compl_alg]),q=0.75,axis=1), 'C'+str(c)+':', zorder=5-c, lw=1)
      else:
        axts[c].plot(sweeps, np.nanmean(np.array(compls_sweep[compl_alg]),axis=1), 'C'+str(c), zorder=5-c)
        axts[c].set_yticks(yticks)
      axts[c].set_zorder(5-c)
      axts[c].patch.set_visible(False)
      if c<4:
        axts[c].tick_params(axis='y', labelcolor='C'+str(c))
      if c==1:
        axts[c].yaxis.set_label_position("left")
        axts[c].yaxis.tick_left()
    else:
      if c==0 and alg=='dt':
        lw=1
      elif c==2:
        lw=2
      else:
        lw=1.5
      
      if FIG2:
        ax.plot(sweeps, np.nanmedian(np.array(compls_sweep[compl_alg]),axis=1), zorder=5-c, lw=lw)
        ax.plot(sweeps, np.nanquantile(np.array(compls_sweep[compl_alg]),q=0.25,axis=1), 'C'+str(c)+':', zorder=5-c, lw=1)
        ax.plot(sweeps, np.nanquantile(np.array(compls_sweep[compl_alg]),q=0.75,axis=1), 'C'+str(c)+':', zorder=5-c, lw=1)
      else:
        ax.plot(sweeps, np.nanmean(np.array(compls_sweep[compl_alg]),axis=1), zorder=5-c, lw=lw)
  if alg=='knn':
    ax.set_ylabel('Complexity')
  
  
  if alg!='rf':
    ax.set_ylim([-0.05, 1.05])
  ax.set_xlabel(xlab)
  ax.set_title(title)
  fig.legend(lines, labs, loc='lower center', ncol=len(lines))
  fig.tight_layout()
  fig.subplots_adjust(bottom=.35)
  fig.savefig('figures/compl_demo_emp'+FIG_APDX+'.pdf')

#d=1
#xlab='n'
#ns=np.unique(np.geomspace(10,100,50).astype(int))
#ns=range(10,100)
#for alg, title,ax in zip(['knn','dt','rf'], ['k-Nearest Neighbors', 'Decision Tree', 'Random Forest'], axs):
#  if alg == 'knn':
#    arg=0.35
#  elif alg == 'dt':
#    arg=5
#  elif alg == 'rf':
#    arg=10
#
#  compls_n={}
#  for compl_alg in COMPL_ALGS:
#    compls_n[compl_alg]=[]
#  for n in ns:
#    arg1=round(arg*n) if alg == 'knn' else arg
#    compls_seed={}
#    for compl_alg in COMPL_ALGS:
#      compls_seed[compl_alg]=[]
#    for seed in range(100):
#      np.random.seed(seed)
#      X_tr=np.random.normal(0,1,(n,d))
#      y_tr=np.random.normal(0,1,(n,1))
#      X_te=np.random.normal(0,1,(n,d))
#      S_tr, S_te = ALGS1_DICT[alg](arg1, X_tr, X_te, y_tr, seed)
#      for compl_alg in COMPL_ALGS:
#        compls_seed[compl_alg].append(ALGS_DICT[compl_alg](S_tr, S_te, y_tr))
#    for compl_alg in COMPL_ALGS:
#      compls_n[compl_alg].append(np.nanmean(compls_seed[compl_alg]))
#  if alg=='rf':
#    axts=[ax, ax.twinx(), ax.twinx(), ax.twinx()]
#    if FIG2:
#      axts.append(ax)
#  for c, compl_alg in enumerate(COMPL_ALGS):
#    if alg=='rf':
#      axts[c].plot(ns, compls_n[compl_alg], 'C'+str(c), zorder=5-c)
#      axts[c].set_zorder(5-c)
#      axts[c].patch.set_visible(False)
#      if c<4:
#        axts[c].tick_params(axis='y', labelcolor='C'+str(c))
#      if c==1:
#        axts[c].yaxis.set_label_position("left")
#        axts[c].yaxis.tick_left()
#    else:
#      if c==0 and alg=='dt':
#        lw=1
#      elif c==2:
#        lw=2
#      else:
#        lw=1.5
#      ax.plot(ns, compls_n[compl_alg], zorder=5-c, lw=lw)
#  if alg=='knn':
#    ax.set_ylabel('Complexity')
#  
#  
#  if alg!='rf':
#    ax.set_ylim([-0.05, 1.05])
#  ax.set_xlabel(xlab)
#  ax.set_title(title)
#  fig.legend(lines, labs, loc='lower center', ncol=len(lines))
#  fig.tight_layout()
#  fig.subplots_adjust(bottom=.35)
#  fig.savefig('figures/compl_demo_emp'+FIG_APDX+'.pdf')


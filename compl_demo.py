import numpy as np
from matplotlib import pyplot as plt
import sys
from matplotlib.lines import Line2D
plt.rcParams.update({'pdf.fonttype': 42, 'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman']})

def kern(X,Y,alg,arg):
  c=1
  if alg=='gauss':
    sigma=arg
    X2=np.sum(X**2,1).reshape((-1,1))
    XY=X.dot(Y.T)
    Y2=np.sum(Y**2,1).reshape((-1,1))
    D2=X2-2*XY+Y2.T
    return np.exp(-0.5*D2/sigma**2)
  elif alg=='pol':
    p=arg
    return (c+X@Y.T)**p
  elif alg=='lin':
    return (c+X@Y.T)


def get_gac(K_tr, K_te, lbda):
  n_tr=K_tr.shape[0]
  Kr=np.diag(np.sqrt(1/(1e-8+np.diag(K_tr))))@K_tr@np.diag(np.sqrt(1/(1e-8+np.diag(K_tr))))
  return 1-np.sum(np.square(np.eye(n)-Kr))/(n_tr**2-n_tr)

def get_vne(K_tr, K_te, lbda):
  n_tr=K_tr.shape[0]
  Kr=np.diag(np.sqrt(1/(1e-8+np.diag(K_tr))))@K_tr@np.diag(np.sqrt(1/(1e-8+np.diag(K_tr))))
  s=np.linalg.svd(Kr)[1]
  s/=np.sum(s)
  return -np.sum(s*np.log(s+1e-8))/np.log(n_tr)

def get_enp(K_tr, K_te, lbda):
  n_tr=K_tr.shape[0]
  s=np.linalg.svd(K_tr)[1]
  return np.sum(s/(s+lbda))/n_tr

def get_genpv(K_tr, K_te, lbda):
  n_te,n_tr=K_te.shape
  S_te=K_te@np.linalg.inv(K_tr+lbda*np.eye(n_tr))
  return np.trace(S_te.T@S_te)/n_te

def get_genprx(K_tr, K_te, lbda):
  n_te,n_tr=K_te.shape
  y_te=np.random.normal(0,1,(n_te,1))
  y_tr=np.random.normal(0,1,(n_tr,1))
  lbda+=np.max(np.abs(np.diag(K_tr)))*1e-12
  alpha=np.linalg.solve(K_tr+lbda*np.eye(n_tr),y_tr)
  err_te=np.mean((y_te-K_te@alpha)**2)
  err_tr=np.mean((y_tr-K_tr@alpha)**2)
  opt=err_te-err_tr
  return 1/n_tr*(2*n_tr-1+n_tr*opt-np.sqrt((2*n_tr-1+n_tr*opt)**2-4*(n_tr-1)*n_tr*opt))/2

def get_norm(K_tr,K_te,lbda):
  n_tr=K_tr.shape[0]
  y_tr=np.random.normal(0,1,(n_tr,1))
  lbda+=np.max(np.abs(np.diag(K_tr)))*1e-12
  alpha=np.linalg.solve(K_tr+lbda*np.eye(n_tr),y_tr)
  return np.squeeze(alpha.T@K_tr@alpha)




LBDA=1e-5


ls=np.geomspace(.001,100,100)
ps=range(1,31)
ds=range(1,101)


COMPL_DICT={'GAC': 'compl_gacs', 'ENP': 'compl_enps', 'GENP-V': 'compl_genps', 'vNE': 'compl_vnes', 'GENP-RX': 'compl_rxs', '$\\|\\hat{\\theta}\\|^2_2$': 'compl_norms'}
LABS_DICT={'gac': 'GAC', 'vne': 'vNE', 'enp': 'ENP', 'genpv': 'GENP-V', 'genprx': 'GENP-RX', 'norm': '$\\|\\hat{\\theta}\\|^2_2$'}
ALGS_DICT={'gac': get_gac, 'vne': get_vne, 'enp': get_enp, 'genpv': get_genpv, 'genprx': get_genprx, 'norm': get_norm}

COMPL_ALGS=['gac', 'enp', 'genpv', 'genprx']
FIG_APDX=''
FIG2=False

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

if FIG2:
  COMPL_ALGS=['gac', 'enp', 'genpv', 'genprx', 'vne', 'norm']
  FIG_APDX='2'


lines=[]
labs=[]
for c, compl_alg in enumerate(COMPL_ALGS):
  labs.append(LABS_DICT[compl_alg])
  lines.append(Line2D([0],[0],color='C'+str(c),lw=2))

fig, ax_mat=plt.subplots(2,3,figsize=(11,4))

for alg, title, xlab, sweeps, axs in zip(['lin','pol','gauss'], ['Linear', 'Polynomial', 'Gaussian'], ['d', 'p', 'l'], [ds,ps,ls],[ax_mat[:,0],ax_mat[:,1],ax_mat[:,2]]):
  d=1
  n=50
  ax0_twin1=axs[0].twinx()
  ax1_twin1=axs[1].twinx()
  ax_dict={'gac': axs[0], 'vne': axs[0], 'enp': axs[0], 'genpv': ax0_twin1, 'genprx': axs[0], 'norm': ax0_twin1}
  compls_sweep={}
  for compl_alg in COMPL_ALGS:
    compls_sweep[compl_alg]=[]
  for sweep in sweeps:
    if alg == 'lin':
      d=sweep
    compls_seed={}
    for compl_alg in COMPL_ALGS:
      compls_seed[compl_alg]=[]
    for seed in range(100):
      np.random.seed(seed)
      X_tr=np.random.normal(0,1,(n,d))
      X_te=np.random.normal(0,1,(n,d))
      K_tr=kern(X_tr,X_tr,alg,sweep)
      K_te=kern(X_te,X_tr,alg,sweep)
      for compl_alg in COMPL_ALGS:
        compls_seed[compl_alg].append(ALGS_DICT[compl_alg](K_tr, K_te, LBDA))
    for compl_alg in COMPL_ALGS:
      compls_sweep[compl_alg].append(compls_seed[compl_alg])
  
  for c, compl_alg in enumerate(COMPL_ALGS):
    if FIG2:
      ax_dict[compl_alg].plot(sweeps, np.median(np.array(compls_sweep[compl_alg]),axis=1), 'C'+str(c), zorder=10-c)
      ax_dict[compl_alg].plot(sweeps, np.quantile(np.array(compls_sweep[compl_alg]),q=0.25,axis=1), 'C'+str(c)+':', zorder=10-c, lw=1)
      ax_dict[compl_alg].plot(sweeps, np.quantile(np.array(compls_sweep[compl_alg]),q=0.75,axis=1), 'C'+str(c)+':', zorder=10-c, lw=1)
    else:
      ax_dict[compl_alg].plot(sweeps, np.mean(np.array(compls_sweep[compl_alg]),axis=1), 'C'+str(c), zorder=10-c)
  ax0_twin1.set_yscale('log')
  if alg=='lin':
    axs[0].set_ylabel('Complexity')
  if alg=='gauss':
    axs[0].set_xscale('log')
  axs[0].set_ylim([-0.05, 1.05])
  axs[0].set_xlabel(xlab)
  axs[0].set_title(title)
  fig.tight_layout()
  fig.savefig('figures/compl_demo'+FIG_APDX+'.pdf')
  
  ax_dict={'gac': axs[1], 'vne': axs[1], 'enp': axs[1], 'genpv': ax1_twin1, 'genprx': axs[1], 'norm': ax1_twin1}
  ns=np.unique(np.geomspace(5,100,50).astype(int))
  if alg == 'gauss':
    arg=1
  elif alg == 'pol':
    arg=5
  elif alg == 'lin':
    arg=1
    d=20
  
  compls_n={}
  for compl_alg in COMPL_ALGS:
    compls_n[compl_alg]=[]
  for n in ns:
    compls_seed={}
    for compl_alg in COMPL_ALGS:
      compls_seed[compl_alg]=[]
    for seed in range(100):
      np.random.seed(seed)
      X_tr=np.random.normal(0,1,(n,d))
      X_te=np.random.normal(0,1,(n,d))
      K_tr=kern(X_tr,X_tr,alg,arg)
      K_te=kern(X_te,X_tr,alg,arg)
      for compl_alg in COMPL_ALGS:
        compls_seed[compl_alg].append(ALGS_DICT[compl_alg](K_tr, K_te, LBDA))
    for compl_alg in COMPL_ALGS:
      compls_n[compl_alg].append(compls_seed[compl_alg])
  
  for c, compl_alg in enumerate(COMPL_ALGS):
    if FIG2:
      ax_dict[compl_alg].plot(ns, np.median(np.array(compls_n[compl_alg]),axis=1), 'C'+str(c), zorder=10-c)
      ax_dict[compl_alg].plot(ns, np.quantile(np.array(compls_n[compl_alg]),q=0.25,axis=1), 'C'+str(c)+':', zorder=10-c, lw=1)
      ax_dict[compl_alg].plot(ns, np.quantile(np.array(compls_n[compl_alg]),q=0.75,axis=1), 'C'+str(c)+':', zorder=10-c, lw=1)
    else:
      ax_dict[compl_alg].plot(ns, np.mean(np.array(compls_n[compl_alg]),axis=1), 'C'+str(c), zorder=10-c)
  
  ax1_twin1.set_yscale('log')
  axs[1].set_xscale('log')
  axs[1].set_xlabel('n')
  axs[1].set_ylim([-0.05, 1.05])
  if alg=='lin':
    axs[1].set_ylabel('Complexity')
  
  fig.legend(lines, labs, loc='lower center', ncol=len(lines))
  fig.tight_layout()
  fig.subplots_adjust(bottom=.19)
  fig.savefig('figures/compl_demo'+FIG_APDX+'.pdf')

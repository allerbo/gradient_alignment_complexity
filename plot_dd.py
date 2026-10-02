import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D
plt.rcParams.update({'pdf.fonttype': 42, 'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman']})
plt.rcParams.update({ "text.latex.preamble": r""" \usepackage[T1]{fontenc} \usepackage{lmodern} \usepackage{amsmath} """ })
  
lines=[]
labs=[]
for c,txt in zip([0,1,2], ['Training MSE (left y-axis)', 'Test MSE (left y-axis)', 'EFC (right y-axis)']):
  lines.append(Line2D([0],[0],color='C'+str(c),lw=2))
  labs.append(txt)
  
def forward(p):
  return 1/(1 - np.minimum(1,p)+1e-10)

def inverse(y):
  return 1 - 1/(np.maximum(0,y)+1e-10)

  
ALGS = ['mnist_rff', 'mnist_rf', 'mnist_nn', 'cifar_nn', 'cifar_boost']
TITLES = ['Random Fourier\nFeatures, MNIST', 'Random Forest\nMNIST', 'Neural Network\nMNIST', 'Neural Network\nCIFAR', 'Gradient Boosting\nCIFAR']
FIG2=False
MAX=False

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

fig, ax_mat = plt.subplots(5, 2, figsize=(8, 8))
for i, (alg,title) in enumerate(zip(ALGS,TITLES)):
  axs=ax_mat[i,:]
  with open(alg+'.pkl', 'rb') as f:
    mse_tr_seeds, mse_te_seeds, tot_efc_seeds, max_efc_seeds, compl_proxy, xlabel, x_ticks, x_ticklabels  = pickle.load(f)
  
  efc_seeds= max_efc_seeds if MAX else tot_efc_seeds
  
  mse_tr_mean=np.mean(np.array(mse_tr_seeds),axis=0)
  mse_te_mean=np.mean(np.array(mse_te_seeds),axis=0)
  efc_mean=np.mean(np.array(efc_seeds),axis=0)
  mse_tr_d1=np.quantile(np.array(mse_tr_seeds),q=0.1,axis=0)
  mse_tr_d9=np.quantile(np.array(mse_tr_seeds),q=0.9,axis=0)
  mse_te_d1=np.quantile(np.array(mse_te_seeds),q=0.1,axis=0)
  mse_te_d9=np.quantile(np.array(mse_te_seeds),q=0.9,axis=0)
  efc_d1=np.quantile(np.array(efc_seeds),q=0.1,axis=0)
  efc_d9=np.quantile(np.array(efc_seeds),q=0.9,axis=0)
  _=axs[0].plot(compl_proxy, mse_tr_mean, 'C0', marker="o", markersize=3)
  _=axs[0].plot(compl_proxy, mse_te_mean, 'C1', marker="o", markersize=3)
  if FIG2:
    _=axs[0].plot(compl_proxy, mse_tr_d1, 'C0:')
    _=axs[0].plot(compl_proxy, mse_tr_d9, 'C0:')
    _=axs[0].plot(compl_proxy, mse_te_d1, 'C1:')
    _=axs[0].plot(compl_proxy, mse_te_d9, 'C1:')
  _=axs[0].set_xlabel(xlabel)
  if isinstance(x_ticks, np.ndarray):
    _=axs[0].set_xticks(x_ticks)
    _=axs[0].set_xticklabels(x_ticklabels, fontsize=9)
  elif x_ticks=='log':
    _=axs[0].set_xscale('log')
  _=axs[0].set_ylabel(title)
  
  ax0t=axs[0].twinx()
  _=ax0t.plot(compl_proxy, efc_mean, 'C2', marker="o", markersize=3)
  if FIG2:
    _=ax0t.plot(compl_proxy, efc_d1, 'C2:')
    _=ax0t.plot(compl_proxy, efc_d9, 'C2:')
  
  _=axs[1].plot(efc_mean, mse_tr_mean, 'C0', marker="o", markersize=3)
  _=axs[1].plot(efc_mean, mse_te_mean, 'C1', marker="o", markersize=3)
  if FIG2:
    _=axs[1].plot(efc_mean, mse_tr_d1, 'C0:')
    _=axs[1].plot(efc_mean, mse_tr_d9, 'C0:')
    _=axs[1].plot(efc_mean, mse_te_d1, 'C1:')
    _=axs[1].plot(efc_mean, mse_te_d9, 'C1:')
  _=axs[1].set_xlabel('EFC')

  _=ax0t.set_yscale('function', functions=(forward, inverse))
  if FIG2:
    y_ticks0=np.linspace(forward(np.min(efc_d1)), forward(np.max(efc_d9)),5)
  else:
    y_ticks0=np.linspace(forward(np.min(efc_mean)), forward(np.max(efc_mean)),5)
  y_ticks1=np.linspace(forward(np.min(efc_mean)), forward(np.max(efc_mean)),5)
  p_ticks0 = inverse(y_ticks0)
  p_ticks1 = inverse(y_ticks1)
  _=ax0t.set_yticks(p_ticks0)
  _=ax0t.set_yticklabels([f"{p:.3f}" for p in p_ticks0], color='#1A601A')
  _=axs[1].set_xscale('function', functions=(forward, inverse))
  _=axs[1].set_xticks(p_ticks1)
  _=axs[1].set_xticklabels([f"{p:.3f}" for p in p_ticks1])
  
  fig.add_artist(plt.Line2D([0.543,0.543],[0.095,.98], transform=fig.transFigure,color='black'))
  fig.legend(lines, labs, loc='lower center', ncol=len(lines))
  fig.tight_layout()
  fig.subplots_adjust(bottom=.095)
  if MAX and FIG2:
    fig.savefig("figures/dd2m.pdf")
  elif MAX:
    fig.savefig("figures/ddm.pdf")
  elif FIG2:
    fig.savefig("figures/dd2.pdf")
  else:
    fig.savefig("figures/dd.pdf")


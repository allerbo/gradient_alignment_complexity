import numpy as np
from matplotlib import pyplot as plt
import sys
from matplotlib.lines import Line2D
plt.rcParams.update({'pdf.fonttype': 42, 'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman']})

def f_sin(x):
  y=np.sin(2*np.pi*x)
  return y

def kern_gauss(x,y,sigma):
  return np.exp(-0.5*((x-y.T)/sigma)**2)


def get_gac(K):
  n=K.shape[0]
  Kr=np.diag(np.sqrt(1/(1e-8+np.diag(K))))@K@np.diag(np.sqrt(1/(1e-8+np.diag(K))))
  return 1-np.sum(np.square(np.eye(n)-Kr))/(n**2-n)

def get_enp(K_tr, lbda):
  n_tr=K_tr.shape[0]
  I=np.eye(x_tr.shape[0])
  S_tr=K_tr@np.linalg.inv(K_tr+lbda*I)
  return 1/n_tr*np.trace(S_tr)

def get_fh_and_compls(x_te, x_tr, y_tr, sigma, lbda):
  I=np.eye(x_tr.shape[0])
  K_tr=kern_gauss(x_tr,x_tr,sigma)
  K_te=kern_gauss(x_te,x_tr,sigma)
  fh_te=K_te@np.linalg.solve(K_tr+lbda*I,y_tr)
  return fh_te, get_gac(K_tr), get_enp(K_tr, lbda)

n_tr=20
n_te=1001

SIGMAS=[0.3, 0.03]
LBDAS=[0.0001, 0.65]

np.random.seed(4)
x_tr=np.random.uniform(-1,1,n_tr).reshape((-1,1))
y_tr=f_sin(x_tr)+np.random.normal(0,.1,x_tr.shape)
x_te=np.linspace(-1,1,n_te).reshape((-1,1))
f_te=f_sin(x_te)

lines=[]
labs=[]
fig,ax=plt.subplots(1,1,figsize=(5,3))
_=ax.plot(x_tr,y_tr,'ok', ms=3)
for c, sigma, lbda in zip([1,2], SIGMAS, LBDAS):
  fh_te, gac, enp = get_fh_and_compls(x_te, x_tr, y_tr, sigma, lbda)
  _=ax.plot(x_te,fh_te,'C'+str(c))
  lines.append(Line2D([0],[0],color='C'+str(c),lw=2))
  labs.append(f'$l$={sigma}, $\\lambda$={lbda}, GAC={gac:.2f}, ENP={enp:.2f}')

_=ax.set_xticks([])
_=ax.set_yticks([])

fig.legend(lines, labs, loc='lower center', ncol=1)
fig.tight_layout()
fig.subplots_adjust(bottom=.28)
fig.savefig('figures/pen_demo.pdf')


import numpy as np
import sys

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from flax.training.train_state import TrainState

from load_mnist_cifar import load_cifar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def ma(x, N):
  if N % 2 != 1:
    raise ValueError("N must be odd.")
  x = np.asarray(x, dtype=float)
  kernel = np.ones(N)
  sums = np.convolve(x, kernel, mode="same")
  counts = np.convolve(np.ones_like(x), kernel, mode="same")
  return sums / counts


def init_model(DIM_X, DIM_H, DIM_Y, lr, gamma, seed=0):
  rng, init_rng = jax.random.split(jax.random.PRNGKey(seed), 2)
  model=reg_fl(DIM_H, DIM_Y)
  theta=model.init(init_rng,jnp.ones([1,DIM_X]))
  opt=optax.sgd(lr, gamma)
  model_state = TrainState.create(apply_fn=model.apply, params=theta, tx=opt)
  return model_state

class reg_fl(nn.Module):
  DIM_H: int
  DIM_Y: int
  @nn.compact
  def __call__(self,x):
    x=nn.Dense(self.DIM_H)(x)
    x=nn.activation.relu(x)
    x=nn.Dense(self.DIM_Y)(x)
    return x


@jax.jit
def train_step(model_state, x, y):
  def L2(theta):
    fh = model_state.apply_fn(theta, x)
    return jnp.mean(jnp.square(fh-y))
  
  loss, grads = jax.value_and_grad(L2)(model_state.params)
  model_state = model_state.apply_gradients(grads=grads)
  return model_state, loss

@jax.jit
def get_efc(X, model_state):
  def fh_th(X,theta,model_state):
    fh = model_state.apply_fn(theta,X)
    return fh
  
  jac_dict= jax.jacrev(fh_th,argnums=1)(X,model_state.params,model_state)['params']
  n=X.shape[0]
  K=jnp.zeros((n,n))
  for k1 in jac_dict.keys():
    for k2 in jac_dict[k1].keys():
      Ph_s=jac_dict[k1][k2].reshape(n,-1) #why no squeeze here?
      K+=Ph_s@Ph_s.T
  
  Kr=jnp.diag(jnp.sqrt(1/(1e-8+jnp.diag(K))))@K@jnp.diag(jnp.sqrt(1/(1e-8+jnp.diag(K))))
  return 1-jnp.sum(jnp.square(jnp.eye(n)-Kr))/(n**2-n)

@jax.jit
def get_mse(y,fh):
  return jnp.mean((y-fh)**2)


N_MBS=10
N_TR=1000
N_TE=1000


lr=0.0001
gamma=0.99
EPOCHS=1501

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

batch_idxs=np.array_split(range(N_TR),N_MBS)

DIM_H=20
efc_seeds=[]
mse_tr_seeds=[]

for seed in range(10):
  print(seed)
  X_tr, y_tr, X_te, y_te=load_cifar(n_tr=N_TR,n_te=N_TE,seed=seed)
  n_tr,p=X_tr.shape
  n_te=X_te.shape[0]
  dim_y=y_tr.shape[1]
  mse_trs=[]
  model_state = init_model(p,DIM_H,dim_y, lr, gamma)
  efcs=[]
  mse_trs=[]
  for epoch in range(EPOCHS):
    for bi in batch_idxs:
      model_state, loss = train_step(model_state, X_tr[bi,:], y_tr[bi,:])
    efcs.append(get_efc(np.random.permutation(X_tr)[:20,:], model_state))
    mse_trs.append(get_mse(y_tr,model_state.apply_fn(model_state.params,X_tr)))
  
  efc_seeds.append(efcs)
  mse_tr_seeds.append(mse_trs)


efc_mean=np.mean(np.array(efc_seeds),axis=0)
efc_d1=np.quantile(np.array(efc_seeds),q=0.1,axis=0)
efc_d9=np.quantile(np.array(efc_seeds),q=0.9,axis=0)
mse_tr_mean=np.mean(np.array(mse_tr_seeds),axis=0)
mse_tr_d1=np.quantile(np.array(mse_tr_seeds),q=0.1,axis=0)
mse_tr_d9=np.quantile(np.array(mse_tr_seeds),q=0.9,axis=0)

lines=[]
labs=[]
for c,txt in zip([0,1], ['Training MSE (left y-axis)', 'EFC (right y-axis)']):
  lines.append(Line2D([0],[0],color='C'+str(c),lw=2))
  labs.append(txt)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
axt=ax.twinx()
_=axt.plot(range(EPOCHS), mse_tr_mean,'C0')
_=axt.plot(range(EPOCHS), mse_tr_d1,'C0:')
_=axt.plot(range(EPOCHS), mse_tr_d9,'C0:')
_=ax.plot(range(EPOCHS), ma(efc_mean,11), 'C1')
_=ax.plot(range(EPOCHS), ma(efc_d1,11), 'C1:')
_=ax.plot(range(EPOCHS), ma(efc_d9,11), 'C1:')
_=ax.set_xlabel('Epoch')
_=ax.set_title('CIFAR')
fig.legend(lines, labs, loc='lower center', ncol=len(lines))
fig.tight_layout()
fig.subplots_adjust(bottom=.2)
fig.savefig('figures/cifar_20.pdf')


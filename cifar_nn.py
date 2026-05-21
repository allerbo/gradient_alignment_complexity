import numpy as np
import sys
import pickle

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from flax.training.train_state import TrainState

from load_mnist_cifar import load_cifar

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
def get_loss(model_state, x, y):
  fh = model_state.apply_fn(model_state.params, x)
  return jnp.mean(jnp.square(fh-y))

@jax.jit
def get_gac(X, model_state):
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
EPOCHS=10000
DIM_HS=np.unique(np.geomspace(1,3000,20).astype(int))
print(DIM_HS)

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

batch_idxs=np.array_split(range(N_TR),N_MBS)

mse_tr_seeds=[]
mse_te_seeds=[]
tot_gac_seeds=[]
max_gac_seeds=[]


for seed in range(10):
  print(seed)
  X_tr, y_tr, X_te, y_te=load_cifar(n_tr=N_TR,n_te=N_TE,seed=seed)
  n_tr,p=X_tr.shape
  n_te=X_te.shape[0]
  dim_y=y_tr.shape[1]
  mse_trs=[]
  mse_tes=[]
  tot_gacs=[]
  max_gacs=[]
  for DIM_H in DIM_HS:
    model_state = init_model(p,DIM_H,dim_y, lr, gamma)
    gacs=[]
    loss_diffs=[]
    loss_old = get_loss(model_state, X_tr, y_tr)
    for epoch in range(EPOCHS):
      for bi in batch_idxs:
        model_state, loss = train_step(model_state, X_tr[bi,:], y_tr[bi,:])
      if epoch % 5 == 0:
        gacs.append(get_gac(np.random.permutation(X_tr)[:20,:], model_state))
        loss_diffs.append(np.maximum(0,loss_old - loss))
        loss_old=loss
    
    mse_trs.append(get_mse(y_tr,model_state.apply_fn(model_state.params,X_tr)))
    mse_tes.append(get_mse(y_te,model_state.apply_fn(model_state.params,X_te)))
    tot_gacs.append(np.mean(np.array(gacs)*np.array(loss_diffs))/np.mean(loss_diffs))
    max_gacs.append(np.max(gacs))
    print(f'seed={seed} | h={DIM_H:<3} | train loss={mse_trs[-1]:.4f} | test loss={mse_tes[-1]:.4f} | tot gac={tot_gacs[-1]:.4f} | max gac={max_gacs[-1]:.4f} | ')
  
  mse_tr_seeds.append(mse_trs)
  mse_te_seeds.append(mse_tes)
  tot_gac_seeds.append(tot_gacs)
  max_gac_seeds.append(max_gacs)
  
  with open('cifar_nn.pkl','wb') as f:
    pickle.dump((mse_tr_seeds, mse_te_seeds, tot_gac_seeds, max_gac_seeds, DIM_HS, 'Hidden Units', 'log', None), f)


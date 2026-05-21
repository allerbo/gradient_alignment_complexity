import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
import sys
from load_mnist_cifar import load_mnist
import pickle


class MLP(nn.Module):
  hidden: int
  out: int
  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.hidden)(x)
    x = nn.relu(x)
    x = nn.Dense(self.out)(x)
    return x

def create_state(rng, model, lr, momentum, sample):
  params = model.init(rng, sample)["params"]
  param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
  tx = optax.sgd(lr, momentum=momentum)
  return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx), param_count


def squared_loss(logits, labels):
  y = jax.nn.one_hot(labels, logits.shape[-1])
  return jnp.mean((logits - y) ** 2)


@jax.jit
def get_gac(state,x):
  def fh(x,params,state):
    logits = state.apply_fn({"params": params}, x)
    return logits
  
  jac_dict= jax.jacrev(fh,argnums=1)(x,state.params,state)
  n=x.shape[0]
  K=jnp.zeros((n,n))
  for k1 in jac_dict.keys():
    for k2 in jac_dict[k1].keys():
      Ph_s=jac_dict[k1][k2].reshape(n,-1) #why no squeeze here?
      K+=Ph_s@Ph_s.T
  
  Kr=jnp.diag(jnp.sqrt(1/(1e-8+jnp.diag(K))))@K@jnp.diag(jnp.sqrt(1/(1e-8+jnp.diag(K))))
  return 1-jnp.sum(jnp.square(jnp.eye(n)-Kr))/(n**2-n)


@jax.jit
def train_step(state, x, y):
  def loss_fn(params):
    logits = state.apply_fn({"params": params}, x)
    loss = squared_loss(logits, y)
    return loss
  
  loss, grads = jax.value_and_grad(loss_fn)(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss


def train_model(rng, X_tr, y_tr, X_te, y_te, hidden, dim_y, epochs, lr, gamma, reuse_params=None):
  model = MLP(hidden=hidden, out=dim_y)
  state, param_count = create_state(rng, model, lr, gamma, jnp.ones((1, X_tr.shape[1])))
  
  if param_count<10000 and reuse_params is not None:
    params = state.params
    new_params = {}
    for layer in params:
      new_params[layer] = {}
      for name in params[layer]:
        new_w = params[layer][name]
        old_w = reuse_params.get(layer, {}).get(name, None)
        if old_w is None:
          new_params[layer][name] = new_w
          continue
        
        if new_w.shape == old_w.shape:
          new_params[layer][name] = old_w
          continue
        # partial reuse (copy overlapping block)
        slices = tuple(slice(0, min(a, b)) for a, b in zip(old_w.shape, new_w.shape))
        updated = new_w.at[slices].set(old_w[slices])
        new_params[layer][name] = updated
    state = state.replace(params=new_params)
  
  gacs=[]
  loss_diffs=[]
  loss_old=squared_loss(state.apply_fn({"params": state.params}, X_tr), y_tr)
  for epoch in range(epochs):
    state, loss = train_step(state, X_tr, y_tr)
    if epoch % 5 == 0:
      gacs.append(get_gac(state, np.random.permutation(X_tr)[:20,:]))
      loss_diffs.append(np.maximum(0,loss_old-loss))
      loss_old=loss
  
  logits_tr = model.apply({"params": state.params}, X_tr)
  logits_te = model.apply({"params": state.params}, X_te)
  mse_tr = float(squared_loss(logits_tr, y_tr))
  mse_te = float(squared_loss(logits_te, y_te))
  return state.params, mse_tr, mse_te, param_count, np.mean(np.array(gacs)*np.array(loss_diffs))/np.mean(loss_diffs), np.max(gacs)



def run_sweep(seed, X_tr, y_tr, X_te, y_te, dim_hs, dim_y, epochs, lr, gamma, reuse=False):
  rng = jax.random.PRNGKey(0)
  mse_trs = []
  mse_tes = []
  tot_gacs = []
  max_gacs = []
  prev_params = None
  for h in dim_hs:
    rng, sub = jax.random.split(rng)
    params, mse_tr, mse_te, param_count, tot_gac, max_gac = train_model(sub, X_tr, y_tr, X_te, y_te, h, dim_y, epochs, lr, gamma, reuse_params=prev_params if reuse else None)
    prev_params = params
    mse_trs.append(mse_tr)
    mse_tes.append(mse_te)
    tot_gacs.append(tot_gac)
    max_gacs.append(max_gac)
    print(f'seed={seed} | h={h:<4} | params={param_count:<5} | train loss={mse_tr:.4f} | test loss={mse_te:.4f} | tot gac={tot_gac:.4f} | max gac={max_gac:.4f} | ')
  return np.array(mse_trs), np.array(mse_tes), np.array(tot_gacs), np.array(max_gacs)


seed=0
DIM_HS =np.unique(np.geomspace(3,500,20).astype(int))
EPOCHS=100000
DIM_Y=10

LR=0.01
GAMMA=0.95

print(DIM_HS)

mse_tr_seeds = []
mse_te_seeds = []
tot_gac_seeds = []
max_gac_seeds = []
for seed in range(10):
  X_tr, y_tr, X_te, y_te = load_mnist(n_tr=1000, n_te=1000,seed=seed, down=True)
  mse_trs, mse_tes, tot_gacs, max_gacs = run_sweep(seed, X_tr, y_tr, X_te, y_te, DIM_HS, DIM_Y, EPOCHS, LR, GAMMA, reuse=True)
  mse_tr_seeds.append(mse_trs)
  mse_te_seeds.append(mse_tes)
  tot_gac_seeds.append(tot_gacs)
  max_gac_seeds.append(max_gacs)
  
  with open('mnist_nn.pkl','wb') as f:
    pickle.dump((mse_tr_seeds, mse_te_seeds, tot_gac_seeds, max_gac_seeds, DIM_HS, 'Hidden Units', 'log', None), f)


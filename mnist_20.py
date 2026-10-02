import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
import sys
from load_mnist_cifar import load_mnist
import pickle
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
def get_efc(state,x):
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


def train_model(rng, X_tr, y_tr, X_te, y_te, hidden, dim_y, epochs, lr, gamma):
  model = MLP(hidden=hidden, out=dim_y)
  state, param_count = create_state(rng, model, lr, gamma, jnp.ones((1, X_tr.shape[1])))
  
  efcs=[]
  losses=[]
  for epoch in range(epochs):
    state, loss = train_step(state, X_tr, y_tr)
    efcs.append(get_efc(state, np.random.permutation(X_tr)[:20,:]))
    losses.append(loss)
  
  return state.params, efcs, losses



def run_sweep(seed, X_tr, y_tr, X_te, y_te, dim_h, dim_y, epochs, lr, gamma):
  rng = jax.random.PRNGKey(0)
  rng, sub = jax.random.split(rng)
  params, efcs, losses = train_model(sub, X_tr, y_tr, X_te, y_te, dim_h, dim_y, epochs, lr, gamma)
  return efcs, losses


EPOCHS=50001
DIM_Y=10

LR=0.01
GAMMA=0.95


DIM_H=20
efc_seeds = []
loss_seeds = []
for seed in range(10):
  print(seed)
  X_tr, y_tr, X_te, y_te = load_mnist(n_tr=1000, n_te=1000,seed=seed, down=True)
  efcs, losses = run_sweep(seed, X_tr, y_tr, X_te, y_te, DIM_H, DIM_Y, EPOCHS, LR, GAMMA)
  efc_seeds.append(efcs)
  loss_seeds.append(losses)


efc_mean=np.mean(np.array(efc_seeds),axis=0)
efc_d1=np.quantile(np.array(efc_seeds),q=0.1,axis=0)
efc_d9=np.quantile(np.array(efc_seeds),q=0.9,axis=0)
mse_tr_mean=np.mean(np.array(loss_seeds),axis=0)
mse_tr_d1=np.quantile(np.array(loss_seeds),q=0.1,axis=0)
mse_tr_d9=np.quantile(np.array(loss_seeds),q=0.9,axis=0)

lines=[]
labs=[]
for c,txt in zip([0,1], ['Training MSE (left y-axis)', 'EFC (right y-axis)']):
  lines.append(Line2D([0],[0],color='C'+str(c),lw=2))
  labs.append(txt)

idx=np.arange(0,EPOCHS,100)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
axt=ax.twinx()
_=axt.plot(range(EPOCHS), mse_tr_mean,'C0')
_=axt.plot(range(EPOCHS), mse_tr_d1,'C0:')
_=axt.plot(range(EPOCHS), mse_tr_d9,'C0:')
_=ax.plot(range(EPOCHS), ma(efc_mean,31), 'C1')
_=ax.plot(np.arange(EPOCHS)[idx], ma(efc_d1,31)[idx], 'C1:')
_=ax.plot(np.arange(EPOCHS)[idx], ma(efc_d9,31)[idx], 'C1:')
_=ax.set_xlabel('Epoch')
_=ax.set_title('MNIST')
fig.legend(lines, labs, loc='lower center', ncol=len(lines))
fig.tight_layout()
fig.subplots_adjust(bottom=.2)
fig.savefig('figures/mnist_20.pdf')


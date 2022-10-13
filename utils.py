import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from flax import struct
from flax.jax_utils import replicate
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
import optax
from typing import Any, Callable
from itertools import product
from tqdm import tqdm

import matplotlib.pyplot as plt
# plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (4, 3)
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

'''
See ./1_construct_nn_with_haiku !
'''

class Trainer(struct.PyTreeNode):
    apply_fn: Callable = struct.field(pytree_node=False)
    tx: Callable = struct.field(pytree_node=False)
    step: int
    params: Any = None
    state: Any = None
    opt_state: Any = None
    
    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        opt_state = tx.init(params)
        return cls(step=0, apply_fn=apply_fn, params=params, tx=tx, opt_state=opt_state, **kwargs)
    
    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step+1, params=new_params, opt_state=new_opt_state, **kwargs)
      
def save_ckpt(trainer, path):
  save_checkpoint(path, trainer, trainer.step, overwrite=True)

load_ckpt = restore_checkpoint

def make_forward(model):

  def _forward(*args, **kwargs):
    return model()(*args, **kwargs)

  return hk.transform(_forward)

def make_forward_with_state(model):

  def _forward(*args, **kwargs):
    return model()(*args, **kwargs)

  return hk.transform_with_state(_forward)

def params_to_vec(param, unravel=False):
    vec_param, unravel_fn = jax.flatten_util.ravel_pytree(param)
    if unravel:
        return vec_param, unravel_fn
    else:
        return vec_param
      
def unreplicate(tree, i=0):
  """Returns a single instance of a replicated array."""
  return jax.tree_util.tree_map(lambda x: x[i], tree)

'''
See ./3_image_classification !
'''

def create_lr_sched(num_epoch, num_train, batch_size, warmup_ratio, peak_lr):
  total_step = num_epoch * (num_train // batch_size)
  warmup_step = int(total_step * warmup_ratio)
  return optax.warmup_cosine_decay_schedule(0.0, peak_lr, warmup_step, total_step, 0.0)


def compute_acc_batch(trainer, batch):  
  logit, state = trainer.apply_fn(trainer.params, trainer.state, None, batch['x'], train=False)
  
  acc = (jnp.argmax(logit, axis=-1) == jnp.argmax(batch['y'], axis=-1)).astype(int).mean()
  
  return acc

compute_acc_batch_pmapped = jax.pmap(compute_acc_batch, axis_name='batch')

def compute_acc_dataset(trainer, dataset):
  acc = 0
  for batch in dataset:
      acc_batch = compute_acc_batch_pmapped(trainer, batch)
      acc += np.mean(acc_batch)
  acc /= len(dataset)
  return acc

def compute_loss_batch(trainer, batch):  
  logit, state = trainer.apply_fn(trainer.params, trainer.state, None, batch['x'], train=True)
  
  log_prob = jax.nn.log_softmax(logit)
  loss = - (log_prob * batch['y']).sum(axis=-1).mean()
  
  return loss

compute_loss_batch_pmapped = jax.pmap(compute_loss_batch, axis_name='batch')

def compute_loss_dataset(trainer, dataset):
  loss = 0
  for batch in dataset:
      loss_batch = compute_loss_batch_pmapped(trainer, batch)
      loss += np.mean(loss_batch)
  loss /= len(dataset)
  return loss

'''
See ./4_visualizing_loss_landscapes !
'''

def filter_normalize_module(module_name, name, value):
    # filter normalization for Conv & FC layers
    if (name=='w') and (len(value.shape)==4):
        scale = 1./jnp.sqrt(jnp.sum(value**2, axis=(0,1,2), keepdims=True))
    elif (name=='w') and (len(value.shape)==2):
        scale = 1./jnp.sqrt(jnp.sum(value**2, axis=0, keepdims=True))
    else:
        # mask bias & params of normalization layers
        scale = 0.
    value = value * scale
    return value

def scale_pert(pert, param):
    if len(pert.shape)==4:
        scale = jnp.sqrt(jnp.sum(param**2, axis=(0,1,2), keepdims=True))
    elif len(pert.shape)==2:
        scale = jnp.sqrt(jnp.sum(param**2, axis=0, keepdims=True))
    else:
        scale = 0.
    pert = pert * scale
    return pert

def filter_normalize_pert(pert, params):
    pert = hk.data_structures.map(filter_normalize_module, pert)
    pert = jax.tree_util.tree_map(scale_pert, pert, params)
    return pert

def loss_landscape_visualization(
  trainer, 
  dataset, 
  seed=42,
  num_points=10,
  x_vec=None,
  y_vec=None,
  filter_normalized=False,
  title='loss landscape',
  ):
  
  x = np.linspace(-1, 1, num_points)
  y = np.linspace(-1, 1, num_points)
  xv, yv = np.meshgrid(x, y)
  
  rng = jax.random.PRNGKey(seed)
  
  # sample random direction
  vec_params, unravel_fn = params_to_vec(trainer.params, True)
  rng, rng_x, rng_y = jax.random.split(rng, 3)
  x_vec = x_vec or jax.random.normal(rng_x, vec_params.shape)
  y_vec = y_vec or jax.random.normal(rng_y, vec_params.shape)

  if filter_normalized:
    # normalize random vector
    x_vec = params_to_vec(filter_normalize_pert(unravel_fn(x_vec), trainer.params))
    y_vec = params_to_vec(filter_normalize_pert(unravel_fn(y_vec), trainer.params))

  z = np.zeros_like(xv)
  for i,j in tqdm(list(product(range(num_points), repeat=2))):
    # define perturbation
    alpha, beta = x[i], y[j]
    pert = alpha * x_vec + beta * y_vec
    perturbed_params = vec_params + pert
    perturbed_trainer = trainer.replace(params=unravel_fn(perturbed_params))
    acc_te = compute_loss_dataset(replicate(perturbed_trainer), dataset)
    z[i][j] = acc_te
      
  contour = plt.contour(xv, yv, z, cmap='coolwarm_r')
  plt.clabel(contour, inline=True, fontsize=8)
  plt.title(f'{title}', fontsize=10)
  plt.show()
  
  fig = plt.figure(figsize=(6, 4))
  ax = fig.add_subplot(projection='3d')
  ax.plot_surface(xv, yv, z, cmap='coolwarm_r', rstride=1, cstride=1)
  ax.plot_wireframe(xv, yv, z, color='white', linewidth=0.1, rstride=1, cstride=1)
  ax.set_title(f'{title}', fontsize=10)
  plt.tight_layout()
  plt.show()
  return None

def hvp_batch(v, trainer, batch, use_connect=False):
  vec_params, unravel_fn = params_to_vec(trainer.params, True)
  
  if use_connect:
    multiplier = vec_params
  else:
    multiplier = jnp.ones_like(vec_params)
  
  def loss(params):
    logit, state = trainer.apply_fn(params, trainer.state, None, batch['x'], train=True)
    log_prob = jax.nn.log_softmax(logit)
    return - (log_prob * batch['y']).sum(axis=-1).mean()
  
  gvp, hvp = jax.jvp(jax.grad(loss), [trainer.params], [unravel_fn(v*multiplier)])
  return params_to_vec(hvp)
  
hvp_batch_p = jax.pmap(hvp_batch, static_broadcasted_argnums=(3,))

def hvp(v, trainer, dataset, use_batch, use_connect=False):
    if use_batch:
      res = hvp_batch_p(replicate(v), trainer, dataset[0], use_connect).mean(axis=0)
    else:
      res = 0.
      for batch in dataset:
        res += hvp_batch_p(replicate(v), trainer, batch, use_connect).mean(axis=0)
      res = res / len(dataset)
    return res

def lanczos(trainer, dataset, rand_proj_dim=10, seed=42, use_batch=False, use_connect=False):
    
    rng = jax.random.PRNGKey(seed)
    vec_params, unravel_fn = params_to_vec(unreplicate(trainer).params, True)
    
    tridiag = jnp.zeros((rand_proj_dim, rand_proj_dim))
    vecs = jnp.zeros((rand_proj_dim, len(vec_params)))
    
    init_vec = jax.random.normal(rng, shape=vec_params.shape)
    init_vec = init_vec / jnp.linalg.norm(init_vec)
    vecs = vecs.at[0].set(init_vec)
    
    beta = 0
    for i in tqdm(range(rand_proj_dim)):
      v = vecs[i, :]
      if i == 0:
        v_old = 0
      else:
        v_old = vecs[i-1, :]
      
      w = hvp(v, trainer, dataset, use_batch, use_connect)
      w = w - beta * v_old
      
      alpha = jnp.dot(w, v)
      tridiag = tridiag.at[i, i].set(alpha)
      w = w - alpha * v
      
      for j in range(i):
        tau = vecs[j, :]
        coef = np.dot(w, tau)
        w += - coef * tau
          
      beta = jnp.linalg.norm(w)
      
      if (i + 1) < rand_proj_dim:
        tridiag = tridiag.at[i, i+1].set(beta)
        tridiag = tridiag.at[i+1, i].set(beta)
        vecs = vecs.at[i+1].set(w/beta)
        
    return tridiag, vecs
  
def visualize_eigenspectrum(
  trainer, 
  dataset, 
  num_iter=100,
  seed=42,
  use_batch=False,
  use_connect=False,
  title='Eigenspectrum of Hessian'
  ):
  
  tridiag, vecs = lanczos(replicate(trainer), dataset, num_iter, seed, use_batch, use_connect)
  eigval, eigvec = np.linalg.eigh(tridiag)
  eigval = np.sort(eigval)
  
  import seaborn as sns
  plt.style.use('ggplot')
  
  sns.histplot(eigval, color='teal', bins=30, kde=True)
  plt.title(f'{title}', fontsize=10)
  plt.tight_layout()
  plt.show()
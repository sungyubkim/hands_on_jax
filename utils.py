import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from flax import struct
from flax.jax_utils import replicate
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
import optax
from typing import Any, Callable

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
import haiku as hk
import jax
from typing import Any, Callable
from flax import struct
import optax
from flax.jax_utils import replicate

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
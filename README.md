# hands_on_jax
Simple codes for JAX in practice

## 0. JAX as numpy with auto-differentiation

* Linearization of function
  
  ![](./figs/linearization.png)

## 1. Construct Neural Networks with Haiku

* Visualizing MLP structure
  
  ![](./figs/mlp.svg)

## 2. Stochastic Models

* Monte-Carlo DropOut (MCDO)
  
  ![](./figs/mcdo.png)

## 3. Image Classification

* Learning rate scheduling
  
  ![](./figs/lr_sched.png)

## 4. Loss landscape

* Visualizing loss landscape with filter normalization
  
  | Visualization method | with weight decay | withour weight decay |
  |:---:|:---:|:---:|
  | Filter Normalization | ![](./figs/wd_fn.png) | ![](./figs/wowd_fn.png) |
  | Random vector | ![](./figs/wd_rv.png) | ![](./figs/wowd_rv.png) |

* Eigenspectrum of Hessian with stochastic Lanczos iteration

  ![](figs/hessian_eigenspectrum.png)
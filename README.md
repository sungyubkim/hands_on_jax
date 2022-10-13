# hands_on_jax
Simple codes for JAX in practice

## 0. JAX as numpy with auto-differentiation

* Linearization of function
  
  ![Alt text](./figs/linearization.png)

## 1. Construct Neural Networks with Haiku

* Visualizing MLP structure
  
  ![Alt text](./figs/mlp.svg)

## 2. Stochastic Models

* Monte-Carlo DropOut (MCDO)
  
  ![Alt text](./figs/mcdo.png)

## 3. Image Classification

* Learning rate scheduling
  
  ![Alt text](./figs/lr_sched.png)

## 4. Loss landscape

* Visualizing loss landscape with filter normalization
  
  | Visualization method | with weight decay | without weight decay |
  |:---:|:---:|:---:|
  | Filter Normalization | ![Alt text](./figs/wd_fn.png) | ![Alt text](./figs/wowd_fn.png) |
  | Random vector | ![Alt text](./figs/wd_rv.png) | ![Alt text](./figs/wowd_rv.png) |

* Eigenspectrum of Hessian with stochastic Lanczos iteration

  | Eigenspectrum | with weight decay | without weight decay |
  |:---:|:---:|:---:|
  | [Connectivitiy Hessian](https://arxiv.org/abs/2209.15208) | ![Alt text](./figs/wd_chess.png) | ![Alt text](./figs/wowd_chess.png) |
  | Hessian | ![Alt text](./figs/wd_hess.png) | ![Alt text](./figs/wowd_hess.png) |
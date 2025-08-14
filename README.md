# PINN for the 1D Heat Equation

This project implements a **Physics-Informed Neural Network (PINN)** to solve the 1D heat/diffusion equation with fixed temperature boundaries and a sinusoidal initial condition.  
The PINN leverages both the **governing PDE** and **known conditions** to learn an accurate solution without requiring large labeled datasets.

## Problem Statement
We solve:
$[
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}, \quad x \in (0,L),\ t > 0
]$
with:
$[
u(x,0) = 30 \sin\left(\frac{\pi x}{L}\right), \quad u(0,t) = u(L,t) = 0
]$
and validate against the analytical solution.

## Features
- **Data sampling**:
  - Random collocation points for PDE residual minimization
  - Initial condition (IC) points at \( t=0 \)
  - Boundary condition (BC) points at \( x=0 \) and \( x=L \)
- **Model**: Fully-connected neural network with Tanh activations
- **Loss Function**: Combines physics loss, IC loss, and BC loss
- **Training**:
  - Adam optimizer with learning rate scheduler
  - L-BFGS refinement for convergence
- **Evaluation**:
  - Compare with analytical solution
  - Plot predicted field, analytical field, and absolute error
  - Report mean squared error

## Reported metrics :
- Final total loss after L-BFGS: $\sim10^{-6}$
- Mean squared error vs. analytical: $\sim9\times10^{-6}$

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib


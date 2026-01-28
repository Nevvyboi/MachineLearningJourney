<div align="center">

# ⚙️ Optimization Deep Dive

![Chapter](https://img.shields.io/badge/Chapter-13-blue?style=for-the-badge)
![Topic](https://img.shields.io/badge/Topic-Training%20%7C%20Tuning-green?style=for-the-badge)

*Optimizers, Learning Rates, Regularization & Best Practices*

---

</div>

# Part XVI: Deep Dive into Optimization and Training

---

## Chapter 52: Optimization Algorithms

### 52.1 Gradient Descent Variants

```
┌─────────────────────────────────────────────────────────────────────┐
│                GRADIENT DESCENT COMPARISON                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  BATCH GRADIENT DESCENT                                            │
│  - Uses ALL samples per update                                     │
│  - Stable but slow                                                 │
│  - Memory intensive for large datasets                             │
│                                                                     │
│  STOCHASTIC GRADIENT DESCENT (SGD)                                 │
│  - Uses ONE sample per update                                      │
│  - Noisy but fast                                                  │
│  - Can escape local minima                                         │
│                                                                     │
│  MINI-BATCH GRADIENT DESCENT                                       │
│  - Uses BATCH_SIZE samples per update                              │
│  - Best of both worlds                                             │
│  - Typical batch sizes: 32, 64, 128, 256                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 52.2 Optimizer Implementations

```python
import numpy as np
import torch

class SGD:
    """
    Stochastic Gradient Descent with momentum.
    
    v_t = β * v_{t-1} + (1-β) * g_t
    θ_t = θ_{t-1} - α * v_t
    """
    
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # Initialize velocity
        self.velocity = [torch.zeros_like(p) for p in self.params]
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad.data
            
            # Weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data
            
            # Momentum
            if self.momentum != 0:
                self.velocity[i] = self.momentum * self.velocity[i] + grad
                
                if self.nesterov:
                    grad = grad + self.momentum * self.velocity[i]
                else:
                    grad = self.velocity[i]
            
            # Update parameters
            p.data -= self.lr * grad


class Adam:
    """
    Adam: Adaptive Moment Estimation.
    
    m_t = β1 * m_{t-1} + (1-β1) * g_t           (first moment)
    v_t = β2 * v_{t-1} + (1-β2) * g_t²          (second moment)
    m̂_t = m_t / (1 - β1^t)                      (bias correction)
    v̂_t = v_t / (1 - β2^t)                      (bias correction)
    θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
    """
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moments
        self.m = [torch.zeros_like(p) for p in self.params]  # First moment
        self.v = [torch.zeros_like(p) for p in self.params]  # Second moment
        self.t = 0  # Time step
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self):
        self.t += 1
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad.data
            
            # Weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)
            
            # Bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)


class AdamW:
    """
    AdamW: Adam with decoupled weight decay.
    
    Properly implements weight decay (not L2 regularization).
    """
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self):
        self.t += 1
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad.data
            
            # Update moments (without weight decay in gradient)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Weight decay (decoupled)
            p.data -= self.lr * self.weight_decay * p.data
            
            # Adam update
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)


class RMSprop:
    """
    RMSprop: Root Mean Square Propagation.
    
    v_t = β * v_{t-1} + (1-β) * g_t²
    θ_t = θ_{t-1} - α * g_t / (√v_t + ε)
    """
    
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad.data
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data
            
            # Update running average of squared gradients
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * grad.pow(2)
            
            # Update parameters
            p.data -= self.lr * grad / (self.v[i].sqrt() + self.eps)


class LAMB:
    """
    LAMB: Layer-wise Adaptive Moments optimizer for Batch training.
    
    Good for large batch training.
    """
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self):
        self.t += 1
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad.data
            
            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Adam-style update
            update = m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * p.data
            
            # Layer-wise trust ratio
            param_norm = p.data.norm()
            update_norm = update.norm()
            
            if param_norm > 0 and update_norm > 0:
                trust_ratio = param_norm / update_norm
            else:
                trust_ratio = 1.0
            
            # Update
            p.data -= self.lr * trust_ratio * update


# Comparison function
def compare_optimizers(model_fn, data, epochs=100):
    """Compare different optimizers on the same problem."""
    import matplotlib.pyplot as plt
    
    optimizers = {
        'SGD': lambda p: SGD(p, lr=0.1, momentum=0.9),
        'Adam': lambda p: Adam(p, lr=0.001),
        'AdamW': lambda p: AdamW(p, lr=0.001, weight_decay=0.01),
        'RMSprop': lambda p: RMSprop(p, lr=0.01),
    }
    
    results = {}
    
    for name, opt_fn in optimizers.items():
        model = model_fn()
        optimizer = opt_fn(model.parameters())
        
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        results[name] = losses
    
    # Plot
    plt.figure(figsize=(10, 6))
    for name, losses in results.items():
        plt.plot(losses, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimizer Comparison')
    plt.legend()
    plt.yscale('log')
    
    return results


print("Optimizer Comparison:")
print("=" * 60)
print("""
SGD + Momentum:
- Simple and effective
- Requires tuning learning rate
- Good generalization

Adam:
- Adaptive learning rates per parameter
- Less sensitive to learning rate choice
- May not generalize as well as SGD

AdamW:
- Proper weight decay implementation
- Better than Adam for transformers
- Default choice for many tasks

RMSprop:
- Good for RNNs
- Handles non-stationary objectives

LAMB:
- For very large batch training
- Layer-wise trust ratios
""")
```

### 52.3 Learning Rate Schedulers

```python
import math

class LearningRateScheduler:
    """Base class for learning rate schedulers."""
    
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [self.optimizer.lr]
    
    def get_lr(self):
        raise NotImplementedError
    
    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        
        lr = self.get_lr()
        self.optimizer.lr = lr
        return lr


class StepLR(LearningRateScheduler):
    """
    Step learning rate decay.
    
    Decay LR by gamma every step_size epochs.
    """
    
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return self.base_lrs[0] * (self.gamma ** (self.last_epoch // self.step_size))


class ExponentialLR(LearningRateScheduler):
    """
    Exponential learning rate decay.
    
    LR_t = LR_0 * gamma^t
    """
    
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return self.base_lrs[0] * (self.gamma ** self.last_epoch)


class CosineAnnealingLR(LearningRateScheduler):
    """
    Cosine annealing learning rate.
    
    LR_t = LR_min + 0.5 * (LR_max - LR_min) * (1 + cos(π * t / T))
    """
    
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return self.eta_min + (self.base_lrs[0] - self.eta_min) * \
               (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2


class WarmupCosineScheduler(LearningRateScheduler):
    """
    Linear warmup followed by cosine annealing.
    
    Common for transformers.
    """
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return self.base_lrs[0] * (self.last_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return self.min_lr + (self.base_lrs[0] - self.min_lr) * \
                   (1 + math.cos(math.pi * progress)) / 2


class OneCycleLR(LearningRateScheduler):
    """
    One Cycle Learning Rate Policy.
    
    1. Warmup: LR increases from initial to max
    2. Annealing: LR decreases from max to min
    """
    
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, 
                 div_factor=25, final_div_factor=1e4, last_epoch=-1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.min_lr = max_lr / final_div_factor
        
        super().__init__(optimizer, last_epoch)
        self.base_lrs = [self.initial_lr]
    
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.total_steps * self.pct_start:
            # Warmup phase
            progress = step / (self.total_steps * self.pct_start)
            return self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Annealing phase
            progress = (step - self.total_steps * self.pct_start) / \
                      (self.total_steps * (1 - self.pct_start))
            return self.min_lr + (self.max_lr - self.min_lr) * \
                   (1 + math.cos(math.pi * progress)) / 2


class ReduceLROnPlateau:
    """
    Reduce learning rate when metric stops improving.
    """
    
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, 
                 threshold=1e-4, min_lr=0):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
    
    def step(self, metric):
        if self.mode == 'min':
            improved = metric < self.best - self.threshold
        else:
            improved = metric > self.best + self.threshold
        
        if improved:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
            if self.num_bad_epochs >= self.patience:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                self.optimizer.lr = new_lr
                self.num_bad_epochs = 0
                print(f'Reducing LR to {new_lr}')


# Visualize schedulers
def visualize_schedulers(total_epochs=100):
    """Visualize different learning rate schedules."""
    import matplotlib.pyplot as plt
    
    class DummyOptimizer:
        def __init__(self):
            self.lr = 0.1
    
    schedulers = {
        'StepLR': StepLR(DummyOptimizer(), step_size=30, gamma=0.1),
        'ExponentialLR': ExponentialLR(DummyOptimizer(), gamma=0.95),
        'CosineAnnealing': CosineAnnealingLR(DummyOptimizer(), T_max=total_epochs),
        'WarmupCosine': WarmupCosineScheduler(DummyOptimizer(), warmup_epochs=10, total_epochs=total_epochs),
        'OneCycle': OneCycleLR(DummyOptimizer(), max_lr=0.1, total_steps=total_epochs),
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, scheduler in schedulers.items():
        scheduler.optimizer.lr = 0.1
        scheduler.base_lrs = [0.1]
        scheduler.last_epoch = -1
        
        lrs = []
        for epoch in range(total_epochs):
            lr = scheduler.step()
            lrs.append(lr)
        
        ax.plot(lrs, label=name)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedulers Comparison')
    ax.legend()
    ax.set_yscale('log')
    
    return fig


print("\nLearning Rate Schedulers:")
print("=" * 60)
print("""
StepLR: Decay by factor every N epochs
ExponentialLR: Exponential decay
CosineAnnealing: Smooth cosine decay
WarmupCosine: Warmup + cosine (transformers)
OneCycle: Up then down (super-convergence)
ReduceLROnPlateau: Decay when stuck
""")
```

---

## Chapter 53: Regularization Techniques

### 53.1 Weight Regularization

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class L1Regularization:
    """
    L1 Regularization (Lasso).
    
    Loss = Original Loss + λ * Σ|w|
    
    Promotes sparsity.
    """
    
    def __init__(self, model, lambda_l1=0.01):
        self.model = model
        self.lambda_l1 = lambda_l1
    
    def __call__(self):
        l1_loss = 0
        for param in self.model.parameters():
            l1_loss += torch.abs(param).sum()
        return self.lambda_l1 * l1_loss


class L2Regularization:
    """
    L2 Regularization (Ridge/Weight Decay).
    
    Loss = Original Loss + λ * Σw²
    
    Prevents large weights.
    """
    
    def __init__(self, model, lambda_l2=0.01):
        self.model = model
        self.lambda_l2 = lambda_l2
    
    def __call__(self):
        l2_loss = 0
        for param in self.model.parameters():
            l2_loss += torch.pow(param, 2).sum()
        return self.lambda_l2 * l2_loss


class ElasticNet:
    """
    Elastic Net: Combination of L1 and L2.
    
    Loss = Original Loss + α * L1 + (1-α) * L2
    """
    
    def __init__(self, model, lambda_reg=0.01, l1_ratio=0.5):
        self.model = model
        self.lambda_reg = lambda_reg
        self.l1_ratio = l1_ratio
    
    def __call__(self):
        l1_loss = 0
        l2_loss = 0
        
        for param in self.model.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()
        
        return self.lambda_reg * (
            self.l1_ratio * l1_loss + (1 - self.l1_ratio) * l2_loss
        )


class SpectralNorm(nn.Module):
    """
    Spectral Normalization: Normalize weights by spectral norm.
    
    Constrains Lipschitz constant of layer.
    """
    
    def __init__(self, layer, n_power_iterations=1):
        super().__init__()
        self.layer = layer
        self.n_power_iterations = n_power_iterations
        
        # Initialize u and v vectors
        weight = layer.weight.data
        h, w = weight.shape[0], weight.view(weight.shape[0], -1).shape[1]
        
        self.register_buffer('u', torch.randn(h).requires_grad_(False))
        self.register_buffer('v', torch.randn(w).requires_grad_(False))
    
    def _power_iteration(self, W):
        """Compute spectral norm using power iteration."""
        for _ in range(self.n_power_iterations):
            self.v = F.normalize(torch.mv(W.t(), self.u), dim=0)
            self.u = F.normalize(torch.mv(W, self.v), dim=0)
        
        sigma = torch.dot(self.u, torch.mv(W, self.v))
        return sigma
    
    def forward(self, x):
        W = self.layer.weight.view(self.layer.weight.shape[0], -1)
        sigma = self._power_iteration(W)
        
        # Normalize weight
        self.layer.weight.data = self.layer.weight.data / sigma
        
        return self.layer(x)
```

### 53.2 Dropout Variants

```python
class Dropout(nn.Module):
    """
    Standard Dropout.
    
    Randomly zero elements with probability p during training.
    Scale by 1/(1-p) to maintain expected value.
    """
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) > self.p).float()
            return x * mask / (1 - self.p)
        return x


class Dropout2d(nn.Module):
    """
    Spatial Dropout for CNNs.
    
    Drops entire feature maps instead of individual elements.
    """
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training:
            # x: (batch, channels, height, width)
            mask = (torch.rand(x.size(0), x.size(1), 1, 1, device=x.device) > self.p).float()
            return x * mask / (1 - self.p)
        return x


class DropConnect(nn.Module):
    """
    DropConnect: Drop weights instead of activations.
    
    Each weight has probability p of being zeroed.
    """
    
    def __init__(self, layer, p=0.5):
        super().__init__()
        self.layer = layer
        self.p = p
    
    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(self.layer.weight) > self.p).float()
            weight = self.layer.weight * mask / (1 - self.p)
            
            return F.linear(x, weight, self.layer.bias)
        
        return self.layer(x)


class DropPath(nn.Module):
    """
    Drop Path (Stochastic Depth).
    
    Drop entire residual branch with probability p.
    Used in ResNets and Vision Transformers.
    """
    
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        keep_prob = 1 - self.p
        
        # Random tensor for each sample in batch
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        
        return x / keep_prob * binary_mask


class AlphaDropout(nn.Module):
    """
    Alpha Dropout for SELU activations.
    
    Maintains self-normalizing property of SELU networks.
    """
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
        # SELU parameters
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        
        # Dropout parameters
        self.a_prime = -self.alpha * self.scale
        q = 1 - self.p
        self.a = (q + self.a_prime ** 2 * q * (1 - q)) ** (-0.5)
        self.b = -self.a * (1 - q) * self.a_prime
    
    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) > self.p).float()
            
            # Set dropped values to a' instead of 0
            x = mask * x + (1 - mask) * self.a_prime
            
            # Affine transformation to maintain mean and variance
            return self.a * x + self.b
        
        return x


print("\nDropout Variants:")
print("=" * 60)
print("""
Standard Dropout: Drop random elements
Spatial Dropout: Drop entire feature maps (CNNs)
DropConnect: Drop random weights
DropPath: Drop entire layers/branches
AlphaDropout: For SELU networks
""")
```

### 53.3 Normalization Techniques

```python
class BatchNorm1d(nn.Module):
    """
    Batch Normalization for 1D inputs.
    
    y = γ * (x - μ_B) / √(σ²_B + ε) + β
    
    Where μ_B and σ²_B are batch statistics.
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics for inference
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        if self.training:
            # Compute batch statistics
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.affine:
            x_norm = self.gamma * x_norm + self.beta
        
        return x_norm


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    
    Normalizes across features (not batch).
    Used in transformers.
    """
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        # Normalize over last dimensions
        dims = tuple(range(-len(self.normalized_shape), 0))
        
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.elementwise_affine:
            x_norm = self.gamma * x_norm + self.beta
        
        return x_norm


class InstanceNorm(nn.Module):
    """
    Instance Normalization.
    
    Normalizes each sample independently.
    Used in style transfer.
    """
    
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        # x: (batch, channels, height, width)
        # Normalize over H, W for each channel independently
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.affine:
            x_norm = self.gamma.view(1, -1, 1, 1) * x_norm + self.beta.view(1, -1, 1, 1)
        
        return x_norm


class GroupNorm(nn.Module):
    """
    Group Normalization.
    
    Divides channels into groups and normalizes within each group.
    Works well with small batch sizes.
    """
    
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x):
        # x: (batch, channels, height, width)
        batch_size, C, H, W = x.shape
        
        # Reshape to (batch, groups, channels_per_group, H, W)
        x = x.view(batch_size, self.num_groups, C // self.num_groups, H, W)
        
        # Normalize over (channels_per_group, H, W)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back
        x_norm = x_norm.view(batch_size, C, H, W)
        
        if self.affine:
            x_norm = self.gamma.view(1, -1, 1, 1) * x_norm + self.beta.view(1, -1, 1, 1)
        
        return x_norm


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    
    Simpler than LayerNorm - only divides by RMS.
    Used in some transformer variants.
    """
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * x / rms


print("\nNormalization Comparison:")
print("=" * 60)
print("""
┌─────────────────────────────────────────────────────────────┐
│ Normalization │ Normalize Over        │ Use Case            │
├─────────────────────────────────────────────────────────────┤
│ Batch Norm    │ Batch                 │ CNNs, large batch   │
│ Layer Norm    │ Features              │ Transformers, RNNs  │
│ Instance Norm │ Each sample (H, W)    │ Style transfer      │
│ Group Norm    │ Channel groups        │ Small batch CNNs    │
│ RMS Norm      │ Features (simplified) │ LLMs                │
└─────────────────────────────────────────────────────────────┘
""")
```

---

## Chapter 54: Training Best Practices

### 54.1 Weight Initialization

```python
import torch.nn.init as init

class WeightInitializer:
    """Various weight initialization methods."""
    
    @staticmethod
    def xavier_uniform(layer):
        """
        Xavier/Glorot Uniform Initialization.
        
        Good for: tanh, sigmoid activations
        Var(W) = 2 / (fan_in + fan_out)
        """
        if hasattr(layer, 'weight'):
            init.xavier_uniform_(layer.weight)
        if hasattr(layer, 'bias') and layer.bias is not None:
            init.zeros_(layer.bias)
    
    @staticmethod
    def xavier_normal(layer):
        """Xavier/Glorot Normal Initialization."""
        if hasattr(layer, 'weight'):
            init.xavier_normal_(layer.weight)
        if hasattr(layer, 'bias') and layer.bias is not None:
            init.zeros_(layer.bias)
    
    @staticmethod
    def kaiming_uniform(layer, mode='fan_in', nonlinearity='relu'):
        """
        Kaiming/He Uniform Initialization.
        
        Good for: ReLU, LeakyReLU activations
        Var(W) = 2 / fan_in (for ReLU)
        """
        if hasattr(layer, 'weight'):
            init.kaiming_uniform_(layer.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias') and layer.bias is not None:
            init.zeros_(layer.bias)
    
    @staticmethod
    def kaiming_normal(layer, mode='fan_in', nonlinearity='relu'):
        """Kaiming/He Normal Initialization."""
        if hasattr(layer, 'weight'):
            init.kaiming_normal_(layer.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias') and layer.bias is not None:
            init.zeros_(layer.bias)
    
    @staticmethod
    def orthogonal(layer, gain=1):
        """
        Orthogonal Initialization.
        
        Good for: RNNs
        Preserves gradient norms.
        """
        if hasattr(layer, 'weight'):
            init.orthogonal_(layer.weight, gain=gain)
        if hasattr(layer, 'bias') and layer.bias is not None:
            init.zeros_(layer.bias)
    
    @staticmethod
    def truncated_normal(layer, mean=0, std=0.02):
        """
        Truncated Normal Initialization.
        
        Good for: Transformers, Embeddings
        """
        if hasattr(layer, 'weight'):
            init.trunc_normal_(layer.weight, mean=mean, std=std)
        if hasattr(layer, 'bias') and layer.bias is not None:
            init.zeros_(layer.bias)


def initialize_model(model, init_type='kaiming'):
    """Apply initialization to entire model."""
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if init_type == 'xavier':
                WeightInitializer.xavier_normal(m)
            elif init_type == 'kaiming':
                WeightInitializer.kaiming_normal(m)
            elif init_type == 'orthogonal':
                WeightInitializer.orthogonal(m)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                init.ones_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)


print("\nWeight Initialization Guide:")
print("=" * 60)
print("""
Xavier (Glorot):
- For tanh, sigmoid, softmax
- Maintains variance across layers

Kaiming (He):
- For ReLU, LeakyReLU
- Accounts for ReLU zeroing half the activations

Orthogonal:
- For RNNs
- Prevents gradient vanishing/exploding

Truncated Normal:
- For Transformers
- Small std (0.02) for stability
""")
```

### 54.2 Gradient Management

```python
class GradientManagement:
    """Utilities for managing gradients during training."""
    
    @staticmethod
    def gradient_clipping_norm(model, max_norm, norm_type=2):
        """
        Clip gradients by global norm.
        
        Scales all gradients so total norm <= max_norm.
        """
        parameters = [p for p in model.parameters() if p.grad is not None]
        
        if len(parameters) == 0:
            return 0.0
        
        # Compute total norm
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        
        # Clip
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
        
        return total_norm
    
    @staticmethod
    def gradient_clipping_value(model, clip_value):
        """
        Clip gradients by value.
        
        Clamps each gradient element to [-clip_value, clip_value].
        """
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-clip_value, clip_value)
    
    @staticmethod
    def gradient_accumulation(loss, optimizer, accumulation_steps, step):
        """
        Gradient accumulation for larger effective batch size.
        """
        # Scale loss
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            return True
        return False
    
    @staticmethod
    def compute_gradient_norm(model, norm_type=2):
        """Compute gradient norm for monitoring."""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        return total_norm ** (1. / norm_type)


class GradientCheckpointing:
    """
    Gradient Checkpointing (Activation Checkpointing).
    
    Trade compute for memory by recomputing activations during backward.
    """
    
    @staticmethod
    def checkpoint(function, *args):
        """
        Checkpoint a function to save memory.
        
        During backward, recomputes activations instead of storing them.
        """
        import torch.utils.checkpoint as cp
        return cp.checkpoint(function, *args)
    
    @staticmethod
    def checkpoint_sequential(functions, segments, input):
        """
        Checkpoint a sequential model.
        
        Divides model into segments, checkpoints each segment.
        """
        import torch.utils.checkpoint as cp
        return cp.checkpoint_sequential(functions, segments, input)


class MixedPrecisionTraining:
    """
    Mixed Precision Training with automatic scaling.
    """
    
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, data, target, criterion):
        """Single training step with mixed precision."""
        self.optimizer.zero_grad()
        
        # Forward pass with autocast (FP16)
        with torch.cuda.amp.autocast():
            output = self.model(data)
            loss = criterion(output, target)
        
        # Backward pass with scaling
        self.scaler.scale(loss).backward()
        
        # Unscale and clip gradients
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update weights
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()


print("\nGradient Management:")
print("=" * 60)
print("""
GRADIENT CLIPPING:
- By Norm: Scale all gradients proportionally
- By Value: Clamp individual gradient values
- Prevents exploding gradients

GRADIENT ACCUMULATION:
- Simulate larger batch with limited memory
- Accumulate over N steps, then update
- Effective batch = batch_size * accumulation_steps

CHECKPOINTING:
- Save memory by recomputing activations
- ~30% slowdown for ~50% memory reduction

MIXED PRECISION:
- Use FP16 for speed, FP32 for stability
- 2-3x speedup on modern GPUs
- Use loss scaling to prevent underflow
""")
```

### 54.3 Early Stopping and Model Selection

```python
class EarlyStopping:
    """
    Early Stopping to prevent overfitting.
    
    Stop training when validation metric stops improving.
    """
    
    def __init__(self, patience=10, min_delta=0, mode='min', restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.early_stop = False
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict().copy()
        
        elif self._is_improvement(score):
            self.best_score = score
            self.best_state = model.state_dict().copy()
            self.counter = 0
        
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best:
                    model.load_state_dict(self.best_state)
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


class ModelCheckpoint:
    """
    Model Checkpointing - save best models during training.
    """
    
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = float('inf') if mode == 'min' else float('-inf')
    
    def __call__(self, score, model, optimizer=None, epoch=None):
        is_best = (self.mode == 'min' and score < self.best_score) or \
                  (self.mode == 'max' and score > self.best_score)
        
        if is_best:
            self.best_score = score
        
        if not self.save_best_only or is_best:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'score': score,
                'epoch': epoch,
            }
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            torch.save(checkpoint, self.filepath)
            print(f'Model saved: {self.filepath} (score: {score:.4f})')


class TrainingCallback:
    """Base class for training callbacks."""
    
    def on_epoch_begin(self, epoch, logs=None):
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        pass
    
    def on_batch_begin(self, batch, logs=None):
        pass
    
    def on_batch_end(self, batch, logs=None):
        pass
    
    def on_train_begin(self, logs=None):
        pass
    
    def on_train_end(self, logs=None):
        pass


class ProgressCallback(TrainingCallback):
    """Print training progress."""
    
    def on_epoch_end(self, epoch, logs=None):
        metrics = ' - '.join([f'{k}: {v:.4f}' for k, v in logs.items()])
        print(f'Epoch {epoch}: {metrics}')


class TensorBoardCallback(TrainingCallback):
    """Log to TensorBoard."""
    
    def __init__(self, log_dir='runs'):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
    
    def on_epoch_end(self, epoch, logs=None):
        for name, value in logs.items():
            self.writer.add_scalar(name, value, epoch)
    
    def on_train_end(self, logs=None):
        self.writer.close()


# Complete training loop with callbacks
def train_with_callbacks(model, train_loader, val_loader, optimizer, criterion, 
                         epochs, callbacks=None):
    """Training loop with callback support."""
    if callbacks is None:
        callbacks = []
    
    for cb in callbacks:
        cb.on_train_begin()
    
    for epoch in range(epochs):
        for cb in callbacks:
            cb.on_epoch_begin(epoch)
        
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            for cb in callbacks:
                cb.on_batch_begin(batch_idx)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            for cb in callbacks:
                cb.on_batch_end(batch_idx, {'loss': loss.item()})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / len(val_loader.dataset)
        
        logs = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        
        for cb in callbacks:
            cb.on_epoch_end(epoch, logs)
    
    for cb in callbacks:
        cb.on_train_end()


print("\nTraining Best Practices Summary:")
print("=" * 60)
print("""
1. INITIALIZATION
   - Xavier for tanh/sigmoid
   - Kaiming for ReLU
   
2. OPTIMIZATION
   - AdamW for transformers
   - SGD+momentum for CNNs
   - Warmup + cosine schedule

3. REGULARIZATION
   - Dropout (0.1-0.5)
   - Weight decay (1e-4 to 1e-2)
   - Data augmentation

4. MONITORING
   - Early stopping
   - Model checkpointing
   - Gradient monitoring

5. EFFICIENCY
   - Mixed precision
   - Gradient accumulation
   - Checkpointing
""")
```

---

## Summary: Optimization and Training

```
┌─────────────────────────────────────────────────────────────────────┐
│              OPTIMIZATION AND TRAINING SUMMARY                      │
├─────────────────────────────────────────────────────────────────────┤
│  OPTIMIZERS                                                         │
│  ├── SGD + Momentum: Classic, good generalization                  │
│  ├── Adam: Adaptive, fast convergence                              │
│  ├── AdamW: Decoupled weight decay                                 │
│  └── LAMB: Large batch training                                    │
│                                                                     │
│  LEARNING RATE SCHEDULES                                            │
│  ├── Step decay: Simple, effective                                 │
│  ├── Cosine annealing: Smooth decay                                │
│  ├── Warmup + decay: For transformers                              │
│  └── One cycle: Super-convergence                                  │
│                                                                     │
│  REGULARIZATION                                                     │
│  ├── L1/L2/Elastic Net: Weight penalties                           │
│  ├── Dropout variants: Activation masking                          │
│  └── Normalization: Batch/Layer/Group/RMS                          │
│                                                                     │
│  TRAINING TECHNIQUES                                                │
│  ├── Gradient clipping: Prevent explosion                          │
│  ├── Mixed precision: Speed + memory                               │
│  ├── Early stopping: Prevent overfitting                           │
│  └── Checkpointing: Save best models                               │
└─────────────────────────────────────────────────────────────────────┘
```


---

<div align="center">

[⬅️ Previous: Responsible AI](12-responsible-ai.md) | [📚 Table of Contents](../README.md) | [Next: Graph Neural Networks ➡️](14-graph-neural-networks.md)

</div>

<div align="center">

# 🔬 Advanced Algorithms

![Chapter](https://img.shields.io/badge/Chapter-17-blue?style=for-the-badge)
![Topic](https://img.shields.io/badge/Topic-Bayesian%20%7C%20Meta-green?style=for-the-badge)

*Bayesian ML, Meta-Learning, Compression & Continual Learning*

---

</div>

# Part XX: Additional Algorithms and Advanced Techniques

---

## Chapter 80: Bayesian Machine Learning

### 80.1 Bayesian Foundations

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BAYESIAN INFERENCE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  BAYES' THEOREM:                                                    │
│                                                                     │
│  P(θ|D) = P(D|θ) × P(θ) / P(D)                                     │
│                                                                     │
│  Where:                                                             │
│  - P(θ|D): Posterior - our updated belief about θ given data       │
│  - P(D|θ): Likelihood - probability of data given θ                │
│  - P(θ): Prior - our initial belief about θ                        │
│  - P(D): Evidence - normalizing constant                           │
│                                                                     │
│  FREQUENTIST vs BAYESIAN:                                           │
│                                                                     │
│  Frequentist:                                                       │
│  - Parameters are fixed, unknown constants                         │
│  - Estimate point values                                           │
│  - Confidence intervals                                             │
│                                                                     │
│  Bayesian:                                                          │
│  - Parameters are random variables with distributions              │
│  - Full posterior distribution                                     │
│  - Credible intervals                                              │
│  - Naturally handles uncertainty                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 80.2 Bayesian Linear Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class BayesianLinearRegression:
    """
    Bayesian Linear Regression with conjugate prior.
    
    Prior: p(w) = N(w | m_0, S_0)
    Posterior: p(w | X, y) = N(w | m_N, S_N)
    
    With closed-form update formulas.
    """
    
    def __init__(self, alpha=1.0, beta=25.0):
        """
        Args:
            alpha: Prior precision (1/variance)
            beta: Noise precision (1/noise_variance)
        """
        self.alpha = alpha
        self.beta = beta
        self.mean = None
        self.cov = None
    
    def fit(self, X, y):
        """
        Compute posterior distribution of weights.
        """
        # Add bias term
        X = self._add_bias(X)
        n_features = X.shape[1]
        
        # Prior
        S_0_inv = self.alpha * np.eye(n_features)
        m_0 = np.zeros(n_features)
        
        # Posterior
        S_N_inv = S_0_inv + self.beta * X.T @ X
        self.cov = np.linalg.inv(S_N_inv)
        self.mean = self.cov @ (S_0_inv @ m_0 + self.beta * X.T @ y)
        
        return self
    
    def predict(self, X, return_std=False):
        """
        Predict with uncertainty.
        
        Returns:
            mean: Predicted mean
            std: Predictive standard deviation (if return_std=True)
        """
        X = self._add_bias(X)
        
        # Predictive mean
        mean = X @ self.mean
        
        if return_std:
            # Predictive variance = noise + model uncertainty
            var = 1/self.beta + np.sum(X @ self.cov * X, axis=1)
            return mean, np.sqrt(var)
        
        return mean
    
    def _add_bias(self, X):
        """Add bias column."""
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    def sample_weights(self, n_samples=100):
        """Sample weights from posterior."""
        return np.random.multivariate_normal(self.mean, self.cov, n_samples)


class GaussianProcess:
    """
    Gaussian Process Regression.
    
    Non-parametric Bayesian approach - places prior over functions.
    """
    
    def __init__(self, kernel='rbf', length_scale=1.0, noise=1e-6):
        self.kernel = kernel
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
    
    def _kernel_func(self, X1, X2):
        """Compute kernel matrix."""
        if self.kernel == 'rbf':
            # RBF (Squared Exponential) kernel
            sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
                     np.sum(X2**2, 1) - 2 * X1 @ X2.T
            return np.exp(-0.5 * sqdist / self.length_scale**2)
        elif self.kernel == 'linear':
            return X1 @ X2.T
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y):
        """Store training data and compute K inverse."""
        self.X_train = X
        self.y_train = y
        
        K = self._kernel_func(X, X)
        K += self.noise * np.eye(len(X))
        
        self.K_inv = np.linalg.inv(K)
        return self
    
    def predict(self, X, return_std=False, return_cov=False):
        """
        Predict at new points.
        
        Returns full predictive distribution.
        """
        K_s = self._kernel_func(self.X_train, X)
        K_ss = self._kernel_func(X, X)
        
        # Predictive mean
        mean = K_s.T @ self.K_inv @ self.y_train
        
        # Predictive covariance
        cov = K_ss - K_s.T @ self.K_inv @ K_s
        
        if return_std:
            std = np.sqrt(np.diag(cov))
            return mean, std
        elif return_cov:
            return mean, cov
        
        return mean
    
    def sample(self, X, n_samples=5):
        """Sample functions from posterior."""
        mean, cov = self.predict(X, return_cov=True)
        return np.random.multivariate_normal(mean, cov + 1e-6*np.eye(len(X)), n_samples)


# Example usage
print("Bayesian Machine Learning:")
print("=" * 60)

# Generate data
np.random.seed(42)
X = np.linspace(0, 10, 20).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.randn(20) * 0.1

# Bayesian Linear Regression with polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)[:, 1:]  # Remove bias (BLR adds it)

blr = BayesianLinearRegression(alpha=0.01, beta=100)
blr.fit(X_poly, y)

# Gaussian Process
gp = GaussianProcess(kernel='rbf', length_scale=1.0)
gp.fit(X, y)

X_test = np.linspace(0, 10, 100).reshape(-1, 1)
mean, std = gp.predict(X_test, return_std=True)

print(f"GP Mean at x=5: {gp.predict(np.array([[5]]))[0]:.3f}")
print(f"True value at x=5: {np.sin(5):.3f}")
```

### 80.3 Variational Inference

```python
class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE).
    
    Uses variational inference to learn latent representations.
    
    ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
    """
    
    def __init__(self, input_dim, hidden_dim=256, latent_dim=32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for backprop through sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent variable to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss(self, x, x_recon, mu, logvar):
        """
        ELBO loss.
        
        Reconstruction loss + KL divergence.
        """
        # Reconstruction loss (binary cross entropy)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        
        # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss
    
    def sample(self, n_samples):
        """Generate samples from prior."""
        z = torch.randn(n_samples, self.fc_mu.out_features)
        return self.decode(z)


class BayesianNeuralNetwork(nn.Module):
    """
    Bayesian Neural Network using variational inference.
    
    Places distributions over weights instead of point estimates.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # Weight means
        self.w1_mu = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.w2_mu = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.1)
        
        # Weight log-variances
        self.w1_logvar = nn.Parameter(torch.zeros(input_dim, hidden_dim) - 5)
        self.w2_logvar = nn.Parameter(torch.zeros(hidden_dim, output_dim) - 5)
        
        # Bias means
        self.b1_mu = nn.Parameter(torch.zeros(hidden_dim))
        self.b2_mu = nn.Parameter(torch.zeros(output_dim))
        
        # Bias log-variances
        self.b1_logvar = nn.Parameter(torch.zeros(hidden_dim) - 5)
        self.b2_logvar = nn.Parameter(torch.zeros(output_dim) - 5)
    
    def _sample_weights(self):
        """Sample weights from variational distribution."""
        w1_std = torch.exp(0.5 * self.w1_logvar)
        w2_std = torch.exp(0.5 * self.w2_logvar)
        b1_std = torch.exp(0.5 * self.b1_logvar)
        b2_std = torch.exp(0.5 * self.b2_logvar)
        
        w1 = self.w1_mu + w1_std * torch.randn_like(w1_std)
        w2 = self.w2_mu + w2_std * torch.randn_like(w2_std)
        b1 = self.b1_mu + b1_std * torch.randn_like(b1_std)
        b2 = self.b2_mu + b2_std * torch.randn_like(b2_std)
        
        return w1, w2, b1, b2
    
    def forward(self, x, n_samples=1):
        """Forward pass with weight sampling."""
        outputs = []
        
        for _ in range(n_samples):
            w1, w2, b1, b2 = self._sample_weights()
            
            h = F.relu(x @ w1 + b1)
            out = h @ w2 + b2
            outputs.append(out)
        
        return torch.stack(outputs).mean(dim=0)
    
    def kl_divergence(self):
        """KL divergence between q(w) and p(w)."""
        kl = 0
        
        for mu, logvar in [(self.w1_mu, self.w1_logvar),
                           (self.w2_mu, self.w2_logvar),
                           (self.b1_mu, self.b1_logvar),
                           (self.b2_mu, self.b2_logvar)]:
            # KL(N(mu, sigma) || N(0, 1))
            kl += 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        
        return kl


print("\nVariational Inference:")
print("=" * 60)
print("""
Key Concepts:
- Replace intractable posterior with tractable approximation
- Optimize ELBO (Evidence Lower Bound)
- Reparameterization trick enables backprop through sampling

Applications:
- VAE: Generative modeling
- Bayesian Neural Networks: Uncertainty quantification
- Bayesian Optimization: Hyperparameter tuning
""")
```

---

## Chapter 81: Meta-Learning

### 81.1 Learning to Learn

```python
class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML).
    
    Learn initial parameters that can be quickly adapted to new tasks.
    
    Inner loop: Adapt to task with few gradient steps
    Outer loop: Update meta-parameters
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, inner_lr=0.01):
        super().__init__()
        
        self.inner_lr = inner_lr
        
        # Base model parameters
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, params=None):
        """Forward pass with optional custom parameters."""
        if params is None:
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return self.fc3(h)
        else:
            # Use provided parameters
            h = F.relu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
            h = F.relu(F.linear(h, params['fc2.weight'], params['fc2.bias']))
            return F.linear(h, params['fc3.weight'], params['fc3.bias'])
    
    def adapt(self, support_x, support_y, num_steps=1):
        """
        Adapt model parameters to a specific task.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            num_steps: Number of inner loop gradient steps
        
        Returns:
            adapted_params: Parameters adapted to the task
        """
        # Get current parameters
        params = {name: param.clone() for name, param in self.named_parameters()}
        
        for _ in range(num_steps):
            # Forward pass
            pred = self.forward(support_x, params)
            loss = F.cross_entropy(pred, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, params.values(), create_graph=True)
            
            # Update parameters
            params = {name: param - self.inner_lr * grad 
                     for (name, param), grad in zip(params.items(), grads)}
        
        return params
    
    def meta_loss(self, tasks, num_inner_steps=1):
        """
        Compute meta-loss across tasks.
        
        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples
        """
        total_loss = 0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Adapt to task
            adapted_params = self.adapt(support_x, support_y, num_inner_steps)
            
            # Evaluate on query set
            pred = self.forward(query_x, adapted_params)
            loss = F.cross_entropy(pred, query_y)
            
            total_loss += loss
        
        return total_loss / len(tasks)


class ProtoNet(nn.Module):
    """
    Prototypical Networks for few-shot learning.
    
    Learn embedding space where classification is done by 
    computing distances to class prototypes.
    """
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def compute_prototypes(self, support_x, support_y, n_classes):
        """
        Compute prototype for each class.
        
        Prototype = mean embedding of support examples.
        """
        embeddings = self.encoder(support_x)
        
        prototypes = []
        for c in range(n_classes):
            class_embeddings = embeddings[support_y == c]
            prototypes.append(class_embeddings.mean(dim=0))
        
        return torch.stack(prototypes)
    
    def forward(self, support_x, support_y, query_x, n_classes):
        """
        Classify query examples.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            n_classes: Number of classes
        """
        # Compute prototypes
        prototypes = self.compute_prototypes(support_x, support_y, n_classes)
        
        # Embed query examples
        query_embeddings = self.encoder(query_x)
        
        # Compute distances to prototypes
        dists = torch.cdist(query_embeddings, prototypes)
        
        # Return negative distances as logits
        return -dists


class MatchingNetwork(nn.Module):
    """
    Matching Networks for one-shot learning.
    
    Uses attention over support set for prediction.
    """
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, support_x, support_y, query_x, n_classes):
        """
        Classify using attention over support set.
        """
        # Embed support and query
        support_emb = self.encoder(support_x)
        query_emb = self.encoder(query_x)
        
        # Compute attention (cosine similarity)
        support_norm = F.normalize(support_emb, dim=1)
        query_norm = F.normalize(query_emb, dim=1)
        
        attention = query_norm @ support_norm.t()  # (n_query, n_support)
        attention = F.softmax(attention, dim=1)
        
        # Convert support labels to one-hot
        support_onehot = F.one_hot(support_y, n_classes).float()
        
        # Weighted sum of one-hot labels
        logits = attention @ support_onehot
        
        return logits


print("\nMeta-Learning Approaches:")
print("=" * 60)
print("""
MAML (Model-Agnostic Meta-Learning):
- Learn good initialization
- Few gradient steps to adapt
- Works with any gradient-based model

Prototypical Networks:
- Metric learning approach
- Class = mean of support embeddings
- Simple and effective

Matching Networks:
- Attention over support set
- Non-parametric at test time
- Good for very few shots
""")
```

---

## Chapter 82: Neural Network Compression

### 82.1 Pruning

```python
class NetworkPruner:
    """
    Neural network pruning techniques.
    """
    
    @staticmethod
    def magnitude_pruning(model, sparsity=0.5):
        """
        Prune weights with smallest magnitude.
        
        Args:
            model: Neural network
            sparsity: Fraction of weights to prune
        """
        # Collect all weights
        all_weights = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                all_weights.append(param.data.abs().view(-1))
        
        all_weights = torch.cat(all_weights)
        
        # Find threshold
        threshold = torch.quantile(all_weights, sparsity)
        
        # Apply masks
        masks = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                mask = param.data.abs() > threshold
                param.data *= mask.float()
                masks[name] = mask
        
        return masks
    
    @staticmethod
    def structured_pruning(model, prune_ratio=0.3):
        """
        Prune entire filters/neurons (structured sparsity).
        
        Maintains hardware efficiency.
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Compute filter importance (L1 norm)
                importance = module.weight.data.abs().sum(dim=(1, 2, 3))
                
                # Find filters to prune
                n_prune = int(len(importance) * prune_ratio)
                _, indices = importance.sort()
                prune_indices = indices[:n_prune]
                
                # Zero out pruned filters
                module.weight.data[prune_indices] = 0
                if module.bias is not None:
                    module.bias.data[prune_indices] = 0
            
            elif isinstance(module, nn.Linear):
                # Prune neurons (output dimension)
                importance = module.weight.data.abs().sum(dim=1)
                
                n_prune = int(len(importance) * prune_ratio)
                _, indices = importance.sort()
                prune_indices = indices[:n_prune]
                
                module.weight.data[prune_indices] = 0
                if module.bias is not None:
                    module.bias.data[prune_indices] = 0


class LotteryTicketHypothesis:
    """
    Lottery Ticket Hypothesis: Find sparse subnetworks that train well.
    
    Algorithm:
    1. Train network
    2. Prune smallest weights
    3. Reset remaining weights to initial values
    4. Retrain
    """
    
    def __init__(self, model, prune_ratio=0.2):
        self.model = model
        self.prune_ratio = prune_ratio
        
        # Save initial weights
        self.initial_weights = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        self.masks = None
    
    def find_winning_ticket(self, train_fn, iterations=5):
        """
        Iteratively find winning ticket.
        
        Args:
            train_fn: Function to train the model
            iterations: Number of prune-retrain iterations
        """
        current_sparsity = 0
        
        for i in range(iterations):
            # Train
            train_fn(self.model)
            
            # Prune
            current_sparsity = 1 - (1 - self.prune_ratio) ** (i + 1)
            self.masks = NetworkPruner.magnitude_pruning(self.model, current_sparsity)
            
            # Reset to initial weights (but keep pruning)
            for name, param in self.model.named_parameters():
                if name in self.initial_weights:
                    param.data = self.initial_weights[name].clone()
                    if name in self.masks:
                        param.data *= self.masks[name].float()
            
            print(f"Iteration {i+1}: Sparsity = {current_sparsity:.2%}")
        
        return self.masks
```

### 82.2 Quantization

```python
class Quantizer:
    """
    Neural network quantization techniques.
    """
    
    @staticmethod
    def quantize_tensor(tensor, num_bits=8):
        """
        Uniform quantization of a tensor.
        
        Maps floating point values to integers.
        """
        # Find range
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Compute scale and zero point
        qmin = 0
        qmax = 2**num_bits - 1
        
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        zero_point = int(round(zero_point))
        zero_point = max(qmin, min(qmax, zero_point))
        
        # Quantize
        q_tensor = torch.round(tensor / scale + zero_point)
        q_tensor = torch.clamp(q_tensor, qmin, qmax).to(torch.int8)
        
        return q_tensor, scale, zero_point
    
    @staticmethod
    def dequantize_tensor(q_tensor, scale, zero_point):
        """Dequantize back to floating point."""
        return scale * (q_tensor.float() - zero_point)
    
    @staticmethod
    def quantize_model(model, num_bits=8):
        """
        Post-training quantization of entire model.
        """
        quantized_params = {}
        
        for name, param in model.named_parameters():
            q_param, scale, zp = Quantizer.quantize_tensor(param.data, num_bits)
            quantized_params[name] = {
                'data': q_param,
                'scale': scale,
                'zero_point': zp
            }
        
        return quantized_params


class QuantizationAwareTraining(nn.Module):
    """
    Quantization-aware training module.
    
    Simulates quantization during training for better accuracy.
    """
    
    def __init__(self, num_bits=8):
        super().__init__()
        self.num_bits = num_bits
        self.qmin = 0
        self.qmax = 2**num_bits - 1
    
    def forward(self, x):
        """Fake quantize: quantize then immediately dequantize."""
        # This allows gradients to flow through
        min_val = x.min()
        max_val = x.max()
        
        scale = (max_val - min_val) / (self.qmax - self.qmin)
        zero_point = self.qmin - min_val / scale
        
        # Fake quantize
        x_q = torch.round(x / scale + zero_point)
        x_q = torch.clamp(x_q, self.qmin, self.qmax)
        x_dq = scale * (x_q - zero_point)
        
        # Straight-through estimator for gradients
        return x + (x_dq - x).detach()


print("\nNetwork Compression:")
print("=" * 60)
print("""
PRUNING:
- Magnitude pruning: Remove small weights
- Structured pruning: Remove entire neurons/filters
- Lottery ticket: Find sparse trainable subnetworks

QUANTIZATION:
- Post-training: Quantize after training
- Quantization-aware: Simulate during training
- Common: INT8 (8-bit), INT4 (4-bit)

KNOWLEDGE DISTILLATION:
- Train small "student" to mimic large "teacher"
- Soft labels carry more information
- Temperature scaling for softer distributions

Compression Benefits:
- Smaller model size (4-10x)
- Faster inference (2-4x)
- Lower power consumption
- Edge deployment
""")
```

### 82.3 Knowledge Distillation

```python
class KnowledgeDistillation:
    """
    Knowledge Distillation: Transfer knowledge from teacher to student.
    """
    
    def __init__(self, teacher, student, temperature=4.0, alpha=0.5):
        """
        Args:
            teacher: Large, trained teacher model
            student: Smaller student model to train
            temperature: Softmax temperature (higher = softer)
            alpha: Weight for distillation loss vs hard loss
        """
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Combined loss: soft targets + hard targets.
        
        L = α * KL(softmax(t/T), softmax(s/T)) * T² + (1-α) * CE(s, y)
        """
        # Soft targets (from teacher)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        distill_loss *= self.temperature ** 2  # Scale by T²
        
        # Hard targets (ground truth)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        return self.alpha * distill_loss + (1 - self.alpha) * hard_loss
    
    def train_step(self, x, labels, optimizer):
        """Single training step."""
        optimizer.zero_grad()
        
        # Teacher predictions (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        
        # Student predictions
        student_logits = self.student(x)
        
        # Loss and update
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        loss.backward()
        optimizer.step()
        
        return loss.item()


class FeatureDistillation(nn.Module):
    """
    Feature-based distillation: Match intermediate representations.
    """
    
    def __init__(self, teacher, student, layer_pairs):
        """
        Args:
            layer_pairs: List of (teacher_layer_name, student_layer_name) tuples
        """
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.layer_pairs = layer_pairs
        
        # Feature extractors
        self.teacher_features = {}
        self.student_features = {}
        
        # Register hooks
        for t_name, s_name in layer_pairs:
            self._register_hook(teacher, t_name, self.teacher_features)
            self._register_hook(student, s_name, self.student_features)
    
    def _register_hook(self, model, layer_name, storage):
        """Register forward hook to capture features."""
        def hook(module, input, output):
            storage[layer_name] = output
        
        for name, module in model.named_modules():
            if name == layer_name:
                module.register_forward_hook(hook)
                break
    
    def feature_loss(self):
        """Compute feature matching loss."""
        total_loss = 0
        
        for t_name, s_name in self.layer_pairs:
            t_feat = self.teacher_features[t_name]
            s_feat = self.student_features[s_name]
            
            # Adapt dimensions if needed
            if t_feat.shape != s_feat.shape:
                # Use 1x1 conv or linear to match
                pass
            
            total_loss += F.mse_loss(s_feat, t_feat)
        
        return total_loss


print("\nKnowledge Distillation:")
print("=" * 60)
print("""
Distillation Types:
1. Response-based: Match output logits
2. Feature-based: Match intermediate features
3. Relation-based: Match relationships between samples

Key Insights:
- Soft labels contain "dark knowledge"
- Temperature > 1 reveals class similarities
- Student can sometimes exceed teacher!

Applications:
- Model compression for deployment
- Transfer to different architectures
- Multi-task learning
""")
```

---

## Chapter 83: Continual Learning

### 83.1 Overcoming Catastrophic Forgetting

```python
class EWC(nn.Module):
    """
    Elastic Weight Consolidation.
    
    Prevents catastrophic forgetting by penalizing changes to important weights.
    """
    
    def __init__(self, model, importance_weight=1000):
        super().__init__()
        self.model = model
        self.importance_weight = importance_weight
        
        self.fisher = {}  # Fisher information (importance)
        self.optimal_params = {}  # Optimal parameters for previous tasks
    
    def compute_fisher(self, dataloader, criterion):
        """
        Estimate Fisher information matrix.
        
        Fisher approximates importance of each parameter.
        """
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        
        for x, y in dataloader:
            self.model.zero_grad()
            output = self.model(x)
            loss = criterion(output, y)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
        
        # Normalize by number of samples
        for n in fisher:
            fisher[n] /= len(dataloader.dataset)
        
        return fisher
    
    def register_task(self, dataloader, criterion):
        """Register Fisher and optimal params after training on a task."""
        # Compute Fisher
        new_fisher = self.compute_fisher(dataloader, criterion)
        
        # Accumulate Fisher
        for n, f in new_fisher.items():
            if n in self.fisher:
                self.fisher[n] += f
            else:
                self.fisher[n] = f
        
        # Store optimal parameters
        self.optimal_params = {
            n: p.data.clone() for n, p in self.model.named_parameters()
        }
    
    def ewc_loss(self):
        """
        EWC penalty term.
        
        Penalizes moving away from optimal params proportional to importance.
        """
        loss = 0
        
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.optimal_params[n]).pow(2)).sum()
        
        return self.importance_weight * loss


class PackNet(nn.Module):
    """
    PackNet: Progressive pruning for continual learning.
    
    After each task:
    1. Prune unimportant weights
    2. Freeze pruned weights
    3. Use remaining capacity for new tasks
    """
    
    def __init__(self, model, prune_ratio=0.5):
        super().__init__()
        self.model = model
        self.prune_ratio = prune_ratio
        self.masks = {}  # Binary masks for each task
        self.current_task = 0
    
    def prune_and_freeze(self, task_id):
        """Prune after training on a task."""
        task_mask = {}
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # Get available (unfrozen) weights
                if name in self.masks:
                    available = self.masks[name] == 0  # Not yet assigned
                else:
                    available = torch.ones_like(param.data, dtype=torch.bool)
                
                # Find important weights among available
                importance = param.data.abs() * available.float()
                threshold = torch.quantile(
                    importance[available], 
                    1 - self.prune_ratio
                )
                
                # Create mask for this task
                task_mask[name] = (importance >= threshold) & available
                
                # Update global mask
                if name in self.masks:
                    self.masks[name][task_mask[name]] = task_id + 1
                else:
                    self.masks[name] = torch.zeros_like(param.data, dtype=torch.int)
                    self.masks[name][task_mask[name]] = task_id + 1
        
        self.current_task = task_id + 1
        return task_mask
    
    def forward(self, x, task_id=None):
        """Forward pass with task-specific masks."""
        # Apply masks if doing inference on specific task
        if task_id is not None:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in self.masks:
                        mask = (self.masks[name] == task_id + 1) | (self.masks[name] == 0)
                        param.data *= mask.float()
        
        return self.model(x)


class ReplayBuffer:
    """
    Experience replay for continual learning.
    
    Store and replay samples from previous tasks.
    """
    
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, x, y, task_id):
        """Add samples to buffer."""
        for i in range(len(x)):
            sample = (x[i], y[i], task_id)
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.position] = sample
            
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample random batch from buffer."""
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        
        x = torch.stack([self.buffer[i][0] for i in indices])
        y = torch.tensor([self.buffer[i][1] for i in indices])
        tasks = torch.tensor([self.buffer[i][2] for i in indices])
        
        return x, y, tasks


print("\nContinual Learning:")
print("=" * 60)
print("""
CHALLENGE: Catastrophic Forgetting
- Neural networks forget previous tasks when learning new ones
- New gradients overwrite old knowledge

SOLUTIONS:

1. Regularization-based:
   - EWC: Penalize changes to important weights
   - SI: Online importance estimation
   
2. Architecture-based:
   - PackNet: Dedicate capacity to each task
   - Progressive Networks: Add new columns
   
3. Replay-based:
   - Experience Replay: Store and replay old samples
   - Generative Replay: Train generator to produce old samples

4. Parameter Isolation:
   - Task-specific heads/modules
   - Separate parameters per task
""")
```

---

## Summary: Advanced Algorithms

```
┌─────────────────────────────────────────────────────────────────────┐
│              ADVANCED ALGORITHMS SUMMARY                            │
├─────────────────────────────────────────────────────────────────────┤
│  BAYESIAN ML                                                        │
│  ├── Full posterior distributions over parameters                  │
│  ├── Natural uncertainty quantification                            │
│  ├── Gaussian Processes: Non-parametric, function-level prior      │
│  └── Variational Inference: Tractable approximations               │
│                                                                     │
│  META-LEARNING                                                      │
│  ├── MAML: Learn good initialization                               │
│  ├── Prototypical Networks: Metric learning                        │
│  └── Applications: Few-shot learning, fast adaptation              │
│                                                                     │
│  COMPRESSION                                                        │
│  ├── Pruning: Remove unnecessary weights                           │
│  ├── Quantization: Reduce precision                                │
│  ├── Distillation: Transfer to smaller model                       │
│  └── Benefits: Faster, smaller, deployable                         │
│                                                                     │
│  CONTINUAL LEARNING                                                 │
│  ├── EWC: Protect important weights                                │
│  ├── PackNet: Dedicate capacity                                    │
│  ├── Replay: Remember old samples                                  │
│  └── Goal: Learn sequentially without forgetting                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

# END OF TEXTBOOK

**Total Coverage:**
- 19 Parts
- 83+ Chapters  
- Comprehensive code implementations
- Conceptual explanations
- Practical exercises
- Quick reference guides

**Thank you for reading this comprehensive ML textbook!**


---

<div align="center">

[⬅️ Previous: Foundation Models](16-foundation-models.md) | [📚 Table of Contents](../README.md) | [Next: Interview Preparation ➡️](18-interview-prep.md)

</div>

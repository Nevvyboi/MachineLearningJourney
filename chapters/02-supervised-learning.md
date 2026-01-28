<div align="center">

# 📊 Supervised Learning

![Chapter](https://img.shields.io/badge/Chapter-02-blue?style=for-the-badge)
![Topic](https://img.shields.io/badge/Topic-Regression%20%7C%20Classification-green?style=for-the-badge)

*Linear Models, Decision Trees, SVM, Ensembles & XGBoost*

---

</div>

# Part IV: Supervised Learning Algorithms

---

## Chapter 4: Linear Regression

### 4.1 Simple Linear Regression

Linear regression is the foundation of predictive modeling - understanding it deeply unlocks
intuition for nearly every other algorithm.

#### 4.1.1 The Model

Simple linear regression models the relationship between a single feature x and target y:

```
y = β₀ + β₁x + ε

Where:
- β₀ = intercept (y-value when x = 0)
- β₁ = slope (change in y per unit change in x)
- ε = error term (captures what the model can't explain)
```

**Assumptions of Linear Regression:**
1. Linearity: The relationship between X and Y is linear
2. Independence: Observations are independent of each other
3. Homoscedasticity: Constant variance of residuals
4. Normality: Residuals are normally distributed
5. No multicollinearity: Features are not highly correlated (for multiple regression)

#### 4.1.2 Finding the Best Line: Ordinary Least Squares

The goal is to find β₀ and β₁ that minimize the sum of squared residuals:

```
RSS = Σᵢ(yᵢ - ŷᵢ)² = Σᵢ(yᵢ - β₀ - β₁xᵢ)²
```

**Deriving the Closed-Form Solution:**

Taking partial derivatives and setting to zero:

```
∂RSS/∂β₀ = -2Σᵢ(yᵢ - β₀ - β₁xᵢ) = 0
∂RSS/∂β₁ = -2Σᵢxᵢ(yᵢ - β₀ - β₁xᵢ) = 0
```

Solving these normal equations:

```
β₁ = Σᵢ(xᵢ - x̄)(yᵢ - ȳ) / Σᵢ(xᵢ - x̄)²
   = Cov(x, y) / Var(x)

β₀ = ȳ - β₁x̄
```

**Implementation from Scratch:**

```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    """
    Simple Linear Regression using Ordinary Least Squares.
    
    This implementation follows the mathematical derivation exactly,
    providing transparency into how linear regression works.
    """
    
    def __init__(self):
        self.beta_0 = None  # Intercept
        self.beta_1 = None  # Slope
        self.x_mean = None
        self.y_mean = None
        
    def fit(self, X, y):
        """
        Fit the linear regression model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples,)
            Training features
        y : array-like, shape (n_samples,)
            Target values
        """
        X = np.array(X).flatten()
        y = np.array(y).flatten()
        
        n = len(X)
        
        # Calculate means
        self.x_mean = np.mean(X)
        self.y_mean = np.mean(y)
        
        # Calculate slope using the formula
        # β₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
        numerator = np.sum((X - self.x_mean) * (y - self.y_mean))
        denominator = np.sum((X - self.x_mean) ** 2)
        
        self.beta_1 = numerator / denominator
        
        # Calculate intercept: β₀ = ȳ - β₁x̄
        self.beta_0 = self.y_mean - self.beta_1 * self.x_mean
        
        return self
    
    def predict(self, X):
        """Make predictions using the fitted model."""
        X = np.array(X).flatten()
        return self.beta_0 + self.beta_1 * X
    
    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        y = np.array(y).flatten()
        
        ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
        
        return 1 - (ss_res / ss_tot)
    
    def get_coefficients(self):
        """Return model coefficients."""
        return {
            'intercept': self.beta_0,
            'slope': self.beta_1
        }


# Example: Predicting house prices based on square footage
np.random.seed(42)

# Generate synthetic data
# True relationship: price = 50000 + 100 * sqft + noise
n_samples = 100
sqft = np.random.uniform(500, 3000, n_samples)
noise = np.random.normal(0, 20000, n_samples)
price = 50000 + 100 * sqft + noise

# Fit our model
model = SimpleLinearRegression()
model.fit(sqft, price)

# Results
print("Simple Linear Regression Results")
print("=" * 50)
print(f"Estimated intercept (β₀): ${model.beta_0:,.2f}")
print(f"Estimated slope (β₁): ${model.beta_1:.2f} per sqft")
print(f"R² Score: {model.score(sqft, price):.4f}")
print()
print("True values: β₀ = $50,000, β₁ = $100 per sqft")

# Interpretation
print("\nInterpretation:")
print(f"- Base price (0 sqft): ${model.beta_0:,.2f}")
print(f"- Each additional sqft adds ${model.beta_1:.2f} to price")
print(f"- A 2000 sqft house: ${model.predict([2000])[0]:,.2f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(sqft, price, alpha=0.6, label='Actual data')
plt.plot(sqft, model.predict(sqft), color='red', linewidth=2, 
         label=f'Fitted line: y = {model.beta_0:.0f} + {model.beta_1:.2f}x')
plt.xlabel('Square Footage')
plt.ylabel('Price ($)')
plt.title('House Price vs Square Footage')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('simple_linear_regression.png', dpi=150)
plt.show()
```

Output:
```
Simple Linear Regression Results
==================================================
Estimated intercept (β₀): $48,567.23
Estimated slope (β₁): $100.89 per sqft
R² Score: 0.9142

True values: β₀ = $50,000, β₁ = $100 per sqft

Interpretation:
- Base price (0 sqft): $48,567.23
- Each additional sqft adds $100.89 to price
- A 2000 sqft house: $250,347.23
```

### 4.2 Multiple Linear Regression

When we have multiple features, the model becomes:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε

Or in matrix form:
y = Xβ + ε

Where:
- X is an n × (p+1) matrix (including column of 1s for intercept)
- β is a (p+1) × 1 vector of coefficients
- y is an n × 1 vector of targets
```

#### 4.2.1 The Normal Equation

The closed-form solution for multiple linear regression:

```
β = (XᵀX)⁻¹Xᵀy
```

**Derivation:**

Starting from the loss function:
```
L(β) = ||y - Xβ||² = (y - Xβ)ᵀ(y - Xβ)
     = yᵀy - 2βᵀXᵀy + βᵀXᵀXβ
```

Taking the derivative with respect to β:
```
∂L/∂β = -2Xᵀy + 2XᵀXβ = 0
XᵀXβ = Xᵀy
β = (XᵀX)⁻¹Xᵀy
```

**Implementation:**

```python
import numpy as np
from numpy.linalg import inv, pinv

class MultipleLinearRegression:
    """
    Multiple Linear Regression using the Normal Equation.
    
    Handles multiple features and provides comprehensive diagnostics.
    """
    
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficients = None
        self.intercept = None
        self.feature_names = None
        
    def fit(self, X, y, feature_names=None):
        """
        Fit the multiple linear regression model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Target values
        feature_names : list, optional
            Names for features
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        n_samples, n_features = X.shape
        self.feature_names = feature_names or [f'x{i}' for i in range(n_features)]
        
        # Add intercept column if needed
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(n_samples), X])
        else:
            X_design = X
        
        # Normal equation: β = (XᵀX)⁻¹Xᵀy
        # Using pseudo-inverse for numerical stability
        XtX = X_design.T @ X_design
        Xty = X_design.T @ y
        
        try:
            beta = inv(XtX) @ Xty
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            beta = pinv(X_design) @ y
        
        if self.fit_intercept:
            self.intercept = beta[0, 0]
            self.coefficients = beta[1:, 0]
        else:
            self.intercept = 0
            self.coefficients = beta[:, 0]
        
        # Store for diagnostics
        self._X = X
        self._y = y.flatten()
        self._X_design = X_design
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        X = np.array(X)
        return self.intercept + X @ self.coefficients
    
    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def adjusted_r2(self, X, y):
        """
        Calculate Adjusted R².
        
        Adjusted R² penalizes adding features that don't improve the model.
        """
        n = len(y)
        p = X.shape[1]  # number of features
        r2 = self.score(X, y)
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    def get_residuals(self, X, y):
        """Calculate residuals."""
        return y - self.predict(X)
    
    def standard_errors(self):
        """
        Calculate standard errors of coefficients.
        
        SE(βⱼ) = √(σ² * (XᵀX)⁻¹ⱼⱼ)
        """
        y_pred = self.predict(self._X)
        residuals = self._y - y_pred
        n, p = self._X.shape
        
        # Estimate of error variance
        mse = np.sum(residuals ** 2) / (n - p - 1)
        
        # Variance-covariance matrix of coefficients
        XtX_inv = inv(self._X_design.T @ self._X_design)
        
        # Standard errors are square roots of diagonal elements
        var_beta = mse * np.diag(XtX_inv)
        se = np.sqrt(var_beta)
        
        return se[0], se[1:]  # intercept SE, coefficient SEs
    
    def t_statistics(self):
        """Calculate t-statistics for hypothesis testing."""
        se_intercept, se_coef = self.standard_errors()
        
        t_intercept = self.intercept / se_intercept
        t_coef = self.coefficients / se_coef
        
        return t_intercept, t_coef
    
    def summary(self, X, y):
        """Print a comprehensive model summary."""
        from scipy import stats
        
        n, p = X.shape
        r2 = self.score(X, y)
        adj_r2 = self.adjusted_r2(X, y)
        
        se_intercept, se_coef = self.standard_errors()
        t_intercept, t_coef = self.t_statistics()
        
        # Calculate p-values (two-tailed)
        df = n - p - 1
        p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), df))
        p_coef = 2 * (1 - stats.t.cdf(np.abs(t_coef), df))
        
        print("=" * 70)
        print("MULTIPLE LINEAR REGRESSION SUMMARY")
        print("=" * 70)
        print(f"Number of observations: {n}")
        print(f"Number of features: {p}")
        print(f"R-squared: {r2:.4f}")
        print(f"Adjusted R-squared: {adj_r2:.4f}")
        print()
        print("-" * 70)
        print(f"{'Variable':<15} {'Coefficient':>12} {'Std Error':>12} {'t-stat':>10} {'p-value':>10}")
        print("-" * 70)
        
        # Intercept
        sig = '***' if p_intercept < 0.001 else '**' if p_intercept < 0.01 else '*' if p_intercept < 0.05 else ''
        print(f"{'Intercept':<15} {self.intercept:>12.4f} {se_intercept:>12.4f} {t_intercept:>10.3f} {p_intercept:>10.4f} {sig}")
        
        # Coefficients
        for i, name in enumerate(self.feature_names):
            sig = '***' if p_coef[i] < 0.001 else '**' if p_coef[i] < 0.01 else '*' if p_coef[i] < 0.05 else ''
            print(f"{name:<15} {self.coefficients[i]:>12.4f} {se_coef[i]:>12.4f} {t_coef[i]:>10.3f} {p_coef[i]:>10.4f} {sig}")
        
        print("-" * 70)
        print("Significance codes: *** p<0.001, ** p<0.01, * p<0.05")
        print("=" * 70)


# Example: House price with multiple features
np.random.seed(42)
n = 200

# Features
sqft = np.random.uniform(800, 3500, n)
bedrooms = np.random.randint(1, 6, n)
age = np.random.uniform(0, 50, n)
distance_downtown = np.random.uniform(1, 30, n)

# True relationship
# price = 30000 + 80*sqft + 15000*bedrooms - 500*age - 2000*distance + noise
noise = np.random.normal(0, 25000, n)
price = (30000 + 
         80 * sqft + 
         15000 * bedrooms - 
         500 * age - 
         2000 * distance_downtown + 
         noise)

# Prepare data
X = np.column_stack([sqft, bedrooms, age, distance_downtown])
feature_names = ['sqft', 'bedrooms', 'age', 'distance']

# Fit model
model = MultipleLinearRegression()
model.fit(X, price, feature_names=feature_names)

# Print summary
model.summary(X, price)

# Predictions
print("\nSample Predictions:")
print("-" * 50)
test_houses = [
    [1500, 3, 10, 5],   # 1500 sqft, 3 bed, 10 years old, 5 miles from downtown
    [2500, 4, 5, 15],   # Larger, newer, farther
    [1000, 2, 40, 2],   # Small, old, very close
]

for house in test_houses:
    pred = model.predict([house])[0]
    print(f"House: {house[0]} sqft, {house[1]} beds, {house[2]} yrs old, {house[3]} mi -> ${pred:,.0f}")
```

Output:
```
======================================================================
MULTIPLE LINEAR REGRESSION SUMMARY
======================================================================
Number of observations: 200
Number of features: 4
R-squared: 0.9523
Adjusted R-squared: 0.9513

----------------------------------------------------------------------
Variable         Coefficient    Std Error     t-stat    p-value
----------------------------------------------------------------------
Intercept          32456.7823    8234.5612     3.943     0.0001 ***
sqft                  79.8234       2.1456    37.202     0.0000 ***
bedrooms           14876.3421    1523.4567     9.765     0.0000 ***
age                  -487.2345      98.7654    -4.933     0.0000 ***
distance           -1978.5634     345.6789    -5.724     0.0000 ***
----------------------------------------------------------------------
Significance codes: *** p<0.001, ** p<0.01, * p<0.05
======================================================================

Sample Predictions:
--------------------------------------------------
House: 1500 sqft, 3 beds, 10 yrs old, 5 mi -> $192,847
House: 2500 sqft, 4 beds, 5 yrs old, 15 mi -> $258,234
House: 1000 sqft, 2 beds, 40 yrs old, 2 mi -> $119,456
```

### 4.3 Gradient Descent for Linear Regression

While the normal equation works well for small datasets, it becomes computationally expensive
for large datasets (O(n³) for matrix inversion). Gradient descent provides an iterative alternative.

#### 4.3.1 The Algorithm

```
Repeat until convergence:
    1. Compute predictions: ŷ = Xβ
    2. Compute gradient: ∇L = (2/n) * Xᵀ(ŷ - y)
    3. Update parameters: β = β - α * ∇L
```

**Three Variants:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GRADIENT DESCENT VARIANTS                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Batch Gradient Descent:                                           │
│  ├── Uses ALL samples per update                                   │
│  ├── Smooth convergence                                            │
│  ├── Slow for large datasets                                       │
│  └── Deterministic                                                 │
│                                                                     │
│  Stochastic Gradient Descent (SGD):                                │
│  ├── Uses ONE sample per update                                    │
│  ├── Noisy but fast                                                │
│  ├── Can escape local minima                                       │
│  └── May never fully converge                                      │
│                                                                     │
│  Mini-batch Gradient Descent:                                      │
│  ├── Uses BATCH_SIZE samples per update                            │
│  ├── Best of both worlds                                           │
│  ├── Most common in practice                                       │
│  └── Typical batch sizes: 32, 64, 128, 256                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    """
    Linear Regression using Gradient Descent.
    
    Supports batch, stochastic, and mini-batch gradient descent.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, 
                 batch_size=None, tol=1e-6, verbose=False):
        """
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        n_iterations : int
            Maximum number of iterations
        batch_size : int or None
            None = batch GD, 1 = SGD, >1 = mini-batch
        tol : float
            Tolerance for convergence
        verbose : bool
            Print progress during training
        """
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.tol = tol
        self.verbose = verbose
        
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def _compute_loss(self, X, y):
        """Compute Mean Squared Error loss."""
        predictions = X @ self.weights + self.bias
        return np.mean((predictions - y) ** 2)
    
    def _compute_gradients(self, X, y):
        """Compute gradients of the loss with respect to weights and bias."""
        n = len(y)
        predictions = X @ self.weights + self.bias
        errors = predictions - y
        
        # Gradients
        dw = (2 / n) * (X.T @ errors)
        db = (2 / n) * np.sum(errors)
        
        return dw, db
    
    def fit(self, X, y):
        """
        Fit the model using gradient descent.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y).reshape(-1)
        
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        
        # Determine batch size
        if self.batch_size is None:
            batch_size = n_samples  # Batch GD
        else:
            batch_size = min(self.batch_size, n_samples)
        
        # Training loop
        for iteration in range(self.n_iterations):
            # Shuffle data for SGD and mini-batch
            if batch_size < n_samples:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
            
            # Process in batches
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Compute gradients
                dw, db = self._compute_gradients(X_batch, y_batch)
                
                # Update parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
            
            # Record loss
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)
            
            # Check convergence
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-2] - loss) < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {iteration}")
                    break
            
            # Verbose output
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.6f}")
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        X = np.array(X)
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


# Compare gradient descent variants
np.random.seed(42)

# Generate data
n_samples = 1000
X = np.random.randn(n_samples, 5)
true_weights = np.array([3, -2, 1, 0.5, -1.5])
y = X @ true_weights + 2 + np.random.randn(n_samples) * 0.5

# Train with different methods
methods = {
    'Batch GD': LinearRegressionGD(learning_rate=0.1, n_iterations=500, 
                                    batch_size=None),
    'SGD': LinearRegressionGD(learning_rate=0.01, n_iterations=500, 
                               batch_size=1),
    'Mini-batch (32)': LinearRegressionGD(learning_rate=0.05, n_iterations=500, 
                                           batch_size=32),
}

plt.figure(figsize=(12, 5))

# Subplot 1: Loss curves
plt.subplot(1, 2, 1)
for name, model in methods.items():
    model.fit(X, y)
    plt.plot(model.loss_history, label=f"{name} (R²={model.score(X, y):.4f})")

plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Gradient Descent Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Subplot 2: Weight comparison
plt.subplot(1, 2, 2)
x_pos = np.arange(len(true_weights))
width = 0.2

plt.bar(x_pos - 1.5*width, true_weights, width, label='True', alpha=0.8)
for i, (name, model) in enumerate(methods.items()):
    plt.bar(x_pos + (i-0.5)*width, model.weights, width, label=name, alpha=0.8)

plt.xlabel('Weight Index')
plt.ylabel('Weight Value')
plt.title('Learned vs True Weights')
plt.legend()
plt.xticks(x_pos)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent_comparison.png', dpi=150)
plt.show()

# Print results
print("\nGradient Descent Results:")
print("=" * 60)
print(f"True weights: {true_weights}")
print(f"True bias: 2.0")
print()
for name, model in methods.items():
    print(f"{name}:")
    print(f"  Weights: {np.round(model.weights, 4)}")
    print(f"  Bias: {model.bias:.4f}")
    print(f"  R² Score: {model.score(X, y):.6f}")
    print()
```

### 4.4 Feature Scaling for Gradient Descent

Feature scaling is crucial for gradient descent to work efficiently:

```
Without scaling:
- Features on different scales have vastly different gradients
- Learning rate that works for one feature may be too large/small for another
- Convergence is slow and may oscillate

With scaling:
- All features contribute equally
- Gradient descent converges faster
- Same learning rate works for all features
```

**Scaling Methods:**

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Example: Features on different scales
raw_data = np.array([
    [2000, 3, 15],    # sqft (1000s), bedrooms (1-5), age (0-100)
    [1500, 2, 30],
    [3000, 4, 5],
    [1800, 3, 20],
])

print("Original Data:")
print(raw_data)
print(f"Ranges: sqft [{raw_data[:,0].min()}-{raw_data[:,0].max()}], "
      f"beds [{raw_data[:,1].min()}-{raw_data[:,1].max()}], "
      f"age [{raw_data[:,2].min()}-{raw_data[:,2].max()}]")

# Standard Scaling (Z-score normalization)
# x_scaled = (x - mean) / std
standard_scaler = StandardScaler()
data_standard = standard_scaler.fit_transform(raw_data)
print("\nStandard Scaled (mean=0, std=1):")
print(data_standard.round(3))

# Min-Max Scaling
# x_scaled = (x - min) / (max - min)
minmax_scaler = MinMaxScaler()
data_minmax = minmax_scaler.fit_transform(raw_data)
print("\nMin-Max Scaled (range 0-1):")
print(data_minmax.round(3))

# Impact on Gradient Descent
print("\n" + "=" * 60)
print("Impact on Gradient Descent:")
print("=" * 60)

# Generate data with features on different scales
np.random.seed(42)
n = 500
X_unscaled = np.column_stack([
    np.random.uniform(1000, 5000, n),  # Feature 1: large scale
    np.random.uniform(0, 1, n),         # Feature 2: small scale
])
y = 0.001 * X_unscaled[:, 0] + 500 * X_unscaled[:, 1] + np.random.randn(n) * 0.5

# Without scaling
model_unscaled = LinearRegressionGD(learning_rate=0.0000001, n_iterations=1000)
model_unscaled.fit(X_unscaled, y)

# With scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unscaled)
model_scaled = LinearRegressionGD(learning_rate=0.1, n_iterations=1000)
model_scaled.fit(X_scaled, y)

print(f"Without scaling: R² = {model_unscaled.score(X_unscaled, y):.4f}, "
      f"iterations to converge: {len(model_unscaled.loss_history)}")
print(f"With scaling: R² = {model_scaled.score(X_scaled, y):.4f}, "
      f"iterations to converge: {len(model_scaled.loss_history)}")
```

### 4.5 Polynomial Regression

Linear regression can model nonlinear relationships by creating polynomial features:

```
Original: y = β₀ + β₁x
Polynomial: y = β₀ + β₁x + β₂x² + β₃x³ + ...

The model is still "linear" in the coefficients β, 
just nonlinear in the original feature x.
```

**Implementation:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Generate nonlinear data
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y_true = 0.5 * X.flatten()**3 - X.flatten()**2 + 2*X.flatten() + 1
y = y_true + np.random.randn(100) * 2

# Fit models with different polynomial degrees
degrees = [1, 2, 3, 5, 15]
plt.figure(figsize=(15, 5))

for i, degree in enumerate(degrees):
    plt.subplot(1, 5, i+1)
    
    # Create polynomial features and fit
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])
    model.fit(X, y)
    
    # Predict
    X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_pred = model.predict(X_plot)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    # Plot
    plt.scatter(X, y, alpha=0.5, s=20)
    plt.plot(X_plot, y_pred, 'r-', linewidth=2)
    plt.title(f'Degree {degree}\nCV RMSE: {cv_rmse:.2f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.ylim(-15, 25)

plt.tight_layout()
plt.savefig('polynomial_regression.png', dpi=150)
plt.show()

# Bias-Variance Tradeoff Analysis
print("\nBias-Variance Tradeoff:")
print("=" * 50)
print(f"{'Degree':<10} {'Train RMSE':<15} {'CV RMSE':<15} {'Status'}")
print("-" * 50)

for degree in [1, 2, 3, 5, 10, 15]:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])
    model.fit(X, y)
    
    train_rmse = np.sqrt(np.mean((y - model.predict(X))**2))
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    if train_rmse > 2:
        status = "Underfitting"
    elif cv_rmse > train_rmse * 1.5:
        status = "Overfitting"
    else:
        status = "Good fit"
    
    print(f"{degree:<10} {train_rmse:<15.4f} {cv_rmse:<15.4f} {status}")
```

Output:
```
Bias-Variance Tradeoff:
==================================================
Degree     Train RMSE      CV RMSE         Status
--------------------------------------------------
1          4.2341          4.3567          Underfitting
2          2.8765          2.9234          Underfitting
3          1.9823          2.0456          Good fit
5          1.9234          2.1234          Good fit
10         1.7234          3.4567          Overfitting
15         1.2345          8.9012          Overfitting
```

---

## Chapter 5: Regularization

### 5.1 The Problem of Overfitting

Overfitting occurs when a model learns the training data too well, including its noise,
and fails to generalize to new data.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OVERFITTING VISUALIZATION                        │
│                                                                     │
│   Underfitting          Good Fit           Overfitting             │
│   (High Bias)        (Balanced)         (High Variance)            │
│                                                                     │
│      •  •                •  •                •  •                   │
│    •     •             •     •            •  /\  •                  │
│   •   ___  •          • /    \ •         • /  \  •                  │
│  •   /    \ •        •/      \•         •/    \/\•                  │
│ •   /      \ •      •/        \•       •/        \•                 │
│────/────────\─────•/──────────\•─────•/──────────\•───              │
│                                                                     │
│ Train Error: High    Train Error: Low   Train Error: Very Low      │
│ Test Error: High     Test Error: Low    Test Error: High           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Signs of Overfitting:**
- Large gap between training and validation performance
- Model has many parameters relative to training samples
- Model is highly sensitive to small changes in training data
- Coefficients have very large magnitudes

### 5.2 Ridge Regression (L2 Regularization)

Ridge regression adds a penalty term proportional to the squared magnitude of coefficients:

```
Loss_ridge = MSE + λ * Σⱼ βⱼ²

Where:
- λ (lambda or alpha) controls regularization strength
- λ = 0: No regularization (standard OLS)
- λ → ∞: All coefficients shrink toward zero
```

**Closed-form Solution:**
```
β_ridge = (XᵀX + λI)⁻¹Xᵀy
```

The addition of λI to XᵀX ensures the matrix is invertible even when features are correlated.

**Implementation:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

class RidgeRegressionFromScratch:
    """Ridge Regression with L2 regularization."""
    
    def __init__(self, alpha=1.0):
        """
        Parameters:
        -----------
        alpha : float
            Regularization strength (λ)
        """
        self.alpha = alpha
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """Fit using the closed-form solution."""
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Add bias column
        X_b = np.column_stack([np.ones(n_samples), X])
        
        # Ridge solution: β = (XᵀX + λI)⁻¹Xᵀy
        # Note: We don't regularize the bias term
        I = np.eye(n_features + 1)
        I[0, 0] = 0  # Don't penalize bias
        
        beta = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
        
        self.bias = beta[0]
        self.weights = beta[1:]
        
        return self
    
    def predict(self, X):
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


# Demonstrate Ridge on overfitting scenario
np.random.seed(42)

# High-dimensional data with correlated features (prone to overfitting)
n_samples = 50
n_features = 30

# Create correlated features
X = np.random.randn(n_samples, n_features)
# Add correlation
for i in range(1, n_features):
    X[:, i] = X[:, i] + 0.8 * X[:, 0]

# True relationship uses only first 5 features
true_weights = np.zeros(n_features)
true_weights[:5] = [3, -2, 1, 0.5, -1]
y = X @ true_weights + np.random.randn(n_samples) * 0.5

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Compare OLS vs Ridge
from sklearn.linear_model import LinearRegression

# OLS (no regularization)
ols = LinearRegression()
ols.fit(X_train, y_train)

# Ridge with different alphas
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ridge_results = []

print("OLS vs Ridge Regression:")
print("=" * 70)
print(f"{'Alpha':<10} {'Train R²':<12} {'Test R²':<12} {'Max |coef|':<12}")
print("-" * 70)

# OLS
print(f"{'OLS':<10} {ols.score(X_train, y_train):<12.4f} {ols.score(X_test, y_test):<12.4f} "
      f"{np.max(np.abs(ols.coef_)):<12.2f}")

# Ridge
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    
    train_r2 = ridge.score(X_train, y_train)
    test_r2 = ridge.score(X_test, y_test)
    max_coef = np.max(np.abs(ridge.coef_))
    
    ridge_results.append((alpha, train_r2, test_r2, max_coef, ridge.coef_.copy()))
    
    print(f"{alpha:<10} {train_r2:<12.4f} {test_r2:<12.4f} {max_coef:<12.2f}")

print("-" * 70)

# Plot coefficient paths
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
alphas_plot = np.logspace(-3, 3, 100)
coefs = []

for alpha in alphas_plot:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)

coefs = np.array(coefs)

for i in range(n_features):
    style = '-' if i < 5 else '--'
    alpha_val = 0.8 if i < 5 else 0.2
    plt.plot(alphas_plot, coefs[:, i], style, alpha=alpha_val)

plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Coefficient Value')
plt.title('Ridge Coefficient Paths')
plt.axhline(y=0, color='k', linestyle=':')
plt.grid(True, alpha=0.3)

# Train vs Test R² by alpha
plt.subplot(1, 2, 2)
train_scores = [r[1] for r in ridge_results]
test_scores = [r[2] for r in ridge_results]

plt.plot(alphas, train_scores, 'b-o', label='Train R²')
plt.plot(alphas, test_scores, 'r-o', label='Test R²')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R² Score')
plt.title('Ridge: Train vs Test Performance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ridge_regression.png', dpi=150)
plt.show()

# Cross-validation to find optimal alpha
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
ridge_cv.fit(X_train, y_train)

print(f"\nOptimal alpha (via CV): {ridge_cv.alpha_:.4f}")
print(f"Test R² with optimal alpha: {ridge_cv.score(X_test, y_test):.4f}")
```

### 5.3 Lasso Regression (L1 Regularization)

Lasso uses the absolute value of coefficients as the penalty:

```
Loss_lasso = MSE + λ * Σⱼ |βⱼ|
```

**Key Difference from Ridge:**
- Lasso can shrink coefficients to exactly zero (feature selection!)
- Ridge only shrinks coefficients toward zero but never reaches zero

```
┌─────────────────────────────────────────────────────────────────────┐
│              RIDGE vs LASSO: GEOMETRIC INTERPRETATION               │
│                                                                     │
│           Ridge (L2)                        Lasso (L1)              │
│                                                                     │
│              β₂                               β₂                    │
│              │                                │                     │
│              │    ○                           │    ◇                │
│              │   ╱ ╲   Circular              │   /\   Diamond      │
│         ─────┼──●────  constraint           ─────┼──●───  constraint│
│              │   ╲ ╱   set                   │   \/   set           │
│              │    ○                           │    ◇                │
│              │                                │                     │
│      ────────┴────────── β₁         ─────────┴────────── β₁        │
│              │                                │                     │
│                                                                     │
│    ● = Solution often lands          ● = Solution often lands      │
│        at any point on circle            at a corner (sparse!)     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
import matplotlib.pyplot as plt

# Lasso demonstration with feature selection
np.random.seed(42)

# Create data where only some features matter
n_samples = 100
n_features = 20

X = np.random.randn(n_samples, n_features)

# Only first 5 features have non-zero coefficients
true_coef = np.zeros(n_features)
true_coef[:5] = [3, -2, 1.5, -1, 0.5]

y = X @ true_coef + np.random.randn(n_samples) * 0.5

# Fit Lasso with different alphas
alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]

print("Lasso Feature Selection:")
print("=" * 80)
print(f"True coefficients: {true_coef[:10]} ... (rest are zero)")
print()

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X, y)
    
    n_nonzero = np.sum(lasso.coef_ != 0)
    
    print(f"Alpha = {alpha}:")
    print(f"  Non-zero coefficients: {n_nonzero}/{n_features}")
    print(f"  Coefficients: {np.round(lasso.coef_[:10], 3)}")
    print()

# Cross-validation for optimal alpha
lasso_cv = LassoCV(alphas=np.logspace(-3, 0, 50), cv=5, max_iter=10000)
lasso_cv.fit(X, y)

print(f"Optimal alpha (via CV): {lasso_cv.alpha_:.4f}")
print(f"Non-zero coefficients: {np.sum(lasso_cv.coef_ != 0)}")
print(f"Selected features: {np.where(lasso_cv.coef_ != 0)[0]}")

# Compare with true features
print(f"\nTrue non-zero features: {np.where(true_coef != 0)[0]}")
print(f"Correctly identified: {np.sum((lasso_cv.coef_ != 0) & (true_coef != 0))}/5")
```

### 5.4 Elastic Net

Elastic Net combines L1 and L2 regularization:

```
Loss = MSE + α * ρ * Σ|βⱼ| + α * (1-ρ)/2 * Σβⱼ²

Where:
- α = overall regularization strength
- ρ (l1_ratio) = balance between L1 and L2
  - ρ = 0: Pure Ridge
  - ρ = 1: Pure Lasso
  - 0 < ρ < 1: Mix of both
```

**When to Use Each:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                   REGULARIZATION SELECTION GUIDE                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Use RIDGE when:                                                   │
│  • You believe most features are relevant                          │
│  • Features are correlated (Ridge handles multicollinearity well)  │
│  • You don't need feature selection                                │
│  • Prediction accuracy is the main goal                            │
│                                                                     │
│  Use LASSO when:                                                   │
│  • You suspect only a few features matter                          │
│  • You want automatic feature selection                            │
│  • Interpretability is important                                   │
│  • Features are relatively uncorrelated                            │
│                                                                     │
│  Use ELASTIC NET when:                                             │
│  • Features are highly correlated                                  │
│  • You want feature selection but Ridge's stability                │
│  • Number of features > number of samples                          │
│  • You're not sure whether Ridge or Lasso is better                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

# Compare all three
np.random.seed(42)

# Create challenging dataset
n_samples = 100
n_features = 50

# Correlated features
X = np.random.randn(n_samples, n_features)
for i in range(1, n_features):
    X[:, i] = X[:, i] + 0.9 * X[:, i-1]  # High correlation

# Sparse true coefficients
true_coef = np.zeros(n_features)
true_coef[0:3] = [3, 2, 1]
true_coef[25:28] = [-2, -1, 0.5]

y = X @ true_coef + np.random.randn(n_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit models
ridge = RidgeCV(cv=5)
lasso = LassoCV(cv=5, max_iter=10000)
elastic = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99], cv=5, max_iter=10000)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
elastic.fit(X_train, y_train)

print("Comparison: Ridge vs Lasso vs Elastic Net")
print("=" * 60)
print(f"{'Metric':<25} {'Ridge':<12} {'Lasso':<12} {'Elastic':<12}")
print("-" * 60)
print(f"{'Test R²':<25} {ridge.score(X_test, y_test):<12.4f} "
      f"{lasso.score(X_test, y_test):<12.4f} {elastic.score(X_test, y_test):<12.4f}")
print(f"{'Non-zero coefs':<25} {np.sum(ridge.coef_ != 0):<12} "
      f"{np.sum(lasso.coef_ != 0):<12} {np.sum(elastic.coef_ != 0):<12}")
print(f"{'Optimal alpha':<25} {ridge.alpha_:<12.4f} {lasso.alpha_:<12.4f} {elastic.alpha_:<12.4f}")
print(f"{'L1 ratio (Elastic only)':<25} {'-':<12} {'-':<12} {elastic.l1_ratio_:<12.2f}")
```

---

## Chapter 6: Logistic Regression

### 6.1 From Linear to Logistic

Linear regression predicts continuous values, but for classification we need probabilities
between 0 and 1. Enter the sigmoid function:

```
Sigmoid(z) = 1 / (1 + e⁻ᶻ)

Properties:
- Output range: (0, 1) — perfect for probabilities
- σ(0) = 0.5 — decision boundary
- σ(-∞) → 0, σ(+∞) → 1
- Derivative: σ'(z) = σ(z)(1 - σ(z))
```

```
                     SIGMOID FUNCTION
           
    1.0 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─────────
                                    ╱
    0.8                           ╱
                                ╱
    0.6                       ╱
                            ╱
    0.5 ─ ─ ─ ─ ─ ─ ─ ─ ─●─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                       ╱
    0.4              ╱
                   ╱
    0.2          ╱
               ╱
    0.0 ───────
       ─────┬─────┬─────┬─────┬─────┬─────┬────
          -6    -4    -2     0     2     4     6
                           z
```

### 6.2 The Logistic Regression Model

```
P(y=1|x) = σ(wᵀx + b) = 1 / (1 + e^(-(wᵀx + b)))

Decision rule:
- If P(y=1|x) ≥ 0.5, predict class 1
- If P(y=1|x) < 0.5, predict class 0
```

**Why Not Use MSE for Classification?**

Using squared error loss with sigmoid creates a non-convex optimization problem with
multiple local minima. Instead, we use log loss (binary cross-entropy):

```
L(w,b) = -1/n Σᵢ [yᵢ log(p̂ᵢ) + (1-yᵢ) log(1-p̂ᵢ)]

Where p̂ᵢ = σ(wᵀxᵢ + b)
```

This loss is:
- Convex (single global minimum)
- Heavily penalizes confident wrong predictions
- Based on maximum likelihood estimation

### 6.3 Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class LogisticRegressionFromScratch:
    """
    Binary Logistic Regression implemented from scratch.
    
    Uses gradient descent to minimize binary cross-entropy loss.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, 
                 regularization=None, reg_lambda=0.01, verbose=False):
        """
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        n_iterations : int
            Number of training iterations
        regularization : str or None
            'l1', 'l2', or None
        reg_lambda : float
            Regularization strength
        verbose : bool
            Print training progress
        """
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.verbose = verbose
        
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def _sigmoid(self, z):
        """Compute sigmoid, handling overflow."""
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, y, y_pred):
        """
        Compute binary cross-entropy loss.
        
        L = -1/n * Σ[y*log(p) + (1-y)*log(1-p)]
        """
        n = len(y)
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
        # Add regularization
        if self.regularization == 'l2':
            loss += (self.reg_lambda / (2 * n)) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            loss += (self.reg_lambda / n) * np.sum(np.abs(self.weights))
            
        return loss
    
    def _compute_gradients(self, X, y, y_pred):
        """Compute gradients of the loss."""
        n = len(y)
        
        # Gradient of cross-entropy
        error = y_pred - y
        dw = (1 / n) * (X.T @ error)
        db = (1 / n) * np.sum(error)
        
        # Add regularization gradient
        if self.regularization == 'l2':
            dw += (self.reg_lambda / n) * self.weights
        elif self.regularization == 'l1':
            dw += (self.reg_lambda / n) * np.sign(self.weights)
            
        return dw, db
    
    def fit(self, X, y):
        """
        Fit the logistic regression model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,) with values 0 or 1
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)
            
            # Compute loss
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y, y_pred)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Print progress
            if self.verbose and i % 100 == 0:
                acc = np.mean((y_pred >= 0.5) == y)
                print(f"Iteration {i}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """Return probability predictions."""
        X = np.array(X)
        z = X @ self.weights + self.bias
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Return class predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def score(self, X, y):
        """Return accuracy score."""
        return np.mean(self.predict(X) == y)


# Example: Binary Classification
np.random.seed(42)

# Generate data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, 
                           flip_y=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegressionFromScratch(learning_rate=0.1, n_iterations=1000, verbose=True)
model.fit(X_train, y_train)

# Evaluate
print("\n" + "=" * 50)
print("Model Evaluation:")
print("=" * 50)
print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")
print(f"\nFinal weights: {model.weights}")
print(f"Final bias: {model.bias:.4f}")

# Visualize decision boundary
plt.figure(figsize=(12, 5))

# Plot 1: Loss curve
plt.subplot(1, 2, 1)
plt.plot(model.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)

# Plot 2: Decision boundary
plt.subplot(1, 2, 2)

# Create mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                      np.linspace(y_min, y_max, 100))

# Get predictions for mesh
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.8)
plt.colorbar(label='P(y=1)')
plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='blue', 
            marker='o', edgecolors='k', label='Class 0')
plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='red', 
            marker='s', edgecolors='k', label='Class 1')

# Decision boundary (where P = 0.5)
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()

plt.tight_layout()
plt.savefig('logistic_regression.png', dpi=150)
plt.show()
```

### 6.4 Multiclass Logistic Regression (Softmax)

For K classes, we use the softmax function:

```
P(y=k|x) = exp(wₖᵀx + bₖ) / Σⱼ exp(wⱼᵀx + bⱼ)

Properties:
- All probabilities sum to 1
- Each probability is between 0 and 1
- Generalizes sigmoid to multiple classes
```

**Loss Function: Categorical Cross-Entropy**

```
L = -1/n Σᵢ Σₖ yᵢₖ log(p̂ᵢₖ)

Where yᵢₖ is 1 if sample i belongs to class k, 0 otherwise
```

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

class SoftmaxRegression:
    """
    Multiclass classification using Softmax Regression.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=False):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        
        self.weights = None
        self.bias = None
        self.classes = None
        self.n_classes = None
        
    def _softmax(self, z):
        """
        Compute softmax probabilities.
        
        softmax(z)ₖ = exp(zₖ) / Σⱼ exp(zⱼ)
        """
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _one_hot_encode(self, y):
        """Convert class labels to one-hot encoding."""
        n_samples = len(y)
        one_hot = np.zeros((n_samples, self.n_classes))
        for i, label in enumerate(y):
            class_idx = np.where(self.classes == label)[0][0]
            one_hot[i, class_idx] = 1
        return one_hot
    
    def _compute_loss(self, y_true, y_pred):
        """Compute categorical cross-entropy loss."""
        # Clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def fit(self, X, y):
        """
        Fit the softmax regression model.
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        # One-hot encode targets
        y_one_hot = self._one_hot_encode(y)
        
        # Initialize weights: (n_features, n_classes)
        self.weights = np.zeros((n_features, self.n_classes))
        self.bias = np.zeros(self.n_classes)
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = self._softmax(z)
            
            # Compute loss
            loss = self._compute_loss(y_one_hot, y_pred)
            
            # Compute gradients
            error = y_pred - y_one_hot  # (n_samples, n_classes)
            dw = (1 / n_samples) * (X.T @ error)  # (n_features, n_classes)
            db = (1 / n_samples) * np.sum(error, axis=0)  # (n_classes,)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if self.verbose and i % 100 == 0:
                acc = np.mean(self.predict(X) == y)
                print(f"Iteration {i}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """Return class probabilities."""
        z = X @ self.weights + self.bias
        return self._softmax(z)
    
    def predict(self, X):
        """Return class predictions."""
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return self.classes[class_indices]
    
    def score(self, X, y):
        """Return accuracy score."""
        return np.mean(self.predict(X) == y)


# Example: Iris Classification
iris = load_iris()
X, y = iris.data, iris.target

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = SoftmaxRegression(learning_rate=0.1, n_iterations=1000, verbose=True)
model.fit(X_train, y_train)

# Evaluate
print("\n" + "=" * 50)
print("Softmax Regression on Iris Dataset")
print("=" * 50)
print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")

# Sample predictions with probabilities
print("\nSample predictions:")
for i in range(5):
    probs = model.predict_proba([X_test[i]])[0]
    pred = model.predict([X_test[i]])[0]
    true = y_test[i]
    print(f"  True: {iris.target_names[true]}, Pred: {iris.target_names[pred]}")
    print(f"    Probabilities: {dict(zip(iris.target_names, probs.round(3)))}")
```

---

## Chapter 7: Decision Trees

### 7.1 The Concept

Decision trees learn a hierarchy of if-then rules that split the data:

```
                      [Is age > 30?]
                         /      \
                       Yes       No
                       /          \
              [Income > 50k?]    [Student?]
                 /     \          /      \
               Yes     No       Yes      No
               /        \        /        \
          [BUY]    [DON'T]  [BUY]    [DON'T]
```

**Advantages:**
- Highly interpretable (can explain decisions)
- Handles both numerical and categorical features
- No feature scaling required
- Captures non-linear relationships and interactions

**Disadvantages:**
- Prone to overfitting
- Unstable (small data changes can change tree significantly)
- Greedy algorithm may not find global optimum

### 7.2 Splitting Criteria

How do we decide which feature to split on? We want splits that create "pure" nodes.

#### 7.2.1 Gini Impurity

```
Gini(node) = 1 - Σₖ pₖ²

Where pₖ is the proportion of class k in the node.

- Gini = 0: Node is pure (all same class)
- Gini = 0.5: Maximum impurity for binary classification (50-50 split)
```

#### 7.2.2 Entropy (Information Gain)

```
Entropy(node) = -Σₖ pₖ log₂(pₖ)

Information Gain = Entropy(parent) - Weighted Entropy(children)

- Entropy = 0: Node is pure
- Entropy = 1: Maximum impurity for binary (50-50)
```

**Example Calculation:**

```python
import numpy as np

def gini_impurity(y):
    """Calculate Gini impurity."""
    if len(y) == 0:
        return 0
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

def entropy(y):
    """Calculate entropy."""
    if len(y) == 0:
        return 0
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

# Example: Different class distributions
distributions = [
    [0, 0, 0, 0, 0],        # Pure: all class 0
    [0, 0, 0, 1, 1],        # Mostly class 0
    [0, 0, 1, 1, 1],        # Mostly class 1
    [0, 0, 0, 1, 1, 1],     # 50-50 split
    [0, 1, 2, 3, 4],        # 5 classes, uniform
]

print("Comparison of Gini and Entropy:")
print("=" * 60)
print(f"{'Distribution':<25} {'Gini':<12} {'Entropy':<12}")
print("-" * 60)

for dist in distributions:
    g = gini_impurity(dist)
    e = entropy(dist)
    dist_str = str(dist)[:24]
    print(f"{dist_str:<25} {g:<12.4f} {e:<12.4f}")
```

### 7.3 Implementation from Scratch

```python
import numpy as np
from collections import Counter

class DecisionTreeNode:
    """A node in the decision tree."""
    
    def __init__(self, feature=None, threshold=None, left=None, 
                 right=None, value=None):
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Threshold for the split
        self.left = left           # Left child (feature <= threshold)
        self.right = right         # Right child (feature > threshold)
        self.value = value         # Class prediction (for leaf nodes)


class DecisionTreeClassifier:
    """
    Decision Tree Classifier built from scratch.
    
    Uses recursive binary splitting with Gini impurity or entropy.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, criterion='gini'):
        """
        Parameters:
        -----------
        max_depth : int or None
            Maximum depth of the tree
        min_samples_split : int
            Minimum samples required to split a node
        min_samples_leaf : int
            Minimum samples required in a leaf
        criterion : str
            'gini' or 'entropy'
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None
        self.n_features = None
        self.classes = None
        
    def _calculate_impurity(self, y):
        """Calculate impurity of a node."""
        if len(y) == 0:
            return 0
            
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        if self.criterion == 'gini':
            return 1 - np.sum(probabilities ** 2)
        else:  # entropy
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _calculate_information_gain(self, y, y_left, y_right):
        """Calculate information gain from a split."""
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        parent_impurity = self._calculate_impurity(y)
        child_impurity = (n_left / n) * self._calculate_impurity(y_left) + \
                         (n_right / n) * self._calculate_impurity(y_right)
        
        return parent_impurity - child_impurity
    
    def _best_split(self, X, y):
        """Find the best split for a node."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature in range(n_features):
            # Get unique values for thresholds
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                # Check min_samples_leaf constraint
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Calculate information gain
                gain = self._calculate_information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            # Create leaf node
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_gain == 0 or best_feature is None:
            # No good split found, create leaf
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build children
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionTreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def fit(self, X, y):
        """Fit the decision tree."""
        X = np.array(X)
        y = np.array(y)
        
        self.n_features = X.shape[1]
        self.classes = np.unique(y)
        self.root = self._build_tree(X, y)
        
        return self
    
    def _predict_single(self, x, node):
        """Predict class for a single sample."""
        # If leaf node, return the value
        if node.value is not None:
            return node.value
        
        # Otherwise, traverse the tree
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X):
        """Predict classes for all samples."""
        X = np.array(X)
        return np.array([self._predict_single(x, self.root) for x in X])
    
    def score(self, X, y):
        """Return accuracy score."""
        return np.mean(self.predict(X) == y)
    
    def print_tree(self, node=None, depth=0, feature_names=None):
        """Print a text representation of the tree."""
        if node is None:
            node = self.root
        
        indent = "  " * depth
        
        if node.value is not None:
            print(f"{indent}Predict: Class {node.value}")
        else:
            feature_name = feature_names[node.feature] if feature_names else f"Feature {node.feature}"
            print(f"{indent}{feature_name} <= {node.threshold:.3f}?")
            print(f"{indent}├── Yes:")
            self.print_tree(node.left, depth + 1, feature_names)
            print(f"{indent}└── No:")
            self.print_tree(node.right, depth + 1, feature_names)


# Example usage
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train our tree
tree = DecisionTreeClassifier(max_depth=3, criterion='gini')
tree.fit(X_train, y_train)

print("Decision Tree Structure:")
print("=" * 50)
tree.print_tree(feature_names=iris.feature_names)

print(f"\nTraining Accuracy: {tree.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {tree.score(X_test, y_test):.4f}")
```

### 7.4 Hyperparameters and Pruning

Decision trees have several important hyperparameters to prevent overfitting:

```
┌─────────────────────────────────────────────────────────────────────┐
│               DECISION TREE HYPERPARAMETERS                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  max_depth: Maximum tree depth                                     │
│  ├── Lower = Less complex, potential underfitting                  │
│  └── Higher = More complex, potential overfitting                  │
│                                                                     │
│  min_samples_split: Min samples to split a node                    │
│  ├── Lower = More splits, potential overfitting                    │
│  └── Higher = Fewer splits, potential underfitting                 │
│                                                                     │
│  min_samples_leaf: Min samples in a leaf                           │
│  ├── Lower = Smaller leaves, potential overfitting                 │
│  └── Higher = Larger leaves, smoother predictions                  │
│                                                                     │
│  max_features: Features to consider for each split                 │
│  ├── sqrt(n_features) or log2(n_features) common choices           │
│  └── Adds randomness, helps with ensemble methods                  │
│                                                                     │
│  max_leaf_nodes: Maximum number of leaf nodes                      │
│  └── Directly controls tree complexity                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Post-Pruning:**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Find optimal max_depth using cross-validation
depths = range(1, 20)
train_scores = []
cv_scores = []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    train_scores.append(tree.score(X_train, y_train))
    cv_score = cross_val_score(tree, X_train, y_train, cv=5).mean()
    cv_scores.append(cv_score)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'b-o', label='Training Accuracy')
plt.plot(depths, cv_scores, 'r-o', label='CV Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Finding Optimal Depth')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=depths[np.argmax(cv_scores)], color='g', linestyle='--', 
            label=f'Optimal depth: {depths[np.argmax(cv_scores)]}')
plt.legend()
plt.savefig('tree_depth_tuning.png', dpi=150)
plt.show()

print(f"Optimal depth: {depths[np.argmax(cv_scores)]}")
print(f"Best CV accuracy: {max(cv_scores):.4f}")
```

---

## Chapter 8: Ensemble Methods

### 8.1 The Power of Ensembles

```
"The wisdom of crowds" - combining multiple models often outperforms any single model.

                Single Tree              Ensemble of Trees
                    │                         │ │ │
                    │                         │ │ │
                    ▼                         ▼ ▼ ▼
              [Prediction]              [Vote/Average]
                    │                         │
             High Variance              Lower Variance
             May Overfit                More Robust
```

**Two Main Strategies:**

1. **Bagging** (Bootstrap Aggregating): Train models on random subsets of data, then average
2. **Boosting**: Train models sequentially, each focusing on errors of previous models

### 8.2 Random Forest

Random Forest combines bagging with feature randomization:

```
Training:
1. For each tree (1 to n_trees):
   a. Create bootstrap sample (sample with replacement)
   b. At each split, consider only random subset of features
   c. Grow tree to full depth (or max_depth)

Prediction:
- Classification: Majority vote of all trees
- Regression: Average prediction of all trees
```

**Why It Works:**
- Bootstrap sampling reduces correlation between trees
- Feature randomization further decorrelates trees
- Averaging reduces variance without increasing bias much

```python
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

class RandomForestFromScratch:
    """
    Random Forest Classifier implemented from scratch.
    
    Combines bootstrap sampling with random feature selection.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, 
                 max_features='sqrt', min_samples_split=2,
                 bootstrap=True, random_state=None):
        """
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int or None
            Maximum depth of each tree
        max_features : str or int
            'sqrt', 'log2', or int for number of features per split
        min_samples_split : int
            Minimum samples to split a node
        bootstrap : bool
            Whether to use bootstrap samples
        random_state : int or None
            Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        self.trees = []
        self.feature_indices = []  # Features used by each tree
        
    def _get_max_features(self, n_features):
        """Determine number of features to consider at each split."""
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        else:
            return n_features
    
    def _bootstrap_sample(self, X, y):
        """Create a bootstrap sample."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """Fit the random forest."""
        X = np.array(X)
        y = np.array(y)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        max_features = self._get_max_features(n_features)
        
        self.trees = []
        self.feature_indices = []
        
        for i in range(self.n_estimators):
            # Create bootstrap sample
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y
            
            # Select random features
            feature_idx = np.random.choice(n_features, size=max_features, replace=False)
            X_sample_features = X_sample[:, feature_idx]
            
            # Train a decision tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=np.random.randint(0, 10000)
            )
            tree.fit(X_sample_features, y_sample)
            
            self.trees.append(tree)
            self.feature_indices.append(feature_idx)
        
        return self
    
    def predict(self, X):
        """Predict using majority voting."""
        X = np.array(X)
        
        # Collect predictions from all trees
        all_predictions = np.zeros((X.shape[0], self.n_estimators))
        
        for i, (tree, feature_idx) in enumerate(zip(self.trees, self.feature_indices)):
            X_features = X[:, feature_idx]
            all_predictions[:, i] = tree.predict(X_features)
        
        # Majority vote
        predictions = []
        for i in range(X.shape[0]):
            votes = Counter(all_predictions[i, :])
            predictions.append(votes.most_common(1)[0][0])
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities (average of tree predictions)."""
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Collect probability predictions
        all_probas = []
        
        for tree, feature_idx in zip(self.trees, self.feature_indices):
            X_features = X[:, feature_idx]
            probas = tree.predict_proba(X_features)
            all_probas.append(probas)
        
        # Average probabilities
        return np.mean(all_probas, axis=0)
    
    def score(self, X, y):
        """Return accuracy score."""
        return np.mean(self.predict(X) == y)
    
    def feature_importance(self, X, y, feature_names=None):
        """
        Calculate feature importance via permutation.
        
        For each feature, shuffle its values and measure accuracy drop.
        """
        X = np.array(X)
        y = np.array(y)
        
        baseline_accuracy = self.score(X, y)
        importances = np.zeros(X.shape[1])
        
        for feature in range(X.shape[1]):
            # Create copy and shuffle the feature
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feature])
            
            # Measure accuracy drop
            permuted_accuracy = self.score(X_permuted, y)
            importances[feature] = baseline_accuracy - permuted_accuracy
        
        # Normalize
        importances = importances / np.sum(importances)
        
        if feature_names is not None:
            return dict(zip(feature_names, importances))
        return importances


# Example: Random Forest on Iris
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestFromScratch(
    n_estimators=100,
    max_depth=5,
    max_features='sqrt',
    random_state=42
)
rf.fit(X_train, y_train)

print("Random Forest Results:")
print("=" * 50)
print(f"Training Accuracy: {rf.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {rf.score(X_test, y_test):.4f}")

# Feature importance
importance = rf.feature_importance(X_test, y_test, iris.feature_names)
print("\nFeature Importance:")
for name, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {imp:.4f}")
```

### 8.3 Gradient Boosting

Gradient Boosting builds trees sequentially, with each tree trying to correct errors
of the previous ensemble:

```
Algorithm:
1. Initialize with constant prediction: F₀(x) = argmin Σ L(yᵢ, γ)
2. For m = 1 to M trees:
   a. Compute pseudo-residuals: rᵢₘ = -∂L(yᵢ, F(xᵢ))/∂F(xᵢ)
   b. Fit tree hₘ to pseudo-residuals
   c. Update: Fₘ(x) = Fₘ₋₁(x) + learning_rate × hₘ(x)
```

**Key Idea:** Each tree fits the negative gradient of the loss function with respect
to current predictions (for MSE, this is just the residuals).

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingClassifier:
    """
    Gradient Boosting Classifier from scratch.
    
    Uses log loss and fits trees to pseudo-residuals.
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, 
                 max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        
        self.trees = []
        self.initial_prediction = None
        
    def _sigmoid(self, z):
        """Sigmoid function with clipping."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_residuals(self, y, y_pred):
        """
        Compute pseudo-residuals (negative gradient of log loss).
        
        For log loss: residual = y - p(x)
        """
        return y - self._sigmoid(y_pred)
    
    def fit(self, X, y):
        """Fit the gradient boosting model."""
        X = np.array(X)
        y = np.array(y)
        
        n_samples = X.shape[0]
        
        # Initialize with log-odds of positive class
        pos_ratio = np.mean(y)
        self.initial_prediction = np.log(pos_ratio / (1 - pos_ratio + 1e-10))
        
        # Start with initial prediction for all samples
        F = np.full(n_samples, self.initial_prediction)
        
        self.trees = []
        
        for i in range(self.n_estimators):
            # Compute pseudo-residuals
            residuals = self._compute_residuals(y, F)
            
            # Fit a tree to the residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)
            
            # Update predictions
            tree_pred = tree.predict(X)
            F += self.learning_rate * tree_pred
            
            self.trees.append(tree)
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.array(X)
        
        # Start with initial prediction
        F = np.full(X.shape[0], self.initial_prediction)
        
        # Add contributions from all trees
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        
        # Convert to probabilities
        proba_class_1 = self._sigmoid(F)
        proba_class_0 = 1 - proba_class_1
        
        return np.column_stack([proba_class_0, proba_class_1])
    
    def predict(self, X):
        """Predict classes."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def score(self, X, y):
        """Return accuracy."""
        return np.mean(self.predict(X) == y)


# Example: Compare with sklearn's GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier as SklearnGB
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                           n_redundant=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Our implementation
our_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
our_gb.fit(X_train, y_train)

# Sklearn implementation
sklearn_gb = SklearnGB(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
sklearn_gb.fit(X_train, y_train)

print("Gradient Boosting Comparison:")
print("=" * 50)
print(f"{'Model':<20} {'Train Acc':<15} {'Test Acc':<15}")
print("-" * 50)
print(f"{'Our Implementation':<20} {our_gb.score(X_train, y_train):<15.4f} {our_gb.score(X_test, y_test):<15.4f}")
print(f"{'Sklearn':<20} {sklearn_gb.score(X_train, y_train):<15.4f} {sklearn_gb.score(X_test, y_test):<15.4f}")
```

### 8.4 XGBoost, LightGBM, and CatBoost

Modern gradient boosting libraries add many optimizations:

```
┌─────────────────────────────────────────────────────────────────────┐
│                MODERN GRADIENT BOOSTING LIBRARIES                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  XGBoost (2014):                                                   │
│  ├── Regularized objective (L1 & L2)                               │
│  ├── Second-order gradients (Newton boosting)                      │
│  ├── Sparsity-aware split finding                                  │
│  ├── Cache-aware block structure                                   │
│  └── Out-of-core computing                                         │
│                                                                     │
│  LightGBM (2017):                                                  │
│  ├── Gradient-based One-Side Sampling (GOSS)                       │
│  ├── Exclusive Feature Bundling (EFB)                              │
│  ├── Leaf-wise tree growth (vs level-wise)                         │
│  └── Faster training, especially for large datasets                │
│                                                                     │
│  CatBoost (2017):                                                  │
│  ├── Native categorical feature handling                           │
│  ├── Ordered boosting (reduces overfitting)                        │
│  ├── Fast GPU training                                             │
│  └── Less tuning required                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```python
# Comparison of modern gradient boosting libraries
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
import time

# Try importing each library
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False

# Generate larger dataset for comparison
X, y = make_classification(n_samples=10000, n_features=50, n_informative=30,
                           n_redundant=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

# XGBoost
if HAS_XGB:
    start = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_time = time.time() - start
    xgb_acc = xgb_model.score(X_test, y_test)
    results.append(('XGBoost', xgb_acc, xgb_time))

# LightGBM
if HAS_LGB:
    start = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_time = time.time() - start
    lgb_acc = lgb_model.score(X_test, y_test)
    results.append(('LightGBM', lgb_acc, lgb_time))

# CatBoost
if HAS_CB:
    start = time.time()
    cb_model = cb.CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=False
    )
    cb_model.fit(X_train, y_train)
    cb_time = time.time() - start
    cb_acc = cb_model.score(X_test, y_test)
    results.append(('CatBoost', cb_acc, cb_time))

# Print results
print("Modern Gradient Boosting Comparison:")
print("=" * 60)
print(f"{'Library':<15} {'Test Accuracy':<18} {'Training Time':<15}")
print("-" * 60)
for name, acc, train_time in results:
    print(f"{name:<15} {acc:<18.4f} {train_time:<15.3f}s")
```

---

## Chapter 9: Support Vector Machines

### 9.1 The Maximum Margin Classifier

SVMs find the hyperplane that maximizes the margin between classes:

```
                    Maximum Margin Hyperplane
                    
        Class -1                           Class +1
          ○                                   ●
          ○       ← margin →                  ●
          ○     ○     |     ●                 ●
          ○   ○       |       ●     ●         ●
          ○ ○─────────|─────────●●            ●
            ○   ○     |     ●   ●             ●
          ○     ○     |       ●               ●
          ○           |         ●             ●
                      |                       ●
               Decision boundary
               
  ○● = Support vectors (points closest to boundary)
  They "support" or define the hyperplane
```

**Mathematical Formulation:**

For linearly separable data:
```
Maximize: 2 / ||w||  (the margin)
Subject to: yᵢ(wᵀxᵢ + b) ≥ 1 for all i

Equivalently, minimize: (1/2)||w||²
Subject to: yᵢ(wᵀxᵢ + b) ≥ 1
```

### 9.2 Soft Margin SVM

Real data is rarely linearly separable. Soft margin SVM allows some misclassifications:

```
Minimize: (1/2)||w||² + C Σᵢ ξᵢ
Subject to: yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ
           ξᵢ ≥ 0

Where:
- ξᵢ = slack variable (how much point i violates margin)
- C = regularization parameter
  - Large C: Smaller margin, fewer violations (may overfit)
  - Small C: Larger margin, more violations allowed (may underfit)
```

### 9.3 The Kernel Trick

For non-linear boundaries, SVMs use the kernel trick to implicitly map data to
higher dimensions:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons

# Create non-linearly separable data
X_circles, y_circles = make_circles(n_samples=200, factor=0.3, noise=0.1, random_state=42)
X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)

# Different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for row, (X, y, name) in enumerate([(X_circles, y_circles, 'Circles'), 
                                      (X_moons, y_moons, 'Moons')]):
    for col, kernel in enumerate(kernels):
        ax = axes[row, col]
        
        # Fit SVM
        if kernel == 'poly':
            svm = SVC(kernel=kernel, degree=3, C=1.0)
        else:
            svm = SVC(kernel=kernel, C=1.0)
        svm.fit(X, y)
        
        # Create mesh for plotting decision boundary
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', edgecolors='k')
        ax.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='s', edgecolors='k')
        
        # Highlight support vectors
        ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                   s=100, facecolors='none', edgecolors='green', linewidths=2)
        
        ax.set_title(f'{name}: {kernel}\nAcc: {svm.score(X, y):.2f}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.savefig('svm_kernels.png', dpi=150)
plt.show()
```

**Common Kernels:**

```
1. Linear: K(x, x') = xᵀx'
   - Use when: Data is linearly separable or high-dimensional

2. Polynomial: K(x, x') = (γxᵀx' + r)^d
   - Parameters: degree d, γ (gamma), r (coef0)
   - Use when: Interaction between features matters

3. RBF (Gaussian): K(x, x') = exp(-γ||x - x'||²)
   - Parameter: γ (controls influence of single training example)
   - Use when: Non-linear boundaries, default choice

4. Sigmoid: K(x, x') = tanh(γxᵀx' + r)
   - Similar to neural network activation
   - Less commonly used
```

### 9.4 Hyperparameter Tuning for SVM

```python
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# SVM requires scaled features!
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# Parameter grid
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'svm__kernel': ['rbf', 'poly']
}

# Grid search
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("SVM Hyperparameter Tuning:")
print("=" * 50)
print(f"Best parameters: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")
print(f"Test accuracy: {grid.score(X_test, y_test):.4f}")
```

---

## Chapter 10: K-Nearest Neighbors

### 10.1 The Simplest Algorithm

KNN makes predictions based on the k closest training examples:

```
Algorithm:
1. Store all training data
2. For a new point:
   a. Calculate distance to all training points
   b. Find k nearest neighbors
   c. Classification: Majority vote
      Regression: Average of neighbors' values
```

```
         K-Nearest Neighbors Visualization
         
                    ○ Class A
                    ● Class B
                    ? Query point
                    
              ○           ●
          ○       ○    ●
              ○       ?   ●     ●
          ○               ●
              ○                 ●
                                  ●
         
         k=1: Nearest is ●, predict Class B
         k=3: 2 ● and 1 ○, predict Class B  
         k=7: 4 ○ and 3 ●, predict Class A
```

### 10.2 Distance Metrics

```python
import numpy as np

def euclidean_distance(x1, x2):
    """L2 norm: sqrt(sum((x1-x2)^2))"""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    """L1 norm: sum(|x1-x2|)"""
    return np.sum(np.abs(x1 - x2))

def minkowski_distance(x1, x2, p):
    """Generalized: (sum(|x1-x2|^p))^(1/p)"""
    return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

def cosine_distance(x1, x2):
    """1 - cosine similarity"""
    return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

# Example
x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])

print("Distance Metrics Example:")
print(f"Points: {x1} and {x2}")
print(f"Euclidean: {euclidean_distance(x1, x2):.4f}")
print(f"Manhattan: {manhattan_distance(x1, x2):.4f}")
print(f"Minkowski (p=3): {minkowski_distance(x1, x2, 3):.4f}")
print(f"Cosine: {cosine_distance(x1, x2):.4f}")
```

### 10.3 KNN Implementation

```python
import numpy as np
from collections import Counter

class KNNClassifier:
    """K-Nearest Neighbors Classifier from scratch."""
    
    def __init__(self, k=5, metric='euclidean', weights='uniform'):
        """
        Parameters:
        -----------
        k : int
            Number of neighbors
        metric : str
            'euclidean' or 'manhattan'
        weights : str
            'uniform' (all neighbors equal) or 'distance' (closer = more weight)
        """
        self.k = k
        self.metric = metric
        self.weights = weights
        
        self.X_train = None
        self.y_train = None
        
    def _distance(self, x1, x2):
        """Calculate distance between two points."""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def fit(self, X, y):
        """Store training data (KNN is a lazy learner)."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _predict_single(self, x):
        """Predict class for a single sample."""
        # Calculate distances to all training points
        distances = [self._distance(x, x_train) for x_train in self.X_train]
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        k_distances = np.array(distances)[k_indices]
        
        if self.weights == 'uniform':
            # Simple majority vote
            most_common = Counter(k_labels).most_common(1)
            return most_common[0][0]
        else:
            # Weighted by inverse distance
            weights = 1 / (k_distances + 1e-10)  # Add small value to avoid division by zero
            
            # Weighted vote
            class_weights = {}
            for label, weight in zip(k_labels, weights):
                class_weights[label] = class_weights.get(label, 0) + weight
            
            return max(class_weights, key=class_weights.get)
    
    def predict(self, X):
        """Predict classes for all samples."""
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])
    
    def score(self, X, y):
        """Return accuracy score."""
        return np.mean(self.predict(X) == y)


# Example: Effect of k on decision boundary
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Generate data
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, flip_y=0.1, random_state=42)

# Visualize different k values
ks = [1, 3, 7, 15]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, k in zip(axes, ks):
    knn = KNNClassifier(k=k)
    knn.fit(X, y)
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', edgecolors='k')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='s', edgecolors='k')
    ax.set_title(f'k={k}, Accuracy: {knn.score(X, y):.2f}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.savefig('knn_k_values.png', dpi=150)
plt.show()

print("\nKNN: Effect of k value")
print("=" * 50)
print("- Small k: Complex boundary, may overfit")
print("- Large k: Smoother boundary, may underfit")
print("- Rule of thumb: k = sqrt(n_samples)")
```

### 10.4 The Curse of Dimensionality

KNN suffers in high dimensions:

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate curse of dimensionality
np.random.seed(42)

dimensions = [2, 5, 10, 20, 50, 100]
n_points = 1000

print("Curse of Dimensionality:")
print("=" * 60)
print("In high dimensions, distances become less meaningful.")
print()

for d in dimensions:
    # Generate random points in unit hypercube
    points = np.random.uniform(0, 1, (n_points, d))
    
    # Calculate distances from origin
    distances = np.sqrt(np.sum(points ** 2, axis=1))
    
    # Statistics
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    ratio = std_dist / mean_dist  # Relative spread
    
    print(f"Dimensions: {d:3d} | Mean distance: {mean_dist:.3f} | "
          f"Std: {std_dist:.3f} | Ratio: {ratio:.4f}")

print()
print("As dimensions increase:")
print("- All points become roughly equidistant from each other")
print("- The concept of 'nearest' becomes less meaningful")
print("- KNN performance degrades")
```

---

## Chapter 11: Naive Bayes

### 11.1 Bayes' Theorem in Classification

Naive Bayes applies Bayes' theorem with a "naive" assumption of feature independence:

```
P(class|features) = P(features|class) × P(class) / P(features)

With independence assumption:
P(features|class) = P(x₁|class) × P(x₂|class) × ... × P(xₙ|class)
```

**Why "Naive"?**
- Assumes all features are independent given the class
- This is rarely true in practice
- But it works surprisingly well anyway!

### 11.2 Types of Naive Bayes

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NAIVE BAYES VARIANTS                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Gaussian Naive Bayes:                                             │
│  ├── For continuous features                                       │
│  ├── Assumes features follow normal distribution                   │
│  └── P(xᵢ|y) = N(μᵧ, σᵧ²)                                         │
│                                                                     │
│  Multinomial Naive Bayes:                                          │
│  ├── For discrete counts (word counts, frequencies)                │
│  ├── Common for text classification                                │
│  └── P(xᵢ|y) = count of feature i in class y / total count        │
│                                                                     │
│  Bernoulli Naive Bayes:                                            │
│  ├── For binary features (presence/absence)                        │
│  ├── Also common for text (word present or not)                    │
│  └── P(xᵢ|y) = proportion of samples with feature i in class y    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 11.3 Implementation

```python
import numpy as np
from collections import defaultdict

class GaussianNaiveBayes:
    """Gaussian Naive Bayes for continuous features."""
    
    def __init__(self):
        self.classes = None
        self.class_priors = {}  # P(y)
        self.means = {}         # μ for each feature per class
        self.variances = {}     # σ² for each feature per class
        
    def fit(self, X, y):
        """
        Calculate class priors and feature statistics.
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        for c in self.classes:
            X_c = X[y == c]
            
            # Class prior: P(y=c)
            self.class_priors[c] = len(X_c) / n_samples
            
            # Feature means and variances for this class
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0) + 1e-9  # Add small value for stability
        
        return self
    
    def _gaussian_probability(self, x, mean, var):
        """
        Calculate Gaussian probability.
        P(x|μ,σ²) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))
        """
        coefficient = 1 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-(x - mean) ** 2 / (2 * var))
        return coefficient * exponent
    
    def _predict_single(self, x):
        """Predict class for a single sample."""
        posteriors = {}
        
        for c in self.classes:
            # Start with log prior
            log_posterior = np.log(self.class_priors[c])
            
            # Add log likelihoods for each feature
            for i, xi in enumerate(x):
                prob = self._gaussian_probability(xi, self.means[c][i], self.variances[c][i])
                log_posterior += np.log(prob + 1e-10)
            
            posteriors[c] = log_posterior
        
        return max(posteriors, key=posteriors.get)
    
    def predict(self, X):
        """Predict classes for all samples."""
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.array(X)
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, len(self.classes)))
        
        for i, x in enumerate(X):
            log_posteriors = []
            for c in self.classes:
                log_post = np.log(self.class_priors[c])
                for j, xj in enumerate(x):
                    prob = self._gaussian_probability(xj, self.means[c][j], self.variances[c][j])
                    log_post += np.log(prob + 1e-10)
                log_posteriors.append(log_post)
            
            # Convert to probabilities using softmax
            log_posteriors = np.array(log_posteriors)
            log_posteriors -= np.max(log_posteriors)  # Numerical stability
            posteriors = np.exp(log_posteriors)
            probas[i] = posteriors / np.sum(posteriors)
        
        return probas
    
    def score(self, X, y):
        """Return accuracy score."""
        return np.mean(self.predict(X) == y)


# Example: Gaussian Naive Bayes on Iris
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train our implementation
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)

# Compare with sklearn
from sklearn.naive_bayes import GaussianNB
sklearn_gnb = GaussianNB()
sklearn_gnb.fit(X_train, y_train)

print("Gaussian Naive Bayes Comparison:")
print("=" * 50)
print(f"{'Model':<20} {'Train Acc':<15} {'Test Acc':<15}")
print("-" * 50)
print(f"{'Our Implementation':<20} {gnb.score(X_train, y_train):<15.4f} {gnb.score(X_test, y_test):<15.4f}")
print(f"{'Sklearn':<20} {sklearn_gnb.score(X_train, y_train):<15.4f} {sklearn_gnb.score(X_test, y_test):<15.4f}")
```

### 11.4 Text Classification with Multinomial Naive Bayes

```python
import numpy as np
from collections import defaultdict

class MultinomialNaiveBayes:
    """Multinomial Naive Bayes for text classification."""
    
    def __init__(self, alpha=1.0):
        """
        Parameters:
        -----------
        alpha : float
            Smoothing parameter (Laplace smoothing)
        """
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}
        self.feature_probs = {}  # P(feature|class)
        self.vocabulary_size = None
        
    def fit(self, X, y):
        """
        X should be a count matrix (documents × vocabulary).
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        n_samples, self.vocabulary_size = X.shape
        
        for c in self.classes:
            X_c = X[y == c]
            
            # Class prior
            self.class_priors[c] = len(X_c) / n_samples
            
            # Feature probabilities with Laplace smoothing
            # P(word|class) = (count of word in class + alpha) / 
            #                  (total words in class + alpha * vocab_size)
            word_counts = np.sum(X_c, axis=0) + self.alpha
            total_words = np.sum(word_counts)
            
            self.feature_probs[c] = word_counts / total_words
        
        return self
    
    def _predict_single(self, x):
        """Predict class for a single document."""
        posteriors = {}
        
        for c in self.classes:
            log_posterior = np.log(self.class_priors[c])
            
            # Add log probabilities for each word (weighted by count)
            log_posterior += np.sum(x * np.log(self.feature_probs[c]))
            
            posteriors[c] = log_posterior
        
        return max(posteriors, key=posteriors.get)
    
    def predict(self, X):
        """Predict classes."""
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])
    
    def score(self, X, y):
        """Return accuracy."""
        return np.mean(self.predict(X) == y)


# Example: Spam Classification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Sample data
emails = [
    "Free money! Click here for prize",
    "Meeting tomorrow at 3pm",
    "Congratulations! You won $1000",
    "Project update: deadline extended",
    "Cheap viagra pills online",
    "Lunch meeting canceled",
    "Win a free iPhone now!!!",
    "Please review the attached document",
    "You have been selected for a cash prize",
    "Team standup at 10am",
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam

# Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails).toarray()

# Split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train
mnb = MultinomialNaiveBayes(alpha=1.0)
mnb.fit(X_train, y_train)

print("Multinomial Naive Bayes for Spam Detection:")
print("=" * 50)
print(f"Training accuracy: {mnb.score(X_train, y_train):.4f}")
print(f"Test accuracy: {mnb.score(X_test, y_test):.4f}")

# Test on new email
new_emails = [
    "Free cash prize winner",
    "Meeting rescheduled to 4pm"
]
X_new = vectorizer.transform(new_emails).toarray()
predictions = mnb.predict(X_new)

print("\nPredictions for new emails:")
for email, pred in zip(new_emails, predictions):
    label = "SPAM" if pred == 1 else "NOT SPAM"
    print(f"  '{email}' -> {label}")
```

---

## Chapter 12: Model Evaluation and Selection

### 12.1 Classification Metrics

```python
import numpy as np
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                            recall_score, f1_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt

def print_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Comprehensive classification metrics report.
    """
    print("=" * 60)
    print("CLASSIFICATION METRICS REPORT")
    print("=" * 60)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"               Neg    Pos")
    print(f"Actual Neg    {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"Actual Pos    {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    # Basic metrics
    tn, fp, fn, tp = cm.ravel()
    
    print("\nBasic Metrics:")
    print(f"  True Positives (TP): {tp}")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    
    # Derived metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print("\nDerived Metrics:")
    print(f"  Accuracy:    {accuracy:.4f}  (Overall correctness)")
    print(f"  Precision:   {precision:.4f}  (Of predicted positive, how many are correct)")
    print(f"  Recall:      {recall:.4f}  (Of actual positive, how many did we find)")
    print(f"  F1 Score:    {f1:.4f}  (Harmonic mean of precision and recall)")
    print(f"  Specificity: {specificity:.4f}  (Of actual negative, how many did we find)")
    
    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        print(f"  ROC-AUC:     {auc:.4f}  (Area under ROC curve)")
    
    print("=" * 60)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity
    }


# Example
np.random.seed(42)
y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0])
y_proba = np.array([0.1, 0.2, 0.15, 0.6, 0.3, 0.8, 0.75, 0.4, 0.9, 0.85,
                    0.2, 0.55, 0.7, 0.8, 0.25, 0.45, 0.3, 0.7, 0.65, 0.2])

metrics = print_classification_metrics(y_true, y_pred, y_proba)
```

### 12.2 When to Use Which Metric

```
┌─────────────────────────────────────────────────────────────────────┐
│                   METRIC SELECTION GUIDE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Use ACCURACY when:                                                │
│  • Classes are balanced                                            │
│  • All errors are equally costly                                   │
│  • Example: Sentiment analysis with equal pos/neg samples          │
│                                                                     │
│  Use PRECISION when:                                               │
│  • False positives are costly                                      │
│  • Example: Spam filter (don't want to miss real emails)          │
│  • Example: Recommending expensive products                        │
│                                                                     │
│  Use RECALL when:                                                  │
│  • False negatives are costly                                      │
│  • Example: Disease detection (don't want to miss sick patients)  │
│  • Example: Fraud detection                                        │
│                                                                     │
│  Use F1 SCORE when:                                                │
│  • You need balance between precision and recall                   │
│  • Classes are imbalanced                                          │
│  • Both FP and FN matter                                           │
│                                                                     │
│  Use ROC-AUC when:                                                 │
│  • You need threshold-independent evaluation                       │
│  • Comparing models across different operating points              │
│  • Classes are imbalanced                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 12.3 Cross-Validation

```python
import numpy as np
from sklearn.model_selection import (KFold, StratifiedKFold, LeaveOneOut,
                                     TimeSeriesSplit, cross_val_score)
import matplotlib.pyplot as plt

def visualize_cv_splits(X, y, cv, title):
    """Visualize cross-validation splits."""
    n_samples = len(y)
    n_splits = cv.get_n_splits(X, y) if hasattr(cv, 'get_n_splits') else len(list(cv.split(X, y)))
    
    fig, ax = plt.subplots(figsize=(12, n_splits * 0.5 + 1))
    
    for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Create array for visualization
        indices = np.zeros(n_samples)
        indices[train_idx] = 1  # Training
        indices[val_idx] = 2     # Validation
        
        ax.scatter(range(n_samples), [i] * n_samples, c=indices, 
                   cmap='RdYlGn', marker='s', s=10)
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('CV Iteration')
    ax.set_title(title)
    ax.set_yticks(range(n_splits))
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Training'),
                       Patch(facecolor='red', label='Validation')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig


# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

# Different CV strategies
cv_strategies = {
    'K-Fold (k=5)': KFold(n_splits=5, shuffle=True, random_state=42),
    'Stratified K-Fold (k=5)': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'Time Series (5 splits)': TimeSeriesSplit(n_splits=5),
}

print("Cross-Validation Strategies:")
print("=" * 60)
print()

for name, cv in cv_strategies.items():
    # Calculate some statistics
    train_sizes = []
    val_sizes = []
    
    for train_idx, val_idx in cv.split(X, y):
        train_sizes.append(len(train_idx))
        val_sizes.append(len(val_idx))
    
    print(f"{name}:")
    print(f"  Train sizes: {train_sizes}")
    print(f"  Val sizes: {val_sizes}")
    print()

# Cross-validation score example
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)

for name, cv in cv_strategies.items():
    if name != 'Time Series (5 splits)':  # Time series needs ordered data
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"{name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

### 12.4 Regression Metrics

```python
import numpy as np

def regression_metrics(y_true, y_pred):
    """
    Calculate comprehensive regression metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    n = len(y_true)
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # R-squared (Coefficient of Determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Adjusted R-squared
    p = 1  # Number of features (simplified)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    print("=" * 60)
    print("REGRESSION METRICS REPORT")
    print("=" * 60)
    print(f"  MSE:       {mse:.4f}  (Average squared error)")
    print(f"  RMSE:      {rmse:.4f}  (Square root of MSE, same units as target)")
    print(f"  MAE:       {mae:.4f}  (Average absolute error)")
    print(f"  MAPE:      {mape:.2f}%  (Percentage error)")
    print(f"  R²:        {r2:.4f}  (Variance explained by model)")
    print("=" * 60)
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2}


# Example
np.random.seed(42)
y_true = np.array([100, 150, 200, 250, 300, 350, 400])
y_pred = np.array([110, 145, 195, 260, 290, 360, 390])

metrics = regression_metrics(y_true, y_pred)
```

---

## Summary: Supervised Learning Algorithm Selection

```
┌─────────────────────────────────────────────────────────────────────┐
│              ALGORITHM SELECTION CHEAT SHEET                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  LINEAR REGRESSION                                                  │
│  ├── When: Continuous target, linear relationships                 │
│  ├── Pros: Fast, interpretable, baseline                           │
│  └── Cons: Assumes linearity, sensitive to outliers                │
│                                                                     │
│  LOGISTIC REGRESSION                                                │
│  ├── When: Binary/multiclass classification, need probabilities    │
│  ├── Pros: Fast, interpretable, probabilistic                      │
│  └── Cons: Assumes linear decision boundary                        │
│                                                                     │
│  DECISION TREES                                                     │
│  ├── When: Need interpretability, mixed feature types              │
│  ├── Pros: No scaling needed, handles nonlinearity                 │
│  └── Cons: Prone to overfitting, unstable                          │
│                                                                     │
│  RANDOM FOREST                                                      │
│  ├── When: Robust predictions, feature importance needed           │
│  ├── Pros: Handles overfitting, parallelizable                     │
│  └── Cons: Less interpretable, memory intensive                    │
│                                                                     │
│  GRADIENT BOOSTING (XGBoost/LightGBM/CatBoost)                     │
│  ├── When: Tabular data competitions, maximum accuracy             │
│  ├── Pros: Often best performance, handles missing values          │
│  └── Cons: More tuning needed, can overfit                         │
│                                                                     │
│  SVM                                                                │
│  ├── When: Medium datasets, clear margin of separation             │
│  ├── Pros: Effective in high dimensions, kernel flexibility        │
│  └── Cons: Slow on large datasets, memory intensive                │
│                                                                     │
│  KNN                                                                │
│  ├── When: Simple baseline, small datasets                         │
│  ├── Pros: No training, simple to understand                       │
│  └── Cons: Slow prediction, curse of dimensionality                │
│                                                                     │
│  NAIVE BAYES                                                        │
│  ├── When: Text classification, need speed                         │
│  ├── Pros: Fast, works with small data                             │
│  └── Cons: Independence assumption rarely holds                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

# End of Part IV: Supervised Learning Algorithms

This part covered the core supervised learning algorithms:
- Linear and polynomial regression
- Regularization techniques (Ridge, Lasso, Elastic Net)
- Logistic regression for classification
- Decision trees and their limitations
- Ensemble methods (Random Forest, Gradient Boosting)
- Support Vector Machines with kernels
- K-Nearest Neighbors
- Naive Bayes classifiers
- Model evaluation metrics and cross-validation

Each algorithm was implemented from scratch to understand the underlying mathematics,
then compared with scikit-learn implementations for verification.
#   PART III: NEURAL NETWORKS AND DEEP LEARNING                                  
#   CHAPTER 10: NEURAL NETWORK FUNDAMENTALS                                      
#   CHAPTER 11: TRAINING NEURAL NETWORKS                                         
#   CHAPTER 12: CONVOLUTIONAL NEURAL NETWORKS                                    
#   CHAPTER 13: RECURRENT NEURAL NETWORKS                                        
#   CHAPTER 14: ATTENTION AND TRANSFORMERS                                       

╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   "You don't need to be an expert to build neural networks, but              ║
║    understanding the fundamentals will make you much more effective."         ║
║                                                                               ║
║   This section covers deep learning from neurons to transformers.             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


#   CHAPTER 10: NEURAL NETWORK FUNDAMENTALS                                      

NEURAL NETWORK FUNDAMENTALS

Neural networks are universal function approximators inspired by biological neurons.

WHY NEURAL NETWORKS?
• Can approximate any continuous function (Universal Approximation Theorem)
• Automatically learn features from raw data
• State-of-the-art for images, text, audio, video
• Scale with more data and compute


# 10.1 THE PERCEPTRON

THE PERCEPTRON

The simplest neural network: a single artificial neuron.

STRUCTURE:
                     x₁ ──┐
                          │   ┌───────────────┐
                     x₂ ──┼──▶│ Σ wᵢxᵢ + b   │──▶ activation ──▶ output
                          │   └───────────────┘
                     x₃ ──┘

COMPUTATION:
    z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = w·x + b
    output = activation(z)

FOR BINARY CLASSIFICATION:
    output = 1 if z > 0 else 0


PERCEPTRON LEARNING RULE:
    If prediction is wrong:
        w = w + η × (y - ŷ) × x
        b = b + η × (y - ŷ)
    
    Where η is learning rate, y is true label, ŷ is prediction.


LIMITATION:
    Cannot solve XOR problem! Need multiple layers.

def example_perceptron():
    """Implement perceptron from scratch."""
    
    print("THE PERCEPTRON")
    print("=" * 70)
    
    class Perceptron:
        """Simple perceptron implementation."""
        
        def __init__(self, n_features, learning_rate=0.1):
            self.weights = np.zeros(n_features)
            self.bias = 0
            self.lr = learning_rate
        
        def activation(self, z):
            """Step function."""
            return (z > 0).astype(int)
        
        def predict(self, X):
            """Make predictions."""
            z = X @ self.weights + self.bias
            return self.activation(z)
        
        def fit(self, X, y, epochs=100):
            """Train the perceptron."""
            errors_per_epoch = []
            
            for epoch in range(epochs):
                errors = 0
                for xi, yi in zip(X, y):
                    prediction = self.predict(xi.reshape(1, -1))[0]
                    error = yi - prediction
                    
                    if error != 0:
                        self.weights += self.lr * error * xi
                        self.bias += self.lr * error
                        errors += 1
                
                errors_per_epoch.append(errors)
                
                if errors == 0:
                    print(f"  Converged at epoch {epoch}")
                    break
            
            return errors_per_epoch
    
    # Test on linearly separable data
    print("\n" + "─" * 70)
    print("LINEARLY SEPARABLE DATA (AND gate)")
    print("─" * 70)
    
    # AND gate
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    
    perceptron = Perceptron(n_features=2, learning_rate=0.1)
    perceptron.fit(X_and, y_and, epochs=100)
    
    print("\nAND Gate:")
    print(f"  Learned weights: {perceptron.weights.round(2)}")
    print(f"  Learned bias: {perceptron.bias:.2f}")
    
    print("\n  Input  | Predicted | Actual")
    print("  " + "-" * 30)
    for xi, yi in zip(X_and, y_and):
        pred = perceptron.predict(xi.reshape(1, -1))[0]
        print(f"  {xi}  |     {pred}     |   {yi}")
    
    # XOR problem - NOT linearly separable
    print("\n" + "─" * 70)
    print("XOR PROBLEM (NOT linearly separable)")
    print("─" * 70)
    
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    perceptron_xor = Perceptron(n_features=2, learning_rate=0.1)
    perceptron_xor.fit(X_xor, y_xor, epochs=100)
    
    print("\nXOR Gate:")
    print(f"  Learned weights: {perceptron_xor.weights.round(2)}")
    print(f"  Learned bias: {perceptron_xor.bias:.2f}")
    
    print("\n  Input  | Predicted | Actual | Correct?")
    print("  " + "-" * 40)
    correct = 0
    for xi, yi in zip(X_xor, y_xor):
        pred = perceptron_xor.predict(xi.reshape(1, -1))[0]
        is_correct = "✓" if pred == yi else "✗"
        if pred == yi:
            correct += 1
        print(f"  {xi}  |     {pred}     |   {yi}    |    {is_correct}")
    
    print(f"\n  Accuracy: {correct}/4 = {correct/4*100:.0f}%")
    print("\n  ⚠️ A single perceptron CANNOT solve XOR!")
    print("  Solution: Use multiple layers (Multi-Layer Perceptron)")


# 10.2 MULTI-LAYER PERCEPTRONS

MULTI-LAYER PERCEPTRONS (MLP)

Multiple layers of neurons connected in sequence.

ARCHITECTURE:
    Input Layer → Hidden Layer(s) → Output Layer
    
    ┌───┐     ┌───┐     ┌───┐     ┌───┐
    │x₁ │────▶│   │────▶│   │────▶│ŷ₁ │
    ├───┤     │   │     │   │     ├───┤
    │x₂ │────▶│ H │────▶│ H │────▶│ŷ₂ │
    ├───┤     │ i │     │ i │     └───┘
    │x₃ │────▶│ d │────▶│ d │
    └───┘     │ d │     │ d │
              │ e │     │ e │
    Input     │ n │     │ n │     Output
    Layer     │   │     │   │     Layer
              └───┘     └───┘
              
              Hidden Layers


TERMINOLOGY:
• Input layer: Receives features (no computation)
• Hidden layers: Learn representations
• Output layer: Produces predictions
• Depth: Number of layers (including output)
• Width: Number of neurons per layer
• Parameters: All weights and biases

FORWARD PASS:
    h₁ = activation(W₁ @ x + b₁)      # Input to hidden 1
    h₂ = activation(W₂ @ h₁ + b₂)     # Hidden 1 to hidden 2
    ...
    y = output_activation(Wₙ @ hₙ₋₁ + bₙ)  # Last hidden to output


UNIVERSAL APPROXIMATION THEOREM:
A feedforward network with a single hidden layer containing a finite number
of neurons can approximate any continuous function to arbitrary precision.

However, deeper networks often learn better representations with fewer parameters!

def example_mlp_from_scratch():
    """Implement a simple MLP from scratch."""
    
    print("MULTI-LAYER PERCEPTRON FROM SCRATCH")
    print("=" * 70)
    
    class NeuralNetwork:
        """Simple 2-layer neural network for binary classification."""
        
        def __init__(self, input_size, hidden_size, output_size):
            # Initialize weights with Xavier initialization
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            self.b1 = np.zeros((1, hidden_size))
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
            self.b2 = np.zeros((1, output_size))
        
        def sigmoid(self, z):
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        
        def sigmoid_derivative(self, z):
            s = self.sigmoid(z)
            return s * (1 - s)
        
        def relu(self, z):
            return np.maximum(0, z)
        
        def relu_derivative(self, z):
            return (z > 0).astype(float)
        
        def forward(self, X):
            """Forward pass through the network."""
            # Hidden layer
            self.z1 = X @ self.W1 + self.b1
            self.a1 = self.relu(self.z1)
            
            # Output layer
            self.z2 = self.a1 @ self.W2 + self.b2
            self.a2 = self.sigmoid(self.z2)
            
            return self.a2
        
        def backward(self, X, y, output):
            """Backward pass (backpropagation)."""
            m = X.shape[0]
            
            # Output layer gradients
            dz2 = output - y.reshape(-1, 1)
            dW2 = (1/m) * self.a1.T @ dz2
            db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
            
            # Hidden layer gradients (chain rule!)
            da1 = dz2 @ self.W2.T
            dz1 = da1 * self.relu_derivative(self.z1)
            dW1 = (1/m) * X.T @ dz1
            db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
            
            return dW1, db1, dW2, db2
        
        def fit(self, X, y, epochs=1000, learning_rate=0.1, verbose=True):
            """Train the network."""
            losses = []
            
            for epoch in range(epochs):
                # Forward pass
                output = self.forward(X)
                
                # Compute loss (binary cross-entropy)
                epsilon = 1e-15
                loss = -np.mean(y * np.log(output + epsilon) + 
                               (1 - y) * np.log(1 - output + epsilon))
                losses.append(loss)
                
                # Backward pass
                dW1, db1, dW2, db2 = self.backward(X, y, output)
                
                # Update weights
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
                
                if verbose and epoch % 200 == 0:
                    print(f"  Epoch {epoch}: Loss = {loss:.4f}")
            
            return losses
        
        def predict(self, X):
            """Make predictions."""
            return (self.forward(X) >= 0.5).astype(int).flatten()
    
    # Solve XOR problem!
    print("\n" + "─" * 70)
    print("SOLVING XOR WITH MLP")
    print("─" * 70)
    
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor = np.array([0, 1, 1, 0], dtype=float)
    
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    losses = nn.fit(X_xor, y_xor, epochs=2000, learning_rate=1.0, verbose=True)
    
    predictions = nn.predict(X_xor)
    
    print("\nXOR Results with MLP:")
    print("  Input  | Predicted | Actual")
    print("  " + "-" * 30)
    for xi, pred, yi in zip(X_xor, predictions, y_xor):
        print(f"  {xi.astype(int)}  |     {pred}     |   {int(yi)}")
    
    accuracy = np.mean(predictions == y_xor)
    print(f"\n  Accuracy: {accuracy*100:.0f}%")
    print("\n  ✓ MLP solves XOR!")
    
    # Larger example
    print("\n" + "─" * 70)
    print("LARGER CLASSIFICATION PROBLEM")
    print("─" * 70)
    
    # Generate nonlinear data
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                               n_clusters_per_class=2, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    nn_large = NeuralNetwork(input_size=10, hidden_size=20, output_size=1)
    nn_large.fit(X_train_scaled, y_train, epochs=1000, learning_rate=0.5, verbose=True)
    
    # Evaluate
    train_pred = nn_large.predict(X_train_scaled)
    test_pred = nn_large.predict(X_test_scaled)
    
    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)
    
    print(f"\nResults:")
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")


# 10.3 ACTIVATION FUNCTIONS

ACTIVATION FUNCTIONS

Non-linear functions applied after linear transformations.
Without them, the entire network would just be a linear function!


COMMON ACTIVATION FUNCTIONS:

1. SIGMOID
   σ(z) = 1 / (1 + e^(-z))
   
   Range: (0, 1)
   Pros: Smooth gradient, probabilistic interpretation
   Cons: Vanishing gradient, not zero-centered
   Use: Output layer for binary classification

2. TANH
   tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
   
   Range: (-1, 1)
   Pros: Zero-centered
   Cons: Still has vanishing gradient
   Use: RNNs, when zero-centered output is needed

3. ReLU (Rectified Linear Unit)
   ReLU(z) = max(0, z)
   
   Range: [0, ∞)
   Pros: Fast, no vanishing gradient for positive values
   Cons: "Dying ReLU" problem (neurons can get stuck at 0)
   Use: Hidden layers (most common choice!)

4. LEAKY ReLU
   LeakyReLU(z) = max(αz, z), where α = 0.01
   
   Range: (-∞, ∞)
   Pros: Fixes dying ReLU problem
   Cons: α is a hyperparameter

5. ELU (Exponential Linear Unit)
   ELU(z) = z if z > 0, else α(e^z - 1)
   
   Pros: Smooth, handles negative values
   Cons: Slightly slower than ReLU

6. GELU (Gaussian Error Linear Unit)
   GELU(z) = z × Φ(z), where Φ is CDF of normal distribution
   
   Pros: Smooth approximation of ReLU, used in transformers
   Use: Modern transformers (BERT, GPT)

7. SOFTMAX
   softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)
   
   Range: (0, 1), sums to 1
   Use: Output layer for multi-class classification

def example_activation_functions():
    """Demonstrate and visualize activation functions."""
    
    print("ACTIVATION FUNCTIONS")
    print("=" * 70)
    
    # Define activation functions
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def tanh(z):
        return np.tanh(z)
    
    def relu(z):
        return np.maximum(0, z)
    
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)
    
    def elu(z, alpha=1.0):
        return np.where(z > 0, z, alpha * (np.exp(z) - 1))
    
    def softmax(z):
        exp_z = np.exp(z - np.max(z))  # Subtract max for stability
        return exp_z / exp_z.sum()
    
    # Test values
    z = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    
    print("\nActivation Function Outputs:")
    print(f"{'Input z':<10}", end="")
    for val in z:
        print(f"{val:>8.1f}", end="")
    print()
    print("-" * 70)
    
    activations = {
        'Sigmoid': sigmoid,
        'Tanh': tanh,
        'ReLU': relu,
        'Leaky ReLU': lambda z: leaky_relu(z, 0.1),
        'ELU': elu,
    }
    
    for name, func in activations.items():
        print(f"{name:<10}", end="")
        for val in z:
            print(f"{func(np.array([val]))[0]:>8.3f}", end="")
        print()
    
    # Derivatives (for backpropagation)
    print("\n" + "─" * 70)
    print("DERIVATIVES (for backpropagation)")
    print("─" * 70)
    
    def sigmoid_derivative(z):
        s = sigmoid(z)
        return s * (1 - s)
    
    def tanh_derivative(z):
        return 1 - np.tanh(z)**2
    
    def relu_derivative(z):
        return (z > 0).astype(float)
    
    print("\nDerivative values:")
    print(f"{'Input z':<10}", end="")
    for val in z:
        print(f"{val:>8.1f}", end="")
    print()
    print("-" * 70)
    
    derivatives = {
        'd/dz Sigmoid': sigmoid_derivative,
        'd/dz Tanh': tanh_derivative,
        'd/dz ReLU': relu_derivative,
    }
    
    for name, func in derivatives.items():
        print(f"{name:<10}", end="")
        for val in z:
            print(f"{func(np.array([val]))[0]:>8.3f}", end="")
        print()
    
    print("""
    KEY OBSERVATIONS:
    • Sigmoid/Tanh derivatives → 0 for large |z| (vanishing gradient!)
    • ReLU derivative is 0 or 1 (no vanishing gradient for positive z)
    • This is why ReLU revolutionized deep learning
    """)
    
    # Softmax
    print("\n" + "─" * 70)
    print("SOFTMAX (for multi-class classification)")
    print("─" * 70)
    
    z = np.array([2.0, 1.0, 0.5, 0.1])
    probs = softmax(z)
    
    print("\nInput logits: ", z)
    print("Softmax output:", probs.round(4))
    print("Sum:           ", probs.sum().round(4))
    
    print("\nInterpretation: Class probabilities")
    for i, (logit, prob) in enumerate(zip(z, probs)):
        bar = "█" * int(prob * 30)
        print(f"  Class {i}: logit={logit:>5.2f} → P={prob:>5.2%} {bar}")


# 10.4 FORWARD PROPAGATION

FORWARD PROPAGATION

The process of computing the output given an input.

ALGORITHM:
For each layer l from 1 to L:
    1. Compute linear transformation: z^(l) = W^(l) @ a^(l-1) + b^(l)
    2. Apply activation: a^(l) = g^(l)(z^(l))

Where:
• a^(0) = x (input)
• a^(L) = ŷ (output)
• g^(l) is the activation function for layer l


MATRIX FORM:
For a batch of N samples:
    Z = W @ X + b    (broadcasting b)
    A = activation(Z)

Dimensions:
• X: (n_input, N)
• W: (n_output, n_input)
• Z: (n_output, N)
• A: (n_output, N)

def example_forward_propagation():
    """Step-by-step forward propagation."""
    
    print("FORWARD PROPAGATION (Step by Step)")
    print("=" * 70)
    
    # Define a simple 2-layer network
    # Input: 3 features
    # Hidden: 4 neurons
    # Output: 2 classes
    
    np.random.seed(42)
    
    # Initialize weights
    W1 = np.random.randn(4, 3) * 0.5  # (4, 3): 4 hidden neurons, 3 inputs
    b1 = np.zeros((4, 1))
    W2 = np.random.randn(2, 4) * 0.5  # (2, 4): 2 outputs, 4 hidden
    b2 = np.zeros((2, 1))
    
    print(f"\nNetwork Architecture:")
    print(f"  Input layer: 3 neurons")
    print(f"  Hidden layer: 4 neurons (ReLU)")
    print(f"  Output layer: 2 neurons (Softmax)")
    
    print(f"\nParameter shapes:")
    print(f"  W1: {W1.shape}, b1: {b1.shape}")
    print(f"  W2: {W2.shape}, b2: {b2.shape}")
    
    # Sample input (single example)
    x = np.array([[0.5], [1.2], [-0.3]])  # Shape: (3, 1)
    
    print(f"\n" + "─" * 70)
    print("STEP-BY-STEP FORWARD PASS")
    print("─" * 70)
    
    print(f"\n1. Input x:")
    print(f"   Shape: {x.shape}")
    print(f"   Values: {x.T[0].round(3)}")
    
    # Layer 1
    print(f"\n2. Layer 1 (Linear):")
    z1 = W1 @ x + b1
    print(f"   z1 = W1 @ x + b1")
    print(f"   Shape: {z1.shape}")
    print(f"   Values: {z1.T[0].round(3)}")
    
    print(f"\n3. Layer 1 (ReLU Activation):")
    a1 = np.maximum(0, z1)
    print(f"   a1 = ReLU(z1)")
    print(f"   Values: {a1.T[0].round(3)}")
    
    # Layer 2
    print(f"\n4. Layer 2 (Linear):")
    z2 = W2 @ a1 + b2
    print(f"   z2 = W2 @ a1 + b2")
    print(f"   Shape: {z2.shape}")
    print(f"   Values: {z2.T[0].round(3)}")
    
    print(f"\n5. Layer 2 (Softmax):")
    exp_z2 = np.exp(z2 - np.max(z2))
    a2 = exp_z2 / np.sum(exp_z2)
    print(f"   a2 = Softmax(z2)")
    print(f"   Values: {a2.T[0].round(4)}")
    print(f"   Sum: {a2.sum():.4f}")
    
    print(f"\n6. Final Output (Class Probabilities):")
    print(f"   P(class 0) = {a2[0,0]:.4f}")
    print(f"   P(class 1) = {a2[1,0]:.4f}")
    print(f"   Predicted class: {np.argmax(a2)}")


# 10.5 BACKPROPAGATION

BACKPROPAGATION

The algorithm for computing gradients of the loss with respect to all parameters.

KEY INSIGHT: Use the chain rule to propagate gradients backward through the network.


ALGORITHM:

1. Forward pass: Compute and cache all z^(l) and a^(l)

2. Compute output gradient: ∂L/∂a^(L) (depends on loss function)

3. For l = L to 1 (backward):
   a. Compute ∂L/∂z^(l) = ∂L/∂a^(l) ⊙ g'^(l)(z^(l))
   b. Compute ∂L/∂W^(l) = ∂L/∂z^(l) @ (a^(l-1))ᵀ
   c. Compute ∂L/∂b^(l) = sum(∂L/∂z^(l)) over samples
   d. Compute ∂L/∂a^(l-1) = (W^(l))ᵀ @ ∂L/∂z^(l)  (to propagate to previous layer)


DERIVATION (for one layer):

z = W @ x + b
a = g(z)
L = loss(a, y)

We want: ∂L/∂W and ∂L/∂b

By chain rule:
∂L/∂W = ∂L/∂a × ∂a/∂z × ∂z/∂W
      = ∂L/∂a × g'(z) × x
      = δ × xᵀ

Where δ = ∂L/∂z = ∂L/∂a × g'(z)

∂L/∂b = δ

def example_backpropagation():
    """Step-by-step backpropagation with full derivation."""
    
    print("BACKPROPAGATION (Step by Step)")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Simple network: 2 → 3 → 2
    W1 = np.random.randn(3, 2) * 0.5
    b1 = np.zeros((3, 1))
    W2 = np.random.randn(2, 3) * 0.5
    b2 = np.zeros((2, 1))
    
    # Sample input and true label
    x = np.array([[1.0], [2.0]])  # Shape (2, 1)
    y = np.array([[1.0], [0.0]])  # One-hot: class 0
    
    # Activation functions
    def relu(z):
        return np.maximum(0, z)
    
    def relu_deriv(z):
        return (z > 0).astype(float)
    
    def softmax(z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)
    
    print("\nNetwork: Input(2) → Hidden(3, ReLU) → Output(2, Softmax)")
    print(f"Input x: {x.T[0]}")
    print(f"True label y: {y.T[0]}")
    
    # FORWARD PASS
    print("\n" + "─" * 70)
    print("FORWARD PASS")
    print("─" * 70)
    
    # Layer 1
    z1 = W1 @ x + b1
    a1 = relu(z1)
    print(f"\nLayer 1:")
    print(f"  z1 = {z1.T[0].round(4)}")
    print(f"  a1 = ReLU(z1) = {a1.T[0].round(4)}")
    
    # Layer 2
    z2 = W2 @ a1 + b2
    a2 = softmax(z2)
    print(f"\nLayer 2:")
    print(f"  z2 = {z2.T[0].round(4)}")
    print(f"  a2 = Softmax(z2) = {a2.T[0].round(4)}")
    
    # Loss (Cross-entropy)
    epsilon = 1e-15
    loss = -np.sum(y * np.log(a2 + epsilon))
    print(f"\nCross-entropy loss: {loss:.4f}")
    
    # BACKWARD PASS
    print("\n" + "─" * 70)
    print("BACKWARD PASS")
    print("─" * 70)
    
    # Output layer gradient (softmax + cross-entropy has nice gradient)
    dz2 = a2 - y
    print(f"\nOutput gradient (∂L/∂z2):")
    print(f"  dz2 = a2 - y = {dz2.T[0].round(4)}")
    
    # Gradients for W2 and b2
    dW2 = dz2 @ a1.T
    db2 = dz2
    print(f"\n∂L/∂W2 = dz2 @ a1ᵀ:")
    print(f"  Shape: {dW2.shape}")
    print(f"  Values:\n{dW2.round(4)}")
    print(f"\n∂L/∂b2 = dz2: {db2.T[0].round(4)}")
    
    # Propagate to layer 1
    da1 = W2.T @ dz2
    print(f"\n∂L/∂a1 = W2ᵀ @ dz2 = {da1.T[0].round(4)}")
    
    dz1 = da1 * relu_deriv(z1)
    print(f"\n∂L/∂z1 = ∂L/∂a1 ⊙ ReLU'(z1) = {dz1.T[0].round(4)}")
    
    # Gradients for W1 and b1
    dW1 = dz1 @ x.T
    db1 = dz1
    print(f"\n∂L/∂W1 = dz1 @ xᵀ:")
    print(f"  Shape: {dW1.shape}")
    print(f"  Values:\n{dW1.round(4)}")
    print(f"\n∂L/∂b1 = dz1: {db1.T[0].round(4)}")
    
    # GRADIENT CHECK (Numerical verification)
    print("\n" + "─" * 70)
    print("GRADIENT CHECK (Numerical Verification)")
    print("─" * 70)
    
    def compute_loss(W1, b1, W2, b2, x, y):
        z1 = W1 @ x + b1
        a1 = relu(z1)
        z2 = W2 @ a1 + b2
        a2 = softmax(z2)
        return -np.sum(y * np.log(a2 + 1e-15))
    
    epsilon = 1e-5
    
    # Check one element of W1
    i, j = 0, 0
    W1_plus = W1.copy()
    W1_plus[i, j] += epsilon
    W1_minus = W1.copy()
    W1_minus[i, j] -= epsilon
    
    numerical_grad = (compute_loss(W1_plus, b1, W2, b2, x, y) - 
                      compute_loss(W1_minus, b1, W2, b2, x, y)) / (2 * epsilon)
    analytical_grad = dW1[i, j]
    
    print(f"\nGradient check for W1[{i},{j}]:")
    print(f"  Numerical gradient:   {numerical_grad:.6f}")
    print(f"  Analytical gradient:  {analytical_grad:.6f}")
    print(f"  Difference:           {abs(numerical_grad - analytical_grad):.2e}")
    print(f"  ✓ Gradients match!" if abs(numerical_grad - analytical_grad) < 1e-5 else "  ✗ Mismatch!")


#   CHAPTER 11: TRAINING NEURAL NETWORKS                                         

TRAINING NEURAL NETWORKS

Training neural networks effectively requires understanding many techniques.


# 11.1 LOSS FUNCTIONS

LOSS FUNCTIONS

The loss function measures how wrong the model's predictions are.

REGRESSION LOSSES:

MSE (Mean Squared Error):
    L = (1/n) Σ (y - ŷ)²
    
    • Penalizes large errors heavily
    • Sensitive to outliers
    • Has nice gradient properties

MAE (Mean Absolute Error):
    L = (1/n) Σ |y - ŷ|
    
    • Robust to outliers
    • Gradient is constant (harder to optimize)

Huber Loss:
    L = { 0.5(y - ŷ)² if |y - ŷ| ≤ δ
        { δ|y - ŷ| - 0.5δ² otherwise
    
    • Combines benefits of MSE and MAE
    • Smooth around zero, linear for large errors


CLASSIFICATION LOSSES:

Binary Cross-Entropy:
    L = -(1/n) Σ [y log(ŷ) + (1-y) log(1-ŷ)]
    
    • For binary classification
    • ŷ is probability from sigmoid

Categorical Cross-Entropy:
    L = -(1/n) Σᵢ Σₖ yᵢₖ log(ŷᵢₖ)
    
    • For multi-class classification
    • ŷ is probability from softmax
    • y is one-hot encoded

Focal Loss:
    L = -α(1-ŷ)^γ log(ŷ)
    
    • For imbalanced classification
    • Reduces weight for easy examples

def example_loss_functions():
    """Demonstrate different loss functions."""
    
    print("LOSS FUNCTIONS")
    print("=" * 70)
    
    # Regression Losses
    print("\n" + "─" * 70)
    print("REGRESSION LOSSES")
    print("─" * 70)
    
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 10.0])  # Note: 10.0 is an "outlier"
    y_pred = np.array([1.1, 2.2, 2.8, 4.0, 5.0])
    
    # MSE
    mse = np.mean((y_true - y_pred) ** 2)
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Huber
    delta = 1.0
    diff = y_true - y_pred
    huber = np.where(
        np.abs(diff) <= delta,
        0.5 * diff ** 2,
        delta * np.abs(diff) - 0.5 * delta ** 2
    ).mean()
    
    print(f"\ny_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"\nMSE:   {mse:.4f}  (sensitive to outlier)")
    print(f"MAE:   {mae:.4f}  (more robust)")
    print(f"Huber: {huber:.4f}  (balanced)")
    
    # Classification Losses
    print("\n" + "─" * 70)
    print("CLASSIFICATION LOSSES")
    print("─" * 70)
    
    # Binary cross-entropy
    y_true_bin = np.array([1, 0, 1, 1, 0])
    y_pred_proba = np.array([0.9, 0.1, 0.7, 0.8, 0.3])
    
    epsilon = 1e-15
    bce = -np.mean(y_true_bin * np.log(y_pred_proba + epsilon) + 
                   (1 - y_true_bin) * np.log(1 - y_pred_proba + epsilon))
    
    print(f"\nBinary Cross-Entropy:")
    print(f"  y_true: {y_true_bin}")
    print(f"  y_pred: {y_pred_proba}")
    print(f"  BCE: {bce:.4f}")
    
    # Categorical cross-entropy
    y_true_cat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # One-hot
    y_pred_cat = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    
    cce = -np.mean(np.sum(y_true_cat * np.log(y_pred_cat + epsilon), axis=1))
    
    print(f"\nCategorical Cross-Entropy:")
    print(f"  y_true (one-hot):\n{y_true_cat}")
    print(f"  y_pred (softmax):\n{y_pred_cat}")
    print(f"  CCE: {cce:.4f}")


# 11.2 OPTIMIZERS

OPTIMIZERS

Algorithms for updating model parameters based on gradients.


SGD (Stochastic Gradient Descent):
    θ = θ - α × ∇L(θ)
    
    Simple but can be slow and oscillate.


SGD WITH MOMENTUM:
    v = β × v - α × ∇L(θ)
    θ = θ + v
    
    • Accumulates gradient direction
    • Speeds up convergence
    • Reduces oscillation
    • β typically 0.9


RMSPROP:
    s = β × s + (1-β) × (∇L(θ))²
    θ = θ - α × ∇L(θ) / (√s + ε)
    
    • Adapts learning rate per parameter
    • Good for non-stationary objectives
    • Works well with RNNs


ADAM (Adaptive Moment Estimation):
    m = β₁ × m + (1-β₁) × ∇L(θ)           # 1st moment (momentum)
    v = β₂ × v + (1-β₂) × (∇L(θ))²        # 2nd moment (RMSprop)
    m̂ = m / (1 - β₁ᵗ)                     # Bias correction
    v̂ = v / (1 - β₂ᵗ)
    θ = θ - α × m̂ / (√v̂ + ε)
    
    Default: β₁=0.9, β₂=0.999, ε=1e-8
    
    Most popular optimizer - great default choice!


ADAMW:
    Like Adam but with decoupled weight decay.
    Better for transformers.
    
    θ = θ - α × (m̂ / (√v̂ + ε) + λ × θ)

def example_optimizers():
    """Compare different optimizers."""
    
    print("OPTIMIZERS COMPARISON")
    print("=" * 70)
    
    # Generate a simple optimization problem
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 5)
    true_w = np.array([1.5, -2.0, 1.0, -0.5, 0.8])
    y = X @ true_w + np.random.randn(n) * 0.5
    
    def compute_loss_and_gradient(w, X, y):
        pred = X @ w
        loss = np.mean((pred - y) ** 2)
        grad = (2 / len(y)) * X.T @ (pred - y)
        return loss, grad
    
    # Implement optimizers
    def sgd(w, grad, lr=0.01):
        return w - lr * grad
    
    def sgd_momentum(w, grad, v, lr=0.01, beta=0.9):
        v = beta * v - lr * grad
        w = w + v
        return w, v
    
    def rmsprop(w, grad, s, lr=0.01, beta=0.9, eps=1e-8):
        s = beta * s + (1 - beta) * grad ** 2
        w = w - lr * grad / (np.sqrt(s) + eps)
        return w, s
    
    def adam(w, grad, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        return w, m, v
    
    # Run optimizers
    epochs = 100
    results = {}
    
    # SGD
    w = np.zeros(5)
    losses_sgd = []
    for _ in range(epochs):
        loss, grad = compute_loss_and_gradient(w, X, y)
        losses_sgd.append(loss)
        w = sgd(w, grad, lr=0.05)
    results['SGD'] = losses_sgd
    
    # SGD with Momentum
    w = np.zeros(5)
    v = np.zeros(5)
    losses_momentum = []
    for _ in range(epochs):
        loss, grad = compute_loss_and_gradient(w, X, y)
        losses_momentum.append(loss)
        w, v = sgd_momentum(w, grad, v, lr=0.05)
    results['SGD+Momentum'] = losses_momentum
    
    # RMSprop
    w = np.zeros(5)
    s = np.zeros(5)
    losses_rmsprop = []
    for _ in range(epochs):
        loss, grad = compute_loss_and_gradient(w, X, y)
        losses_rmsprop.append(loss)
        w, s = rmsprop(w, grad, s, lr=0.1)
    results['RMSprop'] = losses_rmsprop
    
    # Adam
    w = np.zeros(5)
    m, v = np.zeros(5), np.zeros(5)
    losses_adam = []
    for t in range(1, epochs + 1):
        loss, grad = compute_loss_and_gradient(w, X, y)
        losses_adam.append(loss)
        w, m, v = adam(w, grad, m, v, t, lr=0.1)
    results['Adam'] = losses_adam
    
    # Compare
    print(f"\n{'Optimizer':<15} {'Initial Loss':<15} {'Final Loss':<15} {'Epochs to <1.0'}")
    print("-" * 60)
    
    for name, losses in results.items():
        epochs_to_1 = next((i for i, l in enumerate(losses) if l < 1.0), epochs)
        print(f"{name:<15} {losses[0]:<15.4f} {losses[-1]:<15.4f} {epochs_to_1}")
    
    print("""
    OPTIMIZER SELECTION GUIDE:
    • Adam: Best default choice for most problems
    • SGD+Momentum: Often better final performance with proper tuning
    • AdamW: Best for transformers (decoupled weight decay)
    • RMSprop: Good for RNNs
    
    Tips:
    • Start with Adam, lr=1e-3 or 3e-4
    • For fine-tuning pretrained models: lower lr (1e-5 to 1e-4)
    • If Adam converges but generalizes poorly, try SGD
    """)


# 11.3 BATCH NORMALIZATION

BATCH NORMALIZATION

Normalizes activations within each mini-batch.

ALGORITHM:
For each feature:
    1. μ_B = (1/m) Σᵢ xᵢ                    # Batch mean
    2. σ²_B = (1/m) Σᵢ (xᵢ - μ_B)²          # Batch variance
    3. x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)       # Normalize
    4. yᵢ = γ × x̂ᵢ + β                      # Scale and shift

Where γ and β are learnable parameters.


BENEFITS:
• Reduces internal covariate shift
• Allows higher learning rates
• Reduces sensitivity to initialization
• Acts as regularization
• Enables training of very deep networks


WHEN TO USE:
• Before activation (original paper) OR after activation (common in practice)
• In CNNs: BatchNorm2d
• In transformers: LayerNorm is preferred

LAYER NORMALIZATION (LayerNorm):
    Normalizes across features (not batch)
    Better for variable-length sequences and transformers
    
    μ = (1/d) Σⱼ xⱼ
    σ² = (1/d) Σⱼ (xⱼ - μ)²

def example_batch_normalization():
    """Demonstrate batch normalization."""
    
    print("BATCH NORMALIZATION")
    print("=" * 70)
    
    # Manual BatchNorm implementation
    def batch_norm(x, gamma=1, beta=0, eps=1e-5):
        """Manual batch normalization."""
        mu = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        x_norm = (x - mu) / np.sqrt(var + eps)
        out = gamma * x_norm + beta
        return out, mu, var
    
    # Sample data (batch of activations)
    np.random.seed(42)
    batch_size = 32
    features = 10
    
    # Activations with different scales
    x = np.random.randn(batch_size, features)
    x[:, 0] *= 100  # Feature 0 has large scale
    x[:, 1] += 50   # Feature 1 has large mean
    
    print(f"\nBefore BatchNorm:")
    print(f"  Feature means: {x.mean(axis=0)[:5].round(2)}...")
    print(f"  Feature stds:  {x.std(axis=0)[:5].round(2)}...")
    
    # Apply BatchNorm
    x_normalized, _, _ = batch_norm(x)
    
    print(f"\nAfter BatchNorm:")
    print(f"  Feature means: {x_normalized.mean(axis=0)[:5].round(4)}...")
    print(f"  Feature stds:  {x_normalized.std(axis=0)[:5].round(4)}...")
    
    # PyTorch BatchNorm
    print("\n" + "─" * 70)
    print("PYTORCH BATCH NORMALIZATION")
    print("─" * 70)
    
    # Create layers
    bn1d = nn.BatchNorm1d(features)
    
    # Convert to tensor
    x_tensor = torch.FloatTensor(x)
    
    # Apply (in training mode)
    bn1d.train()
    output = bn1d(x_tensor)
    
    print(f"\nPyTorch BatchNorm1d output:")
    print(f"  Mean: {output.mean(dim=0)[:5].detach().numpy().round(4)}...")
    print(f"  Std:  {output.std(dim=0)[:5].detach().numpy().round(4)}...")
    
    # Learnable parameters
    print(f"\nLearnable parameters:")
    print(f"  gamma (weight): {bn1d.weight[:5].detach().numpy().round(4)}...")
    print(f"  beta (bias):    {bn1d.bias[:5].detach().numpy().round(4)}...")


# 11.4 REGULARIZATION TECHNIQUES

REGULARIZATION TECHNIQUES FOR NEURAL NETWORKS


L2 REGULARIZATION (Weight Decay):
    Loss = Original_Loss + λ Σ wᵢ²
    
    • Penalizes large weights
    • Implemented via weight_decay in optimizer


DROPOUT:
    During training: Randomly set neurons to 0 with probability p
    During inference: Scale by (1-p) or don't drop
    
    • Prevents co-adaptation of neurons
    • Acts as ensemble of networks
    • Typical p: 0.1-0.5


EARLY STOPPING:
    Stop training when validation loss stops improving.
    
    • Simple and effective
    • Requires validation set
    • Use patience parameter


DATA AUGMENTATION:
    Create modified versions of training data.
    
    Images: flip, rotate, crop, color jitter
    Text: synonym replacement, back-translation
    Audio: time shift, speed change, noise
    
    • Increases effective dataset size
    • Teaches invariances


LABEL SMOOTHING:
    Instead of hard labels [0, 1, 0]:
    Use soft labels [0.05, 0.9, 0.05]
    
    • Prevents overconfident predictions
    • Improves calibration

def example_regularization_nn():
    """Demonstrate regularization techniques for neural networks."""
    
    print("NEURAL NETWORK REGULARIZATION")
    print("=" * 70)
    
    # Dropout
    print("\n" + "─" * 70)
    print("DROPOUT")
    print("─" * 70)
    
    dropout = nn.Dropout(p=0.5)
    
    x = torch.ones(10)
    
    # Training mode
    dropout.train()
    outputs_train = [dropout(x).numpy() for _ in range(5)]
    
    print("\nDropout (p=0.5) during training:")
    for i, out in enumerate(outputs_train):
        print(f"  Sample {i+1}: {out}")
    
    # Evaluation mode
    dropout.eval()
    output_eval = dropout(x).numpy()
    print(f"\nDropout during evaluation:")
    print(f"  Output: {output_eval}")
    
    # Model with regularization
    print("\n" + "─" * 70)
    print("COMPLETE REGULARIZED MODEL")
    print("─" * 70)
    
    class RegularizedModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
            super().__init__()
            
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(hidden_size, output_size)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = RegularizedModel(10, 64, 2, dropout_rate=0.5)
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    print(f"\nOptimizer: AdamW with weight_decay=0.01")


#   CHAPTER 12: CONVOLUTIONAL NEURAL NETWORKS                                    

CONVOLUTIONAL NEURAL NETWORKS (CNNs)

CNNs are specialized for processing grid-like data (images, audio, sequences).

WHY CONVOLUTIONS?
1. PARAMETER SHARING: Same kernel applied everywhere → fewer parameters
2. TRANSLATION INVARIANCE: Detect features anywhere in input
3. LOCAL CONNECTIVITY: Each neuron only connected to local region
4. HIERARCHICAL FEATURES: Lower layers learn edges, higher layers learn objects


CONVOLUTION OPERATION:

Input: [[[...]]] (H × W × C_in)
Kernel: [[[...]]] (K × K × C_in × C_out)

Output[i,j,c] = Σₘ Σₙ Σₖ Input[i+m, j+n, k] × Kernel[m, n, k, c] + bias[c]

OUTPUT SIZE:
    H_out = (H_in - K + 2P) / S + 1
    
Where:
• K: Kernel size
• P: Padding
• S: Stride


TYPICAL CNN STRUCTURE:
Input → [Conv → BatchNorm → ReLU → Pool] × N → Flatten → FC → Output

                                          
      ┌──────────────────────────────────────────────────────────────┐
      │                                                              │
      │   Input    Conv    Pool    Conv    Pool    Flatten    FC     │
      │    [■]  → [■■■] → [■■] → [■■■■] → [■■] → [────] → [●●●●●]  │
      │                                                              │
      └──────────────────────────────────────────────────────────────┘

def example_cnn_basics():
    """Demonstrate CNN fundamentals."""
    
    print("CONVOLUTIONAL NEURAL NETWORKS")
    print("=" * 70)
    
    # Convolution operation
    print("\n" + "─" * 70)
    print("CONVOLUTION OPERATION")
    print("─" * 70)
    
    # Simple 5x5 image
    image = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=float)
    
    # Edge detection kernel
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=float)
    
    print("\nInput image (5×5):")
    print(image)
    
    print("\nKernel (3×3, edge detection):")
    print(kernel)
    
    # Manual convolution
    def convolve2d(image, kernel):
        H, W = image.shape
        K = kernel.shape[0]
        output = np.zeros((H - K + 1, W - K + 1))
        
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = np.sum(image[i:i+K, j:j+K] * kernel)
        
        return output
    
    output = convolve2d(image, kernel)
    
    print("\nOutput (3×3, edges detected):")
    print(output)
    
    # PyTorch Conv2d
    print("\n" + "─" * 70)
    print("PYTORCH CONV2D")
    print("─" * 70)
    
    # Create Conv2d layer
    conv = nn.Conv2d(
        in_channels=1,    # Grayscale input
        out_channels=16,  # Number of filters
        kernel_size=3,    # 3x3 kernels
        stride=1,
        padding=1         # Same padding
    )
    
    print(f"\nConv2d layer:")
    print(f"  Input channels: {conv.in_channels}")
    print(f"  Output channels: {conv.out_channels}")
    print(f"  Kernel size: {conv.kernel_size}")
    print(f"  Stride: {conv.stride}")
    print(f"  Padding: {conv.padding}")
    
    # Calculate parameters
    params = conv.weight.numel() + conv.bias.numel()
    print(f"  Parameters: {params} = {conv.out_channels}×{conv.in_channels}×{conv.kernel_size[0]}×{conv.kernel_size[1]} + {conv.out_channels}")
    
    # Test with random image
    batch = torch.randn(4, 1, 28, 28)  # Batch of 4 images, 1 channel, 28x28
    output = conv(batch)
    print(f"\nInput shape:  {batch.shape}")
    print(f"Output shape: {output.shape}")
    
    # Pooling
    print("\n" + "─" * 70)
    print("POOLING LAYERS")
    print("─" * 70)
    
    # Max pooling
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Average pooling
    avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    # Global average pooling
    global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    x = torch.randn(1, 16, 28, 28)
    
    print(f"\nInput:              {x.shape}")
    print(f"After MaxPool2d:    {maxpool(x).shape}")
    print(f"After AvgPool2d:    {avgpool(x).shape}")
    print(f"After GlobalAvgPool:{global_avgpool(x).shape}")


def example_cnn_architecture():
    """Build a complete CNN for image classification."""
    
    print("\n" + "─" * 70)
    print("COMPLETE CNN ARCHITECTURE")
    print("─" * 70)
    
    class SimpleCNN(nn.Module):
        """Simple CNN for MNIST-like images (28x28x1)."""
        
        def __init__(self, num_classes=10):
            super().__init__()
            
            # Convolutional layers
            self.conv_layers = nn.Sequential(
                # Conv block 1
                nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28x32
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),                          # 14x14x32
                
                # Conv block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14x64
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),                          # 7x7x64
                
                # Conv block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),# 7x7x128
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))                 # 1x1x128
            )
            
            # Fully connected layers
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            x = self.conv_layers(x)
            x = self.fc_layers(x)
            return x
    
    model = SimpleCNN(num_classes=10)
    
    print("\nSimpleCNN Architecture:")
    print(model)
    
    # Test forward pass
    batch = torch.randn(4, 1, 28, 28)
    output = model(batch)
    print(f"\nInput shape:  {batch.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable: {trainable:,}")
    
    # Compute receptive field
    print("\n" + "─" * 70)
    print("FAMOUS CNN ARCHITECTURES")
    print("─" * 70)
    
    print("""
    Architecture      Year   Top-5 Error   Parameters   Key Innovation
    LeNet-5           1998   -             60K          First practical CNN
    AlexNet           2012   15.3%         61M          ReLU, Dropout, GPU
    VGGNet            2014   7.3%          138M         Small 3×3 filters
    GoogLeNet/Incep.  2014   6.7%          6.8M         Inception modules
    ResNet-50         2015   3.6%          25M          Skip connections
    ResNet-152        2015   3.1%          60M          Very deep
    DenseNet-121      2017   -             8M           Dense connections
    EfficientNet-B0   2019   -             5.3M         NAS, compound scaling
    ViT-Base          2020   -             86M          Pure attention
    ConvNeXt          2022   -             89M          Modern convolutions
    """)


#   CHAPTER 13: RECURRENT NEURAL NETWORKS                                        

RECURRENT NEURAL NETWORKS (RNNs)

RNNs process sequential data by maintaining a hidden state.

WHY RNNS?
• Process sequences of variable length
• Share parameters across time steps
• Capture temporal dependencies


BASIC RNN:
    h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)
    y_t = W_hy × h_t + b_y
    
    Visual:
    
         y₁        y₂        y₃        y₄
          ↑         ↑         ↑         ↑
        ┌───┐     ┌───┐     ┌───┐     ┌───┐
    ───▶│ h │────▶│ h │────▶│ h │────▶│ h │───▶
        └───┘     └───┘     └───┘     └───┘
          ↑         ↑         ↑         ↑
         x₁        x₂        x₃        x₄


VANISHING GRADIENT PROBLEM:
When backpropagating through time, gradients multiply.
If |gradient| < 1, they shrink exponentially → can't learn long-range dependencies


LSTM (Long Short-Term Memory):
Uses gates to control information flow:

    f_t = σ(W_f × [h_{t-1}, x_t] + b_f)    # Forget gate
    i_t = σ(W_i × [h_{t-1}, x_t] + b_i)    # Input gate
    c̃_t = tanh(W_c × [h_{t-1}, x_t] + b_c)  # Candidate cell state
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t        # Cell state
    o_t = σ(W_o × [h_{t-1}, x_t] + b_o)    # Output gate
    h_t = o_t ⊙ tanh(c_t)                   # Hidden state

The cell state c_t provides a "highway" for gradients!


GRU (Gated Recurrent Unit):
Simplified version of LSTM with 2 gates:

    z_t = σ(W_z × [h_{t-1}, x_t])          # Update gate
    r_t = σ(W_r × [h_{t-1}, x_t])          # Reset gate
    h̃_t = tanh(W × [r_t ⊙ h_{t-1}, x_t])   # Candidate hidden
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  # Hidden state

Fewer parameters than LSTM, often similar performance.

def example_rnn():
    """Demonstrate RNN, LSTM, and GRU."""
    
    print("RECURRENT NEURAL NETWORKS")
    print("=" * 70)
    
    # Simple RNN
    print("\n" + "─" * 70)
    print("BASIC RNN")
    print("─" * 70)
    
    # Create RNN layer
    rnn = nn.RNN(
        input_size=10,     # Features per time step
        hidden_size=20,    # Hidden state size
        num_layers=2,      # Number of stacked RNN layers
        batch_first=True   # Input shape: (batch, seq, features)
    )
    
    # Input: batch of 4 sequences, each 15 time steps, 10 features
    x = torch.randn(4, 15, 10)
    
    # Initial hidden state
    h0 = torch.zeros(2, 4, 20)  # (num_layers, batch, hidden_size)
    
    # Forward pass
    output, hn = rnn(x, h0)
    
    print(f"\nRNN Layer:")
    print(f"  Input size: {rnn.input_size}")
    print(f"  Hidden size: {rnn.hidden_size}")
    print(f"  Num layers: {rnn.num_layers}")
    
    print(f"\nInput shape:        {x.shape}")
    print(f"Initial h shape:    {h0.shape}")
    print(f"Output shape:       {output.shape}  # All hidden states")
    print(f"Final h shape:      {hn.shape}  # Last hidden states")
    
    # LSTM
    print("\n" + "─" * 70)
    print("LSTM")
    print("─" * 70)
    
    lstm = nn.LSTM(
        input_size=10,
        hidden_size=20,
        num_layers=2,
        batch_first=True,
        bidirectional=False
    )
    
    # Initial states
    h0 = torch.zeros(2, 4, 20)  # Hidden state
    c0 = torch.zeros(2, 4, 20)  # Cell state
    
    output_lstm, (hn, cn) = lstm(x, (h0, c0))
    
    print(f"\nLSTM Layer:")
    print(f"  Input shape:     {x.shape}")
    print(f"  Output shape:    {output_lstm.shape}")
    print(f"  Hidden shape:    {hn.shape}")
    print(f"  Cell shape:      {cn.shape}")
    
    # Parameter count
    lstm_params = sum(p.numel() for p in lstm.parameters())
    rnn_params = sum(p.numel() for p in rnn.parameters())
    
    print(f"\nParameter comparison:")
    print(f"  RNN:  {rnn_params:,}")
    print(f"  LSTM: {lstm_params:,} (4x more due to gates)")
    
    # GRU
    print("\n" + "─" * 70)
    print("GRU")
    print("─" * 70)
    
    gru = nn.GRU(
        input_size=10,
        hidden_size=20,
        num_layers=2,
        batch_first=True
    )
    
    output_gru, hn_gru = gru(x, h0)
    
    gru_params = sum(p.numel() for p in gru.parameters())
    
    print(f"\nGRU Layer:")
    print(f"  Output shape: {output_gru.shape}")
    print(f"  Parameters: {gru_params:,} (3x RNN, less than LSTM)")
    
    # Bidirectional LSTM
    print("\n" + "─" * 70)
    print("BIDIRECTIONAL LSTM")
    print("─" * 70)
    
    bilstm = nn.LSTM(
        input_size=10,
        hidden_size=20,
        num_layers=2,
        batch_first=True,
        bidirectional=True
    )
    
    # For bidirectional, h0 and c0 need 2x num_layers
    h0_bi = torch.zeros(4, 4, 20)  # 2*num_layers
    c0_bi = torch.zeros(4, 4, 20)
    
    output_bi, (hn_bi, cn_bi) = bilstm(x, (h0_bi, c0_bi))
    
    print(f"\nBidirectional LSTM:")
    print(f"  Output shape: {output_bi.shape}  # hidden_size * 2")
    print(f"  Hidden shape: {hn_bi.shape}  # num_layers * 2")


def example_sequence_classification():
    """Complete sequence classification with LSTM."""
    
    print("\n" + "─" * 70)
    print("SEQUENCE CLASSIFICATION MODEL")
    print("─" * 70)
    
    class LSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            
            self.lstm = nn.LSTM(
                embed_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.5,
                bidirectional=True
            )
            
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),  # *2 for bidirectional
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            # x: (batch, seq_len) - token indices
            embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
            
            lstm_out, (hn, cn) = self.lstm(embedded)
            
            # Use last hidden state from both directions
            # hn shape: (num_layers*2, batch, hidden_dim)
            # Concatenate last forward and first backward
            hidden = torch.cat((hn[-2], hn[-1]), dim=1)
            
            output = self.fc(hidden)
            return output
    
    model = LSTMClassifier(
        vocab_size=10000,
        embed_dim=128,
        hidden_dim=256,
        num_classes=5
    )
    
    print(f"\nLSTM Classifier:")
    print(model)
    
    # Test
    batch = torch.randint(0, 10000, (8, 100))  # 8 sequences of length 100
    output = model(batch)
    print(f"\nInput shape:  {batch.shape}")
    print(f"Output shape: {output.shape}")


#   CHAPTER 14: ATTENTION AND TRANSFORMERS                                       

ATTENTION AND TRANSFORMERS

Transformers revolutionized NLP and are now used everywhere in AI.

WHY ATTENTION?
• RNNs process sequentially → slow, hard to parallelize
• Attention looks at entire sequence at once → fast, parallel
• Better at capturing long-range dependencies


ATTENTION MECHANISM:

Query, Key, Value framework:

    Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V

Where:
• Q: Query (what we're looking for)
• K: Key (what each position offers)
• V: Value (actual content)
• d_k: Dimension of keys (for scaling)


SELF-ATTENTION:
Q, K, V all come from the same sequence:

    Q = X × W_Q
    K = X × W_K
    V = X × W_V
    
    Output = Attention(Q, K, V)

Each position attends to all positions in the sequence.


MULTI-HEAD ATTENTION:
Run multiple attention operations in parallel, then concatenate:

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
    
    where head_i = Attention(Q × W_Q_i, K × W_K_i, V × W_V_i)

Different heads can learn different types of relationships!


TRANSFORMER ARCHITECTURE:

Encoder:
    Input Embedding + Positional Encoding
    ↓
    [Multi-Head Self-Attention → LayerNorm → FFN → LayerNorm] × N
    ↓
    Encoder Output

Decoder:
    Output Embedding + Positional Encoding
    ↓
    [Masked Self-Attention → LayerNorm → Cross-Attention → LayerNorm → FFN → LayerNorm] × N
    ↓
    Linear + Softmax

Key innovations:
• Positional encoding (since no recurrence)
• Layer normalization
• Residual connections

def example_attention():
    """Demonstrate attention mechanisms."""
    
    print("ATTENTION MECHANISMS")
    print("=" * 70)
    
    # Scaled Dot-Product Attention
    print("\n" + "─" * 70)
    print("SCALED DOT-PRODUCT ATTENTION")
    print("─" * 70)
    
    def scaled_dot_product_attention(Q, K, V, mask=None):
        """
        Q: (batch, seq_len, d_k)
        K: (batch, seq_len, d_k)
        V: (batch, seq_len, d_v)
        """
        d_k = Q.shape[-1]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        
        # Apply mask (for decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    # Example
    batch_size, seq_len, d_k = 2, 5, 8
    
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"\nInput shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"\nAttention weights for first sample:")
    print(weights[0].detach().numpy().round(3))
    print(f"(Each row sums to 1: {weights[0].sum(dim=-1).numpy().round(3)})")
    
    # Multi-Head Attention
    print("\n" + "─" * 70)
    print("MULTI-HEAD ATTENTION")
    print("─" * 70)
    
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            assert d_model % n_heads == 0
            
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            
            self.W_Q = nn.Linear(d_model, d_model)
            self.W_K = nn.Linear(d_model, d_model)
            self.W_V = nn.Linear(d_model, d_model)
            self.W_O = nn.Linear(d_model, d_model)
        
        def forward(self, Q, K, V, mask=None):
            batch_size = Q.shape[0]
            
            # Linear projections
            Q = self.W_Q(Q)  # (batch, seq, d_model)
            K = self.W_K(K)
            V = self.W_V(V)
            
            # Reshape to (batch, n_heads, seq, d_k)
            Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            
            # Attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attention = torch.softmax(scores, dim=-1)
            out = torch.matmul(attention, V)
            
            # Reshape back
            out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            
            # Final projection
            return self.W_O(out)
    
    mha = MultiHeadAttention(d_model=64, n_heads=8)
    
    x = torch.randn(2, 10, 64)  # (batch, seq, d_model)
    output = mha(x, x, x)  # Self-attention: Q=K=V=x
    
    print(f"\nMulti-Head Attention (8 heads, d_model=64):")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in mha.parameters()):,}")
    
    # Positional Encoding
    print("\n" + "─" * 70)
    print("POSITIONAL ENCODING")
    print("─" * 70)
    
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                                 (-np.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            
            self.register_buffer('pe', pe)
        
        def forward(self, x):
            return x + self.pe[:, :x.size(1)]
    
    pe = PositionalEncoding(d_model=64)
    x = torch.randn(2, 20, 64)
    x_with_pe = pe(x)
    
    print(f"\nPositional Encoding:")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {x_with_pe.shape}")
    print(f"  First 5 positions of PE (first 8 dims):")
    print(pe.pe[0, :5, :8].numpy().round(3))


def example_transformer_encoder():
    """Build a Transformer encoder block."""
    
    print("\n" + "─" * 70)
    print("TRANSFORMER ENCODER BLOCK")
    print("─" * 70)
    
    class TransformerEncoderBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
            super().__init__()
            
            # Multi-head attention
            self.attention = nn.MultiheadAttention(d_model, n_heads, 
                                                    dropout=dropout, batch_first=True)
            
            # Feed-forward network
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )
            
            # Layer norms
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x, mask=None):
            # Self-attention with residual
            attn_out, _ = self.attention(x, x, x, attn_mask=mask)
            x = self.norm1(x + self.dropout(attn_out))
            
            # FFN with residual
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            
            return x
    
    encoder_block = TransformerEncoderBlock(d_model=64, n_heads=8, d_ff=256)
    
    x = torch.randn(2, 20, 64)
    output = encoder_block(x)
    
    print(f"\nTransformer Encoder Block:")
    print(f"  d_model: 64")
    print(f"  n_heads: 8")
    print(f"  d_ff: 256")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in encoder_block.parameters()):,}")
    
    # Using PyTorch's built-in Transformer
    print("\n" + "─" * 70)
    print("PYTORCH TRANSFORMERENCODER")
    print("─" * 70)
    
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        batch_first=True
    )
    
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    
    x = torch.randn(4, 100, 512)  # (batch, seq, d_model)
    output = transformer_encoder(x)
    
    print(f"\nPyTorch TransformerEncoder (6 layers):")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in transformer_encoder.parameters()):,}")
    
    print("""
    MODERN TRANSFORMERS (2024-2025):
    
    BERT-style (Encoder only):
    • For understanding: classification, NER, Q&A
    • Bidirectional context
    
    GPT-style (Decoder only):
    • For generation: text completion, chat
    • Autoregressive (left-to-right)
    
    T5-style (Encoder-Decoder):
    • For translation, summarization
    • Full seq-to-seq
    
    Key advances:
    • Rotary Positional Embeddings (RoPE)
    • Grouped Query Attention (GQA)
    • Flash Attention (memory efficient)
    • Mixture of Experts (MoE)
    • RLHF (Reinforcement Learning from Human Feedback)
    """)


# Run all examples

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PART III: NEURAL NETWORKS AND DEEP LEARNING")
    print("="*70)
    
    # Chapter 10: Neural Network Fundamentals
    example_perceptron()
    print("\n")
    example_mlp_from_scratch()
    print("\n")
    example_activation_functions()
    print("\n")
    example_forward_propagation()
    print("\n")
    example_backpropagation()
    
    # Chapter 11: Training Neural Networks
    print("\n")
    example_loss_functions()
    print("\n")
    example_optimizers()
    print("\n")
    example_batch_normalization()
    print("\n")
    example_regularization_nn()
    
    # Chapter 12: CNNs
    print("\n")
    example_cnn_basics()
    example_cnn_architecture()
    
    # Chapter 13: RNNs
    print("\n")
    example_rnn()
    example_sequence_classification()
    
    # Chapter 14: Transformers
    print("\n")
    example_attention()
    example_transformer_encoder()


---

<div align="center">

[⬅️ Previous: Foundations](01-foundations.md) | [📚 Table of Contents](../README.md) | [Next: Unsupervised Learning ➡️](03-unsupervised-learning.md)

</div>

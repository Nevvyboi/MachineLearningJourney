<div align="center">

# 📝 Exercises & Quizzes

![Chapter](https://img.shields.io/badge/Chapter-15-blue?style=for-the-badge)
![Topic](https://img.shields.io/badge/Topic-Practice%20%7C%20Challenges-green?style=for-the-badge)

*Coding Challenges, Quizzes & Mini-Projects*

---

</div>

# Part XVIII: Exercises, Quizzes, and Practice Problems

---

## Chapter 59: Conceptual Exercises

### 59.1 Machine Learning Fundamentals

```
┌─────────────────────────────────────────────────────────────────────┐
│                 CONCEPTUAL QUESTIONS                                │
├─────────────────────────────────────────────────────────────────────┤

QUESTION 1: Bias-Variance Tradeoff
---------------------------------
A model has high training accuracy (98%) but low test accuracy (65%).

a) Is this model suffering from high bias or high variance?
b) What techniques could help improve the model?
c) Would increasing model complexity help or hurt?

ANSWER:
a) High variance (overfitting). The model memorizes training data
   but doesn't generalize.
b) - Add regularization (L1/L2, dropout)
   - Get more training data
   - Reduce model complexity
   - Use data augmentation
   - Apply early stopping
c) Increasing complexity would likely hurt - the model is already
   too complex for the available data.


QUESTION 2: Feature Scaling
--------------------------
When should you normalize/standardize features? Which algorithms
require it and which don't?

ANSWER:
REQUIRE scaling:
- Gradient-based (neural networks, logistic regression)
- Distance-based (KNN, K-means, SVM with RBF kernel)
- Regularized models (features must be on same scale)

DON'T require scaling:
- Tree-based (Random Forest, XGBoost, Decision Trees)
- Naive Bayes

Reason: Gradient descent converges faster with scaled features.
Distance metrics are biased by feature magnitude.


QUESTION 3: Train/Val/Test Split
-------------------------------
Why do we need three separate datasets? What happens if we:
a) Only use train and test?
b) Use test set for hyperparameter tuning?

ANSWER:
a) Without validation set, we can't tune hyperparameters without
   leaking information from test set.
b) Test accuracy becomes optimistic - we've implicitly fit to
   test set through hyperparameter selection. This is "data leakage."

Best practice:
- Train: Learn model parameters
- Validation: Tune hyperparameters, early stopping
- Test: Final unbiased evaluation (use ONCE)


QUESTION 4: Regularization
-------------------------
Compare L1 and L2 regularization:
a) Mathematical difference
b) Effect on weights
c) When to use each

ANSWER:
a) L1: λΣ|w|  (sum of absolute values)
   L2: λΣw²   (sum of squared values)

b) L1: Drives weights to exactly zero (sparse solutions)
   L2: Shrinks weights toward zero but rarely exactly zero

c) L1: When you want feature selection (sparse model)
   L2: When all features might be useful but want small weights
   Elastic Net: Combination when you want some of both


QUESTION 5: Gradient Descent
---------------------------
Explain the difference between:
a) Batch gradient descent
b) Stochastic gradient descent
c) Mini-batch gradient descent

Which is typically preferred and why?

ANSWER:
a) Batch GD: Uses ALL data points per update
   - Stable gradients, slow updates
   - Can get stuck in local minima

b) Stochastic GD: Uses ONE data point per update
   - Noisy gradients, fast updates
   - Can escape local minima

c) Mini-batch GD: Uses BATCH_SIZE data points
   - Balance of stability and speed
   - Efficient GPU utilization

Mini-batch is preferred:
- Vectorization benefits (GPU)
- Regularizing effect of noise
- Typical sizes: 32, 64, 128, 256


└─────────────────────────────────────────────────────────────────────┘
```

### 59.2 Deep Learning Concepts

```
┌─────────────────────────────────────────────────────────────────────┐
│                 DEEP LEARNING QUESTIONS                             │
├─────────────────────────────────────────────────────────────────────┤

QUESTION 6: Activation Functions
-------------------------------
a) Why can't we use linear activation functions throughout a network?
b) Why is ReLU preferred over sigmoid for hidden layers?
c) When would you use sigmoid vs softmax?

ANSWER:
a) Composition of linear functions is linear:
   f(x) = W₂(W₁x) = (W₂W₁)x = Wx
   Network would be equivalent to single linear layer.

b) ReLU advantages:
   - No vanishing gradient for positive inputs
   - Sparse activation (zeros for negative)
   - Faster computation
   - Sigmoid: Gradients vanish for large |x|

c) Sigmoid: Binary classification (output in [0,1])
   Softmax: Multi-class classification (outputs sum to 1)


QUESTION 7: Batch Normalization
------------------------------
a) What problem does batch norm solve?
b) Why does it help training?
c) What happens differently during training vs inference?

ANSWER:
a) Internal covariate shift - distribution of layer inputs
   changes during training as previous layers update.

b) Benefits:
   - Allows higher learning rates
   - Reduces dependence on initialization
   - Acts as regularizer
   - Stabilizes training

c) Training: Use batch statistics (μ_batch, σ_batch)
   Inference: Use running averages computed during training
   This ensures consistent behavior for single samples.


QUESTION 8: Dropout
------------------
a) How does dropout work during training?
b) Why does it help prevent overfitting?
c) What changes during inference?

ANSWER:
a) Randomly set neuron outputs to 0 with probability p.
   Scale remaining by 1/(1-p) to maintain expected value.

b) Prevents co-adaptation:
   - Neurons can't rely on specific other neurons
   - Forces learning redundant representations
   - Like training ensemble of networks

c) Inference: No dropout applied.
   All neurons active, no scaling needed
   (because of inverted dropout during training).


QUESTION 9: CNN Architecture
---------------------------
a) Why are convolutions better than fully connected layers for images?
b) What do different layers learn?
c) Why use pooling?

ANSWER:
a) - Parameter efficiency: Weight sharing
   - Translation equivariance: Detect features anywhere
   - Local connectivity: Spatial relationships

b) Layer hierarchy:
   - Early layers: Edges, textures
   - Middle layers: Parts, patterns
   - Later layers: Objects, concepts

c) Pooling benefits:
   - Reduces spatial dimensions
   - Provides translation invariance
   - Reduces parameters and computation


QUESTION 10: Transformers
------------------------
a) What is self-attention and why is it useful?
b) Why do we need positional encoding?
c) What is the computational complexity of attention?

ANSWER:
a) Self-attention: Each position attends to all positions.
   Benefits:
   - Captures long-range dependencies
   - Parallelizable (unlike RNNs)
   - Dynamic weighting based on content

b) Attention is permutation-invariant.
   Without positional encoding, "The cat sat" = "sat cat The".
   Position encodes sequence order information.

c) O(n²) where n is sequence length.
   Each position attends to all n positions.
   This limits maximum sequence length.


└─────────────────────────────────────────────────────────────────────┘
```

---

## Chapter 60: Coding Exercises

### 60.1 Implementation from Scratch

```python
EXERCISE 1: Implement Logistic Regression from Scratch
=====================================================
Complete the following class to implement binary logistic regression.

class LogisticRegressionScratch:
    """
    Binary logistic regression using gradient descent.
    
    TODO: Implement the missing methods.
    """
    
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.lr = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        """TODO: Implement sigmoid function."""
        # SOLUTION:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        """
        TODO: Implement gradient descent training.
        
        Steps:
        1. Initialize weights and bias
        2. For each iteration:
           a. Compute predictions (sigmoid of linear combination)
           b. Compute gradients
           c. Update weights and bias
        """
        n_samples, n_features = X.shape
        
        # SOLUTION:
        # Initialize
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.num_iterations):
            # Forward pass
            linear = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear)
            
            # Gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict_proba(self, X):
        """Return probability of class 1."""
        linear = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear)
    
    def predict(self, X, threshold=0.5):
        """Return class predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)


# Test
print("Exercise 1: Logistic Regression")
print("=" * 50)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegressionScratch(learning_rate=0.1, num_iterations=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
```

```python
EXERCISE 2: Implement K-Means from Scratch
==========================================

class KMeansScratch:
    """
    K-Means clustering algorithm.
    
    TODO: Implement initialization, assignment, and update steps.
    """
    
    def __init__(self, n_clusters=3, max_iterations=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None
    
    def fit(self, X):
        """
        TODO: Implement K-Means algorithm.
        
        Steps:
        1. Initialize centroids (k-means++ or random)
        2. Repeat until convergence:
           a. Assign each point to nearest centroid
           b. Update centroids as mean of assigned points
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # SOLUTION:
        # Initialize centroids randomly from data points
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx].copy()
        
        for iteration in range(self.max_iterations):
            # Assignment step
            distances = self._compute_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # Update step
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                if np.sum(self.labels == k) > 0:
                    new_centroids[k] = X[self.labels == k].mean(axis=0)
                else:
                    new_centroids[k] = self.centroids[k]
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
        
        return self
    
    def _compute_distances(self, X):
        """Compute distances from each point to each centroid."""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.sqrt(np.sum((X - self.centroids[k])**2, axis=1))
        return distances
    
    def predict(self, X):
        """Assign clusters to new points."""
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)
    
    def inertia(self, X):
        """Compute within-cluster sum of squares."""
        distances = self._compute_distances(X)
        return np.sum(np.min(distances, axis=1)**2)


print("\nExercise 2: K-Means Clustering")
print("=" * 50)
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)
kmeans = KMeansScratch(n_clusters=4, random_state=42)
kmeans.fit(X)
print(f"Inertia: {kmeans.inertia(X):.2f}")
```

```python
EXERCISE 3: Implement a Simple Neural Network
=============================================

class NeuralNetworkScratch:
    """
    Simple feed-forward neural network with one hidden layer.
    
    TODO: Implement forward pass, backward pass, and training.
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        TODO: Implement forward pass.
        
        Returns: output probabilities and cache for backprop
        """
        # SOLUTION:
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2
    
    def backward(self, X, y):
        """
        TODO: Implement backward pass.
        
        Compute gradients for all weights and biases.
        """
        m = X.shape[0]
        
        # SOLUTION:
        # Output layer gradient
        dz2 = self.a2.copy()
        dz2[range(m), y] -= 1
        dz2 /= m
        
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradient
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)
        
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def fit(self, X, y, epochs=100):
        """Train the network."""
        for epoch in range(epochs):
            # Forward
            probs = self.forward(X)
            
            # Compute loss
            loss = -np.mean(np.log(probs[range(len(y)), y] + 1e-10))
            
            # Backward
            self.backward(X, y)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """Predict class labels."""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)


print("\nExercise 3: Neural Network")
print("=" * 50)
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
X = (X - X.mean(axis=0)) / X.std(axis=0)  # Standardize

nn = NeuralNetworkScratch(input_size=4, hidden_size=10, output_size=3, learning_rate=0.1)
nn.fit(X, y, epochs=500)
predictions = nn.predict(X)
print(f"Accuracy: {np.mean(predictions == y):.4f}")
```

### 60.2 PyTorch Exercises

```python
EXERCISE 4: Implement a CNN for CIFAR-10
========================================

TODO: Complete the CNN architecture and training loop.

import torch
import torch.nn as nn
import torch.optim as optim

class CIFAR10CNN(nn.Module):
    """
    CNN for CIFAR-10 classification.
    
    TODO: Design a CNN with:
    - 3 convolutional blocks (conv -> bn -> relu -> pool)
    - 2 fully connected layers
    - Dropout for regularization
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # SOLUTION:
        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_cifar10(model, train_loader, test_loader, epochs=10, lr=0.001):
    """
    TODO: Implement training loop with:
    - Adam optimizer
    - Learning rate scheduling
    - Training and validation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Acc: {100*train_correct/train_total:.2f}%, '
              f'Test Acc: {100*test_correct/test_total:.2f}%')


print("\nExercise 4: CIFAR-10 CNN")
print("=" * 50)
model = CIFAR10CNN()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

```python
EXERCISE 5: Implement Attention Mechanism
=========================================

TODO: Implement scaled dot-product attention from scratch.

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    TODO: Implement forward pass with optional masking.
    """
    
    def __init__(self, d_k):
        super().__init__()
        self.scale = np.sqrt(d_k)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_len_q, d_k)
            key: (batch, seq_len_k, d_k)
            value: (batch, seq_len_v, d_v)
            mask: (batch, seq_len_q, seq_len_k) optional
        
        Returns:
            output: (batch, seq_len_q, d_v)
            attention_weights: (batch, seq_len_q, seq_len_k)
        """
        # SOLUTION:
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttentionExercise(nn.Module):
    """
    Multi-Head Attention.
    
    TODO: Implement multi-head attention using the ScaledDotProductAttention.
    """
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # SOLUTION:
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape to (batch, heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(output)


print("\nExercise 5: Attention Mechanism")
print("=" * 50)
mha = MultiHeadAttentionExercise(d_model=64, num_heads=8)
x = torch.randn(2, 10, 64)
output = mha(x, x, x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

---

## Chapter 61: Mini-Projects

### 61.1 Project Ideas

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MINI-PROJECTS                                  │
├─────────────────────────────────────────────────────────────────────┤

PROJECT 1: Sentiment Analysis Pipeline
=====================================
Build an end-to-end sentiment analysis system:
1. Data collection (movie reviews, tweets)
2. Text preprocessing pipeline
3. Compare models: Naive Bayes, LSTM, BERT
4. Build REST API for predictions
5. Create simple web interface

Deliverables:
- Trained models with evaluation metrics
- API documentation
- Performance comparison report


PROJECT 2: Image Classification with Transfer Learning
====================================================
Fine-tune a pre-trained model for custom classification:
1. Collect/curate image dataset (100+ images per class)
2. Implement data augmentation
3. Fine-tune ResNet/EfficientNet
4. Experiment with different strategies:
   - Feature extraction (frozen backbone)
   - Full fine-tuning
   - Gradual unfreezing
5. Deploy as mobile app or web service

Deliverables:
- Training/validation curves
- Confusion matrix analysis
- Deployed model demo


PROJECT 3: Time Series Forecasting
=================================
Build a forecasting system for real-world data:
1. Obtain time series data (stock prices, weather, sales)
2. Perform EDA and stationarity analysis
3. Implement multiple models:
   - ARIMA/SARIMA
   - Prophet
   - LSTM/GRU
   - Transformer
4. Create ensemble predictions
5. Build dashboard for visualization

Deliverables:
- Model comparison with proper backtesting
- Confidence intervals for predictions
- Interactive dashboard


PROJECT 4: Recommendation System
================================
Build a movie/product recommendation system:
1. Use MovieLens or similar dataset
2. Implement multiple approaches:
   - Collaborative filtering (user-based, item-based)
   - Matrix factorization
   - Neural collaborative filtering
   - Content-based filtering
3. Create hybrid system
4. Evaluate with proper metrics

Deliverables:
- Offline evaluation (RMSE, precision@k, recall@k)
- A/B testing framework design
- Working recommendation API


PROJECT 5: Object Detection System
=================================
Build an object detection system:
1. Choose domain (traffic signs, product detection, etc.)
2. Label or obtain annotated dataset
3. Train YOLOv5 or Faster R-CNN
4. Optimize for edge deployment
5. Real-time video processing

Deliverables:
- mAP metrics on test set
- Speed benchmarks (FPS)
- Demo with webcam/video


PROJECT 6: Question Answering System
===================================
Build a QA system using transformer models:
1. Use SQuAD or custom Q&A dataset
2. Fine-tune BERT/RoBERTa for extractive QA
3. Implement retrieval-augmented generation
4. Add conversation context handling
5. Evaluate on held-out questions

Deliverables:
- F1 score and exact match metrics
- Error analysis
- Demo interface


└─────────────────────────────────────────────────────────────────────┘
```

### 61.2 Project Template

```python
PROJECT TEMPLATE
================

Use this structure for your ML projects.

# project/
# ├── data/
# │   ├── raw/
# │   ├── processed/
# │   └── external/
# ├── notebooks/
# │   ├── 01_eda.ipynb
# │   ├── 02_preprocessing.ipynb
# │   └── 03_modeling.ipynb
# ├── src/
# │   ├── __init__.py
# │   ├── data/
# │   │   ├── __init__.py
# │   │   ├── load_data.py
# │   │   └── preprocess.py
# │   ├── features/
# │   │   ├── __init__.py
# │   │   └── build_features.py
# │   ├── models/
# │   │   ├── __init__.py
# │   │   ├── train.py
# │   │   └── predict.py
# │   └── visualization/
# │       ├── __init__.py
# │       └── visualize.py
# ├── tests/
# │   └── test_*.py
# ├── models/
# │   └── trained_model.pkl
# ├── config.yaml
# ├── requirements.txt
# ├── setup.py
# └── README.md

import yaml
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ProjectConfig:
    """Configuration for ML project."""
    
    # Data
    data_path: str
    test_size: float = 0.2
    random_state: int = 42
    
    # Model
    model_type: str = "random_forest"
    model_params: dict = None
    
    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    
    # Paths
    model_save_path: str = "models/"
    log_path: str = "logs/"
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)


class MLPipeline:
    """Base class for ML pipeline."""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.model = None
        self.preprocessor = None
    
    def load_data(self):
        """Load and split data."""
        raise NotImplementedError
    
    def preprocess(self, X):
        """Preprocess features."""
        raise NotImplementedError
    
    def train(self, X_train, y_train):
        """Train model."""
        raise NotImplementedError
    
    def evaluate(self, X_test, y_test):
        """Evaluate model."""
        raise NotImplementedError
    
    def save(self, path: str):
        """Save model and preprocessor."""
        raise NotImplementedError
    
    def load(self, path: str):
        """Load model and preprocessor."""
        raise NotImplementedError
    
    def run(self):
        """Run full pipeline."""
        print("Loading data...")
        X_train, X_test, y_train, y_test = self.load_data()
        
        print("Preprocessing...")
        X_train = self.preprocess(X_train)
        X_test = self.preprocess(X_test)
        
        print("Training...")
        self.train(X_train, y_train)
        
        print("Evaluating...")
        metrics = self.evaluate(X_test, y_test)
        
        print("Saving...")
        self.save(self.config.model_save_path)
        
        return metrics


print("\nProject Template Ready")
print("=" * 50)
print("Follow this structure for organized ML projects")
```

---

## Chapter 62: Quiz Questions

### 62.1 Multiple Choice Questions

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTIPLE CHOICE QUIZ                             │
├─────────────────────────────────────────────────────────────────────┤

Q1. Which regularization technique can lead to sparse weights?
    a) L2 regularization
    b) L1 regularization
    c) Dropout
    d) Batch normalization
    
    ANSWER: b) L1 regularization


Q2. What is the time complexity of self-attention with respect to 
    sequence length n?
    a) O(n)
    b) O(n log n)
    c) O(n²)
    d) O(n³)
    
    ANSWER: c) O(n²)


Q3. In a confusion matrix for binary classification, what does 
    the element at position (1,0) represent?
    a) True Positive
    b) True Negative
    c) False Positive
    d) False Negative
    
    ANSWER: d) False Negative (actual=1, predicted=0)


Q4. Which optimizer adapts learning rate per-parameter?
    a) SGD
    b) SGD with momentum
    c) Adam
    d) All of the above
    
    ANSWER: c) Adam


Q5. What problem does batch normalization primarily address?
    a) Overfitting
    b) Internal covariate shift
    c) Vanishing gradients
    d) Class imbalance
    
    ANSWER: b) Internal covariate shift


Q6. In a GAN, what is the discriminator's objective?
    a) Generate realistic samples
    b) Distinguish real from fake samples
    c) Minimize reconstruction error
    d) Maximize data likelihood
    
    ANSWER: b) Distinguish real from fake samples


Q7. What is the purpose of positional encoding in transformers?
    a) Reduce model size
    b) Speed up training
    c) Encode sequence order information
    d) Prevent overfitting
    
    ANSWER: c) Encode sequence order information


Q8. Which metric is most appropriate for highly imbalanced datasets?
    a) Accuracy
    b) F1 Score
    c) Mean Squared Error
    d) R² Score
    
    ANSWER: b) F1 Score


Q9. What is the main advantage of ResNet's skip connections?
    a) Reduce parameters
    b) Enable training of deeper networks
    c) Faster inference
    d) Better data augmentation
    
    ANSWER: b) Enable training of deeper networks


Q10. In cross-validation, why do we split data multiple times?
     a) To get more training data
     b) To reduce variance in performance estimate
     c) To speed up training
     d) To prevent data leakage
     
     ANSWER: b) To reduce variance in performance estimate


└─────────────────────────────────────────────────────────────────────┘
```

### 62.2 True/False Questions

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRUE/FALSE QUIZ                                  │
├─────────────────────────────────────────────────────────────────────┤

T/F 1: Decision trees are invariant to feature scaling.
       ANSWER: TRUE - Trees use comparison operators, not magnitudes.

T/F 2: Dropout should be applied during both training and inference.
       ANSWER: FALSE - Dropout is only applied during training.

T/F 3: ReLU can suffer from the "dying ReLU" problem.
       ANSWER: TRUE - Neurons can get stuck outputting 0.

T/F 4: More training data always leads to better model performance.
       ANSWER: FALSE - Quality matters; noisy data can hurt.

T/F 5: SGD with momentum can overshoot the minimum.
       ANSWER: TRUE - Momentum can carry past optimal point.

T/F 6: Precision and recall are always inversely related.
       ANSWER: FALSE - They can both improve with better models.

T/F 7: CNNs are translation invariant by design.
       ANSWER: FALSE - They are translation equivariant, not invariant.
       Pooling adds some invariance.

T/F 8: Transfer learning always improves performance.
       ANSWER: FALSE - Negative transfer can occur if domains differ.

T/F 9: K-means always converges to the global optimum.
       ANSWER: FALSE - It converges to local optimum; depends on init.

T/F 10: Softmax output probabilities always sum to 1.
        ANSWER: TRUE - By mathematical construction of softmax.


└─────────────────────────────────────────────────────────────────────┘
```

---

## Summary: Practice and Assessment

```
┌─────────────────────────────────────────────────────────────────────┐
│              PRACTICE RECOMMENDATIONS                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  FOR BEGINNERS:                                                     │
│  - Start with conceptual exercises                                 │
│  - Implement basic algorithms from scratch                         │
│  - Use well-known datasets (MNIST, Iris, Titanic)                 │
│  - Focus on understanding over performance                         │
│                                                                     │
│  FOR INTERMEDIATE:                                                  │
│  - Complete mini-projects end-to-end                               │
│  - Experiment with hyperparameter tuning                           │
│  - Compare multiple approaches                                      │
│  - Practice with Kaggle competitions                               │
│                                                                     │
│  FOR ADVANCED:                                                      │
│  - Read and implement papers                                        │
│  - Contribute to open source                                        │
│  - Design custom architectures                                      │
│  - Focus on production deployment                                   │
│                                                                     │
│  KEY SKILLS TO DEVELOP:                                             │
│  - Problem framing                                                  │
│  - Data exploration and preprocessing                              │
│  - Model selection and evaluation                                  │
│  - Debugging and error analysis                                    │
│  - Communication of results                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```


---

<div align="center">

[⬅️ Previous: Graph Neural Networks](14-graph-neural-networks.md) | [📚 Table of Contents](../README.md) | [Next: Foundation Models ➡️](16-foundation-models.md)

</div>

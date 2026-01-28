<div align="center">

# 💼 Interview Preparation

![Chapter](https://img.shields.io/badge/Chapter-18-blue?style=for-the-badge)
![Topic](https://img.shields.io/badge/Topic-Career%20%7C%20Coding-green?style=for-the-badge)

*System Design, Coding Questions & Behavioral Interviews*

---

</div>

# Part XXI: Interview Preparation and Real-World ML

---

## Chapter 84: ML System Design Interview

### 84.1 System Design Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│              ML SYSTEM DESIGN FRAMEWORK                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  STEP 1: CLARIFY REQUIREMENTS (5 min)                              │
│  ├── What is the business goal?                                    │
│  ├── What metrics matter?                                          │
│  ├── What are the constraints (latency, throughput, cost)?         │
│  ├── What data is available?                                       │
│  └── Who are the users?                                            │
│                                                                     │
│  STEP 2: FRAME THE ML PROBLEM (5 min)                              │
│  ├── What type of ML problem is this?                              │
│  ├── What is the input/output?                                     │
│  ├── What are the key features?                                    │
│  └── How will we evaluate success?                                 │
│                                                                     │
│  STEP 3: DATA PIPELINE (10 min)                                    │
│  ├── Data sources                                                  │
│  ├── Data collection and storage                                   │
│  ├── Feature engineering                                           │
│  ├── Data validation                                               │
│  └── Training/serving data split                                   │
│                                                                     │
│  STEP 4: MODEL DEVELOPMENT (10 min)                                │
│  ├── Baseline model                                                │
│  ├── Model selection                                               │
│  ├── Training pipeline                                             │
│  ├── Hyperparameter tuning                                         │
│  └── Offline evaluation                                            │
│                                                                     │
│  STEP 5: SERVING & DEPLOYMENT (10 min)                             │
│  ├── Online vs batch inference                                     │
│  ├── Model serving infrastructure                                  │
│  ├── A/B testing                                                   │
│  ├── Monitoring and alerting                                       │
│  └── Model updates                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 84.2 Common ML System Design Questions

```
┌─────────────────────────────────────────────────────────────────────┐
│         DESIGN: RECOMMENDATION SYSTEM (e.g., YouTube)              │
├─────────────────────────────────────────────────────────────────────┤

REQUIREMENTS:
- Personalized video recommendations
- Billions of users, millions of videos
- Real-time recommendations
- Optimize for engagement (watch time)

DATA:
- User features: demographics, history, preferences
- Video features: metadata, embeddings, popularity
- Interaction data: clicks, watch time, likes, shares
- Context: time of day, device, location

ARCHITECTURE:
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  [User Request] → [Candidate Generation] → [Ranking] → [Results]  │
│                          │                     │                   │
│                    (1000s of                 (Top N)               │
│                     candidates)                                    │
│                                                                    │
│  Candidate Generation:                                             │
│  - Collaborative filtering (similar users)                         │
│  - Content-based (similar videos)                                  │
│  - Popular/trending videos                                         │
│  - Explore (fresh content)                                         │
│                                                                    │
│  Ranking Model:                                                    │
│  - Two-tower neural network                                        │
│  - User tower: user features → embedding                           │
│  - Item tower: video features → embedding                          │
│  - Score = dot(user_emb, item_emb)                                │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

METRICS:
- Offline: AUC, Precision@K, Recall@K, NDCG
- Online: CTR, Watch time, User retention

CHALLENGES:
- Cold start (new users/videos)
- Filter bubbles
- Scalability
- Freshness vs quality tradeoff

└─────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────┐
│           DESIGN: SEARCH RANKING (e.g., Google)                    │
├─────────────────────────────────────────────────────────────────────┤

REQUIREMENTS:
- Relevant search results
- Sub-100ms latency
- Billions of queries daily
- Handle ambiguous queries

ARCHITECTURE:
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  [Query] → [Query Understanding] → [Retrieval] → [Ranking]        │
│                    │                    │            │             │
│           (spell check,         (inverted      (ML ranker)        │
│            expansion,            index)                            │
│            intent)                                                 │
│                                                                    │
│  Retrieval (fast, broad):                                          │
│  - Inverted index (BM25)                                           │
│  - Dense retrieval (BERT embeddings)                               │
│  - Return top 1000 candidates                                      │
│                                                                    │
│  Ranking (accurate, slow):                                         │
│  - Learning to Rank                                                │
│  - Features: relevance, freshness, authority, personalization     │
│  - BERT-based cross-encoder                                        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

FEATURES:
- Query features: length, intent, entity mentions
- Document features: title match, freshness, PageRank
- Query-document features: BM25, semantic similarity
- User features: history, location, language

TRAINING:
- Click data (implicit feedback)
- Human ratings (explicit relevance)
- Pairwise/listwise loss functions

└─────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────┐
│          DESIGN: FRAUD DETECTION SYSTEM                            │
├─────────────────────────────────────────────────────────────────────┤

REQUIREMENTS:
- Real-time fraud detection for transactions
- High precision (minimize false positives)
- Handle class imbalance (fraud is rare)
- Adapt to evolving fraud patterns

DATA:
- Transaction features: amount, merchant, time, location
- User features: account age, history, device
- Network features: graph of user relationships
- Historical fraud labels

ARCHITECTURE:
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  [Transaction] → [Feature Store] → [Model Ensemble] → [Decision]  │
│        │               │                  │              │         │
│        │          (real-time         (rules +        (approve/    │
│        │           features)          ML models)      block/       │
│        │                                              review)      │
│        ↓                                                           │
│  [Graph Database] → [Network Analysis]                             │
│                                                                    │
│  Model Types:                                                      │
│  - Rule-based (known patterns)                                     │
│  - XGBoost (tabular features)                                      │
│  - Neural network (sequence of transactions)                       │
│  - Graph neural network (relationship patterns)                    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

HANDLING IMBALANCE:
- Oversampling (SMOTE)
- Class weights
- Anomaly detection approach
- Precision-recall tradeoff

DEPLOYMENT:
- Real-time scoring (<100ms)
- Batch retraining (daily/weekly)
- Model versioning
- Shadow mode testing

└─────────────────────────────────────────────────────────────────────┘
```

### 84.3 Design Question: Ad Click Prediction

```python
DESIGN: Ad Click Prediction System (CTR Prediction)

Goal: Predict probability user clicks on ad
Optimize: Revenue (clicks × bid)
Constraint: Latency < 10ms, billions of requests/day

# Feature Engineering
class CTRFeatureEngineering:
    """
    Features for CTR prediction.
    """
    
    @staticmethod
    def user_features():
        return """
        STATIC FEATURES:
        - Demographics: age, gender, location
        - Account: account_age, purchase_history
        - Interests: inferred categories
        
        DYNAMIC FEATURES:
        - Recent searches
        - Recent page views
        - Session duration
        - Device and browser
        """
    
    @staticmethod
    def ad_features():
        return """
        STATIC FEATURES:
        - Ad category
        - Advertiser ID
        - Creative type (text, image, video)
        - Historical CTR
        
        CONTENT FEATURES:
        - Ad text embeddings
        - Image features (from CNN)
        """
    
    @staticmethod
    def context_features():
        return """
        - Time of day
        - Day of week
        - Page context
        - Position on page
        - Other ads on page
        """
    
    @staticmethod
    def interaction_features():
        return """
        - User × advertiser history
        - User × category history
        - User-ad similarity
        - Cross features (user_age × ad_category)
        """


class CTRModel:
    """
    CTR prediction model architecture.
    """
    
    def architecture_options(self):
        return """
        OPTION 1: Logistic Regression
        - Pros: Fast, interpretable, handles sparse features
        - Cons: Limited capacity, needs manual feature engineering
        
        OPTION 2: Gradient Boosting (XGBoost/LightGBM)
        - Pros: Handles feature interactions, good performance
        - Cons: Slower inference, harder to update online
        
        OPTION 3: Deep Learning (Wide & Deep, DeepFM)
        - Pros: Automatic feature learning, handles embeddings
        - Cons: Slower, needs more data
        
        RECOMMENDED: Wide & Deep
        - Wide: Memorization (sparse cross features)
        - Deep: Generalization (dense embeddings)
        """
    
    def wide_and_deep_model(self):
        return """
        class WideAndDeep(nn.Module):
            def __init__(self, sparse_dim, dense_dim, embed_dims):
                # Wide part: Linear on sparse features
                self.wide = nn.Linear(sparse_dim, 1)
                
                # Deep part: MLP on dense features + embeddings
                self.embeddings = nn.ModuleList([
                    nn.Embedding(card, dim) 
                    for card, dim in embed_dims
                ])
                
                total_embed = sum(d for _, d in embed_dims)
                self.deep = nn.Sequential(
                    nn.Linear(dense_dim + total_embed, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                )
            
            def forward(self, sparse_x, dense_x, cat_x):
                # Wide
                wide_out = self.wide(sparse_x)
                
                # Deep
                embeds = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
                deep_input = torch.cat([dense_x] + embeds, dim=1)
                deep_out = self.deep(deep_input)
                
                # Combine
                return torch.sigmoid(wide_out + deep_out)
        """


class CTRServingSystem:
    """
    Production serving architecture.
    """
    
    def serving_architecture(self):
        return """
        ┌─────────────────────────────────────────────────────────────┐
        │                    CTR SERVING SYSTEM                       │
        ├─────────────────────────────────────────────────────────────┤
        │                                                             │
        │  [Ad Request] → [Feature Service] → [Model Service]        │
        │        │              │                   │                 │
        │        ↓              ↓                   ↓                 │
        │  [User/Context]  [Feature Store]   [Model Ensemble]        │
        │        │              │                   │                 │
        │        └──────────────┴───────────────────┘                 │
        │                       ↓                                     │
        │              [Ranking & Selection]                          │
        │                       ↓                                     │
        │                 [Ad Response]                               │
        │                                                             │
        │  Optimizations:                                             │
        │  - Feature caching                                          │
        │  - Model quantization                                       │
        │  - Batched inference                                        │
        │  - GPU serving                                              │
        │                                                             │
        └─────────────────────────────────────────────────────────────┘
        """
    
    def monitoring(self):
        return """
        ONLINE METRICS:
        - CTR (primary)
        - Revenue per impression
        - Latency (p50, p99)
        - Prediction distribution
        
        MODEL HEALTH:
        - Feature drift
        - Prediction drift
        - Training-serving skew
        - Stale features
        
        ALERTING:
        - CTR drops > 10%
        - Latency spikes
        - Error rate increase
        - Feature missing rate
        """


print("CTR Prediction System Design:")
print("=" * 60)
ctr = CTRModel()
print(ctr.architecture_options())
```

---

## Chapter 85: Coding Interview Questions

### 85.1 Classic ML Coding Questions

```python
QUESTION 1: Implement K-Fold Cross Validation

def k_fold_cross_validation(X, y, model_fn, k=5, metric_fn=None):
    """
    Implement k-fold cross validation from scratch.
    
    Args:
        X: Features array
        y: Labels array
        model_fn: Function that returns a new model instance
        k: Number of folds
        metric_fn: Evaluation metric function
    
    Returns:
        scores: List of scores for each fold
    """
    n_samples = len(X)
    fold_size = n_samples // k
    indices = np.random.permutation(n_samples)
    
    scores = []
    
    for i in range(k):
        # Define validation indices for this fold
        val_start = i * fold_size
        val_end = val_start + fold_size if i < k - 1 else n_samples
        val_indices = indices[val_start:val_end]
        
        # Training indices are everything else
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        # Split data
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        # Train and evaluate
        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        if metric_fn:
            score = metric_fn(y_val, y_pred)
        else:
            score = np.mean(y_pred == y_val)  # Accuracy
        
        scores.append(score)
    
    return scores


QUESTION 2: Implement Softmax and Cross-Entropy Loss

def softmax(logits):
    """
    Compute softmax probabilities.
    
    Args:
        logits: (batch_size, num_classes)
    
    Returns:
        probs: (batch_size, num_classes) summing to 1
    """
    # Subtract max for numerical stability
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy_loss(logits, targets):
    """
    Compute cross-entropy loss.
    
    Args:
        logits: (batch_size, num_classes) raw scores
        targets: (batch_size,) class indices
    
    Returns:
        loss: scalar
    """
    batch_size = logits.shape[0]
    probs = softmax(logits)
    
    # Get probability of correct class
    correct_probs = probs[np.arange(batch_size), targets]
    
    # Negative log likelihood
    loss = -np.mean(np.log(correct_probs + 1e-10))
    
    return loss


def cross_entropy_gradient(logits, targets):
    """
    Compute gradient of cross-entropy w.r.t. logits.
    
    Returns:
        gradient: (batch_size, num_classes)
    """
    batch_size = logits.shape[0]
    probs = softmax(logits)
    
    # Gradient is (probs - one_hot_targets) / batch_size
    grad = probs.copy()
    grad[np.arange(batch_size), targets] -= 1
    grad /= batch_size
    
    return grad


QUESTION 3: Implement Batch Normalization

class BatchNormalization:
    """
    Batch Normalization layer implementation.
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Cache for backward pass
        self.cache = None
    
    def forward(self, x, training=True):
        """
        Forward pass.
        
        Args:
            x: (batch_size, num_features)
            training: Whether in training mode
        """
        if training:
            # Compute batch statistics
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            
            # Normalize
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            
            # Scale and shift
            out = self.gamma * x_norm + self.beta
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            # Cache for backward
            self.cache = (x, x_norm, mean, var)
        else:
            # Use running statistics
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        
        Args:
            dout: Gradient from upstream
        
        Returns:
            dx: Gradient w.r.t. input
            dgamma: Gradient w.r.t. gamma
            dbeta: Gradient w.r.t. beta
        """
        x, x_norm, mean, var = self.cache
        batch_size = x.shape[0]
        
        # Gradients of gamma and beta
        dgamma = np.sum(dout * x_norm, axis=0)
        dbeta = np.sum(dout, axis=0)
        
        # Gradient w.r.t. normalized x
        dx_norm = dout * self.gamma
        
        # Gradient w.r.t. variance
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + self.eps)**(-1.5), axis=0)
        
        # Gradient w.r.t. mean
        dmean = np.sum(dx_norm * -1 / np.sqrt(var + self.eps), axis=0)
        dmean += dvar * np.mean(-2 * (x - mean), axis=0)
        
        # Gradient w.r.t. input
        dx = dx_norm / np.sqrt(var + self.eps)
        dx += dvar * 2 * (x - mean) / batch_size
        dx += dmean / batch_size
        
        return dx, dgamma, dbeta


QUESTION 4: Implement Attention Mechanism

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Implement scaled dot-product attention.
    
    Args:
        query: (batch, seq_len_q, d_k)
        key: (batch, seq_len_k, d_k)
        value: (batch, seq_len_v, d_v)
        mask: Optional mask
    
    Returns:
        output: (batch, seq_len_q, d_v)
        attention_weights: (batch, seq_len_q, seq_len_k)
    """
    d_k = query.shape[-1]
    
    # Compute attention scores
    scores = np.matmul(query, key.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # Apply mask
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    
    # Softmax
    attention_weights = softmax(scores)
    
    # Apply attention to values
    output = np.matmul(attention_weights, value)
    
    return output, attention_weights


QUESTION 5: Implement Decision Tree Splitting

def gini_impurity(y):
    """Compute Gini impurity."""
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)


def information_gain(y, y_left, y_right):
    """Compute information gain from a split."""
    n = len(y)
    n_left, n_right = len(y_left), len(y_right)
    
    if n_left == 0 or n_right == 0:
        return 0
    
    parent_impurity = gini_impurity(y)
    left_impurity = gini_impurity(y_left)
    right_impurity = gini_impurity(y_right)
    
    weighted_child_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
    
    return parent_impurity - weighted_child_impurity


def find_best_split(X, y):
    """
    Find the best feature and threshold for splitting.
    
    Returns:
        best_feature: Index of best feature
        best_threshold: Threshold value
        best_gain: Information gain
    """
    best_gain = 0
    best_feature = None
    best_threshold = None
    
    n_features = X.shape[1]
    
    for feature_idx in range(n_features):
        # Get unique values for this feature
        thresholds = np.unique(X[:, feature_idx])
        
        for threshold in thresholds:
            # Split data
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            # Compute information gain
            gain = information_gain(y, y[left_mask], y[right_mask])
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gain


# Test implementations
print("ML Coding Questions - Test Results:")
print("=" * 60)

# Test softmax
logits = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
probs = softmax(logits)
print(f"Softmax test - sums to 1: {np.allclose(probs.sum(axis=1), 1)}")

# Test cross-entropy
targets = np.array([2, 0])
loss = cross_entropy_loss(logits, targets)
print(f"Cross-entropy loss: {loss:.4f}")

# Test batch norm
bn = BatchNormalization(3)
x = np.random.randn(4, 3)
out = bn.forward(x, training=True)
print(f"BatchNorm output mean ~0: {np.allclose(out.mean(axis=0), 0, atol=0.1)}")

# Test attention
q = np.random.randn(2, 4, 8)
k = np.random.randn(2, 4, 8)
v = np.random.randn(2, 4, 8)
output, weights = scaled_dot_product_attention(q, k, v)
print(f"Attention weights sum to 1: {np.allclose(weights.sum(axis=-1), 1)}")
```

### 85.2 Algorithm Implementation Questions

```python
QUESTION 6: Implement Mini-Batch Gradient Descent

def mini_batch_gradient_descent(X, y, model, loss_fn, grad_fn, 
                                 learning_rate=0.01, batch_size=32, 
                                 epochs=100):
    """
    Mini-batch gradient descent implementation.
    
    Args:
        X: Training features
        y: Training labels
        model: Dict of parameters
        loss_fn: Loss function
        grad_fn: Gradient function
        learning_rate: Learning rate
        batch_size: Batch size
        epochs: Number of epochs
    
    Returns:
        model: Updated parameters
        history: Loss history
    """
    n_samples = len(X)
    history = []
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            # Get batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Compute loss and gradients
            loss = loss_fn(model, X_batch, y_batch)
            grads = grad_fn(model, X_batch, y_batch)
            
            # Update parameters
            for key in model:
                model[key] -= learning_rate * grads[key]
            
            epoch_loss += loss
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        history.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    return model, history


QUESTION 7: Implement K-Means++ Initialization

def kmeans_plus_plus_init(X, k):
    """
    K-Means++ initialization for better starting centroids.
    
    Args:
        X: Data points (n_samples, n_features)
        k: Number of clusters
    
    Returns:
        centroids: Initial centroids (k, n_features)
    """
    n_samples = X.shape[0]
    centroids = []
    
    # Choose first centroid randomly
    idx = np.random.randint(n_samples)
    centroids.append(X[idx])
    
    for _ in range(1, k):
        # Compute distances to nearest centroid
        distances = np.zeros(n_samples)
        for i in range(n_samples):
            min_dist = float('inf')
            for centroid in centroids:
                dist = np.sum((X[i] - centroid) ** 2)
                min_dist = min(min_dist, dist)
            distances[i] = min_dist
        
        # Sample proportional to squared distance
        probs = distances / distances.sum()
        idx = np.random.choice(n_samples, p=probs)
        centroids.append(X[idx])
    
    return np.array(centroids)


QUESTION 8: Implement Early Stopping

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    
    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        """
        Args:
            patience: Epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, score, model=None):
        """
        Check if should stop.
        
        Args:
            score: Current validation metric
            model: Optional model to save best weights
        
        Returns:
            should_stop: Boolean
        """
        if self.best_score is None:
            self.best_score = score
            if model is not None:
                self.best_weights = {k: v.copy() for k, v in model.items()}
            return False
        
        # Check for improvement
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            if model is not None:
                self.best_weights = {k: v.copy() for k, v in model.items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


QUESTION 9: Implement ROC Curve and AUC

def compute_roc_curve(y_true, y_scores):
    """
    Compute ROC curve.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
    
    Returns:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Thresholds used
    """
    # Sort by scores
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    # Count positives and negatives
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    
    tpr = [0]
    fpr = [0]
    thresholds = [y_scores_sorted[0] + 1]  # Start above max score
    
    tp = 0
    fp = 0
    
    for i, (score, label) in enumerate(zip(y_scores_sorted, y_true_sorted)):
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
        thresholds.append(score)
    
    return np.array(fpr), np.array(tpr), np.array(thresholds)


def compute_auc(fpr, tpr):
    """
    Compute AUC using trapezoidal rule.
    """
    auc = 0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    return auc


QUESTION 10: Implement Precision-Recall Curve

def precision_recall_curve(y_true, y_scores):
    """
    Compute precision-recall curve.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
    
    Returns:
        precision: Precision values
        recall: Recall values
        thresholds: Thresholds used
    """
    # Sort by scores (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    n_pos = np.sum(y_true)
    
    precision = []
    recall = []
    thresholds = []
    
    tp = 0
    fp = 0
    
    for i, (score, label) in enumerate(zip(y_scores_sorted, y_true_sorted)):
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        precision.append(tp / (tp + fp))
        recall.append(tp / n_pos)
        thresholds.append(score)
    
    return np.array(precision), np.array(recall), np.array(thresholds)


# Test
print("\nAlgorithm Implementation Tests:")
print("=" * 60)

# Test K-Means++ init
X_test = np.random.randn(100, 2)
centroids = kmeans_plus_plus_init(X_test, k=3)
print(f"K-Means++ centroids shape: {centroids.shape}")

# Test ROC/AUC
y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.65, 0.2, 0.9, 0.3, 0.6, 0.7])
fpr, tpr, _ = compute_roc_curve(y_true, y_scores)
auc = compute_auc(fpr, tpr)
print(f"AUC: {auc:.4f}")
```

---

## Chapter 86: Behavioral Interview Questions

### 86.1 Common Questions and Answers

```
┌─────────────────────────────────────────────────────────────────────┐
│              ML BEHAVIORAL INTERVIEW QUESTIONS                      │
├─────────────────────────────────────────────────────────────────────┤

Q1: Tell me about an ML project you're proud of.

STRUCTURE (STAR Method):
- Situation: What was the context?
- Task: What was your specific responsibility?
- Action: What did you do?
- Result: What was the outcome?

EXAMPLE ANSWER:
"At [Company], we had a customer churn problem costing $2M annually.

SITUATION: 15% monthly churn rate with no prediction capability.

TASK: I led development of a churn prediction system.

ACTION:
- Analyzed historical data (100K customers, 50 features)
- Built feature engineering pipeline
- Compared models: logistic regression, XGBoost, neural network
- XGBoost won with 0.85 AUC
- Deployed with real-time scoring API
- Set up monitoring for drift detection

RESULT:
- Reduced churn by 25% through targeted retention
- Saved $500K annually
- Model still in production 2 years later"


Q2: Tell me about a time an ML project failed. What did you learn?

EXAMPLE ANSWER:
"I built a demand forecasting model that performed well offline 
(MAPE 8%) but poorly in production (MAPE 25%).

WHAT WENT WRONG:
- Training data was cleaner than production data
- Missing features in production that existed in training
- Didn't account for seasonality shifts

WHAT I LEARNED:
- Always validate with production-like data
- Set up comprehensive monitoring from day 1
- Test edge cases and data quality issues
- Include domain experts in validation

HOW I IMPROVED:
- Now I always create a production-mirror test set
- Implement feature drift monitoring
- Do shadow deployments before full release"


Q3: How do you prioritize when you have multiple ML projects?

FRAMEWORK:
1. Business Impact: Revenue potential, cost savings
2. Feasibility: Data availability, technical complexity
3. Time to Value: Quick wins vs long-term investments
4. Dependencies: What blocks other projects?

EXAMPLE ANSWER:
"I use an impact/effort matrix:

HIGH IMPACT, LOW EFFORT: Do first (quick wins)
HIGH IMPACT, HIGH EFFORT: Plan carefully, resource well
LOW IMPACT, LOW EFFORT: Fill gaps when available
LOW IMPACT, HIGH EFFORT: Usually deprioritize or kill

I also consider:
- Stakeholder urgency
- Learning opportunities for the team
- Technical debt implications"


Q4: How do you handle disagreements with stakeholders about ML approach?

EXAMPLE ANSWER:
"Recently, a PM wanted to launch a model with 70% accuracy.
I believed we needed 85% to be useful.

MY APPROACH:
1. Understood their perspective (time pressure, competitor launch)
2. Quantified the risk (30% errors = X complaints/day)
3. Proposed alternatives:
   - Launch with human review for uncertain predictions
   - Phase 1 for low-risk segment only
   - A/B test with small traffic
4. Used data to support my position
5. Found middle ground: Limited launch with monitoring

RESULT:
We launched with human review, gathered feedback, improved
to 82% accuracy in 3 weeks, then fully automated."


Q5: How do you explain complex ML concepts to non-technical stakeholders?

TECHNIQUES:
1. Use analogies
2. Focus on business impact, not technical details
3. Visualizations > equations
4. Concrete examples > abstract concepts

EXAMPLE:
"Instead of: 'The model uses gradient boosting with 
100 estimators and max_depth of 6'

I say: 'Think of it like getting opinions from 100 experts, 
each looking at different aspects of the customer. 
We combine their votes to make a final prediction.

It's 85% accurate, which means for every 100 customers 
we flag as likely to churn, about 85 actually will.'"


└─────────────────────────────────────────────────────────────────────┘
```

---

## Summary: Interview Preparation

```
┌─────────────────────────────────────────────────────────────────────┐
│              INTERVIEW PREPARATION CHECKLIST                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CODING PREPARATION:                                                │
│  □ Implement ML algorithms from scratch                            │
│  □ NumPy/Pandas proficiency                                        │
│  □ PyTorch/TensorFlow basics                                       │
│  □ Data structures and algorithms                                  │
│  □ SQL queries                                                     │
│                                                                     │
│  ML CONCEPTS:                                                       │
│  □ Bias-variance tradeoff                                          │
│  □ Evaluation metrics for different tasks                          │
│  □ Regularization techniques                                       │
│  □ Feature engineering                                             │
│  □ Model selection criteria                                        │
│                                                                     │
│  SYSTEM DESIGN:                                                     │
│  □ Recommendation systems                                          │
│  □ Search ranking                                                  │
│  □ Fraud detection                                                 │
│  □ Ad click prediction                                             │
│  □ Real-time ML systems                                            │
│                                                                     │
│  BEHAVIORAL:                                                        │
│  □ Past project stories (STAR method)                              │
│  □ Failure and learning examples                                   │
│  □ Collaboration examples                                          │
│  □ Technical communication examples                                │
│                                                                     │
│  DAY BEFORE:                                                        │
│  □ Review your resume projects                                     │
│  □ Prepare questions for interviewer                               │
│  □ Test your setup (video, audio, IDE)                            │
│  □ Get good sleep!                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```


---

<div align="center">

[⬅️ Previous: Advanced Algorithms](17-advanced-algorithms.md) | [📚 Table of Contents](../README.md) | [Next: Case Studies ➡️](19-case-studies.md)

</div>

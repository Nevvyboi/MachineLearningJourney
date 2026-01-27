# 🧠 THE ULTIMATE MACHINE LEARNING GUIDE
## From Zero to Hero: A Complete Reference Manual

> **Version:** 2025 Edition  
> **Last Updated:** January 2025  
> **Skill Level:** Beginner → Intermediate → Expert  
> **Prerequisites:** Python knowledge (you got this!)

---

## 📋 TABLE OF CONTENTS

1. [Unit 1: Introduction to Machine Learning](#unit-1-introduction-to-machine-learning)
2. [Unit 2: Supervised Learning](#unit-2-supervised-learning)
3. [Unit 3: Neural Networks](#unit-3-neural-networks)
4. [Unit 4: Unsupervised Learning](#unit-4-unsupervised-learning)
5. [Unit 5: Language and Time Series](#unit-5-language-and-time-series)
6. [Unit 6: Applications and Responsible ML](#unit-6-applications-and-responsible-ml)
7. [Quick Reference Cheat Sheets](#quick-reference-cheat-sheets)
8. [Real-World Projects](#real-world-projects)
9. [Resources & Further Learning](#resources--further-learning)

---

# UNIT 1: INTRODUCTION TO MACHINE LEARNING

## 1.1 What is Machine Learning? (The Honest Truth)

**Plain English:** Machine Learning is teaching computers to learn patterns from data instead of explicitly programming every rule.

**Think of it like this:** 
- Traditional Programming: You tell the computer "IF email contains 'Nigerian Prince' THEN it's spam"
- Machine Learning: You show the computer 10,000 spam emails and 10,000 good emails, and it figures out the patterns itself

```python
# Traditional Programming (Hard-coded rules)
def is_spam_traditional(email):
    spam_words = ['nigerian prince', 'free money', 'click here']
    for word in spam_words:
        if word in email.lower():
            return True
    return False

# Machine Learning Approach (Pattern Learning)
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# The model LEARNS what makes spam from examples
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_emails)
classifier = MultinomialNB()
classifier.fit(X_train, labels)  # labels: 0=not spam, 1=spam

# Now it can predict on NEW emails it's never seen
def is_spam_ml(email):
    X_new = vectorizer.transform([email])
    return classifier.predict(X_new)[0] == 1
```

**Why ML beats traditional programming:**
1. **Scalability** - Can't write rules for every possible spam pattern
2. **Adaptability** - Spammers change tactics; ML models can be retrained
3. **Discovery** - ML finds patterns humans might miss

---

## 1.2 Types of Machine Learning

### 🎯 SUPERVISED LEARNING
**What it is:** Learning from labeled examples (you have the answers)

**Analogy:** A teacher grading your homework - you know what the correct answers should be

| Task | Input (X) | Output (y) | Example |
|------|-----------|------------|---------|
| **Classification** | Features | Category | Email → Spam/Not Spam |
| **Regression** | Features | Number | House features → Price |

```python
# Classification Example: Is this tumor malignant?
from sklearn.ensemble import RandomForestClassifier

# X = features (tumor size, texture, etc.)
# y = labels (0 = benign, 1 = malignant)
model = RandomForestClassifier()
model.fit(X_train, y_train)  # Learn patterns
prediction = model.predict(new_tumor_features)  # Predict

# Regression Example: What's this house worth?
from sklearn.linear_model import LinearRegression

# X = features (bedrooms, sqft, location)
# y = price (continuous value like $450,000)
model = LinearRegression()
model.fit(X_train, y_train)
predicted_price = model.predict(new_house_features)
```

### 🔍 UNSUPERVISED LEARNING
**What it is:** Finding hidden patterns in data WITHOUT labels

**Analogy:** Organizing your closet without instructions - you group similar items together naturally

| Task | What It Does | Example |
|------|--------------|---------|
| **Clustering** | Groups similar items | Customer segmentation |
| **Dimensionality Reduction** | Simplifies complex data | Visualizing 100-dimensional data |
| **Anomaly Detection** | Finds outliers | Fraud detection |

```python
# Clustering Example: Group customers by behavior
from sklearn.cluster import KMeans

# X = customer features (no labels!)
kmeans = KMeans(n_clusters=5)
customer_segments = kmeans.fit_predict(customer_data)
# Output: [0, 2, 1, 0, 4, 2, ...] - which cluster each customer belongs to

# Dimensionality Reduction: Compress 100 features to 2 for visualization
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_2d = pca.fit_transform(data_100d)  # Now we can plot it!
```

### 🎮 REINFORCEMENT LEARNING
**What it is:** Learning through trial and error with rewards/penalties

**Analogy:** Training a dog - reward good behavior, ignore/punish bad behavior

```python
# Conceptual RL Loop (not runnable, just shows the idea)
"""
while game_not_over:
    state = observe_environment()
    action = agent.choose_action(state)  # e.g., move left/right
    reward = environment.step(action)     # +1 for good, -1 for bad
    agent.learn(state, action, reward)    # Update strategy
"""

# Real example with OpenAI Gym
import gym

env = gym.make('CartPole-v1')
state = env.reset()

for _ in range(1000):
    action = policy(state)  # Your trained policy decides
    state, reward, done, info = env.step(action)
    if done:
        break
```

### 🆕 SELF-SUPERVISED LEARNING (The 2024+ Revolution)
**What it is:** Creating labels from the data itself (how GPT and BERT learn!)

**Example:** Mask some words in a sentence, train the model to predict them
```
Input:  "The cat sat on the [MASK]"
Model:  Predicts "mat" or "floor"
Label:  Comes from the original sentence!
```

---

## 1.3 The Machine Learning Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1. DATA    │───▶│ 2. PREPARE  │───▶│  3. TRAIN   │───▶│ 4. EVALUATE │───▶│  5. DEPLOY  │
│  COLLECTION │    │  & CLEAN    │    │    MODEL    │    │    MODEL    │    │   & MONITOR │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                  │                  │
   Raw data          Handling           Choose &          Metrics &         Production &
   gathering        missing vals,       train model       validation        maintenance
                    encoding,
                    scaling
```

### Step 1: Data Collection

```python
import pandas as pd

# From CSV files
df = pd.read_csv('data.csv')

# From databases
import sqlalchemy
engine = sqlalchemy.create_engine('postgresql://user:pass@host/db')
df = pd.read_sql('SELECT * FROM customers', engine)

# From APIs
import requests
response = requests.get('https://api.example.com/data')
data = response.json()

# From web scraping (be ethical!)
from bs4 import BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')
```

### Step 2: Data Preparation (80% of Your Time!)

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('messy_data.csv')

# ═══════════════════════════════════════════════════════════════
# HANDLING MISSING VALUES
# ═══════════════════════════════════════════════════════════════

# Check for missing values
print(df.isnull().sum())
print(f"Total missing: {df.isnull().sum().sum()}")

# Strategy 1: Drop rows with missing values (if you have lots of data)
df_clean = df.dropna()

# Strategy 2: Fill with mean/median/mode
df['age'].fillna(df['age'].median(), inplace=True)
df['city'].fillna(df['city'].mode()[0], inplace=True)

# Strategy 3: Use ML to predict missing values (advanced)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# ═══════════════════════════════════════════════════════════════
# ENCODING CATEGORICAL VARIABLES
# ═══════════════════════════════════════════════════════════════

# Label Encoding (for ordinal data like "low", "medium", "high")
le = LabelEncoder()
df['size_encoded'] = le.fit_transform(df['size'])  # small=2, medium=1, large=0

# One-Hot Encoding (for nominal data like "red", "blue", "green")
df_encoded = pd.get_dummies(df, columns=['color'])
# Creates: color_red, color_blue, color_green columns with 0/1 values

# ═══════════════════════════════════════════════════════════════
# FEATURE SCALING (CRITICAL FOR MANY ALGORITHMS!)
# ═══════════════════════════════════════════════════════════════

"""
WHY SCALE?
- Age ranges: 0-100
- Income ranges: 0-10,000,000
- Without scaling, income will dominate because of larger numbers!

Algorithms that NEED scaling: KNN, SVM, Neural Networks, PCA
Algorithms that DON'T need scaling: Decision Trees, Random Forest
"""

# StandardScaler: Mean=0, Std=1 (use for most algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler: Scales to range [0,1] (use for neural networks)
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
X_normalized = minmax.fit_transform(X)

# ═══════════════════════════════════════════════════════════════
# TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════════════

"""
GOLDEN RULE: Never train on data you'll test on!

Common splits:
- Simple: 80% train, 20% test
- With validation: 70% train, 15% validation, 15% test
"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducibility
    stratify=y          # Maintain class proportions (IMPORTANT for imbalanced data)
)
```

### Step 3: Model Training

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(
    n_estimators=100,     # Number of trees
    max_depth=10,         # How deep each tree can go
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
```

### Step 4: Model Evaluation

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score
)

# ═══════════════════════════════════════════════════════════════
# CLASSIFICATION METRICS
# ═══════════════════════════════════════════════════════════════

# Accuracy: % of correct predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Confusion Matrix: Shows what types of errors
"""
                 Predicted
              Negative  Positive
Actual  Neg     TN         FP      ← Type I Error (False Alarm)
        Pos     FN         TP      ← Type II Error (Missed)
"""
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Precision: Of all PREDICTED positives, how many were actually positive?
# HIGH PRECISION = Few false alarms
# Use when: False positives are costly (e.g., spam filter marking important email)
precision = precision_score(y_test, y_pred)

# Recall (Sensitivity): Of all ACTUAL positives, how many did we catch?
# HIGH RECALL = Few missed cases
# Use when: False negatives are costly (e.g., cancer detection)
recall = recall_score(y_test, y_pred)

# F1 Score: Harmonic mean of precision and recall
# Use when: You need balance between precision and recall
f1 = f1_score(y_test, y_pred)

# All-in-one report
print(classification_report(y_test, y_pred))

# ═══════════════════════════════════════════════════════════════
# REGRESSION METRICS
# ═══════════════════════════════════════════════════════════════

# MSE (Mean Squared Error): Average squared difference
# Penalizes large errors more heavily
mse = mean_squared_error(y_test, y_pred)

# RMSE: Square root of MSE (same units as target)
rmse = np.sqrt(mse)

# MAE (Mean Absolute Error): Average absolute difference
# More robust to outliers
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)

# R² Score: Proportion of variance explained (1.0 = perfect)
r2 = r2_score(y_test, y_pred)
```

### Step 5: Model Deployment

```python
import joblib

# Save the model
joblib.dump(model, 'trained_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load and use in production
loaded_model = joblib.load('trained_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

def predict_new_data(raw_features):
    # Preprocess
    scaled_features = loaded_scaler.transform([raw_features])
    # Predict
    prediction = loaded_model.predict(scaled_features)
    return prediction[0]
```

---

## 1.4 Fundamental Concepts Every ML Engineer MUST Know

### The Bias-Variance Tradeoff (THE Most Important Concept)

```
High Bias                                            High Variance
(Underfitting)                                       (Overfitting)
     │                                                    │
     │  Model is too simple                 Model is too complex
     │  Misses the pattern                  Memorizes noise
     │                                                    │
     │        Training Error: HIGH          Training Error: LOW
     │        Test Error: HIGH              Test Error: HIGH
     │                                                    │
     ▼                                                    ▼
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│    ───────                    ~~~~~~~~                       │
│      │                        │      │                       │
│      └─ Straight line         └──────┴─ Wiggly line          │
│         through data            through data                 │
│         (misses curve)          (follows noise)              │
│                                                              │
└──────────────────────────────────────────────────────────────┘

                        SWEET SPOT
                           │
                           │  Good fit
                           │  Captures pattern
                           │  Ignores noise
                           │
                           ▼
                      ~~~~~~~~~~~~
                       (Natural curve)
```

**How to diagnose:**
```python
# Training accuracy much higher than test accuracy? = OVERFITTING
train_acc = model.score(X_train, y_train)  # 99%
test_acc = model.score(X_test, y_test)      # 70%
# ^ BIG GAP = Overfitting!

# Both accuracies low? = UNDERFITTING
train_acc = model.score(X_train, y_train)  # 60%
test_acc = model.score(X_test, y_test)      # 58%
# ^ Both low = Underfitting!
```

**Solutions:**

| Problem | Solutions |
|---------|-----------|
| **Overfitting** | More data, Less complex model, Regularization, Dropout, Early stopping |
| **Underfitting** | More features, More complex model, Less regularization, Longer training |

### Cross-Validation (Don't Trust a Single Train/Test Split!)

```python
from sklearn.model_selection import cross_val_score, KFold

# Simple 5-fold cross-validation
"""
Data gets split 5 times, each time a different 20% is used for testing:

Fold 1: [TEST] [TRAIN] [TRAIN] [TRAIN] [TRAIN]
Fold 2: [TRAIN] [TEST] [TRAIN] [TRAIN] [TRAIN]
Fold 3: [TRAIN] [TRAIN] [TEST] [TRAIN] [TRAIN]
Fold 4: [TRAIN] [TRAIN] [TRAIN] [TEST] [TRAIN]
Fold 5: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [TEST]
"""

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.2%} (+/- {scores.std()*2:.2%})")

# Stratified K-Fold (maintains class distribution)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate...
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search: Try every combination (exhaustive but slow)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1  # Use all CPU cores
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.2%}")
best_model = grid_search.best_estimator_

# Random Search: Sample random combinations (faster for large spaces)
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions,
    n_iter=100,  # Number of random combinations to try
    cv=5,
    random_state=42
)
random_search.fit(X_train, y_train)
```

---

## 1.5 Mathematics Foundations (Just What You Need)

### Linear Algebra Essentials

```python
import numpy as np

# Vectors: 1D array of numbers
features = np.array([1.2, 3.4, 5.6])  # e.g., [age, income, score]

# Matrices: 2D array (rows = samples, columns = features)
X = np.array([
    [1.2, 3.4, 5.6],
    [2.1, 4.3, 6.5],
    [3.0, 5.2, 7.4]
])

# Dot Product: Core of neural networks!
# Each neuron computes: weights · inputs + bias
weights = np.array([0.5, -0.2, 0.8])
bias = 0.1
output = np.dot(features, weights) + bias

# Matrix Multiplication: Process all samples at once
W = np.array([[0.5, 0.3], [-0.2, 0.4], [0.8, -0.1]])
all_outputs = np.dot(X, W)  # Shape: (3 samples, 2 outputs)
```

### Calculus (Gradient Descent - How Models Learn)

```python
"""
GRADIENT DESCENT: How models minimize error

1. Start with random weights
2. Calculate error (loss)
3. Calculate how error changes with each weight (gradient)
4. Adjust weights in opposite direction of gradient
5. Repeat until error is minimized

It's like rolling a ball downhill - it naturally finds the lowest point
"""

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    # Initialize weights randomly
    weights = np.random.randn(X.shape[1])
    bias = 0
    
    for i in range(iterations):
        # Forward pass: make predictions
        predictions = np.dot(X, weights) + bias
        
        # Calculate error (Mean Squared Error)
        error = predictions - y
        mse = np.mean(error ** 2)
        
        # Calculate gradients (derivatives)
        # How much does the error change if we tweak each weight?
        weight_gradients = (2/len(y)) * np.dot(X.T, error)
        bias_gradient = (2/len(y)) * np.sum(error)
        
        # Update weights (move opposite to gradient)
        weights -= learning_rate * weight_gradients
        bias -= learning_rate * bias_gradient
        
        if i % 100 == 0:
            print(f"Iteration {i}, MSE: {mse:.4f}")
    
    return weights, bias
```

### Probability & Statistics

```python
import numpy as np
from scipy import stats

# ═══════════════════════════════════════════════════════════════
# PROBABILITY BASICS
# ═══════════════════════════════════════════════════════════════

# P(A) = Probability of event A
# P(A|B) = Probability of A given B (conditional probability)
# P(A and B) = P(A) * P(B|A)

# Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)
# "Update your beliefs based on new evidence"

# Example: Spam detection
# P(Spam|"free money") = P("free money"|Spam) * P(Spam) / P("free money")

# ═══════════════════════════════════════════════════════════════
# DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════

# Normal (Gaussian) Distribution - The bell curve
# Most common; many ML algorithms assume this
data = np.random.normal(loc=0, scale=1, size=1000)  # mean=0, std=1

# Bernoulli: Binary outcomes (coin flip, yes/no)
# Binomial: Number of successes in n trials
# Poisson: Count of events in fixed time/space

# ═══════════════════════════════════════════════════════════════
# KEY STATISTICS
# ═══════════════════════════════════════════════════════════════

data = [23, 45, 12, 67, 34, 89, 23, 45, 67, 12]

# Central tendency
mean = np.mean(data)      # 41.7 - average
median = np.median(data)  # 39.5 - middle value (robust to outliers)
mode = stats.mode(data)   # 23 - most frequent

# Spread
std = np.std(data)        # 25.7 - how spread out
var = np.var(data)        # 662.4 - squared spread
range_val = max(data) - min(data)  # 77

# Correlation: How related are two variables?
# +1 = perfect positive, 0 = no relation, -1 = perfect negative
correlation = np.corrcoef(x, y)[0, 1]
```

---

# UNIT 2: SUPERVISED LEARNING

## 2.1 Linear Models (The Foundation)

### Linear Regression: Predicting Continuous Values

```python
"""
LINEAR REGRESSION: Fit a line through your data

The model: y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

Where:
- y = prediction (house price)
- x = features (bedrooms, sqft, etc.)
- w = weights (learned from data)
- w₀ = bias/intercept

Goal: Find weights that minimize Mean Squared Error
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt

# Simple example: Predict house price from size
house_sizes = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)
house_prices = np.array([150000, 200000, 280000, 350000, 400000])

# Train model
model = LinearRegression()
model.fit(house_sizes, house_prices)

# Interpret results
print(f"Intercept (base price): ${model.intercept_:,.0f}")
print(f"Coefficient (price per sqft): ${model.coef_[0]:,.0f}")
# Interpretation: Each additional sqft adds $X to the price

# Predict
new_house = np.array([[1800]])
predicted_price = model.predict(new_house)
print(f"1800 sqft house: ${predicted_price[0]:,.0f}")

# ═══════════════════════════════════════════════════════════════
# REGULARIZATION: Prevent overfitting
# ═══════════════════════════════════════════════════════════════

"""
Regularization adds a penalty for large weights, preventing the model
from becoming too complex.

Ridge (L2): Penalty = α * Σ(weights²)
- Shrinks weights toward zero but rarely to exactly zero
- Good when you have many features that might be useful

Lasso (L1): Penalty = α * Σ|weights|
- Can shrink weights to exactly zero (feature selection!)
- Good when you suspect many features are useless

ElasticNet: Combines both L1 and L2
"""

# Ridge Regression
ridge = Ridge(alpha=1.0)  # Higher alpha = more regularization
ridge.fit(X_train, y_train)

# Lasso Regression
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

# See which features Lasso kept (non-zero weights)
important_features = np.where(lasso.coef_ != 0)[0]
print(f"Lasso kept {len(important_features)} of {len(lasso.coef_)} features")

# ElasticNet
from sklearn.linear_model import ElasticNet
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)  # 50% L1, 50% L2
elastic.fit(X_train, y_train)
```

### Logistic Regression: Classification (Despite the Name!)

```python
"""
LOGISTIC REGRESSION: For binary classification

Despite the name, this is for CLASSIFICATION, not regression!
It outputs probabilities between 0 and 1.

The model: P(y=1) = sigmoid(w₀ + w₁x₁ + w₂x₂ + ...)

Sigmoid function: σ(z) = 1 / (1 + e^(-z))
- Takes any number, outputs between 0 and 1
- z = 0 → probability = 0.5
- z → +∞ → probability → 1
- z → -∞ → probability → 0
"""

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict class
y_pred = model.predict(X_test)

# Get probabilities (more useful!)
y_proba = model.predict_proba(X_test)
print(f"Sample probabilities: {y_proba[0]}")
# Output: [0.02, 0.98] = 2% benign, 98% malignant

# Adjust threshold for predictions
"""
Default: if probability > 0.5, predict positive
But you can adjust this based on your needs!

Cancer detection: Lower threshold (catch more positives, accept more false alarms)
Spam filter: Higher threshold (don't accidentally block important emails)
"""
threshold = 0.3
y_pred_custom = (y_proba[:, 1] >= threshold).astype(int)

# Feature importance: Which features matter most?
importance = pd.DataFrame({
    'feature': data.feature_names,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)
print(importance.head(10))
```

### Multiclass Classification

```python
"""
MULTICLASS: More than 2 classes

Strategies:
1. One-vs-Rest (OvR): Train K binary classifiers, each predicting "class k vs all others"
2. One-vs-One (OvO): Train K(K-1)/2 classifiers, each predicting "class i vs class j"
3. Softmax (multinomial): Single model, outputs probability for each class
"""

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load multiclass data (3 flower species)
iris = load_iris()
X, y = iris.data, iris.target

# One-vs-Rest
ovr_model = LogisticRegression(multi_class='ovr')
ovr_model.fit(X, y)

# Softmax (multinomial) - Generally better
softmax_model = LogisticRegression(multi_class='multinomial')
softmax_model.fit(X, y)

# Get probabilities for all classes
proba = softmax_model.predict_proba(X[:1])
print(f"Probabilities: {proba[0]}")  # [0.90, 0.08, 0.02] for 3 classes
```

---

## 2.2 Tree-Based Methods (Powerful and Interpretable)

### Decision Trees

```python
"""
DECISION TREE: Make decisions by asking yes/no questions

Like playing 20 questions:
- Is the animal bigger than a breadbox? YES
- Does it have fur? YES
- Does it bark? YES
- It's a dog!

The tree learns which questions to ask and in what order.
"""

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Train decision tree
dt = DecisionTreeClassifier(
    max_depth=5,           # Limit tree depth (prevent overfitting)
    min_samples_split=10,  # Min samples to split a node
    min_samples_leaf=5,    # Min samples in leaf nodes
    random_state=42
)
dt.fit(X_train, y_train)

# Visualize the tree
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=feature_names, class_names=class_names, filled=True)
plt.savefig('decision_tree.png')

# Feature importance: Which features does the tree use most?
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)
print(importance)

# Make predictions with explanations
"""
For a specific prediction, you can trace the path through the tree:
1. If income > 50000: go left
2. If age > 30: go right
3. Predict: "Approved"
"""
```

### Random Forest (Ensemble of Trees)

```python
"""
RANDOM FOREST: Build many trees, let them vote

Why it works:
1. Each tree sees different random subsets of data (bagging)
2. Each tree considers different random subsets of features
3. Trees make different errors that cancel out when averaging

Result: More accurate and stable than single tree
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Classification
rf_clf = RandomForestClassifier(
    n_estimators=100,        # Number of trees (more = better, but slower)
    max_depth=10,            # Depth of each tree
    min_samples_split=5,     # Min samples to split
    max_features='sqrt',     # Features to consider at each split (sqrt is good default)
    bootstrap=True,          # Sample with replacement
    oob_score=True,          # Out-of-bag accuracy (free validation!)
    n_jobs=-1,               # Use all CPU cores
    random_state=42
)
rf_clf.fit(X_train, y_train)

# Out-of-bag score: Validation without using a separate test set
print(f"OOB Score: {rf_clf.oob_score_:.2%}")

# Feature importance
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance['feature'][:20], importance['importance'][:20])
plt.xlabel('Importance')
plt.title('Top 20 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Regression
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
```

### Gradient Boosting (The Kaggle Winner)

```python
"""
GRADIENT BOOSTING: Build trees sequentially, each fixing the previous errors

Unlike Random Forest (parallel trees, average predictions):
- Trees are built one at a time
- Each tree focuses on the samples the previous trees got wrong
- Final prediction is weighted sum of all trees

Think of it like a relay team:
- First runner (tree) does their best
- Second runner picks up where they left off
- Each runner improves on the previous performance

As of 2025: Best algorithm for tabular data, wins most Kaggle competitions
"""

from sklearn.ensemble import GradientBoostingClassifier

# Scikit-learn implementation (good for learning, slow for production)
gb = GradientBoostingClassifier(
    n_estimators=100,        # Number of boosting stages
    learning_rate=0.1,       # Shrinkage (lower = more regularization)
    max_depth=3,             # Depth of each tree (usually keep shallow!)
    min_samples_split=5,
    subsample=0.8,           # Fraction of samples for each tree (adds randomness)
    random_state=42
)
gb.fit(X_train, y_train)

# ═══════════════════════════════════════════════════════════════
# XGBoost: The Industry Standard
# ═══════════════════════════════════════════════════════════════

import xgboost as xgb

# Create DMatrix (XGBoost's optimized data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',  # or 'multi:softmax' for multiclass
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,           # Row sampling
    'colsample_bytree': 0.8,    # Column sampling
    'eval_metric': 'auc',
    'use_label_encoder': False
}

# Train with early stopping
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
    verbose_eval=100
)

# Sklearn API (easier interface)
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# ═══════════════════════════════════════════════════════════════
# LightGBM: Faster and Often Better
# ═══════════════════════════════════════════════════════════════

import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',    # or 'dart', 'goss'
    'num_leaves': 31,           # Max leaves in one tree
    'learning_rate': 0.05,
    'feature_fraction': 0.9,    # Like colsample_bytree
    'bagging_fraction': 0.8,    # Like subsample
    'bagging_freq': 5,
    'verbose': -1
}

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_test],
    callbacks=[lgb.early_stopping(50)]
)

# Sklearn API
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42
)
lgb_model.fit(X_train, y_train)

# ═══════════════════════════════════════════════════════════════
# CatBoost: Best for Categorical Features
# ═══════════════════════════════════════════════════════════════

from catboost import CatBoostClassifier

# CatBoost handles categorical features automatically!
cat_features = ['color', 'size', 'category']  # Indices or names

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    cat_features=cat_features,  # Tell it which columns are categorical
    verbose=100
)
cat_model.fit(X_train, y_train, eval_set=(X_test, y_test))
```

---

## 2.3 Support Vector Machines (SVM)

```python
"""
SUPPORT VECTOR MACHINE: Find the best line (hyperplane) to separate classes

Key ideas:
1. Find the line that maximizes the "margin" (distance to nearest points)
2. The nearest points are called "support vectors"
3. Use the "kernel trick" to handle non-linear boundaries

When to use SVM:
- Medium-sized datasets (not great for huge data)
- High-dimensional data (like text)
- When you need a clear margin
"""

from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler

# CRITICAL: SVM needs scaled data!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear SVM (for linearly separable data)
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)

# RBF (Radial Basis Function) kernel - Most common for non-linear
svm_rbf = SVC(
    kernel='rbf',
    C=1.0,              # Regularization (lower = simpler model)
    gamma='scale'       # Kernel coefficient (higher = more complex)
)
svm_rbf.fit(X_train_scaled, y_train)

# Polynomial kernel
svm_poly = SVC(kernel='poly', degree=3)

# Get probabilities (slower but useful)
svm_proba = SVC(kernel='rbf', probability=True)
svm_proba.fit(X_train_scaled, y_train)
proba = svm_proba.predict_proba(X_test_scaled)

# Support Vector Regression (for continuous targets)
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_train_scaled, y_train)
```

---

## 2.4 K-Nearest Neighbors (KNN)

```python
"""
K-NEAREST NEIGHBORS: Classify based on what your neighbors are

"You are the average of the 5 people you spend the most time with"

How it works:
1. Find the K closest training points to your new point
2. Take a vote (classification) or average (regression)
3. That's your prediction!

Pros: Simple, no training, works with any distance metric
Cons: Slow at prediction time (must compute all distances)
"""

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# CRITICAL: KNN needs scaled data!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classification
knn = KNeighborsClassifier(
    n_neighbors=5,          # K - how many neighbors to consider
    weights='uniform',      # 'uniform' or 'distance' (closer = more weight)
    metric='minkowski',     # Distance metric
    p=2                     # p=2 is Euclidean, p=1 is Manhattan
)
knn.fit(X_train_scaled, y_train)

# Finding optimal K
from sklearn.model_selection import cross_val_score

k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_temp, X_train_scaled, y_train, cv=5)
    cv_scores.append(scores.mean())

# Plot to find best K
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores)
plt.xlabel('K')
plt.ylabel('Cross-validation accuracy')
plt.title('KNN: Finding Optimal K')
plt.show()

best_k = k_values[np.argmax(cv_scores)]
print(f"Best K: {best_k}")
```

---

## 2.5 Naive Bayes (Fast and Effective for Text)

```python
"""
NAIVE BAYES: Apply Bayes' Theorem with "naive" independence assumption

P(class|features) ∝ P(class) × P(features|class)

"Naive" because it assumes features are independent given the class
(This is almost never true, but it works surprisingly well!)

Best for:
- Text classification (spam, sentiment, topic)
- When you have limited training data
- When you need fast training and prediction
"""

from sklearn.naive_bayes import (
    MultinomialNB,      # For count data (text with word counts)
    GaussianNB,         # For continuous data
    BernoulliNB         # For binary features
)

# Text Classification Example
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample text data
texts = [
    "free money lottery winner",
    "meeting tomorrow at 3pm",
    "claim your prize now",
    "project deadline next week"
]
labels = [1, 0, 1, 0]  # 1=spam, 0=not spam

# Convert text to features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)

# Train Naive Bayes
nb = MultinomialNB(alpha=1.0)  # alpha is smoothing parameter
nb.fit(X, labels)

# Predict new text
new_text = ["claim free prize money now"]
X_new = vectorizer.transform(new_text)
prediction = nb.predict(X_new)
probability = nb.predict_proba(X_new)
print(f"Spam probability: {probability[0][1]:.2%}")

# See which words indicate spam
feature_names = vectorizer.get_feature_names_out()
spam_word_scores = dict(zip(feature_names, nb.feature_log_prob_[1]))
sorted_spam_words = sorted(spam_word_scores.items(), key=lambda x: x[1], reverse=True)
print("Top spam words:", sorted_spam_words[:10])
```

---

# UNIT 3: NEURAL NETWORKS

## 3.1 The Neuron (Building Block of Deep Learning)

```python
"""
ARTIFICIAL NEURON: Inspired by (but not identical to) biological neurons

What it does:
1. Takes weighted sum of inputs
2. Adds a bias
3. Applies an activation function

Mathematically:
    output = activation(Σ(weight_i × input_i) + bias)
           = activation(w·x + b)
"""

import numpy as np

def neuron(inputs, weights, bias, activation='relu'):
    """A single artificial neuron"""
    # Step 1: Weighted sum
    z = np.dot(inputs, weights) + bias
    
    # Step 2: Activation function
    if activation == 'sigmoid':
        return 1 / (1 + np.exp(-z))
    elif activation == 'tanh':
        return np.tanh(z)
    elif activation == 'relu':
        return np.maximum(0, z)
    elif activation == 'linear':
        return z

# Example
inputs = np.array([0.5, 0.3, 0.2])  # Features
weights = np.array([0.4, 0.6, 0.2])  # Learned parameters
bias = 0.1

output = neuron(inputs, weights, bias, 'relu')
print(f"Neuron output: {output}")
```

### Activation Functions (Why They Matter)

```python
"""
ACTIVATION FUNCTIONS: Add non-linearity so networks can learn complex patterns

Without activation functions, a neural network is just linear regression!
"""

import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# SIGMOID: Outputs between 0 and 1
# ═══════════════════════════════════════════════════════════════
"""
σ(x) = 1 / (1 + e^(-x))

Pros: Good for probability outputs
Cons: Vanishing gradient problem, outputs not centered at 0
Use for: Binary classification output layer
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ═══════════════════════════════════════════════════════════════
# TANH: Outputs between -1 and 1
# ═══════════════════════════════════════════════════════════════
"""
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

Pros: Zero-centered (better than sigmoid)
Cons: Still has vanishing gradient
Use for: Hidden layers (historically, now mostly replaced by ReLU)
"""
def tanh(x):
    return np.tanh(x)

# ═══════════════════════════════════════════════════════════════
# RELU: The Modern Default
# ═══════════════════════════════════════════════════════════════
"""
ReLU(x) = max(0, x)

Pros: Fast, avoids vanishing gradient for positive values
Cons: "Dead neurons" (if always negative, gradient is 0)
Use for: Most hidden layers (default choice!)
"""
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU: Fixes dead neuron problem
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# ═══════════════════════════════════════════════════════════════
# SOFTMAX: For multi-class output
# ═══════════════════════════════════════════════════════════════
"""
softmax(x_i) = e^(x_i) / Σ(e^(x_j))

Outputs: Probabilities that sum to 1
Use for: Multi-class classification output layer
"""
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

# Example: Multi-class output
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Softmax: {probs}")  # [0.66, 0.24, 0.10]
print(f"Sum: {probs.sum()}")  # 1.0
```

---

## 3.2 Building Neural Networks with PyTorch

```python
"""
PYTORCH: The most popular deep learning framework (2025)

Why PyTorch over TensorFlow?
- More Pythonic and intuitive
- Better debugging (eager execution)
- Preferred by researchers
- Now has great production support too
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ═══════════════════════════════════════════════════════════════
# SIMPLE NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        # Define layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Forward pass: how data flows through the network
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        return x

# Create model
model = SimpleNN(input_size=10, hidden_size=64, output_size=2)

# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

# Prepare data
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Set to training mode
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

model.eval()  # Set to evaluation mode
with torch.no_grad():  # Don't compute gradients during evaluation
    X_test_tensor = torch.FloatTensor(X_test)
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted.numpy() == y_test).mean()
    print(f'Test Accuracy: {accuracy:.2%}')

# ═══════════════════════════════════════════════════════════════
# SAVE AND LOAD MODEL
# ═══════════════════════════════════════════════════════════════

# Save
torch.save(model.state_dict(), 'model.pth')

# Load
loaded_model = SimpleNN(input_size=10, hidden_size=64, output_size=2)
loaded_model.load_state_dict(torch.load('model.pth'))
loaded_model.eval()
```

---

## 3.3 Convolutional Neural Networks (CNNs) - Computer Vision

```python
"""
CONVOLUTIONAL NEURAL NETWORK: Designed for image data

Why CNNs for images?
1. Traditional NN: 256×256 image = 65,536 inputs × hidden neurons = EXPLOSION of parameters
2. CNN: Uses small filters that slide across the image = WAY fewer parameters

Key components:
- Convolution layers: Detect local patterns (edges, textures)
- Pooling layers: Reduce spatial dimensions
- Fully connected layers: Final classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # Convolutional layers
        # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)   # 3 channels (RGB) -> 32 filters
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)  # 32 -> 64 filters
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1) # 64 -> 128 filters
        
        # Batch normalization (stabilizes training)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layer (reduce spatial dimensions)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling, stride 2
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 3 pooling layers: 224 -> 112 -> 56 -> 28 (assuming 224x224 input)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Input: (batch_size, 3, 224, 224)
        
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> (batch, 32, 112, 112)
        
        # Second conv block  
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> (batch, 64, 56, 56)
        
        # Third conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> (batch, 128, 28, 28)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # -> (batch, 128*28*28)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# ═══════════════════════════════════════════════════════════════
# TRANSFER LEARNING: Use pre-trained models (THE SMART WAY!)
# ═══════════════════════════════════════════════════════════════

"""
Why train from scratch when someone already did the hard work?

ImageNet pretrained models have already learned:
- Edge detection
- Texture patterns
- Object parts
- Abstract concepts

Just replace the final layer for your specific task!
"""

import torchvision.models as models

# Load pretrained ResNet (trained on ImageNet)
resnet = models.resnet50(pretrained=True)

# Freeze all layers (don't update during training)
for param in resnet.parameters():
    param.requires_grad = False

# Replace the final layer for your task
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, num_classes)

# Only train the new final layer
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)

# Or fine-tune the whole model with lower learning rate
for param in resnet.parameters():
    param.requires_grad = True
optimizer = optim.Adam(resnet.parameters(), lr=0.0001)  # Lower LR for fine-tuning

# ═══════════════════════════════════════════════════════════════
# DATA AUGMENTATION: Artificially expand your dataset
# ═══════════════════════════════════════════════════════════════

from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## 3.4 Recurrent Neural Networks (RNNs) - Sequential Data

```python
"""
RECURRENT NEURAL NETWORK: Designed for sequential data (text, time series, audio)

Why RNNs for sequences?
- Traditional NN sees inputs independently
- RNN maintains a "memory" (hidden state) across time steps

The key insight: Output depends on current input AND previous hidden state
    h_t = f(W_h * h_{t-1} + W_x * x_t + b)
"""

import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════
# VANILLA RNN (Rarely used - just for understanding)
# ═══════════════════════════════════════════════════════════════

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # RNN forward pass
        out, hidden = self.rnn(x, h0)  # out: (batch, seq_len, hidden)
        
        # Use last time step's output
        out = self.fc(out[:, -1, :])  # (batch, output_size)
        return out

# ═══════════════════════════════════════════════════════════════
# LSTM: Long Short-Term Memory (The Standard)
# ═══════════════════════════════════════════════════════════════

"""
LSTM solves the vanishing gradient problem with gates:
1. Forget gate: What to throw away from cell state
2. Input gate: What new information to add
3. Output gate: What to output

This allows information to flow unchanged through time when needed
"""

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 num_layers=2, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        
        # Embedding layer: Convert word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer(s)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Process sequence in both directions
        )
        
        # Output layer (×2 for bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, seq_len) - word indices
        
        # Embed words: (batch, seq_len, embedding_dim)
        embedded = self.dropout(self.embedding(x))
        
        # LSTM: (batch, seq_len, hidden_dim * 2)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate final hidden states from both directions
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Output
        output = self.fc(self.dropout(hidden))
        return output

# ═══════════════════════════════════════════════════════════════
# GRU: Gated Recurrent Unit (Simpler than LSTM)
# ═══════════════════════════════════════════════════════════════

"""
GRU is a simplified version of LSTM:
- Combines forget and input gates
- No separate cell state
- Fewer parameters, often comparable performance
"""

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden)
```

---

## 3.5 The Transformer Architecture (THE Revolution)

```python
"""
TRANSFORMER: The architecture behind GPT, BERT, and all modern AI

Key innovation: ATTENTION - "Look at everything at once"

Unlike RNNs that process sequentially:
- Transformers process all positions in parallel
- Each position can attend to all other positions
- This captures long-range dependencies much better

The paper: "Attention Is All You Need" (2017)
"""

import torch
import torch.nn as nn
import math

# ═══════════════════════════════════════════════════════════════
# SELF-ATTENTION: The Core Mechanism
# ═══════════════════════════════════════════════════════════════

"""
Self-attention computes:
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q (Query): "What am I looking for?"
- K (Key): "What do I contain?"
- V (Value): "What information do I provide?"

Each position creates Q, K, V by transforming its input
Then attention scores determine how much each position
attends to every other position.
"""

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"
        
        # Linear layers for Q, K, V
        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        
        # Final output projection
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, query, key, value, mask=None):
        N = query.shape[0]  # Batch size
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]
        
        # Apply linear transformations
        Q = self.queries(query)
        K = self.keys(key)
        V = self.values(value)
        
        # Split into multiple heads
        Q = Q.view(N, query_len, self.heads, self.head_dim)
        K = K.view(N, key_len, self.heads, self.head_dim)
        V = V.view(N, value_len, self.heads, self.head_dim)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # Using einsum for clarity (can also use matmul)
        scores = torch.einsum("nqhd,nkhd->nhqk", [Q, K])
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # Softmax to get attention weights
        attention = torch.softmax(scores / math.sqrt(self.head_dim), dim=-1)
        
        # Apply attention to values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, V])
        
        # Concatenate heads
        out = out.reshape(N, query_len, self.embed_size)
        
        return self.fc_out(out)

# ═══════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK
# ═══════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention
        self.attention = SelfAttention(embed_size, heads)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),  # Modern activation
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

# ═══════════════════════════════════════════════════════════════
# POSITIONAL ENCODING: Add position information
# ═══════════════════════════════════════════════════════════════

"""
Unlike RNNs, Transformers have no notion of position.
We must explicitly add position information.

The original paper uses sine/cosine functions:
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
"""

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * 
                           -(math.log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ═══════════════════════════════════════════════════════════════
# USING PRETRAINED TRANSFORMERS (THE PRACTICAL WAY!)
# ═══════════════════════════════════════════════════════════════

"""
In practice, use HuggingFace Transformers library!
Never train from scratch unless you have Google-level resources.
"""

from transformers import (
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)

# Load pretrained BERT
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # Binary classification
)

# Tokenize text
text = "This movie was absolutely fantastic!"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Get prediction
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(f"Positive: {predictions[0][1]:.2%}")
```

---

## 3.6 Advanced Architectures (2024-2025)

### Modern CNN Architectures

```python
"""
EVOLUTION OF CNN ARCHITECTURES:

2012: AlexNet - Deep CNNs work!
2014: VGGNet - Smaller filters, deeper networks
2014: GoogLeNet - Inception modules (parallel paths)
2015: ResNet - Residual connections (game changer!)
2016: DenseNet - Dense connections
2019: EfficientNet - Neural architecture search
2020+: Vision Transformers (ViT) - Transformers for images

Key insight from ResNet: Skip connections allow training very deep networks
"""

# Using modern architectures from torchvision
from torchvision import models

# ResNet - The workhorse
resnet50 = models.resnet50(pretrained=True)

# EfficientNet - Best accuracy/efficiency tradeoff
efficientnet = models.efficientnet_b0(pretrained=True)

# Vision Transformer - Transformers for images
from transformers import ViTForImageClassification
vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
```

### State-of-the-Art Language Models (2025)

```python
"""
2025 STATE OF LANGUAGE MODELS:

The key developments:
1. Reasoning models (o1, R1) - Chain-of-thought at inference
2. Mixture of Experts (MoE) - Sparse activation for efficiency
3. RLHF/RLVR - Alignment through reinforcement learning
4. Long context - 128K+ tokens
5. Multimodal - Text, images, audio, video in one model
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

# Using a state-of-the-art model
model_name = "mistralai/Mistral-7B-v0.1"  # Example open model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
print(tokenizer.decode(outputs[0]))
```

---

## 3.7 Training Tips & Best Practices

```python
"""
NEURAL NETWORK TRAINING: THE PRACTICAL GUIDE
"""

# ═══════════════════════════════════════════════════════════════
# LEARNING RATE: The Most Important Hyperparameter
# ═══════════════════════════════════════════════════════════════

"""
Too high: Training diverges (loss explodes)
Too low: Training is slow, may get stuck
Just right: Smooth decrease in loss

Rule of thumb: Start with 3e-4, adjust from there
"""

# Learning rate schedulers
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
)

# Reduce on plateau (most practical)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# In training loop:
# scheduler.step(val_loss)

# One-cycle policy (often best for training from scratch)
scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=num_epochs*len(train_loader))

# ═══════════════════════════════════════════════════════════════
# REGULARIZATION: Prevent Overfitting
# ═══════════════════════════════════════════════════════════════

# 1. Dropout: Randomly zero out neurons during training
self.dropout = nn.Dropout(0.5)

# 2. Weight decay (L2 regularization)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 3. Early stopping
best_loss = float('inf')
patience = 10
counter = 0

for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# 4. Batch normalization
self.bn = nn.BatchNorm1d(hidden_size)

# ═══════════════════════════════════════════════════════════════
# BATCH SIZE CONSIDERATIONS
# ═══════════════════════════════════════════════════════════════

"""
Larger batch:
- More stable gradients
- Better GPU utilization
- May generalize worse
- Needs higher learning rate

Smaller batch:
- More noise in gradients (can help generalization)
- Can escape local minima
- Slower training per epoch

Common sizes: 16, 32, 64, 128, 256
Start with 32, adjust based on GPU memory
"""

# Gradient accumulation: Simulate large batch on small GPU
accumulation_steps = 4
optimizer.zero_grad()

for i, (inputs, labels) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss = loss / accumulation_steps  # Normalize
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# ═══════════════════════════════════════════════════════════════
# MIXED PRECISION TRAINING: Speed up with minimal accuracy loss
# ═══════════════════════════════════════════════════════════════

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, labels in train_loader:
    optimizer.zero_grad()
    
    # Forward pass with automatic mixed precision
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

# UNIT 4: UNSUPERVISED LEARNING

## 4.1 Clustering Algorithms

### K-Means: The Classic

```python
"""
K-MEANS CLUSTERING: Partition data into K clusters

Algorithm:
1. Choose K random points as initial centroids
2. Assign each point to nearest centroid
3. Update centroids to mean of assigned points
4. Repeat 2-3 until convergence

Simple, fast, but requires knowing K in advance
"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Basic K-Means
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Get cluster centers
centers = kmeans.cluster_centers_

# Get distance to cluster center (for anomaly detection)
distances = kmeans.transform(X)  # Distance to each center

# ═══════════════════════════════════════════════════════════════
# FINDING OPTIMAL K: The Elbow Method
# ═══════════════════════════════════════════════════════════════

inertias = []  # Sum of squared distances to centroid
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Silhouette score: Measures cluster quality (-1 to 1, higher is better)
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.3f}")

best_k = range(2, 11)[np.argmax(silhouette_scores)]
print(f"Best K: {best_k}")
```

### DBSCAN: Density-Based Clustering

```python
"""
DBSCAN: Finds clusters based on density

Key parameters:
- eps: Maximum distance between neighbors
- min_samples: Minimum points to form a dense region

Advantages:
- Doesn't require specifying K
- Finds arbitrary-shaped clusters
- Identifies outliers (points in no cluster)
"""

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# CRITICAL: Scale your data first!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# Cluster labels: -1 means outlier/noise
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_outliers = (clusters == -1).sum()

print(f"Number of clusters: {n_clusters}")
print(f"Number of outliers: {n_outliers}")

# Finding optimal eps using k-distance graph
from sklearn.neighbors import NearestNeighbors

k = 5
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X_scaled)
distances, _ = neighbors.kneighbors(X_scaled)

# Sort distances to k-th neighbor
k_distances = np.sort(distances[:, k-1])

plt.figure(figsize=(10, 6))
plt.plot(k_distances)
plt.xlabel('Points sorted by distance')
plt.ylabel('Distance to 5th nearest neighbor')
plt.title('K-Distance Graph (look for "elbow")')
plt.show()
```

### Hierarchical Clustering

```python
"""
HIERARCHICAL CLUSTERING: Build a tree of clusters

Two approaches:
- Agglomerative (bottom-up): Start with each point as cluster, merge
- Divisive (top-down): Start with one cluster, split

Advantage: Creates dendrogram showing hierarchy
"""

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Agglomerative clustering
agg = AgglomerativeClustering(
    n_clusters=5,
    linkage='ward'  # Minimizes variance within clusters
)
clusters = agg.fit_predict(X)

# Create dendrogram
linkage_matrix = linkage(X, method='ward')

plt.figure(figsize=(15, 8))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Cut dendrogram at specific height
from scipy.cluster.hierarchy import fcluster
clusters = fcluster(linkage_matrix, t=10, criterion='distance')
```

---

## 4.2 Dimensionality Reduction

### PCA: Principal Component Analysis

```python
"""
PCA: Find the directions of maximum variance

What it does:
1. Center the data
2. Find orthogonal axes that capture the most variance
3. Project data onto these axes

Use cases:
- Visualization (reduce to 2-3 dimensions)
- Noise reduction
- Speed up other algorithms
- Feature extraction
"""

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Fit PCA
pca = PCA(n_components=2)  # or PCA(n_components=0.95) for 95% variance
X_pca = pca.fit_transform(X)

# Explained variance: How much information is preserved?
print(f"Explained variance per component: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Find number of components for desired variance
pca_full = PCA()
pca_full.fit(X)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")

# Visualize cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA: Choosing Number of Components')
plt.legend()
plt.show()

# Feature importance: Which original features matter most?
components_df = pd.DataFrame(
    pca.components_,
    columns=feature_names,
    index=[f'PC{i+1}' for i in range(pca.n_components_)]
)
print(components_df.abs().sum().sort_values(ascending=False).head(10))
```

### t-SNE: For Visualization

```python
"""
t-SNE: Visualize high-dimensional data in 2D/3D

How it works:
1. Compute pairwise similarities in high-dim space
2. Compute pairwise similarities in low-dim space
3. Minimize difference (KL divergence)

Key point: Only for visualization! Cannot be used for new data.

Important parameter: perplexity (typically 5-50)
- Low perplexity: Focus on local structure
- High perplexity: Focus on global structure
"""

from sklearn.manifold import TSNE

# Basic t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,      # Try different values!
    n_iter=1000,
    random_state=42,
    init='pca',         # More reproducible
    learning_rate='auto'
)
X_tsne = tsne.fit_transform(X)

# Visualize
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()

# Compare different perplexities
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
perplexities = [5, 30, 50, 100]

for ax, perp in zip(axes.flat, perplexities):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_transformed = tsne.fit_transform(X)
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=labels, cmap='tab10', alpha=0.6)
    ax.set_title(f'Perplexity = {perp}')

plt.tight_layout()
plt.show()
```

### UMAP: The Modern Choice

```python
"""
UMAP: Uniform Manifold Approximation and Projection

Better than t-SNE because:
1. Faster (scales better)
2. Preserves more global structure
3. Can be used for new data (has transform method!)
4. Works as dimensionality reduction, not just visualization

2025 recommendation: Use UMAP over t-SNE in most cases
"""

import umap

# Basic UMAP
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,      # Local vs global structure (higher = more global)
    min_dist=0.1,        # How tightly points cluster (lower = tighter)
    metric='euclidean',
    random_state=42
)
X_umap = reducer.fit_transform(X)

# Unlike t-SNE, UMAP can transform new data!
X_new_umap = reducer.transform(X_new)

# For clustering before UMAP
X_umap_clustering = umap.UMAP(n_components=10).fit_transform(X)  # Higher dim for clustering
from sklearn.cluster import HDBSCAN
clusterer = HDBSCAN(min_cluster_size=15)
clusters = clusterer.fit_predict(X_umap_clustering)

# Visualize
plt.figure(figsize=(12, 8))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='tab10', alpha=0.6, s=5)
plt.title('UMAP Visualization')
plt.show()
```

---

## 4.3 Anomaly Detection

```python
"""
ANOMALY DETECTION: Find unusual data points

Use cases:
- Fraud detection
- Network intrusion
- Manufacturing defects
- Medical diagnosis
"""

# ═══════════════════════════════════════════════════════════════
# ISOLATION FOREST: Random Forest for Anomalies
# ═══════════════════════════════════════════════════════════════

from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.1,  # Expected proportion of outliers
    random_state=42
)
predictions = iso_forest.fit_predict(X)
# -1 = anomaly, 1 = normal

# Get anomaly scores (lower = more anomalous)
scores = iso_forest.decision_function(X)

# ═══════════════════════════════════════════════════════════════
# LOCAL OUTLIER FACTOR (LOF)
# ═══════════════════════════════════════════════════════════════

from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.1
)
predictions = lof.fit_predict(X)
scores = lof.negative_outlier_factor_

# ═══════════════════════════════════════════════════════════════
# AUTOENCODERS: Deep Learning for Anomalies
# ═══════════════════════════════════════════════════════════════

"""
Autoencoder approach:
1. Train on normal data to compress and reconstruct
2. Anomalies will have high reconstruction error
"""

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Train on normal data
autoencoder = Autoencoder(input_dim=X.shape[1], encoding_dim=16)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters())

# Training loop...

# Detect anomalies by reconstruction error
autoencoder.eval()
with torch.no_grad():
    reconstructed = autoencoder(torch.FloatTensor(X))
    reconstruction_error = ((X - reconstructed.numpy()) ** 2).mean(axis=1)

threshold = np.percentile(reconstruction_error, 95)
anomalies = reconstruction_error > threshold
```

---

# UNIT 5: LANGUAGE AND TIME SERIES

## 5.1 Natural Language Processing (NLP)

### Text Preprocessing

```python
"""
NLP PREPROCESSING: Transform text into numbers

Pipeline:
1. Tokenization: Split text into words/subwords
2. Cleaning: Remove noise
3. Normalization: Lowercase, stemming, lemmatization
4. Vectorization: Convert to numbers
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatize (better than stemming)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)

# ═══════════════════════════════════════════════════════════════
# TEXT VECTORIZATION
# ═══════════════════════════════════════════════════════════════

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Bag of Words: Simple word counts
count_vec = CountVectorizer(max_features=10000)
X_counts = count_vec.fit_transform(texts)

# TF-IDF: Accounts for word importance
"""
TF-IDF = Term Frequency × Inverse Document Frequency
- High TF: Word appears often in this document
- High IDF: Word appears rarely across all documents
- Rare words in a document are often most informative
"""
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),     # Unigrams and bigrams
    min_df=5,               # Minimum document frequency
    max_df=0.9              # Maximum document frequency
)
X_tfidf = tfidf.fit_transform(texts)

# ═══════════════════════════════════════════════════════════════
# WORD EMBEDDINGS: Dense vector representations
# ═══════════════════════════════════════════════════════════════

"""
Word embeddings capture semantic meaning:
- Similar words have similar vectors
- king - man + woman ≈ queen

Options:
1. Word2Vec: Google, 2013
2. GloVe: Stanford, 2014
3. FastText: Facebook, 2016 (handles subwords)
4. Pretrained transformer embeddings: 2018+ (best!)
"""

from gensim.models import Word2Vec

# Train Word2Vec
sentences = [text.split() for text in texts]
w2v = Word2Vec(
    sentences,
    vector_size=100,    # Embedding dimension
    window=5,           # Context window
    min_count=5,        # Minimum word frequency
    workers=4           # Parallel training
)

# Get word vector
vector = w2v.wv['machine']

# Find similar words
similar = w2v.wv.most_similar('machine', topn=5)

# Using pretrained embeddings (recommended!)
import gensim.downloader as api
glove = api.load('glove-wiki-gigaword-100')
vector = glove['computer']
```

### Modern NLP with Transformers

```python
"""
MODERN NLP: Use pretrained transformers for everything!

The HuggingFace library makes this easy.
"""

from transformers import pipeline

# ═══════════════════════════════════════════════════════════════
# SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════

sentiment_pipeline = pipeline("sentiment-analysis")
result = sentiment_pipeline("I absolutely love this product!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# ═══════════════════════════════════════════════════════════════
# TEXT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# Load pretrained model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3  # Number of classes
)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

trainer.train()

# ═══════════════════════════════════════════════════════════════
# NAMED ENTITY RECOGNITION (NER)
# ═══════════════════════════════════════════════════════════════

ner_pipeline = pipeline("ner", aggregation_strategy="simple")
text = "Apple CEO Tim Cook announced new products in California"
entities = ner_pipeline(text)
# [{'entity_group': 'ORG', 'word': 'Apple', ...},
#  {'entity_group': 'PER', 'word': 'Tim Cook', ...},
#  {'entity_group': 'LOC', 'word': 'California', ...}]

# ═══════════════════════════════════════════════════════════════
# QUESTION ANSWERING
# ═══════════════════════════════════════════════════════════════

qa_pipeline = pipeline("question-answering")
context = "The Eiffel Tower is located in Paris, France. It was built in 1889."
question = "Where is the Eiffel Tower?"
result = qa_pipeline(question=question, context=context)
print(result)  # {'answer': 'Paris, France', 'score': 0.95, ...}

# ═══════════════════════════════════════════════════════════════
# TEXT GENERATION
# ═══════════════════════════════════════════════════════════════

generator = pipeline("text-generation", model="gpt2")
result = generator("The future of AI is", max_length=50, num_return_sequences=1)
print(result[0]['generated_text'])

# ═══════════════════════════════════════════════════════════════
# SUMMARIZATION
# ═══════════════════════════════════════════════════════════════

summarizer = pipeline("summarization")
long_text = "..." # Your long article
summary = summarizer(long_text, max_length=150, min_length=50)
print(summary[0]['summary_text'])

# ═══════════════════════════════════════════════════════════════
# SEMANTIC SIMILARITY
# ═══════════════════════════════════════════════════════════════

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences1 = ["The cat sits outside"]
sentences2 = ["A feline is outdoors"]

embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

similarity = util.cos_sim(embeddings1, embeddings2)
print(f"Similarity: {similarity.item():.4f}")  # High similarity!
```

---

## 5.2 Time Series Analysis

### Classical Methods

```python
"""
TIME SERIES: Data points indexed by time

Components:
1. Trend: Long-term increase/decrease
2. Seasonality: Regular periodic patterns
3. Noise: Random variations
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Load and prepare time series data
df = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')
ts = df['value']

# ═══════════════════════════════════════════════════════════════
# DECOMPOSITION
# ═══════════════════════════════════════════════════════════════

decomposition = seasonal_decompose(ts, model='multiplicative', period=12)

fig, axes = plt.subplots(4, 1, figsize=(15, 10))
decomposition.observed.plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════
# STATIONARITY CHECK
# ═══════════════════════════════════════════════════════════════

"""
Many methods require stationary data (constant mean, variance over time)
"""

def check_stationarity(ts):
    result = adfuller(ts.dropna())
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    if result[1] < 0.05:
        print("Series is stationary (reject null hypothesis)")
    else:
        print("Series is non-stationary (cannot reject null hypothesis)")
        print("Consider differencing the series")

check_stationarity(ts)

# Make series stationary by differencing
ts_diff = ts.diff().dropna()

# ═══════════════════════════════════════════════════════════════
# ARIMA: AutoRegressive Integrated Moving Average
# ═══════════════════════════════════════════════════════════════

"""
ARIMA(p, d, q):
- p: Order of autoregressive component (how many past values)
- d: Degree of differencing (how many times to difference)
- q: Order of moving average component (how many past errors)
"""

from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima  # Automatically finds best parameters

# Auto ARIMA: Finds best parameters
auto_model = auto_arima(
    ts,
    seasonal=True,
    m=12,  # Seasonal period
    stepwise=True,
    suppress_warnings=True,
    trace=True
)
print(auto_model.summary())

# Manual ARIMA
model = ARIMA(ts, order=(1, 1, 1))
fitted = model.fit()

# Forecast
forecast = fitted.forecast(steps=12)
conf_int = fitted.get_forecast(steps=12).conf_int()

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(ts.index, ts, label='Historical')
plt.plot(forecast.index, forecast, color='red', label='Forecast')
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], alpha=0.2)
plt.legend()
plt.show()

# ═══════════════════════════════════════════════════════════════
# PROPHET: Facebook's Time Series Library
# ═══════════════════════════════════════════════════════════════

"""
Prophet: Easy to use, handles:
- Multiple seasonalities
- Holiday effects
- Missing data
- Outliers
"""

from prophet import Prophet

# Prepare data (Prophet requires 'ds' and 'y' columns)
df_prophet = pd.DataFrame({
    'ds': ts.index,
    'y': ts.values
})

# Fit model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
model.fit(df_prophet)

# Create future dates
future = model.make_future_dataframe(periods=365)

# Forecast
forecast = model.predict(future)

# Plot
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)
```

### Deep Learning for Time Series

```python
"""
DEEP LEARNING FOR TIME SERIES

When to use:
- Long sequences
- Complex patterns
- Multiple features
- Non-linear relationships
"""

import torch
import torch.nn as nn
import numpy as np

# ═══════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════

def create_sequences(data, seq_length):
    """Create input sequences and targets for time series"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Normalize data (CRITICAL!)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.reshape(-1, 1))

# Create sequences
seq_length = 60  # Use 60 past values to predict next
X, y = create_sequences(scaled_data, seq_length)

# Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# ═══════════════════════════════════════════════════════════════
# LSTM MODEL
# ═══════════════════════════════════════════════════════════════

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last time step
        out = self.fc(lstm_out[:, -1, :])
        return out

model = LSTMPredictor(
    input_size=1,
    hidden_size=50,
    num_layers=2,
    output_size=1
)

# Training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Predict
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predictions = scaler.inverse_transform(predictions.numpy())

# ═══════════════════════════════════════════════════════════════
# TRANSFORMER FOR TIME SERIES (Modern Approach)
# ═══════════════════════════════════════════════════════════════

class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_size, num_layers, nhead, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=feature_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )
        self.decoder = nn.Linear(feature_size, 1)
        self.positional_encoding = PositionalEncoding(feature_size)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])  # Predict from last position
        return x
```

---

# UNIT 6: APPLICATIONS AND RESPONSIBLE ML

## 6.1 Real-World ML Applications

### Computer Vision Applications

```python
"""
COMPUTER VISION APPLICATIONS:

1. Image Classification: What's in this image?
2. Object Detection: Where are objects? (bounding boxes)
3. Semantic Segmentation: Label every pixel
4. Instance Segmentation: Label every pixel AND distinguish instances
5. Face Recognition
6. Pose Estimation
"""

# Object Detection with YOLO (State-of-the-art)
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')  # Nano model, fast

# Detect objects
results = model('image.jpg')

# Process results
for result in results:
    boxes = result.boxes.xyxy  # Bounding boxes
    classes = result.boxes.cls  # Class IDs
    confidences = result.boxes.conf  # Confidence scores
    
    for box, cls, conf in zip(boxes, classes, confidences):
        print(f"Detected: {model.names[int(cls)]} ({conf:.2%})")
        print(f"  Bounding box: {box}")

# Image Segmentation
from transformers import pipeline

segmenter = pipeline("image-segmentation")
result = segmenter("image.jpg")

# Face Recognition
import face_recognition

# Load images
known_image = face_recognition.load_image_file("known_person.jpg")
unknown_image = face_recognition.load_image_file("unknown.jpg")

# Get face encodings
known_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare faces
results = face_recognition.compare_faces([known_encoding], unknown_encoding)
print(f"Match: {results[0]}")
```

### Recommendation Systems

```python
"""
RECOMMENDATION SYSTEMS:

1. Collaborative Filtering: Users who liked X also liked Y
2. Content-Based: Items similar to what you've liked
3. Hybrid: Combine both approaches
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# ═══════════════════════════════════════════════════════════════
# CONTENT-BASED FILTERING
# ═══════════════════════════════════════════════════════════════

# Create item feature vectors (e.g., TF-IDF of descriptions)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
item_features = tfidf.fit_transform(items['description'])

# Compute similarity
similarity_matrix = cosine_similarity(item_features)

def get_similar_items(item_id, n=5):
    idx = item_id_to_idx[item_id]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return [idx_to_item_id[i[0]] for i in sim_scores[1:n+1]]

# ═══════════════════════════════════════════════════════════════
# COLLABORATIVE FILTERING (Matrix Factorization)
# ═══════════════════════════════════════════════════════════════

# Prepare data for Surprise library
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

# Use SVD (Singular Value Decomposition)
svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train on full dataset
trainset = data.build_full_trainset()
svd.fit(trainset)

# Predict rating for user-item pair
prediction = svd.predict(user_id='user1', item_id='item123')
print(f"Predicted rating: {prediction.est:.2f}")

# Get top N recommendations for a user
def get_recommendations(user_id, n=10):
    all_items = items['item_id'].unique()
    user_rated = ratings[ratings['user_id'] == user_id]['item_id'].values
    items_to_predict = [i for i in all_items if i not in user_rated]
    
    predictions = [svd.predict(user_id, item_id) for item_id in items_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    return [pred.iid for pred in predictions[:n]]
```

---

## 6.2 MLOps: From Notebook to Production

```python
"""
MLOps: Making ML work in production

Key components:
1. Version control (Git)
2. Experiment tracking (MLflow, W&B)
3. Model registry
4. CI/CD pipelines
5. Monitoring
6. Feature stores
"""

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT TRACKING WITH MLFLOW
# ═══════════════════════════════════════════════════════════════

import mlflow
import mlflow.sklearn

# Start experiment
mlflow.set_experiment("my_classification_experiment")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts (plots, data samples, etc.)
    mlflow.log_artifact("confusion_matrix.png")

# Load model from MLflow
model_uri = "runs:/<run_id>/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

# ═══════════════════════════════════════════════════════════════
# MODEL SERVING WITH FASTAPI
# ═══════════════════════════════════════════════════════════════

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model at startup
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

class PredictionInput(BaseModel):
    features: list[float]

class PredictionOutput(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    # Preprocess
    features = np.array(input_data.features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    
    # Predict
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0].max()
    
    return PredictionOutput(
        prediction=int(prediction),
        probability=float(probability)
    )

# Run with: uvicorn main:app --reload

# ═══════════════════════════════════════════════════════════════
# DOCKER CONTAINERIZATION
# ═══════════════════════════════════════════════════════════════

"""
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# ═══════════════════════════════════════════════════════════════
# MONITORING MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════

"""
Monitor for:
1. Data drift: Input distribution changes
2. Model drift: Predictions become less accurate
3. Concept drift: Relationship between input and output changes
"""

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftPreset, TargetDriftPreset

# Compare reference (training) data to current (production) data
report = Report(metrics=[DataDriftPreset()])
report.run(
    reference_data=training_data,
    current_data=production_data
)
report.save_html('drift_report.html')
```

---

## 6.3 Responsible AI & Ethics

### Fairness in ML

```python
"""
FAIRNESS: Ensure your model doesn't discriminate

Types of bias:
1. Historical bias: Past data reflects past discrimination
2. Representation bias: Underrepresented groups in training
3. Measurement bias: Features measured differently for groups
4. Aggregation bias: Single model for diverse populations
"""

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.metrics import accuracy_score

# ═══════════════════════════════════════════════════════════════
# MEASURE FAIRNESS
# ═══════════════════════════════════════════════════════════════

# Check accuracy by sensitive group
metric_frame = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "selection_rate": lambda y, y_pred: y_pred.mean()
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_features_test  # e.g., gender, race
)

print(metric_frame.by_group)
print(f"Demographic Parity Difference: {demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features_test):.4f}")

# ═══════════════════════════════════════════════════════════════
# MITIGATE BIAS
# ═══════════════════════════════════════════════════════════════

# Pre-processing: Rebalance training data
from sklearn.utils import resample

# Separate majority and minority groups
majority = train_data[train_data['group'] == 'majority']
minority = train_data[train_data['group'] == 'minority']

# Upsample minority
minority_upsampled = resample(minority, n_samples=len(majority), random_state=42)
balanced_data = pd.concat([majority, minority_upsampled])

# In-processing: Fairness constraints during training
constraint = DemographicParity()
mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(),
    constraints=constraint
)
mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
y_pred_fair = mitigator.predict(X_test)

# Post-processing: Adjust thresholds per group
from fairlearn.postprocessing import ThresholdOptimizer

postprocess = ThresholdOptimizer(
    estimator=model,
    constraints="demographic_parity",
    prefit=True
)
postprocess.fit(X_val, y_val, sensitive_features=sensitive_val)
y_pred_adjusted = postprocess.predict(X_test, sensitive_features=sensitive_test)
```

### Explainability (XAI)

```python
"""
EXPLAINABLE AI: Understand why models make predictions

Why it matters:
1. Trust: Users need to trust model decisions
2. Debugging: Find and fix model errors
3. Compliance: Regulations may require explanations
4. Improvement: Understand what features matter
"""

import shap
import lime
import lime.lime_tabular

# ═══════════════════════════════════════════════════════════════
# SHAP: SHapley Additive exPlanations
# ═══════════════════════════════════════════════════════════════

"""
SHAP values show the contribution of each feature to a prediction.
Based on game theory (Shapley values).
"""

# For tree-based models (fast!)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot: Feature importance across all predictions
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Dependence plot: How one feature affects predictions
shap.dependence_plot("feature_name", shap_values, X_test)

# Waterfall plot: Explain single prediction
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=feature_names
))

# For any model (slower but universal)
explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
shap_values = explainer.shap_values(X_test[:10])

# ═══════════════════════════════════════════════════════════════
# LIME: Local Interpretable Model-agnostic Explanations
# ═══════════════════════════════════════════════════════════════

"""
LIME explains individual predictions by:
1. Creating perturbed samples around the instance
2. Getting predictions for perturbed samples
3. Fitting a simple interpretable model locally
"""

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=['Class 0', 'Class 1'],
    mode='classification'
)

# Explain a single prediction
explanation = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10
)

# Show explanation
explanation.show_in_notebook()

# For text
from lime.lime_text import LimeTextExplainer

text_explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])
text_explanation = text_explainer.explain_instance(
    "This movie was terrible",
    classifier.predict_proba,
    num_features=10
)
text_explanation.show_in_notebook()

# ═══════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE (Built-in for tree models)
# ═══════════════════════════════════════════════════════════════

# For Random Forest / Gradient Boosting
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Permutation importance (model-agnostic)
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10)
perm_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)
```

### Privacy and Security

```python
"""
ML PRIVACY & SECURITY:

1. Differential Privacy: Add noise to protect individual data
2. Federated Learning: Train on decentralized data
3. Adversarial Robustness: Defend against attacks
"""

# ═══════════════════════════════════════════════════════════════
# DIFFERENTIAL PRIVACY
# ═══════════════════════════════════════════════════════════════

from diffprivlib.models import LogisticRegression as DPLogisticRegression

# Train with differential privacy
dp_model = DPLogisticRegression(epsilon=1.0)  # Lower epsilon = more privacy
dp_model.fit(X_train, y_train)

# ═══════════════════════════════════════════════════════════════
# ADVERSARIAL EXAMPLES
# ═══════════════════════════════════════════════════════════════

"""
Adversarial examples: Small perturbations that fool models

Example: Adding invisible noise to an image makes the model
misclassify a panda as a gibbon with high confidence!
"""

import torch
import torch.nn.functional as F

def fgsm_attack(image, epsilon, gradient):
    """Fast Gradient Sign Method attack"""
    # Perturb in the direction of the gradient
    perturbed_image = image + epsilon * gradient.sign()
    # Clamp to valid range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def generate_adversarial(model, image, label, epsilon=0.1):
    """Generate adversarial example"""
    image.requires_grad = True
    output = model(image)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()
    
    # Create adversarial example
    adversarial = fgsm_attack(image, epsilon, image.grad)
    return adversarial

# Defense: Adversarial training
def adversarial_training(model, train_loader, optimizer, epsilon=0.1):
    for images, labels in train_loader:
        # Generate adversarial examples
        adv_images = generate_adversarial(model, images.clone(), labels, epsilon)
        
        # Train on both clean and adversarial
        optimizer.zero_grad()
        
        # Clean loss
        outputs = model(images)
        clean_loss = F.cross_entropy(outputs, labels)
        
        # Adversarial loss
        adv_outputs = model(adv_images)
        adv_loss = F.cross_entropy(adv_outputs, labels)
        
        # Combined loss
        total_loss = clean_loss + adv_loss
        total_loss.backward()
        optimizer.step()
```

---

# QUICK REFERENCE CHEAT SHEETS

## Algorithm Selection Guide

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WHICH ALGORITHM TO USE?                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SUPERVISED LEARNING                                                         │
│  ├── Classification                                                          │
│  │   ├── Start with: Logistic Regression, Random Forest                     │
│  │   ├── Tabular data: XGBoost, LightGBM, CatBoost (BEST!)                 │
│  │   ├── Images: CNN, Vision Transformer                                    │
│  │   ├── Text: Transformers (BERT, RoBERTa)                                │
│  │   └── Time series: LSTM, Transformer                                     │
│  │                                                                           │
│  └── Regression                                                              │
│      ├── Start with: Linear Regression                                      │
│      ├── Tabular data: XGBoost, LightGBM                                   │
│      ├── Time series: ARIMA, Prophet, LSTM                                 │
│      └── Complex relationships: Neural Networks                             │
│                                                                              │
│  UNSUPERVISED LEARNING                                                       │
│  ├── Clustering                                                              │
│  │   ├── Unknown K: DBSCAN, HDBSCAN                                        │
│  │   ├── Known K: K-Means                                                   │
│  │   └── Hierarchical: Agglomerative                                        │
│  │                                                                           │
│  ├── Dimensionality Reduction                                                │
│  │   ├── Linear: PCA                                                        │
│  │   ├── Visualization: UMAP (preferred), t-SNE                            │
│  │   └── Feature extraction: Autoencoders                                   │
│  │                                                                           │
│  └── Anomaly Detection                                                       │
│      ├── Tabular: Isolation Forest                                          │
│      └── Complex: Autoencoders                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Hyperparameter Tuning Quick Reference

```python
# GENERAL RULES
"""
If overfitting:
  - Reduce model complexity
  - Add regularization
  - Get more data
  - Use dropout
  - Early stopping

If underfitting:
  - Increase model complexity
  - Add features
  - Reduce regularization
  - Train longer
"""

# RANDOM FOREST
# n_estimators: 100-1000 (more is usually better, diminishing returns)
# max_depth: 10-30 (start with None, reduce if overfitting)
# min_samples_split: 2-10
# min_samples_leaf: 1-10

# XGBOOST / LIGHTGBM
# n_estimators: 100-1000 (with early stopping!)
# learning_rate: 0.01-0.3 (lower with more estimators)
# max_depth: 3-10 (lower than Random Forest!)
# subsample: 0.6-1.0
# colsample_bytree: 0.6-1.0

# NEURAL NETWORKS
# learning_rate: 1e-4 to 1e-2 (Adam default: 1e-3)
# batch_size: 16-128 (depends on GPU memory)
# hidden_layers: 1-5 for most tasks
# hidden_units: 32-512 per layer
# dropout: 0.1-0.5

# SVM
# C: 0.1-100 (regularization)
# gamma: 'scale' or 0.001-1 (kernel coefficient)
```

## Common Metrics

```
CLASSIFICATION:
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
  Precision = TP / (TP + FP)  → "How many selected items are relevant?"
  Recall = TP / (TP + FN)     → "How many relevant items are selected?"
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  AUC-ROC = Area under ROC curve (0.5 = random, 1.0 = perfect)

REGRESSION:
  MSE = (1/n) Σ(y - ŷ)²        → Mean Squared Error
  RMSE = √MSE                  → Same units as target
  MAE = (1/n) Σ|y - ŷ|         → Mean Absolute Error
  R² = 1 - (SS_res / SS_tot)   → Proportion of variance explained

CLUSTERING:
  Silhouette Score = (b - a) / max(a, b)  → -1 to 1, higher is better
  Inertia = Sum of squared distances to centroid → Lower is better
```

---

# RESOURCES & FURTHER LEARNING

## Books
- "Hands-On Machine Learning" by Aurélien Géron
- "Deep Learning" by Goodfellow, Bengio, and Courville
- "Pattern Recognition and Machine Learning" by Bishop
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman

## Courses
- Fast.ai - Practical Deep Learning
- Andrew Ng's Machine Learning (Coursera)
- Stanford CS229, CS231n, CS224n (free online)

## Practice
- Kaggle competitions
- Papers With Code
- HuggingFace datasets

## Stay Updated
- arXiv (cs.LG, cs.CV, cs.CL)
- Twitter/X: @kaborojevic, @ylecun, @AndrewYNg
- Newsletters: The Batch, TLDR AI

---

## Final Tips

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                         GOLDEN RULES OF ML                                 ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  1. DATA QUALITY > MODEL COMPLEXITY                                        ║
║     "Garbage in, garbage out"                                              ║
║                                                                            ║
║  2. START SIMPLE                                                           ║
║     Baseline with simple model first, then improve                         ║
║                                                                            ║
║  3. VALIDATION IS EVERYTHING                                               ║
║     Never trust training metrics                                           ║
║                                                                            ║
║  4. UNDERSTAND YOUR DATA                                                   ║
║     EDA before modeling, always                                            ║
║                                                                            ║
║  5. DON'T REINVENT THE WHEEL                                               ║
║     Use pretrained models and transfer learning                            ║
║                                                                            ║
║  6. REPRODUCIBILITY MATTERS                                                ║
║     Set random seeds, version control everything                           ║
║                                                                            ║
║  7. MONITOR IN PRODUCTION                                                  ║
║     Models degrade over time                                               ║
║                                                                            ║
║  8. ETHICS FIRST                                                           ║
║     Check for bias, consider impact                                        ║
║                                                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

**Created:** January 2025  
**Author:** Your ML Learning Companion  
**License:** Free to use, share, and adapt

*"The best way to learn ML is to build things. Start with tutorials, but quickly move to your own projects. The struggle is where the learning happens."*

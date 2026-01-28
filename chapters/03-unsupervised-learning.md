<div align="center">

# 🔮 Unsupervised Learning

![Chapter](https://img.shields.io/badge/Chapter-03-blue?style=for-the-badge)
![Topic](https://img.shields.io/badge/Topic-Clustering%20%7C%20Dimensionality-green?style=for-the-badge)

*K-Means, DBSCAN, PCA, t-SNE & Anomaly Detection*

---

</div>

# Part VI: Unsupervised Learning

---

## Chapter 15: Introduction to Unsupervised Learning

### 15.1 What is Unsupervised Learning?

Unsupervised learning discovers hidden patterns in data without labeled outputs.

```
┌─────────────────────────────────────────────────────────────────────┐
│         SUPERVISED vs UNSUPERVISED LEARNING                         │
├─────────────────────────────────────────────────────────────────────┤
│  SUPERVISED:                                                        │
│  Input: (X, y) pairs                                               │
│  Goal: Learn mapping X → y                                         │
│                                                                     │
│  UNSUPERVISED:                                                      │
│  Input: X only (no labels)                                         │
│  Goal: Find structure in X                                         │
└─────────────────────────────────────────────────────────────────────┘
```

**Common Tasks:**
1. **Clustering:** Group similar data points
2. **Dimensionality Reduction:** Reduce features while preserving information
3. **Anomaly Detection:** Find unusual data points
4. **Association Rules:** Find relationships between features

---

## Chapter 16: Clustering Algorithms

### 16.1 K-Means Clustering

K-Means partitions data into K clusters by minimizing within-cluster variance.

```python
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    """K-Means Clustering from scratch with k-means++ initialization."""
    
    def __init__(self, n_clusters=3, max_iterations=300, tol=1e-4, 
                 init='k-means++', random_state=None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
        
    def _init_centroids_kmeanspp(self, X):
        """K-means++ initialization for better starting centroids."""
        n_samples = X.shape[0]
        centroids = []
        
        # First centroid: random
        idx = np.random.randint(n_samples)
        centroids.append(X[idx])
        
        # Remaining centroids
        for _ in range(1, self.n_clusters):
            distances = np.zeros(n_samples)
            for i, x in enumerate(X):
                min_dist = min(np.sum((x - c) ** 2) for c in centroids)
                distances[i] = min_dist
            
            probabilities = distances / np.sum(distances)
            idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[idx])
        
        return np.array(centroids)
    
    def _assign_clusters(self, X):
        """Assign each point to nearest centroid."""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.sum((X - self.centroids[k]) ** 2, axis=1)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        """Update centroids as mean of assigned points."""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            points = X[labels == k]
            if len(points) > 0:
                new_centroids[k] = np.mean(points, axis=0)
            else:
                new_centroids[k] = X[np.random.randint(X.shape[0])]
        return new_centroids
    
    def fit(self, X):
        """Fit the K-Means model."""
        X = np.array(X)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.centroids = self._init_centroids_kmeanspp(X)
        
        for _ in range(self.max_iterations):
            self.labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, self.labels)
            
            if np.sum((new_centroids - self.centroids) ** 2) < self.tol:
                break
            self.centroids = new_centroids
        
        self.inertia = sum(np.sum((X[self.labels == k] - self.centroids[k]) ** 2) 
                          for k in range(self.n_clusters))
        return self
    
    def predict(self, X):
        return self._assign_clusters(np.array(X))


# Example usage
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

print(f"Inertia: {kmeans.inertia:.2f}")
print(f"Cluster sizes: {[np.sum(kmeans.labels == k) for k in range(4)]}")
```

### 16.2 Choosing K: Elbow Method and Silhouette Score

```python
from sklearn.metrics import silhouette_score

def find_optimal_k(X, max_k=10):
    """Find optimal K using elbow method and silhouette score."""
    inertias = []
    silhouettes = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia)
        silhouettes.append(silhouette_score(X, kmeans.labels))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(range(2, max_k + 1), inertias, 'bo-')
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    
    axes[1].plot(range(2, max_k + 1), silhouettes, 'go-')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')
    
    plt.tight_layout()
    return inertias, silhouettes

inertias, silhouettes = find_optimal_k(X, max_k=8)
print(f"Best K by silhouette: {np.argmax(silhouettes) + 2}")
```

### 16.3 Hierarchical Clustering

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def hierarchical_clustering(X, n_clusters=3, method='ward'):
    """Perform hierarchical clustering and return labels."""
    Z = linkage(X, method=method)
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    return labels, Z

# Example
labels, Z = hierarchical_clustering(X, n_clusters=4)

plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
```

### 16.4 DBSCAN: Density-Based Clustering

```python
class DBSCAN:
    """DBSCAN clustering - finds arbitrary-shaped clusters and outliers."""
    
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        
    def _find_neighbors(self, X, point_idx):
        distances = np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))
        return np.where(distances <= self.eps)[0]
    
    def fit(self, X):
        X = np.array(X)
        n_samples = len(X)
        labels = np.full(n_samples, -1)
        
        neighborhoods = [self._find_neighbors(X, i) for i in range(n_samples)]
        core_points = {i for i in range(n_samples) if len(neighborhoods[i]) >= self.min_samples}
        
        cluster_id = 0
        for i in range(n_samples):
            if labels[i] != -1 or i not in core_points:
                continue
            
            # BFS to expand cluster
            queue = [i]
            labels[i] = cluster_id
            
            while queue:
                current = queue.pop(0)
                if current in core_points:
                    for neighbor in neighborhoods[current]:
                        if labels[neighbor] == -1:
                            labels[neighbor] = cluster_id
                            if neighbor in core_points:
                                queue.append(neighbor)
            cluster_id += 1
        
        self.labels_ = labels
        return self

# Example with moons dataset
from sklearn.datasets import make_moons

X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan.fit(X_moons)

print(f"Clusters found: {len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)}")
print(f"Noise points: {np.sum(dbscan.labels_ == -1)}")
```

### 16.5 Gaussian Mixture Models (GMM)

```python
from scipy.stats import multivariate_normal

class GaussianMixture:
    """Gaussian Mixture Model using EM algorithm."""
    
    def __init__(self, n_components=3, max_iterations=100, tol=1e-6):
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tol = tol
        
    def fit(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Initialize with K-means
        kmeans = KMeans(n_clusters=self.n_components, random_state=42)
        kmeans.fit(X)
        
        self.means = kmeans.centroids
        self.weights = np.ones(self.n_components) / self.n_components
        self.covariances = [np.cov(X[kmeans.labels == k].T) + 1e-6 * np.eye(n_features) 
                           for k in range(self.n_components)]
        
        for _ in range(self.max_iterations):
            # E-step
            resp = self._e_step(X)
            
            # M-step
            self._m_step(X, resp)
        
        return self
    
    def _e_step(self, X):
        resp = np.zeros((len(X), self.n_components))
        for k in range(self.n_components):
            rv = multivariate_normal(self.means[k], self.covariances[k])
            resp[:, k] = self.weights[k] * rv.pdf(X)
        resp /= resp.sum(axis=1, keepdims=True) + 1e-10
        return resp
    
    def _m_step(self, X, resp):
        Nk = resp.sum(axis=0)
        self.weights = Nk / len(X)
        self.means = (resp.T @ X) / Nk[:, np.newaxis]
        
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = (resp[:, k:k+1] * diff).T @ diff / Nk[k]
            self.covariances[k] += 1e-6 * np.eye(X.shape[1])
    
    def predict(self, X):
        return np.argmax(self._e_step(np.array(X)), axis=1)
    
    def predict_proba(self, X):
        return self._e_step(np.array(X))
```

---

## Chapter 17: Dimensionality Reduction

### 17.1 Principal Component Analysis (PCA)

```python
class PCA:
    """Principal Component Analysis from scratch."""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.explained_variance_ratio = None
        self.mean = None
        
    def fit(self, X):
        X = np.array(X)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Covariance matrix and eigendecomposition
        cov_matrix = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.real(eigenvalues[idx])
        eigenvectors = np.real(eigenvectors[:, idx])
        
        if self.n_components is None:
            self.n_components = X.shape[1]
        
        self.components = eigenvectors[:, :self.n_components].T
        self.explained_variance_ratio = eigenvalues[:self.n_components] / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X):
        X_centered = np.array(X) - self.mean
        return X_centered @ self.components.T
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        return X_transformed @ self.components + self.mean


# Example on Iris
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Explained variance ratio: {pca.explained_variance_ratio}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio):.4f}")

plt.figure(figsize=(8, 6))
for i, name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=name, alpha=0.7)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio[1]:.2%})')
plt.title('PCA on Iris Dataset')
plt.legend()
plt.show()
```

### 17.2 t-SNE and UMAP

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Load digits for demonstration
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_digits)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_digits)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, cmap='tab10', alpha=0.6, s=10)
axes[0].set_title('PCA')

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='tab10', alpha=0.6, s=10)
axes[1].set_title('t-SNE')

plt.tight_layout()
plt.show()

print("PCA: Fast, linear, preserves global structure")
print("t-SNE: Slow, nonlinear, preserves local structure")
```

---

## Chapter 18: Anomaly Detection

### 18.1 Statistical Methods

```python
def zscore_anomaly(X, threshold=3):
    """Z-score based anomaly detection."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    z_scores = np.abs((X - mean) / (std + 1e-10))
    return np.max(z_scores, axis=1) > threshold

def iqr_anomaly(X, multiplier=1.5):
    """IQR-based anomaly detection."""
    Q1, Q3 = np.percentile(X, [25, 75], axis=0)
    IQR = Q3 - Q1
    lower, upper = Q1 - multiplier * IQR, Q3 + multiplier * IQR
    return np.any((X < lower) | (X > upper), axis=1)
```

### 18.2 Isolation Forest

```python
from sklearn.ensemble import IsolationForest

# Generate data with anomalies
np.random.seed(42)
X_normal = np.random.randn(200, 2)
X_anomalies = np.random.uniform(-4, 4, (10, 2))
X_all = np.vstack([X_normal, X_anomalies])

# Fit Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
predictions = iso_forest.fit_predict(X_all)

print(f"Detected anomalies: {np.sum(predictions == -1)}")

plt.figure(figsize=(8, 6))
plt.scatter(X_all[predictions == 1, 0], X_all[predictions == 1, 1], 
            c='blue', label='Normal', alpha=0.6)
plt.scatter(X_all[predictions == -1, 0], X_all[predictions == -1, 1], 
            c='red', marker='x', s=100, label='Anomaly')
plt.legend()
plt.title('Isolation Forest Anomaly Detection')
plt.show()
```

### 18.3 One-Class SVM

```python
from sklearn.svm import OneClassSVM

# Train only on normal data
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
ocsvm.fit(X_normal)

# Predict on all data
predictions = ocsvm.predict(X_all)

print(f"Detected anomalies: {np.sum(predictions == -1)}")
```

### 18.4 Local Outlier Factor

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
predictions = lof.fit_predict(X_all)

print(f"LOF detected anomalies: {np.sum(predictions == -1)}")
```

---

## Chapter 19: Association Rule Mining

### 19.1 The Apriori Algorithm

```python
from itertools import combinations

class Apriori:
    """Apriori algorithm for association rule mining."""
    
    def __init__(self, min_support=0.5, min_confidence=0.7):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}
        self.rules = []
        
    def fit(self, transactions):
        """Find frequent itemsets and generate rules."""
        n_transactions = len(transactions)
        
        # Get all unique items
        all_items = set()
        for t in transactions:
            all_items.update(t)
        
        # Find frequent 1-itemsets
        current_itemsets = []
        for item in all_items:
            support = sum(1 for t in transactions if item in t) / n_transactions
            if support >= self.min_support:
                current_itemsets.append(frozenset([item]))
                self.frequent_itemsets[frozenset([item])] = support
        
        # Find larger frequent itemsets
        k = 2
        while current_itemsets:
            candidates = self._generate_candidates(current_itemsets, k)
            current_itemsets = []
            
            for candidate in candidates:
                support = sum(1 for t in transactions if candidate.issubset(set(t))) / n_transactions
                if support >= self.min_support:
                    current_itemsets.append(candidate)
                    self.frequent_itemsets[candidate] = support
            k += 1
        
        # Generate rules
        self._generate_rules()
        
        return self
    
    def _generate_candidates(self, itemsets, k):
        """Generate candidate k-itemsets."""
        candidates = set()
        itemsets_list = list(itemsets)
        
        for i in range(len(itemsets_list)):
            for j in range(i + 1, len(itemsets_list)):
                union = itemsets_list[i] | itemsets_list[j]
                if len(union) == k:
                    candidates.add(union)
        
        return candidates
    
    def _generate_rules(self):
        """Generate association rules from frequent itemsets."""
        for itemset, support in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue
            
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    if antecedent in self.frequent_itemsets:
                        confidence = support / self.frequent_itemsets[antecedent]
                        if confidence >= self.min_confidence:
                            lift = confidence / self.frequent_itemsets[consequent]
                            self.rules.append({
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': support,
                                'confidence': confidence,
                                'lift': lift
                            })


# Example: Market basket analysis
transactions = [
    ['bread', 'milk', 'eggs'],
    ['bread', 'butter'],
    ['milk', 'butter', 'eggs'],
    ['bread', 'milk', 'butter'],
    ['bread', 'milk', 'eggs', 'butter'],
    ['milk', 'eggs'],
    ['bread', 'eggs'],
    ['bread', 'milk'],
]

apriori = Apriori(min_support=0.3, min_confidence=0.6)
apriori.fit(transactions)

print("Frequent Itemsets:")
for itemset, support in sorted(apriori.frequent_itemsets.items(), key=lambda x: -x[1]):
    print(f"  {set(itemset)}: support = {support:.2f}")

print("\nAssociation Rules:")
for rule in sorted(apriori.rules, key=lambda x: -x['confidence']):
    print(f"  {set(rule['antecedent'])} -> {set(rule['consequent'])}")
    print(f"    confidence: {rule['confidence']:.2f}, lift: {rule['lift']:.2f}")
```

---

## Summary: Unsupervised Learning

```
┌─────────────────────────────────────────────────────────────────────┐
│              UNSUPERVISED LEARNING CHEAT SHEET                      │
├─────────────────────────────────────────────────────────────────────┤
│  CLUSTERING                                                         │
│  ├── K-Means: Fast, spherical clusters, need K                     │
│  ├── Hierarchical: Dendrogram, no need for K                       │
│  ├── DBSCAN: Arbitrary shapes, finds outliers                      │
│  └── GMM: Soft clustering, elliptical clusters                     │
│                                                                     │
│  DIMENSIONALITY REDUCTION                                           │
│  ├── PCA: Linear, fast, interpretable                              │
│  ├── t-SNE: Nonlinear, great for visualization                     │
│  └── UMAP: Faster than t-SNE, preserves global structure           │
│                                                                     │
│  ANOMALY DETECTION                                                  │
│  ├── Z-score/IQR: Simple statistical methods                       │
│  ├── Isolation Forest: Tree-based, scalable                        │
│  ├── One-Class SVM: Kernel-based boundary                          │
│  └── LOF: Density-based local outliers                             │
└─────────────────────────────────────────────────────────────────────┘
```


---

<div align="center">

[⬅️ Previous: Supervised Learning](02-supervised-learning.md) | [📚 Table of Contents](../README.md) | [Next: Natural Language Processing ➡️](04-nlp.md)

</div>

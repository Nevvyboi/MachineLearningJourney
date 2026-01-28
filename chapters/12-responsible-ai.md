<div align="center">

# ⚖️ Responsible AI

![Chapter](https://img.shields.io/badge/Chapter-12-blue?style=for-the-badge)
![Topic](https://img.shields.io/badge/Topic-Ethics%20%7C%20Fairness-green?style=for-the-badge)

*Fairness, Interpretability, Privacy & Bias Mitigation*

---

</div>

# Part XV: Responsible AI and Ethics in Machine Learning

---

## Chapter 49: Fairness in Machine Learning

### 49.1 Understanding Bias in ML

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SOURCES OF BIAS IN ML                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DATA BIAS                                                          │
│  ├── Historical Bias: Past discrimination reflected in data        │
│  ├── Representation Bias: Underrepresented groups                  │
│  ├── Measurement Bias: Flawed data collection process              │
│  └── Sampling Bias: Non-representative sample                      │
│                                                                     │
│  ALGORITHMIC BIAS                                                   │
│  ├── Optimization Bias: Optimizing for majority                    │
│  ├── Feature Selection: Proxies for protected attributes           │
│  └── Model Complexity: Overfitting to biased patterns              │
│                                                                     │
│  DEPLOYMENT BIAS                                                    │
│  ├── Feedback Loops: Biased predictions affect future data         │
│  ├── Population Shift: Training ≠ deployment population            │
│  └── Usage Bias: Different impacts across groups                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 49.2 Fairness Metrics

```python
import numpy as np
from sklearn.metrics import confusion_matrix

class FairnessMetrics:
    """
    Calculate various fairness metrics for binary classification.
    """
    
    def __init__(self, y_true, y_pred, protected_attribute):
        """
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            protected_attribute: Binary array indicating protected group membership
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.protected = np.array(protected_attribute)
        
        # Split by protected attribute
        self.group_0_mask = self.protected == 0
        self.group_1_mask = self.protected == 1
    
    def _get_rates(self, mask):
        """Calculate rates for a group."""
        y_true_group = self.y_true[mask]
        y_pred_group = self.y_pred[mask]
        
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        selection_rate = (tp + fp) / len(y_true_group)
        
        return {
            'tpr': tpr,
            'fpr': fpr,
            'ppv': ppv,
            'selection_rate': selection_rate,
            'accuracy': (tp + tn) / len(y_true_group)
        }
    
    def demographic_parity(self):
        """
        Demographic Parity (Statistical Parity):
        P(Y_pred=1 | A=0) = P(Y_pred=1 | A=1)
        
        Selection rates should be equal across groups.
        """
        rates_0 = self._get_rates(self.group_0_mask)
        rates_1 = self._get_rates(self.group_1_mask)
        
        parity_diff = abs(rates_0['selection_rate'] - rates_1['selection_rate'])
        parity_ratio = min(rates_0['selection_rate'], rates_1['selection_rate']) / \
                      max(rates_0['selection_rate'], rates_1['selection_rate']) \
                      if max(rates_0['selection_rate'], rates_1['selection_rate']) > 0 else 0
        
        return {
            'group_0_rate': rates_0['selection_rate'],
            'group_1_rate': rates_1['selection_rate'],
            'parity_difference': parity_diff,
            'parity_ratio': parity_ratio,
            'is_fair': parity_ratio >= 0.8  # 80% rule
        }
    
    def equalized_odds(self):
        """
        Equalized Odds:
        P(Y_pred=1 | A=0, Y=y) = P(Y_pred=1 | A=1, Y=y) for y ∈ {0, 1}
        
        TPR and FPR should be equal across groups.
        """
        rates_0 = self._get_rates(self.group_0_mask)
        rates_1 = self._get_rates(self.group_1_mask)
        
        tpr_diff = abs(rates_0['tpr'] - rates_1['tpr'])
        fpr_diff = abs(rates_0['fpr'] - rates_1['fpr'])
        
        return {
            'group_0_tpr': rates_0['tpr'],
            'group_1_tpr': rates_1['tpr'],
            'group_0_fpr': rates_0['fpr'],
            'group_1_fpr': rates_1['fpr'],
            'tpr_difference': tpr_diff,
            'fpr_difference': fpr_diff,
            'is_fair': tpr_diff < 0.1 and fpr_diff < 0.1
        }
    
    def equal_opportunity(self):
        """
        Equal Opportunity:
        P(Y_pred=1 | A=0, Y=1) = P(Y_pred=1 | A=1, Y=1)
        
        TPR should be equal across groups.
        """
        rates_0 = self._get_rates(self.group_0_mask)
        rates_1 = self._get_rates(self.group_1_mask)
        
        tpr_diff = abs(rates_0['tpr'] - rates_1['tpr'])
        
        return {
            'group_0_tpr': rates_0['tpr'],
            'group_1_tpr': rates_1['tpr'],
            'tpr_difference': tpr_diff,
            'is_fair': tpr_diff < 0.1
        }
    
    def predictive_parity(self):
        """
        Predictive Parity:
        P(Y=1 | Y_pred=1, A=0) = P(Y=1 | Y_pred=1, A=1)
        
        Precision (PPV) should be equal across groups.
        """
        rates_0 = self._get_rates(self.group_0_mask)
        rates_1 = self._get_rates(self.group_1_mask)
        
        ppv_diff = abs(rates_0['ppv'] - rates_1['ppv'])
        
        return {
            'group_0_ppv': rates_0['ppv'],
            'group_1_ppv': rates_1['ppv'],
            'ppv_difference': ppv_diff,
            'is_fair': ppv_diff < 0.1
        }
    
    def generate_report(self):
        """Generate comprehensive fairness report."""
        print("=" * 60)
        print("FAIRNESS EVALUATION REPORT")
        print("=" * 60)
        
        print(f"\nGroup 0 size: {sum(self.group_0_mask)}")
        print(f"Group 1 size: {sum(self.group_1_mask)}")
        
        dp = self.demographic_parity()
        print("\n1. DEMOGRAPHIC PARITY")
        print(f"   Group 0 selection rate: {dp['group_0_rate']:.3f}")
        print(f"   Group 1 selection rate: {dp['group_1_rate']:.3f}")
        print(f"   Parity ratio: {dp['parity_ratio']:.3f}")
        print(f"   Fair (≥0.8): {'✓' if dp['is_fair'] else '✗'}")
        
        eo = self.equalized_odds()
        print("\n2. EQUALIZED ODDS")
        print(f"   Group 0 TPR: {eo['group_0_tpr']:.3f}, FPR: {eo['group_0_fpr']:.3f}")
        print(f"   Group 1 TPR: {eo['group_1_tpr']:.3f}, FPR: {eo['group_1_fpr']:.3f}")
        print(f"   Fair (<0.1 diff): {'✓' if eo['is_fair'] else '✗'}")
        
        eop = self.equal_opportunity()
        print("\n3. EQUAL OPPORTUNITY")
        print(f"   TPR difference: {eop['tpr_difference']:.3f}")
        print(f"   Fair (<0.1): {'✓' if eop['is_fair'] else '✗'}")
        
        pp = self.predictive_parity()
        print("\n4. PREDICTIVE PARITY")
        print(f"   PPV difference: {pp['ppv_difference']:.3f}")
        print(f"   Fair (<0.1): {'✓' if pp['is_fair'] else '✗'}")
        
        print("\n" + "=" * 60)


# Example usage
np.random.seed(42)
n = 1000

# Simulated data with bias
protected = np.random.binomial(1, 0.3, n)  # 30% in protected group
y_true = np.random.binomial(1, 0.5, n)

# Biased predictions (lower accuracy for protected group)
y_pred = y_true.copy()
noise_idx = np.random.choice(n, 100, replace=False)
y_pred[noise_idx] = 1 - y_pred[noise_idx]

# Additional bias against protected group
protected_flip = np.random.choice(np.where(protected == 1)[0], 50, replace=False)
y_pred[protected_flip] = 0

# Evaluate
metrics = FairnessMetrics(y_true, y_pred, protected)
metrics.generate_report()
```

### 49.3 Bias Mitigation Techniques

```python
class PreprocessingMitigation:
    """Pre-processing techniques for bias mitigation."""
    
    @staticmethod
    def reweighing(X, y, protected):
        """
        Reweighing: Assign weights to balance groups.
        
        Weight = P(Y) * P(A) / P(Y, A)
        """
        weights = np.ones(len(y))
        
        for a in [0, 1]:
            for label in [0, 1]:
                mask = (protected == a) & (y == label)
                
                # Expected proportion under independence
                p_a = np.mean(protected == a)
                p_y = np.mean(y == label)
                expected = p_a * p_y
                
                # Observed proportion
                observed = np.mean(mask)
                
                if observed > 0:
                    weights[mask] = expected / observed
        
        return weights
    
    @staticmethod
    def disparate_impact_remover(X, protected, repair_level=1.0):
        """
        Disparate Impact Remover: Transform features to remove correlation 
        with protected attribute.
        """
        X_transformed = X.copy()
        
        for i in range(X.shape[1]):
            # Compute conditional distributions
            group_0_vals = X[protected == 0, i]
            group_1_vals = X[protected == 1, i]
            
            # Compute median of each group
            median_0 = np.median(group_0_vals)
            median_1 = np.median(group_1_vals)
            overall_median = np.median(X[:, i])
            
            # Repair by moving towards overall median
            X_transformed[protected == 0, i] = (
                (1 - repair_level) * group_0_vals + 
                repair_level * (group_0_vals - median_0 + overall_median)
            )
            X_transformed[protected == 1, i] = (
                (1 - repair_level) * group_1_vals + 
                repair_level * (group_1_vals - median_1 + overall_median)
            )
        
        return X_transformed


class InProcessingMitigation:
    """In-processing techniques for bias mitigation."""
    
    @staticmethod
    def adversarial_debiasing_loss(y_pred, y_true, protected_pred, protected_true, lambda_fairness=1.0):
        """
        Adversarial Debiasing: Train to predict target while
        being unable to predict protected attribute.
        
        Loss = L(y_pred, y_true) - λ * L(protected_pred, protected_true)
        """
        import torch.nn.functional as F
        
        # Task loss
        task_loss = F.binary_cross_entropy(y_pred, y_true)
        
        # Adversarial loss (we want to MAXIMIZE this, i.e., make predictor worse)
        adversary_loss = F.binary_cross_entropy(protected_pred, protected_true)
        
        # Combined loss
        total_loss = task_loss - lambda_fairness * adversary_loss
        
        return total_loss, task_loss, adversary_loss
    
    @staticmethod
    def fairness_constraint_loss(y_pred, y_true, protected, fairness_type='demographic_parity'):
        """
        Add fairness constraint as regularization term.
        """
        import torch
        
        group_0_mask = protected == 0
        group_1_mask = protected == 1
        
        if fairness_type == 'demographic_parity':
            # Penalize difference in positive prediction rates
            rate_0 = y_pred[group_0_mask].mean()
            rate_1 = y_pred[group_1_mask].mean()
            fairness_loss = (rate_0 - rate_1) ** 2
            
        elif fairness_type == 'equalized_odds':
            # Penalize difference in TPR and FPR
            # TPR difference (among positive cases)
            pos_mask = y_true == 1
            tpr_0 = y_pred[group_0_mask & pos_mask].mean()
            tpr_1 = y_pred[group_1_mask & pos_mask].mean()
            
            # FPR difference (among negative cases)
            neg_mask = y_true == 0
            fpr_0 = y_pred[group_0_mask & neg_mask].mean()
            fpr_1 = y_pred[group_1_mask & neg_mask].mean()
            
            fairness_loss = (tpr_0 - tpr_1) ** 2 + (fpr_0 - fpr_1) ** 2
        
        else:
            fairness_loss = 0
        
        return fairness_loss


class PostProcessingMitigation:
    """Post-processing techniques for bias mitigation."""
    
    @staticmethod
    def equalized_odds_postprocessing(y_pred_proba, protected, y_true=None):
        """
        Adjust thresholds per group to achieve equalized odds.
        """
        best_thresholds = {0: 0.5, 1: 0.5}
        
        if y_true is not None:
            # Find optimal thresholds for each group
            for group in [0, 1]:
                mask = protected == group
                group_proba = y_pred_proba[mask]
                group_true = y_true[mask]
                
                best_score = -float('inf')
                
                for threshold in np.arange(0.1, 0.9, 0.05):
                    group_pred = (group_proba >= threshold).astype(int)
                    
                    # Calculate accuracy while considering fairness
                    accuracy = np.mean(group_pred == group_true)
                    
                    if accuracy > best_score:
                        best_score = accuracy
                        best_thresholds[group] = threshold
        
        # Apply group-specific thresholds
        y_pred_adjusted = np.zeros(len(y_pred_proba))
        
        for group in [0, 1]:
            mask = protected == group
            y_pred_adjusted[mask] = (y_pred_proba[mask] >= best_thresholds[group]).astype(int)
        
        return y_pred_adjusted, best_thresholds
    
    @staticmethod
    def reject_option_classification(y_pred_proba, protected, threshold=0.5, 
                                     margin=0.1, favorable_label=1):
        """
        Reject Option Classification: In uncertain region, flip predictions
        to favor disadvantaged group.
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Identify uncertain predictions
        lower = threshold - margin
        upper = threshold + margin
        uncertain_mask = (y_pred_proba >= lower) & (y_pred_proba <= upper)
        
        # Determine disadvantaged group (lower selection rate)
        rate_0 = y_pred[protected == 0].mean()
        rate_1 = y_pred[protected == 1].mean()
        disadvantaged = 0 if rate_0 < rate_1 else 1
        
        # Flip predictions in uncertain region for disadvantaged group
        flip_mask = uncertain_mask & (protected == disadvantaged) & (y_pred != favorable_label)
        y_pred[flip_mask] = favorable_label
        
        return y_pred


print("\nBias Mitigation Summary:")
print("=" * 60)
print("""
PRE-PROCESSING (before training):
- Reweighing: Assign sample weights
- Disparate Impact Remover: Transform features
- Fair representation learning

IN-PROCESSING (during training):
- Adversarial debiasing: Can't predict protected attribute
- Fairness constraints: Add regularization term
- Fair loss functions

POST-PROCESSING (after training):
- Threshold adjustment per group
- Reject option classification
- Calibration
""")
```

---

## Chapter 50: Model Interpretability

### 50.1 Feature Importance Methods

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

class FeatureImportance:
    """Various methods for computing feature importance."""
    
    @staticmethod
    def permutation_importance(model, X, y, n_repeats=10):
        """
        Permutation Importance: Measure decrease in performance
        when feature is randomly shuffled.
        """
        baseline_score = model.score(X, y)
        importances = []
        
        for col in range(X.shape[1]):
            scores = []
            
            for _ in range(n_repeats):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, col])
                score = model.score(X_permuted, y)
                scores.append(baseline_score - score)
            
            importances.append({
                'mean': np.mean(scores),
                'std': np.std(scores)
            })
        
        return importances
    
    @staticmethod
    def drop_column_importance(model_fn, X, y):
        """
        Drop Column Importance: Measure decrease when feature is removed.
        """
        # Baseline with all features
        model_full = model_fn()
        model_full.fit(X, y)
        baseline_score = model_full.score(X, y)
        
        importances = []
        
        for col in range(X.shape[1]):
            # Remove column
            X_dropped = np.delete(X, col, axis=1)
            
            model = model_fn()
            model.fit(X_dropped, y)
            score = model.score(X_dropped, y)
            
            importances.append(baseline_score - score)
        
        return importances


class SHAP:
    """
    Simplified SHAP (SHapley Additive exPlanations) implementation.
    """
    
    def __init__(self, model, X_background, n_samples=100):
        self.model = model
        self.X_background = X_background
        self.n_samples = n_samples
        self.n_features = X_background.shape[1]
    
    def explain(self, x):
        """
        Compute SHAP values for a single instance.
        
        Uses sampling approximation of Shapley values.
        """
        shap_values = np.zeros(self.n_features)
        
        for _ in range(self.n_samples):
            # Random permutation
            perm = np.random.permutation(self.n_features)
            
            # Random background sample
            bg_idx = np.random.randint(len(self.X_background))
            x_bg = self.X_background[bg_idx].copy()
            
            x_curr = x_bg.copy()
            
            for i, feat_idx in enumerate(perm):
                # Prediction before adding feature
                pred_before = self.model.predict_proba(x_curr.reshape(1, -1))[0, 1]
                
                # Add feature from instance
                x_curr[feat_idx] = x[feat_idx]
                
                # Prediction after adding feature
                pred_after = self.model.predict_proba(x_curr.reshape(1, -1))[0, 1]
                
                # Marginal contribution
                shap_values[feat_idx] += (pred_after - pred_before)
        
        shap_values /= self.n_samples
        
        return shap_values
    
    def explain_batch(self, X):
        """Compute SHAP values for multiple instances."""
        return np.array([self.explain(x) for x in X])


class LIME:
    """
    Simplified LIME (Local Interpretable Model-agnostic Explanations).
    """
    
    def __init__(self, model, n_samples=1000, kernel_width=0.75):
        self.model = model
        self.n_samples = n_samples
        self.kernel_width = kernel_width
    
    def explain(self, x, feature_names=None):
        """
        Explain a single prediction using local linear approximation.
        """
        from sklearn.linear_model import Ridge
        
        n_features = len(x)
        
        # Generate perturbations
        perturbations = np.random.normal(0, 1, (self.n_samples, n_features))
        X_perturbed = x + perturbations * 0.1  # Scale perturbations
        
        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            y_pred = self.model.predict_proba(X_perturbed)[:, 1]
        else:
            y_pred = self.model.predict(X_perturbed)
        
        # Compute distances and weights
        distances = np.sqrt(np.sum((X_perturbed - x) ** 2, axis=1))
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))
        
        # Fit weighted linear model
        linear_model = Ridge(alpha=1.0)
        linear_model.fit(X_perturbed, y_pred, sample_weight=weights)
        
        # Feature importances are coefficients
        importances = linear_model.coef_
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]
        
        explanation = dict(zip(feature_names, importances))
        
        return explanation, linear_model.intercept_


# Example usage
print("\nModel Interpretability Methods:")
print("=" * 60)
print("""
1. Permutation Importance:
   - Shuffle feature, measure performance drop
   - Model-agnostic
   - Can detect feature interactions

2. SHAP (Shapley Values):
   - Game-theoretic approach
   - Additive feature attributions
   - Satisfies important properties (consistency, efficiency)

3. LIME (Local Interpretable Model-agnostic Explanations):
   - Local linear approximation
   - Generates human-interpretable explanations
   - Works for any black-box model

4. Partial Dependence Plots:
   - Show marginal effect of feature
   - Good for understanding non-linear relationships
""")
```

### 50.2 Attention Visualization

```python
import matplotlib.pyplot as plt

def visualize_attention(attention_weights, tokens, layer=0, head=0):
    """
    Visualize attention weights as heatmap.
    
    Args:
        attention_weights: (layers, heads, seq_len, seq_len)
        tokens: List of token strings
    """
    weights = attention_weights[layer, head].detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(weights, cmap='viridis')
    
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)
    
    ax.set_xlabel('Key Tokens')
    ax.set_ylabel('Query Tokens')
    ax.set_title(f'Attention Weights (Layer {layer}, Head {head})')
    
    plt.colorbar(im)
    plt.tight_layout()
    
    return fig


def visualize_token_importance(tokens, importance_scores, title='Token Importance'):
    """
    Visualize token importance as colored text or bar chart.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    colors = plt.cm.RdYlGn(importance_scores)
    
    for i, (token, score) in enumerate(zip(tokens, importance_scores)):
        ax.bar(i, score, color=colors[i])
        ax.text(i, score + 0.02, token, ha='center', rotation=45)
    
    ax.set_ylabel('Importance Score')
    ax.set_title(title)
    ax.set_xticks([])
    
    return fig


class GradCAM:
    """
    Grad-CAM: Visual explanations for CNNs.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap.
        """
        import torch
        
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # ReLU and normalize
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        import torch.nn.functional as F
        cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)
        
        return cam.squeeze().cpu().numpy()


print("\nVisualization Methods for Deep Learning:")
print("=" * 60)
print("""
ATTENTION VISUALIZATION:
- Heatmaps showing attention weights
- Token importance scores
- Head-by-head analysis

GRAD-CAM:
- Highlights important image regions
- Uses gradients flowing into final conv layer
- Works for any CNN

SALIENCY MAPS:
- Gradient of output w.r.t. input
- Shows which pixels affect prediction

ACTIVATION MAXIMIZATION:
- Generate input that maximizes neuron activation
- Reveals what features a neuron detects
""")
```

---

## Chapter 51: Privacy in Machine Learning

### 51.1 Differential Privacy

```python
import numpy as np

class DifferentialPrivacy:
    """
    Differential Privacy mechanisms for ML.
    """
    
    @staticmethod
    def laplace_mechanism(true_value, sensitivity, epsilon):
        """
        Laplace Mechanism: Add Laplace noise for ε-differential privacy.
        
        Args:
            true_value: True query result
            sensitivity: Maximum change in result from single record
            epsilon: Privacy budget (smaller = more private)
        """
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        return true_value + noise
    
    @staticmethod
    def gaussian_mechanism(true_value, sensitivity, epsilon, delta=1e-5):
        """
        Gaussian Mechanism: Add Gaussian noise for (ε,δ)-differential privacy.
        """
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise = np.random.normal(0, sigma)
        return true_value + noise
    
    @staticmethod
    def randomized_response(true_value, p=0.75):
        """
        Randomized Response: Plausible deniability for sensitive questions.
        
        With probability p: report true value
        With probability 1-p: report random value
        """
        if np.random.random() < p:
            return true_value
        else:
            return np.random.randint(0, 2)


class DPSGDOptimizer:
    """
    Differentially Private Stochastic Gradient Descent.
    """
    
    def __init__(self, model, lr=0.01, noise_multiplier=1.0, max_grad_norm=1.0):
        self.model = model
        self.lr = lr
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
    
    def step(self, loss, batch_size):
        """
        DP-SGD step:
        1. Compute per-sample gradients
        2. Clip gradients
        3. Add noise
        4. Update parameters
        """
        import torch
        
        # Compute gradients
        loss.backward()
        
        # Clip gradients
        total_norm = 0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        # Add noise
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    0, self.noise_multiplier * self.max_grad_norm / batch_size,
                    size=param.grad.shape
                )
                param.grad.data.add_(noise)
        
        # Update parameters
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    param.data -= self.lr * param.grad.data


class PrivacyAccountant:
    """
    Track privacy budget usage.
    """
    
    def __init__(self, epsilon_budget, delta=1e-5):
        self.epsilon_budget = epsilon_budget
        self.delta = delta
        self.epsilon_spent = 0
        self.queries = []
    
    def spend(self, epsilon, query_name=''):
        """Record privacy spending."""
        self.epsilon_spent += epsilon
        self.queries.append({
            'name': query_name,
            'epsilon': epsilon,
            'cumulative': self.epsilon_spent
        })
        
        if self.epsilon_spent > self.epsilon_budget:
            print(f"WARNING: Privacy budget exceeded! ({self.epsilon_spent:.4f} > {self.epsilon_budget})")
        
        return self.epsilon_spent <= self.epsilon_budget
    
    def remaining_budget(self):
        """Return remaining privacy budget."""
        return max(0, self.epsilon_budget - self.epsilon_spent)
    
    def report(self):
        """Print privacy spending report."""
        print("\nPrivacy Budget Report")
        print("=" * 50)
        print(f"Total budget: ε = {self.epsilon_budget}")
        print(f"Spent: ε = {self.epsilon_spent:.4f}")
        print(f"Remaining: ε = {self.remaining_budget():.4f}")
        print("\nQuery History:")
        for q in self.queries:
            print(f"  {q['name']}: ε = {q['epsilon']:.4f} (cumulative: {q['cumulative']:.4f})")


# Example
print("\nDifferential Privacy Example:")
print("=" * 60)

# True average salary
true_avg = 75000
sensitivity = 200000 / 1000  # Max salary / n

accountant = PrivacyAccountant(epsilon_budget=1.0)

# Multiple queries with noise
for i in range(3):
    epsilon = 0.3
    if accountant.spend(epsilon, f'Query {i+1}'):
        private_avg = DifferentialPrivacy.laplace_mechanism(true_avg, sensitivity, epsilon)
        print(f"Query {i+1}: Private average = ${private_avg:,.0f} (ε = {epsilon})")

accountant.report()
```

### 51.2 Federated Learning

```python
import torch
import torch.nn as nn
import copy

class FederatedLearning:
    """
    Simplified Federated Learning implementation.
    """
    
    def __init__(self, global_model, num_clients=10):
        self.global_model = global_model
        self.num_clients = num_clients
        self.client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
    
    def distribute_model(self):
        """Send global model to all clients."""
        for client_model in self.client_models:
            client_model.load_state_dict(self.global_model.state_dict())
    
    def client_update(self, client_id, data_loader, epochs=1, lr=0.01):
        """
        Train model on client's local data.
        """
        model = self.client_models[client_id]
        model.train()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for data, target in data_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        return model.state_dict()
    
    def federated_averaging(self, client_weights=None):
        """
        FedAvg: Average client model parameters.
        
        Args:
            client_weights: Optional weights for each client (e.g., by data size)
        """
        if client_weights is None:
            client_weights = [1.0 / self.num_clients] * self.num_clients
        
        # Initialize averaged state dict
        avg_state_dict = {}
        
        for key in self.global_model.state_dict().keys():
            avg_state_dict[key] = torch.zeros_like(
                self.global_model.state_dict()[key], dtype=torch.float32
            )
            
            for client_id, weight in enumerate(client_weights):
                avg_state_dict[key] += weight * self.client_models[client_id].state_dict()[key].float()
        
        self.global_model.load_state_dict(avg_state_dict)
    
    def train_round(self, client_data_loaders, local_epochs=1, lr=0.01, 
                   participation_rate=1.0):
        """
        Execute one round of federated training.
        """
        # 1. Distribute global model
        self.distribute_model()
        
        # 2. Select participating clients
        num_participating = max(1, int(self.num_clients * participation_rate))
        participating_clients = np.random.choice(
            self.num_clients, num_participating, replace=False
        )
        
        # 3. Client updates
        for client_id in participating_clients:
            if client_id < len(client_data_loaders):
                self.client_update(
                    client_id, 
                    client_data_loaders[client_id],
                    epochs=local_epochs,
                    lr=lr
                )
        
        # 4. Aggregate
        weights = [1.0 / num_participating if i in participating_clients else 0 
                  for i in range(self.num_clients)]
        self.federated_averaging(weights)
        
        return self.global_model


class SecureAggregation:
    """
    Simplified secure aggregation for federated learning.
    """
    
    @staticmethod
    def add_masks(updates, seed=42):
        """
        Add pairwise masks that cancel out in aggregation.
        """
        np.random.seed(seed)
        n = len(updates)
        
        masked_updates = [u.copy() for u in updates]
        
        for i in range(n):
            for j in range(i + 1, n):
                # Generate random mask
                mask_ij = np.random.randn(*updates[i].shape)
                
                # Add to i, subtract from j
                masked_updates[i] += mask_ij
                masked_updates[j] -= mask_ij
        
        return masked_updates
    
    @staticmethod
    def aggregate(masked_updates):
        """
        Aggregate masked updates (masks cancel out).
        """
        return np.mean(masked_updates, axis=0)


print("\nFederated Learning Summary:")
print("=" * 60)
print("""
FEDERATED LEARNING PROCESS:
1. Server sends global model to clients
2. Clients train on local data
3. Clients send updates (not data) to server
4. Server aggregates updates (FedAvg)
5. Repeat

PRIVACY BENEFITS:
- Raw data stays on device
- Only model updates shared
- Can combine with differential privacy

CHALLENGES:
- Non-IID data distribution
- Communication efficiency
- Client dropout
- System heterogeneity
""")
```

---

## Summary: Responsible AI

```
┌─────────────────────────────────────────────────────────────────────┐
│              RESPONSIBLE AI SUMMARY                                 │
├─────────────────────────────────────────────────────────────────────┤
│  FAIRNESS                                                           │
│  ├── Metrics: Demographic parity, equalized odds, etc.            │
│  ├── Pre-processing: Reweighing, representation                   │
│  ├── In-processing: Fairness constraints, adversarial             │
│  └── Post-processing: Threshold adjustment                         │
│                                                                     │
│  INTERPRETABILITY                                                   │
│  ├── Feature importance: Permutation, SHAP, LIME                  │
│  ├── Model-specific: Attention, Grad-CAM                          │
│  └── Global vs Local explanations                                  │
│                                                                     │
│  PRIVACY                                                            │
│  ├── Differential Privacy: Noise mechanisms                        │
│  ├── Federated Learning: Decentralized training                   │
│  └── Secure Aggregation: Cryptographic protection                  │
│                                                                     │
│  BEST PRACTICES                                                     │
│  ├── Document data and model choices                               │
│  ├── Test across demographic groups                                │
│  ├── Provide explanations for decisions                            │
│  └── Regular audits and monitoring                                 │
└─────────────────────────────────────────────────────────────────────┘
```


---

<div align="center">

[⬅️ Previous: Advanced Topics](11-advanced-topics.md) | [📚 Table of Contents](../README.md) | [Next: Optimization Deep Dive ➡️](13-optimization.md)

</div>

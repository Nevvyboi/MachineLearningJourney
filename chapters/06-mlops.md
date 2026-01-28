<div align="center">

# 🚀 MLOps & Deployment

![Chapter](https://img.shields.io/badge/Chapter-06-blue?style=for-the-badge)
![Topic](https://img.shields.io/badge/Topic-Production%20%7C%20CI/CD-green?style=for-the-badge)

*Model Serving, Monitoring, Docker & Kubernetes*

---

</div>

# Part IX: MLOps and Model Deployment

---

## Chapter 29: The ML Lifecycle

### 29.1 From Notebook to Production

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ML PROJECT LIFECYCLE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Problem Definition                                              │
│     └── Define success metrics, constraints, requirements          │
│                                                                     │
│  2. Data Collection & Preparation                                   │
│     └── Gather, clean, label, version data                         │
│                                                                     │
│  3. Feature Engineering                                             │
│     └── Create, select, transform features                         │
│                                                                     │
│  4. Model Development                                               │
│     └── Train, tune, validate models                               │
│                                                                     │
│  5. Model Evaluation                                                │
│     └── Test on held-out data, A/B testing                         │
│                                                                     │
│  6. Deployment                                                      │
│     └── Package, serve, integrate with systems                     │
│                                                                     │
│  7. Monitoring & Maintenance                                        │
│     └── Track performance, detect drift, retrain                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 29.2 Experiment Tracking

```python
import json
import hashlib
from datetime import datetime
from pathlib import Path

class ExperimentTracker:
    """Simple experiment tracking system."""
    
    def __init__(self, experiment_name, base_dir="experiments"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / experiment_name / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {}
        self.params = {}
        self.artifacts = []
        
    def log_params(self, params):
        """Log hyperparameters."""
        self.params.update(params)
        
    def log_metric(self, name, value, step=None):
        """Log a metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })
        
    def log_artifact(self, filepath, artifact_name=None):
        """Log a file artifact."""
        self.artifacts.append({
            'path': str(filepath),
            'name': artifact_name or Path(filepath).name
        })
        
    def save(self):
        """Save experiment data."""
        experiment_data = {
            'experiment_name': self.experiment_name,
            'run_id': self.run_id,
            'params': self.params,
            'metrics': self.metrics,
            'artifacts': self.artifacts,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.run_dir / 'experiment.json', 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        print(f"Experiment saved to {self.run_dir}")


# Example usage
tracker = ExperimentTracker("random_forest_classifier")

tracker.log_params({
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'random_state': 42
})

# Simulate training
for epoch in range(10):
    train_loss = 1.0 / (epoch + 1)
    val_accuracy = 0.7 + 0.03 * epoch
    
    tracker.log_metric('train_loss', train_loss, step=epoch)
    tracker.log_metric('val_accuracy', val_accuracy, step=epoch)

tracker.save()
```

### 29.3 Model Versioning

```python
import pickle
import hashlib
from datetime import datetime

class ModelRegistry:
    """Simple model versioning and registry."""
    
    def __init__(self, registry_dir="model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"
        
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                self.registry = json.load(f)
        else:
            self.registry = {'models': {}}
    
    def _compute_hash(self, model):
        """Compute hash of model for versioning."""
        model_bytes = pickle.dumps(model)
        return hashlib.md5(model_bytes).hexdigest()[:8]
    
    def register_model(self, model, name, metrics, tags=None):
        """Register a new model version."""
        model_hash = self._compute_hash(model)
        version = f"v{len(self.registry['models'].get(name, [])) + 1}"
        
        model_info = {
            'version': version,
            'hash': model_hash,
            'metrics': metrics,
            'tags': tags or [],
            'timestamp': datetime.now().isoformat(),
            'path': str(self.registry_dir / f"{name}_{version}.pkl")
        }
        
        # Save model
        with open(model_info['path'], 'wb') as f:
            pickle.dump(model, f)
        
        # Update registry
        if name not in self.registry['models']:
            self.registry['models'][name] = []
        self.registry['models'][name].append(model_info)
        
        # Save registry
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
        
        print(f"Registered {name} {version}")
        return model_info
    
    def load_model(self, name, version='latest'):
        """Load a model by name and version."""
        if name not in self.registry['models']:
            raise ValueError(f"Model {name} not found")
        
        versions = self.registry['models'][name]
        
        if version == 'latest':
            model_info = versions[-1]
        else:
            model_info = next((v for v in versions if v['version'] == version), None)
            if model_info is None:
                raise ValueError(f"Version {version} not found")
        
        with open(model_info['path'], 'rb') as f:
            model = pickle.load(f)
        
        return model, model_info
    
    def list_models(self):
        """List all registered models."""
        for name, versions in self.registry['models'].items():
            print(f"\n{name}:")
            for v in versions:
                print(f"  {v['version']}: {v['metrics']} ({v['timestamp'][:10]})")


# Example
registry = ModelRegistry()

# Simulate registering models
class DummyModel:
    def __init__(self, accuracy):
        self.accuracy = accuracy

model1 = DummyModel(0.85)
registry.register_model(model1, "fraud_detector", {'accuracy': 0.85, 'f1': 0.82})

model2 = DummyModel(0.88)
registry.register_model(model2, "fraud_detector", {'accuracy': 0.88, 'f1': 0.85})

registry.list_models()
```

---

## Chapter 30: Model Serving

### 30.1 REST API with Flask

```python
Flask API for model serving.

File: app.py

from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model at startup
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions."""
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        
        prediction = model.predict(features)[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0].tolist()
        else:
            probabilities = None
        
        return jsonify({
            'prediction': int(prediction),
            'probabilities': probabilities
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Make batch predictions."""
    try:
        data = request.get_json()
        features = np.array(data['features'])
        
        predictions = model.predict(features).tolist()
        
        return jsonify({'predictions': predictions})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### 30.2 FastAPI (Modern Alternative)

```python
FastAPI for model serving - faster and with automatic docs.

File: main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pickle

app = FastAPI(
    title="ML Model API",
    description="API for serving machine learning predictions",
    version="1.0.0"
)

# Request/Response models
class PredictionRequest(BaseModel):
    features: List[float]
    
class BatchPredictionRequest(BaseModel):
    features: List[List[float]]

class PredictionResponse(BaseModel):
    prediction: int
    probabilities: Optional[List[float]] = None
    
class BatchPredictionResponse(BaseModel):
    predictions: List[int]

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = int(model.predict(features)[0])
        
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0].tolist()
        
        return PredictionResponse(
            prediction=prediction,
            probabilities=probabilities
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest):
    try:
        features = np.array(request.features)
        predictions = model.predict(features).tolist()
        
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run with: uvicorn main:app --reload
```

### 30.3 Docker Containerization

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/model.pkl
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## Chapter 31: Model Monitoring

### 31.1 Data Drift Detection

```python
import numpy as np
from scipy import stats

class DriftDetector:
    """Detect data drift in production."""
    
    def __init__(self, reference_data, feature_names=None):
        self.reference_data = np.array(reference_data)
        self.feature_names = feature_names or [f'feature_{i}' for i in range(reference_data.shape[1])]
        
        # Store reference statistics
        self.reference_stats = {
            'mean': np.mean(reference_data, axis=0),
            'std': np.std(reference_data, axis=0),
            'min': np.min(reference_data, axis=0),
            'max': np.max(reference_data, axis=0)
        }
    
    def detect_drift(self, production_data, method='ks', threshold=0.05):
        """
        Detect drift between reference and production data.
        
        Methods:
        - 'ks': Kolmogorov-Smirnov test
        - 'psi': Population Stability Index
        """
        production_data = np.array(production_data)
        drift_results = {}
        
        for i, feature in enumerate(self.feature_names):
            ref_col = self.reference_data[:, i]
            prod_col = production_data[:, i]
            
            if method == 'ks':
                statistic, p_value = stats.ks_2samp(ref_col, prod_col)
                is_drifted = p_value < threshold
                drift_results[feature] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'is_drifted': is_drifted
                }
            
            elif method == 'psi':
                psi = self._calculate_psi(ref_col, prod_col)
                is_drifted = psi > 0.2  # Common threshold
                drift_results[feature] = {
                    'psi': psi,
                    'is_drifted': is_drifted
                }
        
        return drift_results
    
    def _calculate_psi(self, reference, production, n_bins=10):
        """Calculate Population Stability Index."""
        # Bin the data
        bins = np.histogram_bin_edges(reference, bins=n_bins)
        
        ref_counts = np.histogram(reference, bins=bins)[0] / len(reference)
        prod_counts = np.histogram(production, bins=bins)[0] / len(production)
        
        # Avoid division by zero
        ref_counts = np.clip(ref_counts, 0.001, None)
        prod_counts = np.clip(prod_counts, 0.001, None)
        
        # Calculate PSI
        psi = np.sum((prod_counts - ref_counts) * np.log(prod_counts / ref_counts))
        
        return psi


# Example
np.random.seed(42)

# Reference data (training distribution)
reference = np.random.randn(1000, 3)

# Production data with drift in feature 0
production = np.random.randn(500, 3)
production[:, 0] += 0.5  # Add drift to first feature

detector = DriftDetector(reference, ['feature_a', 'feature_b', 'feature_c'])
results = detector.detect_drift(production, method='ks')

print("Drift Detection Results:")
print("=" * 50)
for feature, result in results.items():
    status = "DRIFT DETECTED" if result['is_drifted'] else "No drift"
    print(f"{feature}: {status} (p-value: {result['p_value']:.4f})")
```

### 31.2 Model Performance Monitoring

```python
from collections import deque
from datetime import datetime

class PerformanceMonitor:
    """Monitor model performance in production."""
    
    def __init__(self, window_size=1000, alert_threshold=0.1):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        self.baseline_accuracy = None
        self.alerts = []
    
    def set_baseline(self, accuracy):
        """Set baseline accuracy from validation."""
        self.baseline_accuracy = accuracy
    
    def log_prediction(self, prediction, actual=None):
        """Log a prediction and optionally the actual value."""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(datetime.now())
        
        # Check for alerts if we have enough data
        if actual is not None and len(self.actuals) >= 100:
            self._check_performance()
    
    def _check_performance(self):
        """Check if performance has degraded."""
        # Calculate current accuracy
        valid_pairs = [(p, a) for p, a in zip(self.predictions, self.actuals) if a is not None]
        
        if len(valid_pairs) < 50:
            return
        
        predictions, actuals = zip(*valid_pairs)
        current_accuracy = np.mean(np.array(predictions) == np.array(actuals))
        
        # Check against baseline
        if self.baseline_accuracy:
            degradation = self.baseline_accuracy - current_accuracy
            
            if degradation > self.alert_threshold:
                alert = {
                    'type': 'performance_degradation',
                    'timestamp': datetime.now().isoformat(),
                    'baseline_accuracy': self.baseline_accuracy,
                    'current_accuracy': current_accuracy,
                    'degradation': degradation
                }
                self.alerts.append(alert)
                print(f"ALERT: Performance degraded by {degradation:.2%}")
    
    def get_metrics(self):
        """Get current performance metrics."""
        valid_pairs = [(p, a) for p, a in zip(self.predictions, self.actuals) if a is not None]
        
        if not valid_pairs:
            return None
        
        predictions, actuals = zip(*valid_pairs)
        
        return {
            'accuracy': np.mean(np.array(predictions) == np.array(actuals)),
            'total_predictions': len(self.predictions),
            'labeled_predictions': len(valid_pairs),
            'recent_alerts': len([a for a in self.alerts if a['timestamp'] > (datetime.now().isoformat()[:10])])
        }


# Example
monitor = PerformanceMonitor()
monitor.set_baseline(0.90)

# Simulate predictions
np.random.seed(42)
for i in range(200):
    prediction = np.random.randint(0, 2)
    # Simulate some ground truth coming in with delay
    actual = prediction if np.random.random() > 0.2 else 1 - prediction  # 80% accuracy
    
    monitor.log_prediction(prediction, actual)

print("\nPerformance Metrics:")
print(monitor.get_metrics())
```

---

## Chapter 32: CI/CD for ML

### 32.1 ML Pipeline

```python
Example ML Pipeline using a simple orchestrator.

class MLPipeline:
    """Simple ML pipeline orchestrator."""
    
    def __init__(self, name):
        self.name = name
        self.steps = []
        self.artifacts = {}
        
    def add_step(self, name, function, inputs=None, outputs=None):
        """Add a step to the pipeline."""
        self.steps.append({
            'name': name,
            'function': function,
            'inputs': inputs or [],
            'outputs': outputs or []
        })
        return self
    
    def run(self, initial_data=None):
        """Execute the pipeline."""
        self.artifacts['initial_data'] = initial_data
        
        print(f"\n{'='*50}")
        print(f"Running Pipeline: {self.name}")
        print(f"{'='*50}\n")
        
        for step in self.steps:
            print(f"Step: {step['name']}")
            
            # Gather inputs
            inputs = {inp: self.artifacts.get(inp) for inp in step['inputs']}
            
            # Execute step
            try:
                result = step['function'](**inputs)
                
                # Store outputs
                if isinstance(result, dict):
                    for out in step['outputs']:
                        if out in result:
                            self.artifacts[out] = result[out]
                elif len(step['outputs']) == 1:
                    self.artifacts[step['outputs'][0]] = result
                
                print(f"  ✓ Completed")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                raise
        
        print(f"\n{'='*50}")
        print(f"Pipeline completed successfully!")
        print(f"{'='*50}")
        
        return self.artifacts


# Define pipeline steps
def load_data(initial_data):
    """Load and validate data."""
    print(f"  Loading {len(initial_data)} samples")
    return {'data': initial_data}

def preprocess(data):
    """Preprocess the data."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    X, y = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'scaler': scaler
    }

def train_model(X_train, y_train):
    """Train the model."""
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"  Model trained with {model.n_estimators} trees")
    
    return {'model': model}

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    from sklearn.metrics import accuracy_score, classification_report
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"  Accuracy: {accuracy:.4f}")
    
    return {'accuracy': accuracy, 'predictions': predictions}

# Build and run pipeline
pipeline = MLPipeline("training_pipeline")

pipeline.add_step(
    "load_data", load_data,
    inputs=['initial_data'],
    outputs=['data']
)

pipeline.add_step(
    "preprocess", preprocess,
    inputs=['data'],
    outputs=['X_train', 'X_test', 'y_train', 'y_test', 'scaler']
)

pipeline.add_step(
    "train_model", train_model,
    inputs=['X_train', 'y_train'],
    outputs=['model']
)

pipeline.add_step(
    "evaluate", evaluate_model,
    inputs=['model', 'X_test', 'y_test'],
    outputs=['accuracy', 'predictions']
)

# Run with sample data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
data = np.column_stack([X, y])

artifacts = pipeline.run(initial_data=data)
```

### 32.2 GitHub Actions for ML

```yaml
# .github/workflows/ml_pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  train:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v
    
    - name: Train model
      run: |
        python train.py
    
    - name: Evaluate model
      run: |
        python evaluate.py
    
    - name: Upload model artifact
      uses: actions/upload-artifact@v2
      with:
        name: model
        path: models/model.pkl
```

---

## Summary: MLOps Best Practices

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MLOPS CHECKLIST                                  │
├─────────────────────────────────────────────────────────────────────┤
│  EXPERIMENT TRACKING                                                │
│  □ Log all hyperparameters                                         │
│  □ Track metrics over training                                     │
│  □ Version datasets and models                                     │
│  □ Store experiment metadata                                       │
│                                                                     │
│  MODEL SERVING                                                      │
│  □ Create API endpoints                                            │
│  □ Add health checks                                               │
│  □ Handle errors gracefully                                        │
│  □ Containerize with Docker                                        │
│                                                                     │
│  MONITORING                                                         │
│  □ Detect data drift                                               │
│  □ Track prediction latency                                        │
│  □ Monitor model accuracy                                          │
│  □ Set up alerts                                                   │
│                                                                     │
│  CI/CD                                                              │
│  □ Automated testing                                               │
│  □ Model validation gates                                          │
│  □ Staged deployments                                              │
│  □ Rollback procedures                                             │
└─────────────────────────────────────────────────────────────────────┘
```


---

<div align="center">

[⬅️ Previous: Time Series Analysis](05-time-series.md) | [📚 Table of Contents](../README.md) | [Next: Appendices ➡️](07-appendices.md)

</div>

<div align="center">

# 🛠️ Practical Projects

![Chapter](https://img.shields.io/badge/Chapter-10-blue?style=for-the-badge)
![Topic](https://img.shields.io/badge/Topic-End%20to%20End-green?style=for-the-badge)

*Complete ML Project Implementations*

---

</div>

# Part XIII: Practical ML Projects and Case Studies

---

## Chapter 41: End-to-End ML Project Workflow

### 41.1 Project Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                   ML PROJECT STRUCTURE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ml_project/                                                        │
│  ├── data/                                                         │
│  │   ├── raw/              # Original data                         │
│  │   ├── processed/        # Cleaned data                          │
│  │   └── external/         # Third-party data                      │
│  ├── notebooks/            # Jupyter notebooks for exploration     │
│  ├── src/                                                          │
│  │   ├── data/             # Data loading and processing           │
│  │   ├── features/         # Feature engineering                   │
│  │   ├── models/           # Model definitions                     │
│  │   └── visualization/    # Plotting utilities                    │
│  ├── models/               # Saved model artifacts                 │
│  ├── configs/              # Configuration files                   │
│  ├── tests/                # Unit tests                            │
│  ├── requirements.txt                                              │
│  ├── setup.py                                                      │
│  └── README.md                                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 41.2 Configuration Management

```python
import yaml
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class DataConfig:
    """Data configuration."""
    train_path: str
    test_path: str
    val_split: float = 0.2
    random_state: int = 42

@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    
@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
@dataclass
class Config:
    """Full project configuration."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    experiment_name: str = "default"
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            experiment_name=config_dict.get('experiment_name', 'default')
        )
    
    def to_yaml(self, path: str):
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'experiment_name': self.experiment_name
        }
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Example config.yaml content
example_config = """
experiment_name: image_classification_v1

data:
  train_path: data/processed/train.csv
  test_path: data/processed/test.csv
  val_split: 0.2
  random_state: 42

model:
  name: resnet18
  hidden_dim: 256
  num_layers: 3
  dropout: 0.2

training:
  batch_size: 64
  learning_rate: 0.001
  num_epochs: 50
  early_stopping_patience: 10

print("Example Configuration:")
print(example_config)
```

### 41.3 Data Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
import pickle

class DataPipeline:
    """
    End-to-end data processing pipeline.
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = None
        self.target_column = None
        
    def load_data(self, path: str) -> pd.DataFrame:
        """Load data from various formats."""
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.parquet'):
            return pd.read_parquet(path)
        elif path.endswith('.json'):
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric with median
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical with mode
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features."""
        # Example: Date features
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        for col in date_cols:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df = df.drop(col, axis=1)
        
        return df
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col == self.target_column:
                continue
                
            if fit:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.encoders:
                    df[col] = self.encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numeric features."""
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c != self.target_column]
        
        if fit:
            self.scalers['standard'] = StandardScaler()
            df[numeric_cols] = self.scalers['standard'].fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scalers['standard'].transform(df[numeric_cols])
        
        return df
    
    def process(self, df: pd.DataFrame, target_col: str, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Full processing pipeline."""
        self.target_column = target_col
        
        df = self.clean_data(df)
        df = self.engineer_features(df)
        df = self.encode_features(df, fit=fit)
        df = self.scale_features(df, fit=fit)
        
        self.feature_columns = [c for c in df.columns if c != target_col]
        
        X = df[self.feature_columns].values
        y = df[target_col].values
        
        return X, y
    
    def save(self, path: str):
        """Save pipeline state."""
        state = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str):
        """Load pipeline state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.scalers = state['scalers']
        self.encoders = state['encoders']
        self.feature_columns = state['feature_columns']
        self.target_column = state['target_column']


# Example usage
print("\nData Pipeline Example:")
print("=" * 50)
print("""
pipeline = DataPipeline(config.data)

# Training
train_df = pipeline.load_data('data/train.csv')
X_train, y_train = pipeline.process(train_df, 'target', fit=True)
pipeline.save('models/pipeline.pkl')

# Inference
pipeline.load('models/pipeline.pkl')
test_df = pipeline.load_data('data/test.csv')
X_test, y_test = pipeline.process(test_df, 'target', fit=False)
""")
```

---

## Chapter 42: Case Study - Image Classification

### 42.1 Complete Image Classification Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm

class ImageDataset(Dataset):
    """Custom dataset for image classification."""
    
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load image paths and labels."""
        classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        for cls in classes:
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ImageClassifier:
    """Complete image classification system."""
    
    def __init__(self, num_classes, model_name='resnet18', pretrained=True):
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained model
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == 'efficientnet':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            self.model.classifier[1] = nn.Linear(
                self.model.classifier[1].in_features, num_classes
            )
        
        self.model = self.model.to(self.device)
        
        # Training transforms
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Validation transforms
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def train(self, train_loader, val_loader, epochs=10, lr=1e-3):
        """Train the model."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        
        best_val_acc = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
        
        return history
    
    def evaluate(self, data_loader):
        """Evaluate the model."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(data_loader), correct / total
    
    def predict(self, image):
        """Predict class for a single image."""
        self.model.eval()
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        image = self.val_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
        
        return predicted.item(), probabilities[0].cpu().numpy()
    
    def save(self, path):
        """Save model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }, path)
    
    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


# Training script example
print("\nImage Classification Training Script:")
print("=" * 50)
print("""
# Create datasets
train_dataset = ImageDataset('data/train', transform=classifier.train_transform)
val_dataset = ImageDataset('data/val', transform=classifier.val_transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Initialize classifier
classifier = ImageClassifier(num_classes=10, model_name='resnet18', pretrained=True)

# Train
history = classifier.train(train_loader, val_loader, epochs=20, lr=1e-3)

# Evaluate
test_dataset = ImageDataset('data/test', transform=classifier.val_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
test_loss, test_acc = classifier.evaluate(test_loader)
print(f'Test Accuracy: {test_acc:.4f}')

# Save model
classifier.save('models/classifier.pth')
""")
```

---

## Chapter 43: Case Study - NLP Sentiment Analysis

### 43.1 Text Classification Pipeline

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re

class TextDataset(Dataset):
    """Dataset for text classification."""
    
    def __init__(self, texts, labels, vocab=None, max_len=256):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        
        if vocab is None:
            self.vocab = self._build_vocab(texts)
        else:
            self.vocab = vocab
    
    def _build_vocab(self, texts, min_freq=2):
        """Build vocabulary from texts."""
        counter = Counter()
        
        for text in texts:
            tokens = self._tokenize(text)
            counter.update(tokens)
        
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        for word, freq in counter.most_common():
            if freq >= min_freq:
                vocab[word] = len(vocab)
        
        return vocab
    
    def _tokenize(self, text):
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def _encode(self, text):
        """Convert text to indices."""
        tokens = self._tokenize(text)
        indices = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        
        # Pad or truncate
        if len(indices) < self.max_len:
            indices = indices + [self.vocab['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return indices
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoded = self._encode(text)
        
        return torch.tensor(encoded), torch.tensor(label)


class LSTMClassifier(nn.Module):
    """LSTM-based text classifier."""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Concatenate last forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        out = self.dropout(hidden)
        out = self.fc(out)
        
        return out


class TransformerClassifier(nn.Module):
    """Transformer-based text classifier."""
    
    def __init__(self, vocab_size, embedding_dim=256, num_heads=8,
                 num_layers=4, num_classes=2, max_len=256, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, embedding_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Create padding mask
        if mask is None:
            mask = (x == 0)
        
        # Embedding + positional encoding
        embedded = self.embedding(x) + self.pos_encoding[:, :x.size(1)]
        
        # Transformer encoding
        encoded = self.transformer(embedded, src_key_padding_mask=mask)
        
        # Use [CLS] token (first position) or mean pooling
        pooled = encoded.mean(dim=1)
        
        out = self.dropout(pooled)
        out = self.fc(out)
        
        return out


class SentimentAnalyzer:
    """Complete sentiment analysis system."""
    
    def __init__(self, model_type='lstm', vocab_size=10000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab = None
        self.model_type = model_type
        
        if model_type == 'lstm':
            self.model = LSTMClassifier(vocab_size)
        else:
            self.model = TransformerClassifier(vocab_size)
        
        self.model = self.model.to(self.device)
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None,
              epochs=10, batch_size=32, lr=1e-3):
        """Train the model."""
        # Create dataset
        train_dataset = TextDataset(train_texts, train_labels)
        self.vocab = train_dataset.vocab
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_texts is not None:
            val_dataset = TextDataset(val_texts, val_labels, vocab=self.vocab)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Update model vocab size
        if self.model_type == 'lstm':
            self.model = LSTMClassifier(len(self.vocab))
        else:
            self.model = TransformerClassifier(len(self.vocab))
        self.model = self.model.to(self.device)
        
        # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for texts, labels in train_loader:
                texts = texts.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            if val_texts is not None:
                val_acc = self.evaluate(val_loader)
                print(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                print(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}')
    
    def evaluate(self, data_loader):
        """Evaluate the model."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for texts, labels in data_loader:
                texts = texts.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(texts)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return correct / total
    
    def predict(self, text):
        """Predict sentiment for a single text."""
        self.model.eval()
        
        dataset = TextDataset([text], [0], vocab=self.vocab)
        encoded, _ = dataset[0]
        encoded = encoded.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(encoded)
            probs = torch.softmax(output, dim=1)
            _, predicted = output.max(1)
        
        sentiment = 'Positive' if predicted.item() == 1 else 'Negative'
        confidence = probs[0][predicted.item()].item()
        
        return sentiment, confidence


# Example usage
print("\nSentiment Analysis Example:")
print("=" * 50)
print("""
# Sample data
texts = [
    "This movie was absolutely amazing! I loved every minute.",
    "Terrible film. Complete waste of time.",
    "Great acting and beautiful cinematography.",
    "Boring and predictable. Would not recommend.",
]
labels = [1, 0, 1, 0]  # 1=positive, 0=negative

# Train
analyzer = SentimentAnalyzer(model_type='lstm')
analyzer.train(texts, labels, epochs=10)

# Predict
sentiment, confidence = analyzer.predict("This is a wonderful movie!")
print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")
""")
```

---

## Chapter 44: Case Study - Time Series Forecasting

### 44.1 Complete Forecasting Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""
    
    def __init__(self, data, seq_length=30, horizon=1):
        self.data = data
        self.seq_length = seq_length
        self.horizon = horizon
    
    def __len__(self):
        return len(self.data) - self.seq_length - self.horizon + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length:idx+self.seq_length+self.horizon]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class LSTMForecaster(nn.Module):
    """LSTM model for time series forecasting."""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 output_size=1, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class TimeSeriesForecaster:
    """Complete time series forecasting system."""
    
    def __init__(self, seq_length=30, horizon=1):
        self.seq_length = seq_length
        self.horizon = horizon
        self.scaler = MinMaxScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_data(self, data, train_ratio=0.8):
        """Prepare data for training."""
        # Scale data
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1))
        
        # Split
        train_size = int(len(data_scaled) * train_ratio)
        train_data = data_scaled[:train_size]
        test_data = data_scaled[train_size:]
        
        return train_data, test_data
    
    def train(self, train_data, epochs=100, batch_size=32, lr=1e-3):
        """Train the forecasting model."""
        # Create dataset and loader
        dataset = TimeSeriesDataset(train_data, self.seq_length, self.horizon)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.model = LSTMForecaster(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=self.horizon
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        history = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y.squeeze(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            history.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}: Loss: {avg_loss:.6f}')
        
        return history
    
    def predict(self, data, steps=1):
        """Generate predictions."""
        self.model.eval()
        
        predictions = []
        current_seq = data[-self.seq_length:].copy()
        
        for _ in range(steps):
            x = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred = self.model(x)
            
            pred_value = pred.cpu().numpy()[0]
            predictions.append(pred_value)
            
            # Update sequence
            current_seq = np.roll(current_seq, -self.horizon)
            current_seq[-self.horizon:] = pred_value.reshape(-1, 1)
        
        # Inverse transform
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def evaluate(self, test_data, actual_values):
        """Evaluate forecasting performance."""
        predictions = self.predict(test_data, steps=len(actual_values))
        
        mae = mean_absolute_error(actual_values, predictions)
        rmse = np.sqrt(mean_squared_error(actual_values, predictions))
        mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'predictions': predictions
        }


# Example usage
print("\nTime Series Forecasting Example:")
print("=" * 50)
print("""
# Generate sample data
np.random.seed(42)
t = np.arange(0, 365)
trend = 0.1 * t
seasonal = 10 * np.sin(2 * np.pi * t / 30)
noise = np.random.randn(len(t)) * 2
data = 50 + trend + seasonal + noise

# Create forecaster
forecaster = TimeSeriesForecaster(seq_length=30, horizon=1)

# Prepare data
train_data, test_data = forecaster.prepare_data(data, train_ratio=0.8)

# Train
history = forecaster.train(train_data, epochs=100)

# Forecast next 30 days
predictions = forecaster.predict(train_data, steps=30)

# Evaluate
results = forecaster.evaluate(test_data[:30], data[-30:])
print(f"MAE: {results['MAE']:.2f}")
print(f"RMSE: {results['RMSE']:.2f}")
print(f"MAPE: {results['MAPE']:.2f}%")
""")
```

---

## Chapter 45: Case Study - Recommendation System

### 45.1 Collaborative Filtering

```python
class MatrixFactorization(nn.Module):
    """Matrix Factorization for collaborative filtering."""
    
    def __init__(self, num_users, num_items, embedding_dim=50):
        super().__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Dot product
        dot_product = (user_emb * item_emb).sum(dim=1)
        
        # Add biases
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()
        
        prediction = dot_product + user_b + item_b + self.global_bias
        
        return prediction


class NeuralCollaborativeFiltering(nn.Module):
    """Neural Collaborative Filtering (NCF)."""
    
    def __init__(self, num_users, num_items, embedding_dim=50, hidden_dims=[64, 32]):
        super().__init__()
        
        # GMF embeddings
        self.gmf_user = nn.Embedding(num_users, embedding_dim)
        self.gmf_item = nn.Embedding(num_items, embedding_dim)
        
        # MLP embeddings
        self.mlp_user = nn.Embedding(num_users, embedding_dim)
        self.mlp_item = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        mlp_layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Output layer
        self.output = nn.Linear(embedding_dim + hidden_dims[-1], 1)
    
    def forward(self, user_ids, item_ids):
        # GMF part
        gmf_user_emb = self.gmf_user(user_ids)
        gmf_item_emb = self.gmf_item(item_ids)
        gmf_output = gmf_user_emb * gmf_item_emb
        
        # MLP part
        mlp_user_emb = self.mlp_user(user_ids)
        mlp_item_emb = self.mlp_item(item_ids)
        mlp_input = torch.cat([mlp_user_emb, mlp_item_emb], dim=1)
        mlp_output = self.mlp(mlp_input)
        
        # Combine
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        prediction = self.output(combined).squeeze()
        
        return prediction


class RecommenderSystem:
    """Complete recommendation system."""
    
    def __init__(self, num_users, num_items, model_type='mf'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 'mf':
            self.model = MatrixFactorization(num_users, num_items)
        else:
            self.model = NeuralCollaborativeFiltering(num_users, num_items)
        
        self.model = self.model.to(self.device)
        self.num_items = num_items
    
    def train(self, user_ids, item_ids, ratings, epochs=20, batch_size=256, lr=1e-3):
        """Train the recommender."""
        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(user_ids),
            torch.LongTensor(item_ids),
            torch.FloatTensor(ratings)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for users, items, targets in loader:
                users = users.to(self.device)
                items = items.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(users, items)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            print(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}')
    
    def recommend(self, user_id, top_k=10, exclude_items=None):
        """Generate recommendations for a user."""
        self.model.eval()
        
        user_tensor = torch.LongTensor([user_id] * self.num_items).to(self.device)
        item_tensor = torch.LongTensor(range(self.num_items)).to(self.device)
        
        with torch.no_grad():
            scores = self.model(user_tensor, item_tensor)
        
        scores = scores.cpu().numpy()
        
        if exclude_items is not None:
            scores[exclude_items] = -np.inf
        
        top_items = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_items]
        
        return list(zip(top_items, top_scores))


print("\nRecommendation System Example:")
print("=" * 50)
print("""
# Sample data
user_ids = [0, 0, 0, 1, 1, 2, 2, 2]
item_ids = [0, 1, 2, 1, 3, 0, 2, 4]
ratings = [5.0, 3.0, 4.0, 4.0, 2.0, 5.0, 4.0, 3.0]

# Create recommender
recommender = RecommenderSystem(num_users=3, num_items=5, model_type='ncf')

# Train
recommender.train(user_ids, item_ids, ratings, epochs=50)

# Get recommendations
recommendations = recommender.recommend(user_id=0, top_k=3, exclude_items=[0, 1, 2])
print("Top recommendations for user 0:")
for item_id, score in recommendations:
    print(f"  Item {item_id}: score = {score:.2f}")
""")
```

---

## Summary: ML Project Best Practices

```
┌─────────────────────────────────────────────────────────────────────┐
│              ML PROJECT BEST PRACTICES                              │
├─────────────────────────────────────────────────────────────────────┤
│  PROJECT SETUP                                                      │
│  ├── Use clear directory structure                                 │
│  ├── Configuration files for reproducibility                       │
│  ├── Version control for code AND data                            │
│  └── Document everything                                           │
│                                                                     │
│  DATA PIPELINE                                                      │
│  ├── Automate data loading and cleaning                           │
│  ├── Version your preprocessing steps                              │
│  ├── Validate data quality                                         │
│  └── Handle missing data consistently                              │
│                                                                     │
│  MODEL DEVELOPMENT                                                  │
│  ├── Start simple, add complexity as needed                        │
│  ├── Use cross-validation                                          │
│  ├── Track all experiments                                         │
│  └── Compare against baselines                                     │
│                                                                     │
│  EVALUATION                                                         │
│  ├── Use appropriate metrics for your task                         │
│  ├── Test on held-out data                                         │
│  ├── Analyze errors and failure cases                              │
│  └── Consider fairness and bias                                    │
│                                                                     │
│  DEPLOYMENT                                                         │
│  ├── Package model with preprocessing                              │
│  ├── Monitor performance in production                             │
│  ├── Plan for model updates                                        │
│  └── Document API and usage                                        │
└─────────────────────────────────────────────────────────────────────┘
```


---

<div align="center">

[⬅️ Previous: Reinforcement Learning](09-reinforcement-learning.md) | [📚 Table of Contents](../README.md) | [Next: Advanced Topics ➡️](11-advanced-topics.md)

</div>

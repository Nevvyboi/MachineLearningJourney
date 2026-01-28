<div align="center">

# 📈 Time Series Analysis

![Chapter](https://img.shields.io/badge/Chapter-05-blue?style=for-the-badge)
![Topic](https://img.shields.io/badge/Topic-Forecasting%20%7C%20ARIMA-green?style=for-the-badge)

*ARIMA, Prophet, LSTM & Forecasting Techniques*

---

</div>

# Part VIII: Time Series Analysis

---

## Chapter 25: Time Series Fundamentals

### 25.1 What is Time Series Data?

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TIME SERIES COMPONENTS                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Time Series = Trend + Seasonality + Residual                      │
│                                                                     │
│  TREND: Long-term increase or decrease                             │
│         ╱                                                          │
│        ╱                                                           │
│       ╱                                                            │
│      ╱                                                             │
│                                                                     │
│  SEASONALITY: Regular periodic patterns                            │
│       /\    /\    /\                                               │
│      /  \  /  \  /  \                                              │
│     /    \/    \/    \                                             │
│                                                                     │
│  RESIDUAL: Random noise after removing trend & seasonality         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 25.2 Time Series Visualization and Decomposition

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def generate_time_series(n_points=365, trend_slope=0.1, seasonal_period=30, noise_std=2):
    """Generate synthetic time series with trend, seasonality, and noise."""
    t = np.arange(n_points)
    
    # Trend component
    trend = trend_slope * t
    
    # Seasonal component
    seasonal = 10 * np.sin(2 * np.pi * t / seasonal_period)
    
    # Noise
    noise = np.random.normal(0, noise_std, n_points)
    
    # Combined series
    series = 50 + trend + seasonal + noise
    
    return t, series, trend, seasonal, noise


def decompose_time_series(series, period):
    """Simple additive decomposition."""
    n = len(series)
    
    # Extract trend using moving average
    trend = np.convolve(series, np.ones(period)/period, mode='same')
    
    # Detrend
    detrended = series - trend
    
    # Extract seasonality by averaging each position in the cycle
    seasonal = np.zeros(n)
    for i in range(period):
        indices = np.arange(i, n, period)
        seasonal[indices] = np.mean(detrended[indices])
    
    # Residual
    residual = series - trend - seasonal
    
    return trend, seasonal, residual


# Generate and decompose
np.random.seed(42)
t, series, true_trend, true_seasonal, true_noise = generate_time_series()

# Decompose
trend, seasonal, residual = decompose_time_series(series, period=30)

# Plot
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(t, series)
axes[0].set_ylabel('Original')
axes[0].set_title('Time Series Decomposition')

axes[1].plot(t, trend)
axes[1].set_ylabel('Trend')

axes[2].plot(t, seasonal)
axes[2].set_ylabel('Seasonal')

axes[3].plot(t, residual)
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Time')

plt.tight_layout()
plt.savefig('time_series_decomposition.png', dpi=150)
plt.show()
```

### 25.3 Stationarity and Differencing

```python
def adf_test_simple(series, max_lags=None):
    """
    Simplified Augmented Dickey-Fuller test concept.
    Tests null hypothesis that series has a unit root (non-stationary).
    """
    # In practice, use statsmodels.tsa.stattools.adfuller
    n = len(series)
    
    # First difference
    diff = np.diff(series)
    
    # Lag 1 of original series
    lag1 = series[:-1]
    
    # Simple regression: diff = alpha + beta * lag1 + error
    # If beta is significantly negative, series is stationary
    
    X = np.column_stack([np.ones(len(lag1)), lag1])
    beta = np.linalg.lstsq(X, diff, rcond=None)[0]
    
    # Calculate t-statistic for beta[1]
    residuals = diff - X @ beta
    mse = np.sum(residuals**2) / (len(diff) - 2)
    var_beta = mse * np.linalg.inv(X.T @ X)
    t_stat = beta[1] / np.sqrt(var_beta[1, 1])
    
    # Critical values (approximate for 5% significance)
    # Real ADF uses special critical values
    critical_value = -2.86
    
    is_stationary = t_stat < critical_value
    
    return t_stat, is_stationary


def make_stationary(series, max_diff=2):
    """Make series stationary through differencing."""
    diff_series = series.copy()
    n_diff = 0
    
    for i in range(max_diff):
        t_stat, is_stationary = adf_test_simple(diff_series)
        print(f"Differencing {i}: t-stat = {t_stat:.3f}, stationary = {is_stationary}")
        
        if is_stationary:
            break
        
        diff_series = np.diff(diff_series)
        n_diff += 1
    
    return diff_series, n_diff


# Example
print("Testing stationarity:")
_, stationary = make_stationary(series)
```

---

## Chapter 26: Classical Time Series Models

### 26.1 Moving Average and Exponential Smoothing

```python
def simple_moving_average(series, window):
    """Simple Moving Average."""
    return np.convolve(series, np.ones(window)/window, mode='valid')


def exponential_moving_average(series, alpha):
    """
    Exponential Moving Average.
    
    EMA_t = alpha * y_t + (1 - alpha) * EMA_{t-1}
    """
    ema = np.zeros(len(series))
    ema[0] = series[0]
    
    for t in range(1, len(series)):
        ema[t] = alpha * series[t] + (1 - alpha) * ema[t-1]
    
    return ema


class HoltWinters:
    """
    Holt-Winters Exponential Smoothing.
    
    Handles trend and seasonality.
    """
    
    def __init__(self, seasonal_period, alpha=0.2, beta=0.1, gamma=0.1):
        self.m = seasonal_period
        self.alpha = alpha  # Level smoothing
        self.beta = beta    # Trend smoothing
        self.gamma = gamma  # Seasonal smoothing
        
    def fit(self, series):
        """Fit the model."""
        n = len(series)
        
        # Initialize
        self.level = np.zeros(n)
        self.trend = np.zeros(n)
        self.seasonal = np.zeros(n)
        
        # Initial level: average of first period
        self.level[0] = np.mean(series[:self.m])
        
        # Initial trend: average change over first two periods
        if n >= 2 * self.m:
            self.trend[0] = (np.mean(series[self.m:2*self.m]) - np.mean(series[:self.m])) / self.m
        else:
            self.trend[0] = 0
        
        # Initial seasonal factors
        for i in range(self.m):
            self.seasonal[i] = series[i] / self.level[0] if self.level[0] != 0 else 1
        
        # Fit
        for t in range(1, n):
            s_idx = (t - 1) % self.m  # Previous seasonal index
            
            # Update level
            self.level[t] = self.alpha * (series[t] / self.seasonal[s_idx]) + \
                           (1 - self.alpha) * (self.level[t-1] + self.trend[t-1])
            
            # Update trend
            self.trend[t] = self.beta * (self.level[t] - self.level[t-1]) + \
                           (1 - self.beta) * self.trend[t-1]
            
            # Update seasonal
            self.seasonal[t] = self.gamma * (series[t] / self.level[t]) + \
                              (1 - self.gamma) * self.seasonal[s_idx]
        
        return self
    
    def forecast(self, steps):
        """Forecast future values."""
        n = len(self.level)
        forecasts = np.zeros(steps)
        
        last_level = self.level[-1]
        last_trend = self.trend[-1]
        
        for h in range(steps):
            s_idx = (n + h) % self.m
            forecasts[h] = (last_level + (h + 1) * last_trend) * self.seasonal[s_idx]
        
        return forecasts


# Example
hw = HoltWinters(seasonal_period=30)
hw.fit(series)

# Forecast
forecast = hw.forecast(60)

plt.figure(figsize=(12, 5))
plt.plot(t, series, label='Historical')
plt.plot(range(len(series), len(series) + len(forecast)), forecast, 'r--', label='Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Holt-Winters Forecast')
plt.legend()
plt.show()
```

### 26.2 ARIMA Models

```python
class ARIMA:
    """
    AutoRegressive Integrated Moving Average model.
    
    ARIMA(p, d, q):
    - p: Order of autoregressive part
    - d: Degree of differencing
    - q: Order of moving average part
    """
    
    def __init__(self, p, d, q):
        self.p = p  # AR order
        self.d = d  # Differencing order
        self.q = q  # MA order
        
        self.ar_params = None
        self.ma_params = None
        self.constant = None
        
    def _difference(self, series, d):
        """Apply differencing d times."""
        diff_series = series.copy()
        for _ in range(d):
            diff_series = np.diff(diff_series)
        return diff_series
    
    def _inverse_difference(self, diff_forecast, original_values, d):
        """Inverse differencing to get original scale."""
        forecast = diff_forecast.copy()
        for i in range(d):
            last_value = original_values[-(d-i)]
            forecast = np.cumsum(np.concatenate([[last_value], forecast]))[1:]
        return forecast
    
    def fit(self, series, learning_rate=0.01, max_iter=1000):
        """
        Fit ARIMA model using gradient descent (simplified).
        
        Note: Real implementations use maximum likelihood estimation.
        """
        # Apply differencing
        self.original = series.copy()
        diff_series = self._difference(series, self.d)
        
        n = len(diff_series)
        max_lag = max(self.p, self.q)
        
        # Initialize parameters
        np.random.seed(42)
        self.ar_params = np.random.randn(self.p) * 0.1
        self.ma_params = np.random.randn(self.q) * 0.1
        self.constant = np.mean(diff_series)
        
        # Store residuals for MA component
        residuals = np.zeros(n)
        
        # Simple gradient descent
        for iteration in range(max_iter):
            predictions = np.zeros(n)
            
            for t in range(max_lag, n):
                pred = self.constant
                
                # AR component
                for i in range(self.p):
                    pred += self.ar_params[i] * diff_series[t - i - 1]
                
                # MA component
                for j in range(self.q):
                    if t - j - 1 >= 0:
                        pred += self.ma_params[j] * residuals[t - j - 1]
                
                predictions[t] = pred
                residuals[t] = diff_series[t] - pred
            
            # Calculate loss
            loss = np.mean(residuals[max_lag:]**2)
            
            if iteration % 200 == 0:
                print(f"Iteration {iteration}, MSE: {loss:.4f}")
        
        self.residuals = residuals
        return self
    
    def forecast(self, steps):
        """Generate forecasts."""
        diff_series = self._difference(self.original, self.d)
        n = len(diff_series)
        
        forecasts = np.zeros(steps)
        residuals = np.concatenate([self.residuals, np.zeros(steps)])
        
        # Extend differenced series for forecasting
        extended = np.concatenate([diff_series, np.zeros(steps)])
        
        for h in range(steps):
            t = n + h
            pred = self.constant
            
            # AR component
            for i in range(self.p):
                pred += self.ar_params[i] * extended[t - i - 1]
            
            # MA component (use 0 for future residuals)
            for j in range(self.q):
                if t - j - 1 < n:
                    pred += self.ma_params[j] * residuals[t - j - 1]
            
            forecasts[h] = pred
            extended[t] = pred
        
        # Inverse differencing
        if self.d > 0:
            forecasts = self._inverse_difference(forecasts, self.original, self.d)
        
        return forecasts


# Example: ARIMA(2, 1, 1)
print("Fitting ARIMA(2, 1, 1) model...")
arima = ARIMA(p=2, d=1, q=1)
arima.fit(series)

forecast = arima.forecast(30)

plt.figure(figsize=(12, 5))
plt.plot(t, series, label='Historical')
plt.plot(range(len(series), len(series) + len(forecast)), forecast, 'r--', label='ARIMA Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()
```

### 26.3 Seasonal ARIMA (SARIMA)

```python
SARIMA(p, d, q)(P, D, Q, m)

Non-seasonal components:
- p: AR order
- d: Differencing order
- q: MA order

Seasonal components:
- P: Seasonal AR order
- D: Seasonal differencing order
- Q: Seasonal MA order
- m: Seasonal period

Example using statsmodels:

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit SARIMA model
model = SARIMAX(series, 
                order=(1, 1, 1),           # ARIMA order
                seasonal_order=(1, 1, 1, 12))  # Seasonal order (monthly data)

results = model.fit()
forecast = results.forecast(steps=12)

print("""
SARIMA Model Selection Guidelines:

1. Plot ACF and PACF to identify orders
   - ACF cuts off at q → MA(q)
   - PACF cuts off at p → AR(p)

2. Check for seasonality in plots

3. Use AIC/BIC for model comparison
   - Lower is better

4. Check residuals for white noise
   - No autocorrelation
   - Normally distributed

5. Common seasonal periods:
   - m=12 for monthly data with yearly seasonality
   - m=4 for quarterly data
   - m=7 for daily data with weekly seasonality
   - m=24 for hourly data with daily seasonality
""")
```

---

## Chapter 27: Deep Learning for Time Series

### 27.1 LSTM for Time Series

```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    """LSTM model for time series forecasting."""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        # Predict
        output = self.fc(last_output)
        return output


def create_sequences(data, seq_length, forecast_horizon=1):
    """Create sequences for supervised learning."""
    X, y = [], []
    
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+forecast_horizon])
    
    return np.array(X), np.array(y)


# Example usage
seq_length = 30
X, y = create_sequences(series, seq_length)

# Reshape for LSTM: (samples, seq_len, features)
X = X.reshape(-1, seq_length, 1)
y = y.reshape(-1, 1)

# Convert to tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Create model
model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2, output_size=1)

print("LSTM Forecaster Architecture:")
print(model)

# Training loop (simplified)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

### 27.2 Transformer for Time Series

```python
class TimeSeriesTransformer(nn.Module):
    """Transformer model for time series forecasting."""
    
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        x = self.transformer_encoder(x)
        
        # Use last position for prediction
        x = x[:, -1, :]
        output = self.fc(x)
        
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# Example
model = TimeSeriesTransformer(
    input_size=1,
    d_model=64,
    nhead=4,
    num_layers=2,
    output_size=1
)

print("\nTime Series Transformer Architecture:")
print(model)
```

### 27.3 TCN (Temporal Convolutional Network)

```python
class CausalConv1d(nn.Module):
    """Causal convolution - doesn't look into the future."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)
        
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding] if self.padding > 0 else x


class TCNBlock(nn.Module):
    """Single TCN block with residual connection."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.residual(x)
        
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        
        return self.relu(x + residual)


class TCN(nn.Module):
    """Temporal Convolutional Network."""
    
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2, output_size=1):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = x[:, :, -1]  # Last time step
        return self.fc(x)


# Example
model = TCN(input_size=1, num_channels=[32, 64, 64], kernel_size=3, output_size=1)
print("\nTCN Architecture:")
print(model)
```

---

## Chapter 28: Time Series Evaluation

### 28.1 Forecasting Metrics

```python
def forecast_metrics(y_true, y_pred):
    """Calculate common forecasting metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # Symmetric MAPE
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100
    
    # Mean Absolute Scaled Error (relative to naive forecast)
    naive_error = np.mean(np.abs(np.diff(y_true)))
    mase = mae / naive_error if naive_error > 0 else np.inf
    
    print("Forecast Evaluation Metrics:")
    print("=" * 50)
    print(f"MAE:   {mae:.4f}")
    print(f"MSE:   {mse:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAPE:  {mape:.2f}%")
    print(f"sMAPE: {smape:.2f}%")
    print(f"MASE:  {mase:.4f}")
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'smape': smape, 'mase': mase}


# Example
y_true = series[-30:]
y_pred = forecast[:30] if len(forecast) >= 30 else forecast

metrics = forecast_metrics(y_true[:len(y_pred)], y_pred)
```

### 28.2 Cross-Validation for Time Series

```python
def time_series_cv(series, model_fn, n_splits=5, test_size=30):
    """
    Time series cross-validation with expanding window.
    
    Unlike regular CV, we can't shuffle - must respect temporal order.
    """
    n = len(series)
    min_train_size = n - n_splits * test_size
    
    scores = []
    
    for i in range(n_splits):
        # Training data grows
        train_end = min_train_size + i * test_size
        test_end = train_end + test_size
        
        train = series[:train_end]
        test = series[train_end:test_end]
        
        # Fit model and forecast
        model = model_fn()
        model.fit(train)
        forecast = model.forecast(test_size)
        
        # Calculate score
        mse = np.mean((test - forecast) ** 2)
        scores.append(mse)
        
        print(f"Fold {i+1}: Train size = {train_end}, Test MSE = {mse:.4f}")
    
    print(f"\nMean MSE: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    return scores


print("""
Time Series Cross-Validation Methods:

1. Expanding Window (Walk-Forward):
   Train: [1...t], Test: [t+1...t+h]
   Train: [1...t+h], Test: [t+h+1...t+2h]
   
2. Sliding Window:
   Train: [1...t], Test: [t+1...t+h]
   Train: [h+1...t+h], Test: [t+h+1...t+2h]
   
3. Blocked Cross-Validation:
   Leave gap between train and test to avoid leakage
""")
```

---

## Summary: Time Series Methods

```
┌─────────────────────────────────────────────────────────────────────┐
│                 TIME SERIES METHODS SUMMARY                         │
├─────────────────────────────────────────────────────────────────────┤
│  CLASSICAL METHODS                                                  │
│  ├── Moving Average: Simple smoothing                              │
│  ├── Exponential Smoothing: Weighted recent values                 │
│  ├── Holt-Winters: Handles trend + seasonality                     │
│  ├── ARIMA: AR + I + MA components                                 │
│  └── SARIMA: ARIMA + seasonal terms                                │
│                                                                     │
│  DEEP LEARNING                                                      │
│  ├── LSTM/GRU: Sequential memory                                   │
│  ├── Transformer: Attention-based                                  │
│  └── TCN: Dilated causal convolutions                              │
│                                                                     │
│  KEY CONCEPTS                                                       │
│  ├── Stationarity: Required for many models                        │
│  ├── Differencing: Make series stationary                          │
│  ├── ACF/PACF: Identify model orders                               │
│  └── Walk-forward CV: Proper time series validation                │
└─────────────────────────────────────────────────────────────────────┘
```


---

<div align="center">

[⬅️ Previous: Natural Language Processing](04-nlp.md) | [📚 Table of Contents](../README.md) | [Next: MLOps & Deployment ➡️](06-mlops.md)

</div>

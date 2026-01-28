<div align="center">

# 📝 Natural Language Processing

![Chapter](https://img.shields.io/badge/Chapter-04-blue?style=for-the-badge)
![Topic](https://img.shields.io/badge/Topic-Text%20%7C%20Transformers-green?style=for-the-badge)

*Text Processing, Word Embeddings, BERT & GPT*

---

</div>

# Part VII: Natural Language Processing

---

## Chapter 20: Text Preprocessing

### 20.1 The NLP Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NLP PREPROCESSING PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Raw Text                                                          │
│      ↓                                                             │
│  Tokenization (split into words/sentences)                         │
│      ↓                                                             │
│  Lowercasing                                                       │
│      ↓                                                             │
│  Stop Word Removal                                                 │
│      ↓                                                             │
│  Stemming / Lemmatization                                          │
│      ↓                                                             │
│  Vectorization (convert to numbers)                                │
│      ↓                                                             │
│  Ready for ML Model                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 20.2 Tokenization

```python
import re
from collections import Counter

def simple_tokenize(text):
    """Basic word tokenization."""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def sentence_tokenize(text):
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# Example
text = "Natural Language Processing is amazing! It helps computers understand human language. NLP is used everywhere."

words = simple_tokenize(text)
sentences = sentence_tokenize(text)

print(f"Tokens: {words}")
print(f"Sentences: {sentences}")
```

### 20.3 Stop Words and Stemming

```python
# Common English stop words
STOP_WORDS = set([
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
    'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
])

def remove_stop_words(tokens):
    """Remove stop words from token list."""
    return [t for t in tokens if t not in STOP_WORDS]

class PorterStemmer:
    """Simplified Porter Stemmer implementation."""
    
    def __init__(self):
        self.vowels = set('aeiou')
        
    def _is_consonant(self, word, i):
        if word[i] in self.vowels:
            return False
        if word[i] == 'y':
            return i == 0 or not self._is_consonant(word, i - 1)
        return True
    
    def _measure(self, word):
        """Count VC sequences."""
        n = 0
        i = 0
        while i < len(word) and self._is_consonant(word, i):
            i += 1
        while i < len(word):
            while i < len(word) and not self._is_consonant(word, i):
                i += 1
            if i < len(word):
                n += 1
                while i < len(word) and self._is_consonant(word, i):
                    i += 1
        return n
    
    def stem(self, word):
        """Apply stemming rules."""
        word = word.lower()
        
        # Step 1: Remove common suffixes
        if word.endswith('sses'):
            word = word[:-2]
        elif word.endswith('ies'):
            word = word[:-2]
        elif word.endswith('ss'):
            pass
        elif word.endswith('s'):
            word = word[:-1]
        
        if word.endswith('eed'):
            if self._measure(word[:-3]) > 0:
                word = word[:-1]
        elif word.endswith('ed'):
            if any(c in self.vowels for c in word[:-2]):
                word = word[:-2]
        elif word.endswith('ing'):
            if any(c in self.vowels for c in word[:-3]):
                word = word[:-3]
        
        # Handle common endings
        if word.endswith('ational'):
            word = word[:-5] + 'e'
        elif word.endswith('tion'):
            word = word[:-3] + 'e'
        elif word.endswith('ness'):
            word = word[:-4]
        elif word.endswith('ment'):
            word = word[:-4]
        elif word.endswith('ful'):
            word = word[:-3]
        elif word.endswith('ly'):
            word = word[:-2]
        
        return word


# Example
stemmer = PorterStemmer()

words = ['running', 'runs', 'ran', 'easily', 'happiness', 'hopeful']
for word in words:
    print(f"{word} -> {stemmer.stem(word)}")
```

### 20.4 Text Cleaning

```python
import html
import unicodedata

def clean_text(text):
    """Comprehensive text cleaning."""
    # Decode HTML entities
    text = html.unescape(text)
    
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# Example
dirty_text = "Check out https://example.com! Email me at test@email.com 😊 #NLP"
clean = clean_text(dirty_text)
print(f"Original: {dirty_text}")
print(f"Cleaned: {clean}")
```

---

## Chapter 21: Text Vectorization

### 21.1 Bag of Words (BoW)

```python
from collections import defaultdict
import numpy as np

class BagOfWords:
    """Bag of Words vectorizer."""
    
    def __init__(self, max_features=None):
        self.max_features = max_features
        self.vocabulary = {}
        
    def fit(self, documents):
        """Build vocabulary from documents."""
        word_counts = Counter()
        
        for doc in documents:
            tokens = simple_tokenize(doc)
            word_counts.update(set(tokens))  # Count each word once per doc
        
        # Select top features
        if self.max_features:
            most_common = word_counts.most_common(self.max_features)
        else:
            most_common = word_counts.most_common()
        
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}
        return self
    
    def transform(self, documents):
        """Convert documents to BoW vectors."""
        vectors = np.zeros((len(documents), len(self.vocabulary)))
        
        for i, doc in enumerate(documents):
            tokens = simple_tokenize(doc)
            for token in tokens:
                if token in self.vocabulary:
                    vectors[i, self.vocabulary[token]] += 1
        
        return vectors
    
    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)


# Example
documents = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are pets",
]

bow = BagOfWords()
X = bow.fit_transform(documents)

print("Vocabulary:", bow.vocabulary)
print("\nBoW Matrix:")
print(X)
```

### 21.2 TF-IDF (Term Frequency-Inverse Document Frequency)

```python
class TfidfVectorizer:
    """TF-IDF Vectorizer from scratch."""
    
    def __init__(self, max_features=None):
        self.max_features = max_features
        self.vocabulary = {}
        self.idf = {}
        
    def fit(self, documents):
        """Compute IDF values."""
        n_docs = len(documents)
        
        # Count document frequency for each word
        doc_freq = Counter()
        word_counts = Counter()
        
        for doc in documents:
            tokens = set(simple_tokenize(doc))
            doc_freq.update(tokens)
            word_counts.update(simple_tokenize(doc))
        
        # Select top features by total frequency
        if self.max_features:
            most_common = word_counts.most_common(self.max_features)
            vocab_words = [word for word, _ in most_common]
        else:
            vocab_words = list(doc_freq.keys())
        
        self.vocabulary = {word: idx for idx, word in enumerate(vocab_words)}
        
        # Compute IDF: log(N / df) + 1
        self.idf = {}
        for word in self.vocabulary:
            df = doc_freq[word]
            self.idf[word] = np.log(n_docs / df) + 1
        
        return self
    
    def transform(self, documents):
        """Convert documents to TF-IDF vectors."""
        vectors = np.zeros((len(documents), len(self.vocabulary)))
        
        for i, doc in enumerate(documents):
            tokens = simple_tokenize(doc)
            token_counts = Counter(tokens)
            
            # Compute TF-IDF
            for token, count in token_counts.items():
                if token in self.vocabulary:
                    tf = count / len(tokens)  # Term frequency
                    tfidf = tf * self.idf[token]
                    vectors[i, self.vocabulary[token]] = tfidf
        
        # L2 normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms
        
        return vectors
    
    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)


# Example
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(documents)

print("TF-IDF Matrix:")
print(X_tfidf.round(3))

print("\nIDF values:")
for word, idf in sorted(tfidf.idf.items(), key=lambda x: -x[1])[:5]:
    print(f"  {word}: {idf:.3f}")
```

### 21.3 Word Embeddings

```python
class Word2VecSkipGram:
    """Simplified Word2Vec Skip-gram implementation."""
    
    def __init__(self, embedding_dim=50, window_size=2, learning_rate=0.01):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.lr = learning_rate
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.W1 = None  # Input embeddings
        self.W2 = None  # Output embeddings
        
    def _build_vocabulary(self, sentences):
        """Build vocabulary from sentences."""
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence)
        
        for idx, (word, _) in enumerate(word_counts.most_common()):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        return len(self.word_to_idx)
    
    def _generate_training_data(self, sentences):
        """Generate skip-gram training pairs."""
        pairs = []
        
        for sentence in sentences:
            indices = [self.word_to_idx[w] for w in sentence if w in self.word_to_idx]
            
            for i, center in enumerate(indices):
                for j in range(max(0, i - self.window_size), 
                              min(len(indices), i + self.window_size + 1)):
                    if i != j:
                        pairs.append((center, indices[j]))
        
        return pairs
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def fit(self, sentences, epochs=100):
        """Train Word2Vec model."""
        vocab_size = self._build_vocabulary(sentences)
        
        # Initialize weights
        np.random.seed(42)
        self.W1 = np.random.randn(vocab_size, self.embedding_dim) * 0.01
        self.W2 = np.random.randn(self.embedding_dim, vocab_size) * 0.01
        
        # Generate training pairs
        training_data = self._generate_training_data(sentences)
        
        # Training
        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(training_data)
            
            for center_idx, context_idx in training_data:
                # Forward pass
                hidden = self.W1[center_idx]
                output = self._softmax(hidden @ self.W2)
                
                # Loss (cross-entropy)
                total_loss -= np.log(output[context_idx] + 1e-10)
                
                # Backward pass
                output[context_idx] -= 1  # Gradient of softmax + cross-entropy
                
                # Update weights
                self.W1[center_idx] -= self.lr * (output @ self.W2.T)
                self.W2 -= self.lr * np.outer(hidden, output)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(training_data):.4f}")
        
        return self
    
    def get_embedding(self, word):
        """Get embedding vector for a word."""
        if word in self.word_to_idx:
            return self.W1[self.word_to_idx[word]]
        return None
    
    def most_similar(self, word, top_k=5):
        """Find most similar words."""
        if word not in self.word_to_idx:
            return []
        
        word_vec = self.get_embedding(word)
        similarities = []
        
        for other_word, idx in self.word_to_idx.items():
            if other_word != word:
                other_vec = self.W1[idx]
                sim = np.dot(word_vec, other_vec) / (np.linalg.norm(word_vec) * np.linalg.norm(other_vec))
                similarities.append((other_word, sim))
        
        return sorted(similarities, key=lambda x: -x[1])[:top_k]


# Example
sentences = [
    ['the', 'cat', 'sat', 'on', 'the', 'mat'],
    ['the', 'dog', 'sat', 'on', 'the', 'log'],
    ['cats', 'and', 'dogs', 'are', 'pets'],
    ['the', 'cat', 'chased', 'the', 'dog'],
    ['dogs', 'and', 'cats', 'play', 'together'],
]

w2v = Word2VecSkipGram(embedding_dim=10, window_size=2)
w2v.fit(sentences, epochs=100)

print("\nMost similar to 'cat':")
for word, sim in w2v.most_similar('cat'):
    print(f"  {word}: {sim:.3f}")
```

---

## Chapter 22: Text Classification

### 22.1 Sentiment Analysis with Naive Bayes

```python
class NaiveBayesClassifier:
    """Naive Bayes for text classification."""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
        self.class_priors = {}
        self.word_probs = {}
        self.vocabulary = set()
        
    def fit(self, documents, labels):
        """Train the classifier."""
        # Count documents per class
        class_counts = Counter(labels)
        total_docs = len(labels)
        
        for cls in class_counts:
            self.class_priors[cls] = class_counts[cls] / total_docs
        
        # Count words per class
        word_counts = defaultdict(lambda: defaultdict(int))
        class_word_totals = defaultdict(int)
        
        for doc, label in zip(documents, labels):
            tokens = simple_tokenize(doc)
            for token in tokens:
                word_counts[label][token] += 1
                class_word_totals[label] += 1
                self.vocabulary.add(token)
        
        # Compute word probabilities with smoothing
        vocab_size = len(self.vocabulary)
        
        for cls in class_counts:
            self.word_probs[cls] = {}
            for word in self.vocabulary:
                count = word_counts[cls][word]
                self.word_probs[cls][word] = (count + self.alpha) / (class_word_totals[cls] + self.alpha * vocab_size)
        
        return self
    
    def predict(self, documents):
        """Predict class labels."""
        predictions = []
        
        for doc in documents:
            tokens = simple_tokenize(doc)
            scores = {}
            
            for cls in self.class_priors:
                score = np.log(self.class_priors[cls])
                for token in tokens:
                    if token in self.vocabulary:
                        score += np.log(self.word_probs[cls].get(token, 1e-10))
                scores[cls] = score
            
            predictions.append(max(scores, key=scores.get))
        
        return predictions


# Example: Simple sentiment analysis
train_docs = [
    "I love this movie it is amazing",
    "Great film wonderful acting",
    "This is the best movie ever",
    "Fantastic story and characters",
    "I hate this terrible movie",
    "Worst film I have ever seen",
    "Awful acting and boring story",
    "Complete waste of time",
]
train_labels = ['positive', 'positive', 'positive', 'positive',
                'negative', 'negative', 'negative', 'negative']

clf = NaiveBayesClassifier()
clf.fit(train_docs, train_labels)

test_docs = [
    "I really enjoyed this amazing film",
    "This was a terrible waste of time",
    "The acting was great but story was weak",
]

predictions = clf.predict(test_docs)
for doc, pred in zip(test_docs, predictions):
    print(f"'{doc[:40]}...' -> {pred}")
```

### 22.2 Text Classification with Deep Learning

```python
import torch
import torch.nn as nn
import torch.optim as optim

class TextCNN(nn.Module):
    """CNN for text classification."""
    
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            c = torch.relu(conv(x))  # (batch_size, num_filters, seq_len - fs + 1)
            c = torch.max(c, dim=2)[0]  # Global max pooling
            conv_outputs.append(c)
        
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class LSTMClassifier(nn.Module):
    """LSTM for text classification."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1, bidirectional=True):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=bidirectional)
        
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use last hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        return self.fc(hidden)


# Example architecture
print("TextCNN Architecture:")
model = TextCNN(vocab_size=10000, embedding_dim=100, num_filters=100, 
                filter_sizes=[3, 4, 5], num_classes=2)
print(model)

print("\nLSTM Classifier Architecture:")
model = LSTMClassifier(vocab_size=10000, embedding_dim=100, hidden_dim=128, num_classes=2)
print(model)
```

---

## Chapter 23: Sequence Models for NLP

### 23.1 Named Entity Recognition (NER)

```python
class BiLSTM_CRF:
    """Bidirectional LSTM with CRF layer for NER (conceptual)."""
    
    def __init__(self, vocab_size, tag_size, embedding_dim, hidden_dim):
        # This is a conceptual implementation
        # Real implementation would use PyTorch/TensorFlow
        
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Transition matrix for CRF
        self.transitions = np.random.randn(tag_size, tag_size)
        
    def viterbi_decode(self, emissions):
        """Viterbi algorithm for finding best tag sequence."""
        seq_len, num_tags = emissions.shape
        
        # DP tables
        dp = np.full((seq_len, num_tags), -np.inf)
        backpointers = np.zeros((seq_len, num_tags), dtype=int)
        
        # Initialize
        dp[0] = emissions[0]
        
        # Forward pass
        for t in range(1, seq_len):
            for j in range(num_tags):
                scores = dp[t-1] + self.transitions[:, j] + emissions[t, j]
                dp[t, j] = np.max(scores)
                backpointers[t, j] = np.argmax(scores)
        
        # Backtrack
        best_path = [np.argmax(dp[-1])]
        for t in range(seq_len - 1, 0, -1):
            best_path.append(backpointers[t, best_path[-1]])
        
        return best_path[::-1]


# NER Tags example
NER_TAGS = {
    'O': 0,      # Outside any entity
    'B-PER': 1,  # Beginning of Person
    'I-PER': 2,  # Inside Person
    'B-ORG': 3,  # Beginning of Organization
    'I-ORG': 4,  # Inside Organization
    'B-LOC': 5,  # Beginning of Location
    'I-LOC': 6,  # Inside Location
}

print("Common NER tags (BIO format):")
for tag, idx in NER_TAGS.items():
    print(f"  {tag}: {idx}")
```

### 23.2 Machine Translation Basics

```python
class Seq2SeqAttention(nn.Module):
    """Sequence-to-sequence model with attention."""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        
        # Encoder
        self.encoder_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Decoder
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        self.decoder = nn.LSTM(embedding_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        
        # Attention
        self.attention = nn.Linear(hidden_dim * 3, 1)
        
        # Output
        self.output = nn.Linear(hidden_dim, tgt_vocab_size)
        
    def forward(self, src, tgt):
        # Encode
        src_embedded = self.encoder_embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder(src_embedded)
        
        # Decode with attention (simplified)
        batch_size, tgt_len = tgt.shape
        outputs = []
        
        decoder_hidden = hidden[-1].unsqueeze(0)
        decoder_cell = cell[-1].unsqueeze(0)
        
        for t in range(tgt_len):
            tgt_embedded = self.decoder_embedding(tgt[:, t:t+1])
            
            # Attention weights
            hidden_expanded = decoder_hidden.permute(1, 0, 2).expand(-1, encoder_outputs.size(1), -1)
            attention_input = torch.cat([encoder_outputs, hidden_expanded], dim=2)
            attention_weights = torch.softmax(self.attention(attention_input).squeeze(-1), dim=1)
            
            # Context vector
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
            
            # Decoder step
            decoder_input = torch.cat([tgt_embedded, context], dim=2)
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
                decoder_input, (decoder_hidden, decoder_cell)
            )
            
            output = self.output(decoder_output)
            outputs.append(output)
        
        return torch.cat(outputs, dim=1)


print("Seq2Seq with Attention for Machine Translation")
print("Architecture: Encoder -> Attention -> Decoder")
```

---

## Chapter 24: Transformers for NLP

### 24.1 Self-Attention Mechanism

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Scaled Dot-Product Attention.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    d_k = query.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    
    # Apply mask (for padding or causal attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism."""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output, attn_weights
```

### 24.2 Transformer Architecture

```python
class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
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


class TransformerClassifier(nn.Module):
    """Transformer for text classification."""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes, max_len=512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Use [CLS] token (first token) for classification
        x = x[:, 0]
        return self.classifier(x)


# Example
print("Transformer Classifier Architecture:")
model = TransformerClassifier(vocab_size=30000, d_model=256, num_heads=8, 
                              num_layers=4, num_classes=2)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 24.3 Using Pre-trained Transformers (BERT, GPT)

```python
# Using Hugging Face Transformers (conceptual example)
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize input
text = "This movie is amazing!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Forward pass
outputs = model(**inputs)
predictions = torch.softmax(outputs.logits, dim=1)

print(f"Positive probability: {predictions[0][1].item():.4f}")

print("""
Pre-trained Transformer Models:

BERT (Bidirectional Encoder):
- Masked Language Modeling + Next Sentence Prediction
- Good for: Classification, NER, Question Answering

GPT (Generative Pre-trained Transformer):
- Autoregressive Language Modeling
- Good for: Text generation, Completion

RoBERTa, ALBERT, DistilBERT:
- BERT variants with different optimizations

T5, BART:
- Encoder-Decoder for seq2seq tasks
""")
```

---

## Summary: NLP Techniques

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NLP TECHNIQUES SUMMARY                           │
├─────────────────────────────────────────────────────────────────────┤
│  PREPROCESSING                                                      │
│  ├── Tokenization, lowercasing, stop words                         │
│  ├── Stemming / Lemmatization                                      │
│  └── Text cleaning (URLs, special chars)                           │
│                                                                     │
│  VECTORIZATION                                                      │
│  ├── Bag of Words: Simple counting                                 │
│  ├── TF-IDF: Weighted by importance                                │
│  └── Word Embeddings: Dense semantic vectors                       │
│                                                                     │
│  CLASSIFICATION                                                     │
│  ├── Naive Bayes: Fast, good baseline                              │
│  ├── TextCNN: Captures n-gram features                             │
│  └── LSTM/GRU: Sequential understanding                            │
│                                                                     │
│  MODERN NLP                                                         │
│  ├── Transformers: Self-attention mechanism                        │
│  ├── BERT: Bidirectional pre-training                              │
│  └── GPT: Autoregressive generation                                │
└─────────────────────────────────────────────────────────────────────┘
```


---

<div align="center">

[⬅️ Previous: Unsupervised Learning](03-unsupervised-learning.md) | [📚 Table of Contents](../README.md) | [Next: Time Series Analysis ➡️](05-time-series.md)

</div>

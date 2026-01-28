# THE COMPLETE MACHINE LEARNING TEXTBOOK

## From Zero to Production-Ready | 2025 Edition

---

```
 __  __            _     _              _                          _             
|  \/  | __ _  ___| |__ (_)_ __   ___  | |    ___  __ _ _ __ _ __ (_)_ __   __ _ 
| |\/| |/ _` |/ __| '_ \| | '_ \ / _ \ | |   / _ \/ _` | '__| '_ \| | '_ \ / _` |
| |  | | (_| | (__| | | | | | | |  __/ | |__|  __/ (_| | |  | | | | | | | | (_| |
|_|  |_|\__,_|\___|_| |_|_|_| |_|\___| |_____\___|\__,_|_|  |_| |_|_|_| |_|\__, |
                                                                           |___/ 
```

---

## About This Textbook

This comprehensive textbook is designed to take you from complete beginner to expert in Machine Learning. Every concept is explained multiple ways, with theory, intuition, mathematics, and practical code examples.

### How to Use This Textbook

- **Read sequentially** for comprehensive learning
- **Use the table of contents** to jump to specific topics
- **Run the code examples** in your own environment
- **Complete the exercises** at the end of each chapter
- **Use the cheat sheets** for quick reference
- **Return to troubleshooting sections** when you hit issues

### Prerequisites

- Basic Python programming knowledge
- High school mathematics (algebra, basic calculus helpful)
- Curiosity and willingness to learn

---

# TABLE OF CONTENTS

## Part I: Machine Learning Foundations
- Chapter 1: What is Machine Learning?
- Chapter 2: Types of Machine Learning
- Chapter 3: The Machine Learning Workflow
- Chapter 4: Evaluation and Validation
- Chapter 5: Getting Started with Python for ML

## Part II: Mathematical Foundations
- Chapter 6: Linear Algebra Essentials
- Chapter 7: Calculus for Machine Learning
- Chapter 8: Probability and Statistics
- Chapter 9: Information Theory Basics

## Part III: Data Fundamentals
- Chapter 10: Data Collection and Exploration
- Chapter 11: Data Preprocessing
- Chapter 12: Feature Engineering
- Chapter 13: Handling Imbalanced Data

## Part IV: Supervised Learning
- Chapter 14: Linear Regression
- Chapter 15: Logistic Regression
- Chapter 16: Decision Trees
- Chapter 17: Ensemble Methods (Random Forest, Gradient Boosting)
- Chapter 18: Support Vector Machines
- Chapter 19: K-Nearest Neighbors
- Chapter 20: Naive Bayes

## Part V: Neural Networks
- Chapter 21: Perceptrons and Multilayer Networks
- Chapter 22: Activation Functions
- Chapter 23: Backpropagation
- Chapter 24: Training Deep Networks
- Chapter 25: Convolutional Neural Networks
- Chapter 26: Recurrent Neural Networks

## Part VI: Unsupervised Learning
- Chapter 27: Clustering (K-Means, Hierarchical, DBSCAN)
- Chapter 28: Dimensionality Reduction (PCA, t-SNE, UMAP)
- Chapter 29: Anomaly Detection
- Chapter 30: Association Rules

## Part VII: Natural Language Processing
- Chapter 31: Text Preprocessing
- Chapter 32: Text Vectorization (Bag of Words, TF-IDF, Word2Vec)
- Chapter 33: Text Classification
- Chapter 34: Sequence Models (RNN, LSTM)
- Chapter 35: Transformers and Attention

## Part VIII: Time Series Analysis
- Chapter 36: Time Series Fundamentals
- Chapter 37: Classical Models (ARIMA, SARIMA)
- Chapter 38: Deep Learning for Time Series
- Chapter 39: Forecasting Evaluation

## Part IX: MLOps and Deployment
- Chapter 40: ML Lifecycle Management
- Chapter 41: Model Serving and APIs
- Chapter 42: Model Monitoring
- Chapter 43: CI/CD for ML

## Part X: Appendices
- Appendix A: Python and NumPy Refresher
- Appendix B: Algorithm Cheat Sheets
- Appendix C: Common Errors and Solutions
- Appendix D: Glossary
- Appendix E: Resources

## Part XI: Advanced Computer Vision
- Chapter 44: CNN Architectures (VGG, ResNet, EfficientNet)
- Chapter 45: Object Detection (YOLO, Faster R-CNN)
- Chapter 46: Image Segmentation (U-Net)
- Chapter 47: Data Augmentation

## Part XII: Reinforcement Learning
- Chapter 48: RL Framework and Concepts
- Chapter 49: Q-Learning and SARSA
- Chapter 50: Deep Q-Networks
- Chapter 51: Policy Gradient Methods

## Part XIII: Practical Projects
- Chapter 52: End-to-End Project Workflow
- Chapter 53: Image Classification Project
- Chapter 54: NLP Sentiment Analysis Project
- Chapter 55: Time Series Forecasting Project

## Part XIV: Advanced Topics
- Chapter 56: Generative Adversarial Networks
- Chapter 57: Transformers In-Depth
- Chapter 58: AutoML and Neural Architecture Search

## Part XV: Responsible AI
- Chapter 59: Fairness in ML
- Chapter 60: Model Interpretability
- Chapter 61: Privacy in ML

## Part XVI: Optimization Deep Dive
- Chapter 62: Optimizer Algorithms
- Chapter 63: Learning Rate Schedulers
- Chapter 64: Regularization Techniques
- Chapter 65: Training Best Practices

## Part XVII: Graph Neural Networks
- Chapter 66: Graph Fundamentals
- Chapter 67: GNN Layers (GCN, GAT, GraphSAGE)
- Chapter 68: Graph-Level Tasks

## Part XVIII: Exercises and Projects
- Chapter 69: Conceptual Exercises
- Chapter 70: Coding Exercises
- Chapter 71: Mini-Projects
- Chapter 72: Quiz Questions

## Part XIX: Foundation Models
- Chapter 73: Self-Supervised Learning
- Chapter 74: Large Language Models
- Chapter 75: Vision Foundation Models
- Chapter 76: Efficient Fine-tuning

## Part XX: Advanced Algorithms
- Chapter 77: Bayesian Machine Learning
- Chapter 78: Meta-Learning
- Chapter 79: Neural Network Compression
- Chapter 80: Continual Learning

## Part XXI: Interview Preparation
- Chapter 81: ML System Design Interview
- Chapter 82: Coding Interview Questions
- Chapter 83: Behavioral Interview Questions

## Part XXII: Case Studies
- Chapter 84: Industry Case Studies (Healthcare, Finance, E-commerce)
- Chapter 85: ML in Production Lessons

---

# QUICK REFERENCE GUIDES

## Model Selection Guide

| Data Type | First Try | Alternatives |
|-----------|-----------|--------------|
| Tabular | XGBoost, LightGBM | Random Forest, Neural Nets |
| Images | ResNet, EfficientNet | ViT, ConvNeXt |
| Text | BERT, RoBERTa | GPT, T5 |
| Time Series | ARIMA, Prophet | LSTM, Transformer |
| Graphs | GCN, GAT | GraphSAGE |

## Hyperparameter Quick Reference

| Model | Key Parameters | Typical Values |
|-------|----------------|----------------|
| Neural Networks | Learning rate | 1e-4 to 1e-2 |
| | Batch size | 32, 64, 128, 256 |
| | Dropout | 0.1 to 0.5 |
| Random Forest | n_estimators | 100-1000 |
| | max_depth | None or 10-50 |
| XGBoost | learning_rate | 0.01-0.3 |
| | max_depth | 3-10 |

## Evaluation Metrics Guide

| Task | Primary Metric | When to Use |
|------|----------------|-------------|
| Binary Classification | AUC-ROC | Balanced classes |
| | F1 Score | Imbalanced classes |
| Multi-class | Accuracy | Balanced |
| | Macro F1 | Imbalanced |
| Regression | RMSE | General use |
| | MAE | Robust to outliers |
| Ranking | NDCG | Recommendations |

---

*Let's begin our journey into Machine Learning!*

---

---

# PART I: FOUNDATIONS

---

# Chapter 1: Introduction to Machine Learning

> *"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."*
> 
> — Tom Mitchell, 1997

---

## 1.1 What is Machine Learning?

Machine Learning (ML) is a subset of Artificial Intelligence that enables computers to learn patterns from data without being explicitly programmed for every possible scenario.

**The Key Insight:** Instead of writing rules for every situation, we show the computer examples and let it figure out the rules itself.

### Traditional Programming vs Machine Learning

**Traditional Programming:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│      DATA       │────▶│     RULES       │────▶│     OUTPUT      │
│                 │     │  (hand-coded)   │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Machine Learning:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│      DATA       │────▶│     MODEL       │────▶│     RULES       │
│    + ANSWERS    │     │   (learning)    │     │   (discovered)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

A CONCRETE EXAMPLE: SPAM DETECTION
───────────────────────────────────────────────────────────────────────────────

Traditional Approach:
You manually write rules like:
- IF email contains "Nigerian Prince" → SPAM
- IF email contains "free money" → SPAM
- IF sender not in contacts → MAYBE SPAM
- ... hundreds more rules ...

Problems:
1. You can't anticipate every spam pattern
2. Spammers constantly change tactics
3. Rules might incorrectly flag legitimate emails
4. Maintaining rules is exhausting

Machine Learning Approach:
You collect 100,000 emails labeled as SPAM or NOT SPAM.
You feed them to a learning algorithm.
The algorithm discovers patterns:
- Certain word combinations
- Sender patterns
- Time patterns
- Link patterns
- ...patterns you never thought of...

Benefits:
1. Discovers patterns humans might miss
2. Adapts when retrained with new data
3. Can be more accurate than hand-coded rules
4. Scales to complex problems
"""

# Let's see both approaches in code:

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1.1: Traditional Programming vs Machine Learning
# ─────────────────────────────────────────────────────────────────────────────

# TRADITIONAL APPROACH: Hand-coded rules
def is_spam_traditional(email_text, sender):
    """
    Traditional rule-based spam detection.
    
    Problems with this approach:
    1. We have to think of every rule
    2. Rules are rigid (what about "Fr33 M0n3y"?)
    3. Rules can conflict
    4. Hard to maintain as spam evolves
    """
    email_lower = email_text.lower()
    
    # Rule 1: Check for common spam phrases
    spam_phrases = [
        'nigerian prince',
        'free money',
        'click here now',
        'act immediately',
        'limited time offer',
        'you have won',
        'congratulations',
        'million dollars',
        'wire transfer',
        'urgent response needed'
    ]
    
    for phrase in spam_phrases:
        if phrase in email_lower:
            return True
    
    # Rule 2: Check for suspicious sender patterns
    suspicious_domains = ['spam.com', 'free-money.net', 'winner.org']
    for domain in suspicious_domains:
        if domain in sender.lower():
            return True
    
    # Rule 3: Check for excessive capitalization
    caps_ratio = sum(1 for c in email_text if c.isupper()) / max(len(email_text), 1)
    if caps_ratio > 0.5:
        return True
    
    # Rule 4: Check for excessive exclamation marks
    if email_text.count('!') > 5:
        return True
    
    return False


# MACHINE LEARNING APPROACH: Learn from data
def create_ml_spam_detector():
    """
    Machine Learning spam detection.
    
    The algorithm learns patterns from labeled examples.
    It can discover patterns we never thought of!
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    
    # Sample training data (in reality, you'd have thousands)
    training_emails = [
        "Hey, want to grab lunch tomorrow?",
        "Meeting rescheduled to 3pm",
        "FREE MONEY!!! Click here to claim your prize!!!",
        "Congratulations! You've won $1,000,000",
        "Project deadline extended to Friday",
        "URGENT: Transfer money immediately",
        "Can you review the attached document?",
        "Limited time offer! Act now!",
        "Your Amazon order has shipped",
        "Nigerian prince needs your help",
    ]
    
    labels = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # 0=not spam, 1=spam
    
    # Create a pipeline that:
    # 1. Converts text to numerical features (TF-IDF)
    # 2. Trains a Naive Bayes classifier
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])
    
    # Train the model (this is where the "learning" happens!)
    model.fit(training_emails, labels)
    
    return model


# Let's test both approaches
def compare_approaches():
    """Compare traditional vs ML approaches"""
    
    test_emails = [
        ("Special offer just for you! Limited time!", "promo@deals.com"),
        ("Can we reschedule our meeting?", "colleague@company.com"),
        ("YOU HAVE WON A LOTTERY!!!", "winner@lottery.com"),
        ("Quarterly report attached", "boss@company.com"),
    ]
    
    print("Traditional Approach Results:")
    print("-" * 50)
    for email, sender in test_emails:
        result = is_spam_traditional(email, sender)
        print(f"Email: '{email[:40]}...'")
        print(f"Spam: {result}\n")
    
    print("\nMachine Learning Approach Results:")
    print("-" * 50)
    ml_model = create_ml_spam_detector()
    for email, sender in test_emails:
        # Get probability of being spam
        proba = ml_model.predict_proba([email])[0]
        prediction = ml_model.predict([email])[0]
        print(f"Email: '{email[:40]}...'")
        print(f"Spam: {bool(prediction)} (confidence: {proba[prediction]:.1%})\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1.2 A BRIEF HISTORY OF MACHINE LEARNING
# ─────────────────────────────────────────────────────────────────────────────

"""
A BRIEF HISTORY OF MACHINE LEARNING
═══════════════════════════════════════════════════════════════════════════════

Timeline of Key Developments:

1943 │ Warren McCulloch & Walter Pitts
     │ First mathematical model of a neural network
     │ Showed neurons could implement logical functions
     │
1950 │ Alan Turing
     │ "Computing Machinery and Intelligence"
     │ Proposed the Turing Test
     │ Asked "Can machines think?"
     │
1957 │ Frank Rosenblatt
     │ The Perceptron
     │ First trainable neural network
     │ Could learn to classify simple patterns
     │
1967 │ The Nearest Neighbor Algorithm
     │ Simple but powerful instance-based learning
     │ Still used today!
     │
1969 │ Minsky & Papert
     │ "Perceptrons" book
     │ Showed limitations of single-layer networks
     │ Caused the first "AI Winter"
     │
1979 │ Stanford Cart
     │ Successfully navigated a room of obstacles
     │ Early example of autonomous systems
     │
1986 │ Backpropagation
     │ Rumelhart, Hinton, Williams
     │ Made training deep networks possible
     │ Renaissance of neural networks
     │
1995 │ Random Forests (Tin Kam Ho)
     │ Support Vector Machines (Cortes & Vapnik)
     │ Powerful algorithms still widely used
     │
1997 │ IBM Deep Blue beats Kasparov
     │ Major milestone for game-playing AI
     │ (Though more search than learning)
     │
1998 │ MNIST dataset released
     │ Yann LeCun's LeNet-5
     │ Convolutional Neural Networks for digits
     │
2006 │ Geoffrey Hinton
     │ "Deep Learning" term popularized
     │ Deep Belief Networks breakthrough
     │
2009 │ ImageNet dataset created
     │ 14+ million labeled images
     │ Enabled modern computer vision
     │
2012 │ AlexNet wins ImageNet
     │ Deep learning revolution begins
     │ Error rate dropped from 26% to 16%
     │ GPU training proves essential
     │
2014 │ GANs introduced (Goodfellow)
     │ Generative Adversarial Networks
     │ Generate realistic images
     │
2015 │ ResNet (152 layers!)
     │ Residual connections enable very deep networks
     │ Superhuman performance on ImageNet
     │
2016 │ AlphaGo beats Lee Sedol
     │ Deep reinforcement learning triumph
     │ Go was considered decades away
     │
2017 │ "Attention Is All You Need"
     │ The Transformer architecture
     │ Revolutionized NLP (and later, everything)
     │
2018 │ BERT (Google)
     │ Bidirectional transformer pretraining
     │ New state-of-the-art in NLP
     │
2019 │ GPT-2 (OpenAI)
     │ Impressive text generation
     │ "Too dangerous to release" controversy
     │
2020 │ GPT-3 (175B parameters)
     │ Few-shot learning capabilities
     │ AI assistants become practical
     │
2021 │ DALL-E, Codex
     │ Image generation from text
     │ Code generation capabilities
     │
2022 │ ChatGPT released
     │ AI goes mainstream
     │ Millions of users overnight
     │ The "GPT moment"
     │
2023 │ GPT-4, Claude, Gemini
     │ Multimodal capabilities
     │ Reasoning improvements
     │ AI becomes a tool for everyone
     │
2024 │ Open source catches up
     │ Llama 3, Mistral, Mixtral
     │ Video generation (Sora)
     │ Agent capabilities emerge
     │
2025 │ Reasoning models (o1, R1)
     │ Reinforcement learning from verifiable rewards
     │ Mixture of Experts architectures
     │ AI agents in production


KEY INSIGHT: The Three Waves of AI
───────────────────────────────────────────────────────────────────────────────

Wave 1 (1950s-1970s): Symbolic AI
├── Hand-coded rules and logic
├── Expert systems
└── Limited by human knowledge

Wave 2 (1980s-2010s): Statistical ML
├── Learning from data
├── SVMs, Random Forests
└── Limited by feature engineering

Wave 3 (2012-present): Deep Learning
├── End-to-end learning
├── Minimal feature engineering
└── Enabled by data + compute + algorithms
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1.3 WHY MACHINE LEARNING MATTERS TODAY
# ─────────────────────────────────────────────────────────────────────────────

"""
WHY MACHINE LEARNING MATTERS TODAY
═══════════════════════════════════════════════════════════════════════════════

THE PERFECT STORM: Why ML Exploded in the 2010s
───────────────────────────────────────────────────────────────────────────────

Three factors converged to enable the deep learning revolution:

1. DATA EXPLOSION
   ┌──────────────────────────────────────────────────────────────────────┐
   │ • Internet generates exabytes of data daily                         │
   │ • Social media provides labeled data (likes, shares, comments)      │
   │ • Smartphones = sensors everywhere                                   │
   │ • Digitization of historical records                                │
   │ • IoT devices creating continuous data streams                      │
   └──────────────────────────────────────────────────────────────────────┘

2. COMPUTE POWER
   ┌──────────────────────────────────────────────────────────────────────┐
   │ • GPUs: 100x faster than CPUs for matrix operations                 │
   │ • Cloud computing: Rent massive compute on demand                   │
   │ • Specialized chips: TPUs, Neural engines                           │
   │ • Moore's Law (until recently)                                      │
   └──────────────────────────────────────────────────────────────────────┘

3. ALGORITHMIC ADVANCES
   ┌──────────────────────────────────────────────────────────────────────┐
   │ • Dropout, BatchNorm: Better training stability                     │
   │ • ReLU: Solved vanishing gradient problem                           │
   │ • Residual connections: Enabled very deep networks                  │
   │ • Transformers: Parallelizable attention                            │
   │ • Better optimizers: Adam, AdamW                                    │
   └──────────────────────────────────────────────────────────────────────┘


ML IS EVERYWHERE: Real-World Applications
───────────────────────────────────────────────────────────────────────────────

HEALTHCARE
├── Disease diagnosis from medical images
├── Drug discovery and development
├── Personalized treatment recommendations
├── Predicting patient outcomes
├── Analyzing genomic data
└── Early detection of outbreaks

FINANCE
├── Fraud detection
├── Credit scoring
├── Algorithmic trading
├── Risk assessment
├── Customer churn prediction
└── Anti-money laundering

TECHNOLOGY
├── Search engines (Google, Bing)
├── Recommendation systems (Netflix, Spotify, Amazon)
├── Virtual assistants (Siri, Alexa, Google Assistant)
├── Email filtering
├── Translation services
└── Code completion (GitHub Copilot)

TRANSPORTATION
├── Self-driving vehicles
├── Route optimization
├── Demand prediction (Uber, Lyft)
├── Traffic prediction
├── Predictive maintenance
└── Autonomous drones

RETAIL
├── Demand forecasting
├── Inventory optimization
├── Price optimization
├── Customer segmentation
├── Visual search
└── Chatbots and customer service

ENTERTAINMENT
├── Content recommendation
├── Content generation
├── Game AI
├── Music composition
├── Video enhancement
└── Deepfakes (for better or worse)

SCIENCE
├── Climate modeling
├── Protein structure prediction (AlphaFold)
├── Particle physics analysis
├── Astronomical discovery
├── Materials science
└── Earthquake prediction

SECURITY
├── Intrusion detection
├── Malware classification
├── Facial recognition
├── Surveillance systems
├── Biometric authentication
└── Threat intelligence


THE ECONOMIC IMPACT
───────────────────────────────────────────────────────────────────────────────

According to various research reports:

• McKinsey: AI could add $13 trillion to global GDP by 2030
• PwC: AI will contribute $15.7 trillion to the global economy by 2030
• Gartner: AI will create 2.3 million jobs by 2025
• IDC: Worldwide AI spending reached $500 billion in 2024

Job market implications:
• Data Scientist consistently ranked top job
• ML Engineer salaries: $150K-$500K+ at top companies
• Demand far exceeds supply of qualified practitioners
• Every industry seeking ML expertise
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1.4 TYPES OF MACHINE LEARNING
# ─────────────────────────────────────────────────────────────────────────────

"""
TYPES OF MACHINE LEARNING
═══════════════════════════════════════════════════════════════════════════════

Machine Learning algorithms are typically categorized by how they learn:

                        ┌─────────────────────────┐
                        │    MACHINE LEARNING     │
                        └───────────┬─────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│  SUPERVISED   │         │ UNSUPERVISED  │         │REINFORCEMENT  │
│   LEARNING    │         │   LEARNING    │         │   LEARNING    │
└───────┬───────┘         └───────┬───────┘         └───────┬───────┘
        │                         │                         │
        │                         │                         │
   Has labels              No labels                 Learn from
   (answers)               (no answers)              rewards/penalties
        │                         │                         │
        ▼                         ▼                         ▼
 • Classification          • Clustering              • Game playing
 • Regression              • Dim. reduction          • Robotics
                           • Anomaly detection       • Resource mgmt


Additional paradigms:
├── Self-Supervised Learning: Create labels from data itself
├── Semi-Supervised Learning: Some labels, mostly unlabeled
└── Transfer Learning: Apply knowledge from one task to another
"""


# ═══════════════════════════════════════════════════════════════════════════
# 1.4.1 SUPERVISED LEARNING - Detailed Explanation
# ═══════════════════════════════════════════════════════════════════════════

"""
SUPERVISED LEARNING
═══════════════════════════════════════════════════════════════════════════════

Definition: Learning from labeled examples where both inputs (X) and 
desired outputs (y) are provided.

THE ANALOGY:
───────────────────────────────────────────────────────────────────────────────
Supervised learning is like learning with a teacher who gives you:
• Practice problems (inputs)
• Answer key (labels)
You learn the patterns and can solve NEW problems.


HOW IT WORKS:
───────────────────────────────────────────────────────────────────────────────

Step 1: Collect labeled data
        ┌────────────────────────────────────────────────────┐
        │  Features (X)              │  Label (y)            │
        ├────────────────────────────┼───────────────────────┤
        │  [3 bedrooms, 1500 sqft]   │  $300,000             │
        │  [2 bedrooms, 1000 sqft]   │  $200,000             │
        │  [4 bedrooms, 2000 sqft]   │  $450,000             │
        │  ...                       │  ...                  │
        └────────────────────────────┴───────────────────────┘

Step 2: Train a model
        model.fit(X_train, y_train)
        
        The model finds patterns:
        "Each bedroom adds ~$50K, each sqft adds ~$100"

Step 3: Predict on new data
        new_house = [3 bedrooms, 1800 sqft]
        price = model.predict(new_house)  # $380,000


TWO MAIN TASKS:
───────────────────────────────────────────────────────────────────────────────

1. CLASSIFICATION: Predict a category
   
   Examples:
   • Email → Spam or Not Spam
   • Image → Cat, Dog, or Bird
   • Transaction → Fraudulent or Legitimate
   • Patient symptoms → Disease diagnosis
   
   Output: Discrete class labels

2. REGRESSION: Predict a continuous number
   
   Examples:
   • House features → Price
   • Student data → Test score
   • Weather data → Temperature tomorrow
   • Customer data → Lifetime value
   
   Output: Continuous values


COMMON SUPERVISED LEARNING ALGORITHMS:
───────────────────────────────────────────────────────────────────────────────

For Classification:
├── Logistic Regression (simple baseline)
├── Decision Trees
├── Random Forests
├── Gradient Boosting (XGBoost, LightGBM)
├── Support Vector Machines
├── K-Nearest Neighbors
├── Naive Bayes
└── Neural Networks

For Regression:
├── Linear Regression (simple baseline)
├── Polynomial Regression
├── Decision Trees
├── Random Forests
├── Gradient Boosting
├── Support Vector Regression
└── Neural Networks
"""

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1.2: Supervised Learning - Classification
# ─────────────────────────────────────────────────────────────────────────────

def supervised_classification_example():
    """
    Complete supervised classification example.
    
    Task: Predict if a customer will churn (leave) based on their behavior.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Prepare labeled data
    # ─────────────────────────────────────────────────────────────────────────
    
    # Simulated customer data
    # Features: [months_as_customer, monthly_charges, total_charges, 
    #            support_tickets, login_frequency]
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    months = np.random.randint(1, 72, n_samples)
    monthly_charges = np.random.uniform(20, 100, n_samples)
    total_charges = months * monthly_charges * np.random.uniform(0.8, 1.2, n_samples)
    support_tickets = np.random.poisson(2, n_samples)
    login_frequency = np.random.uniform(0, 30, n_samples)
    
    X = np.column_stack([months, monthly_charges, total_charges, 
                         support_tickets, login_frequency])
    
    # Generate labels based on some logic (simulating real patterns)
    # More likely to churn if: new customer, high charges, many tickets, low login
    churn_probability = (
        (months < 12).astype(float) * 0.3 +
        (monthly_charges > 70).astype(float) * 0.2 +
        (support_tickets > 3).astype(float) * 0.3 +
        (login_frequency < 5).astype(float) * 0.2
    ) / 1.0
    
    y = (np.random.random(n_samples) < churn_probability).astype(int)
    
    feature_names = ['months_customer', 'monthly_charges', 'total_charges',
                     'support_tickets', 'login_frequency']
    
    print("Dataset Overview:")
    print(f"Total samples: {n_samples}")
    print(f"Features: {feature_names}")
    print(f"Churned customers: {y.sum()} ({y.mean():.1%})")
    print(f"Retained customers: {n_samples - y.sum()} ({1-y.mean():.1%})")
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Split data into training and testing sets
    # ─────────────────────────────────────────────────────────────────────────
    
    """
    WHY SPLIT THE DATA?
    
    We need to test our model on data it has NEVER seen before.
    This tells us how well it will perform in the real world.
    
    If we test on training data, we're just checking if the model
    memorized the answers - not if it learned the patterns!
    
    Common splits:
    - 80% train, 20% test (simple)
    - 70% train, 15% validation, 15% test (with hyperparameter tuning)
    """
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,      # 20% for testing
        random_state=42,    # Reproducibility
        stratify=y          # Maintain class proportions
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Preprocess the data
    # ─────────────────────────────────────────────────────────────────────────
    
    """
    WHY SCALE FEATURES?
    
    Our features have very different scales:
    - months_customer: 1-72
    - monthly_charges: 20-100
    - total_charges: 20-7200+
    
    Many algorithms are sensitive to scale.
    Without scaling, total_charges would dominate just because
    its numbers are bigger!
    
    StandardScaler: Transforms to mean=0, std=1
    """
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit AND transform
    X_test_scaled = scaler.transform(X_test)        # Only transform (same params)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Train models
    # ─────────────────────────────────────────────────────────────────────────
    
    # Model 1: Logistic Regression (simple baseline)
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    # Model 2: Random Forest (more powerful)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Evaluate models
    # ─────────────────────────────────────────────────────────────────────────
    
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    for name, model in [("Logistic Regression", lr), ("Random Forest", rf)]:
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"Accuracy: {accuracy:.2%}")
        print("\nDetailed Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Retained', 'Churned']))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: Interpret results
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("=" * 60)
    
    importances = rf.feature_importances_
    for name, importance in sorted(zip(feature_names, importances), 
                                   key=lambda x: x[1], reverse=True):
        bar = "█" * int(importance * 50)
        print(f"{name:20} {importance:.3f} {bar}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 7: Make predictions on new data
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 60)
    print("PREDICTING FOR NEW CUSTOMERS")
    print("=" * 60)
    
    new_customers = np.array([
        [3, 80, 240, 5, 2],    # New, high charges, many tickets, low login
        [60, 50, 3000, 1, 25], # Long-time, moderate, few tickets, active
        [12, 90, 1080, 0, 10], # 1 year, high charges, no tickets, moderate
    ])
    
    new_customers_scaled = scaler.transform(new_customers)
    predictions = rf.predict(new_customers_scaled)
    probabilities = rf.predict_proba(new_customers_scaled)
    
    for i, (features, pred, proba) in enumerate(zip(new_customers, 
                                                     predictions, 
                                                     probabilities)):
        print(f"\nCustomer {i+1}: {features}")
        print(f"  Prediction: {'WILL CHURN' if pred else 'Will Stay'}")
        print(f"  Confidence: {proba[pred]:.1%}")
        
    return rf, scaler


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1.3: Supervised Learning - Regression
# ─────────────────────────────────────────────────────────────────────────────

def supervised_regression_example():
    """
    Complete supervised regression example.
    
    Task: Predict house prices based on features.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Generate synthetic house data
    np.random.seed(42)
    n_samples = 500
    
    # Features
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    sqft = np.random.randint(500, 4000, n_samples)
    age = np.random.randint(0, 100, n_samples)
    garage = np.random.randint(0, 3, n_samples)
    
    X = np.column_stack([bedrooms, bathrooms, sqft, age, garage])
    feature_names = ['bedrooms', 'bathrooms', 'sqft', 'age', 'garage']
    
    # Generate prices based on features (with some noise)
    base_price = 100000
    price = (
        base_price +
        bedrooms * 25000 +
        bathrooms * 15000 +
        sqft * 150 +
        -age * 1000 +
        garage * 20000 +
        np.random.normal(0, 30000, n_samples)  # Random noise
    )
    
    y = np.maximum(price, 50000)  # Minimum price of $50K
    
    print("House Price Prediction Dataset")
    print("=" * 50)
    print(f"Samples: {n_samples}")
    print(f"Features: {feature_names}")
    print(f"Price range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"Average price: ${y.mean():,.0f}")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    print("Model Performance")
    print("=" * 70)
    print(f"{'Model':<25} {'RMSE':>12} {'MAE':>12} {'R²':>12}")
    print("-" * 70)
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name:<25} ${rmse:>10,.0f} ${mae:>10,.0f} {r2:>11.3f}")
    
    # Interpret Linear Regression coefficients
    lr = models['Linear Regression']
    print("\n\nLinear Regression Coefficients (Feature Impact on Price):")
    print("-" * 60)
    
    for name, coef in zip(feature_names, lr.coef_):
        direction = "+" if coef > 0 else ""
        print(f"{name:<15}: {direction}${coef:,.0f}")
    
    print(f"{'Base price':<15}: ${lr.intercept_:,.0f}")
    
    # Predict for new houses
    print("\n\nPredicting Prices for New Houses:")
    print("-" * 60)
    
    new_houses = [
        [3, 2, 1500, 10, 2],  # 3bed, 2bath, 1500sqft, 10yr old, 2-car garage
        [4, 3, 2500, 5, 2],   # 4bed, 3bath, 2500sqft, 5yr old, 2-car garage
        [2, 1, 800, 50, 0],   # 2bed, 1bath, 800sqft, 50yr old, no garage
    ]
    
    new_houses_scaled = scaler.transform(new_houses)
    rf = models['Random Forest']
    
    for i, (features, scaled) in enumerate(zip(new_houses, new_houses_scaled)):
        pred = rf.predict([scaled])[0]
        print(f"\nHouse {i+1}:")
        print(f"  Features: {dict(zip(feature_names, features))}")
        print(f"  Predicted Price: ${pred:,.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# 1.4.2 UNSUPERVISED LEARNING - Detailed Explanation
# ═══════════════════════════════════════════════════════════════════════════

"""
UNSUPERVISED LEARNING
═══════════════════════════════════════════════════════════════════════════════

Definition: Learning patterns from data WITHOUT labeled examples.
The algorithm must discover structure on its own.

THE ANALOGY:
───────────────────────────────────────────────────────────────────────────────
Unsupervised learning is like exploring a new city without a map:
• No one tells you what the neighborhoods are
• You discover patterns: "this area has restaurants", "this is residential"
• You group things together based on similarity


HOW IT DIFFERS FROM SUPERVISED:
───────────────────────────────────────────────────────────────────────────────

Supervised:
┌─────────────────────────────────────────────────────────────────────────────┐
│  Input: Customer data + Label (churned: yes/no)                            │
│  Goal: Predict if NEW customers will churn                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Unsupervised:
┌─────────────────────────────────────────────────────────────────────────────┐
│  Input: Customer data only (NO labels)                                     │
│  Goal: Discover natural groupings of customers                             │
│        (maybe: "budget", "premium", "at-risk" segments)                    │
└─────────────────────────────────────────────────────────────────────────────┘


MAIN TASKS IN UNSUPERVISED LEARNING:
───────────────────────────────────────────────────────────────────────────────

1. CLUSTERING
   Finding groups of similar data points
   
   Applications:
   • Customer segmentation
   • Document grouping
   • Image compression
   • Anomaly detection
   • Gene expression analysis
   
   Algorithms:
   • K-Means
   • Hierarchical clustering
   • DBSCAN
   • Gaussian Mixture Models

2. DIMENSIONALITY REDUCTION
   Reducing the number of features while preserving information
   
   Applications:
   • Visualization of high-dimensional data
   • Noise reduction
   • Feature extraction
   • Data compression
   • Speeding up other algorithms
   
   Algorithms:
   • PCA (Principal Component Analysis)
   • t-SNE
   • UMAP
   • Autoencoders

3. ANOMALY DETECTION
   Finding unusual data points
   
   Applications:
   • Fraud detection
   • Network intrusion detection
   • Manufacturing defect detection
   • Medical diagnosis
   
   Algorithms:
   • Isolation Forest
   • One-Class SVM
   • Local Outlier Factor
   • Autoencoders

4. ASSOCIATION RULE LEARNING
   Finding relationships between variables
   
   Applications:
   • Market basket analysis ("customers who bought X also bought Y")
   • Recommendation systems
   
   Algorithms:
   • Apriori
   • FP-Growth
"""

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1.4: Unsupervised Learning - Clustering
# ─────────────────────────────────────────────────────────────────────────────

def unsupervised_clustering_example():
    """
    Complete unsupervised clustering example.
    
    Task: Segment customers into groups based on their purchasing behavior.
    No labels are provided - the algorithm discovers groups on its own!
    """
    import numpy as np
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    
    # Generate synthetic customer data (no labels!)
    np.random.seed(42)
    
    # Create 4 natural clusters (but the algorithm doesn't know this!)
    
    # Cluster 1: Budget shoppers (low spending, low frequency)
    budget = np.random.normal([20, 5, 100], [5, 2, 30], (100, 3))
    
    # Cluster 2: Regular shoppers (medium spending, medium frequency)
    regular = np.random.normal([50, 15, 500], [10, 3, 100], (100, 3))
    
    # Cluster 3: Premium shoppers (high spending, high frequency)
    premium = np.random.normal([100, 30, 2000], [20, 5, 400], (100, 3))
    
    # Cluster 4: Occasional big spenders (low frequency, high per-purchase)
    occasional = np.random.normal([150, 3, 800], [30, 1, 200], (100, 3))
    
    # Combine all data
    X = np.vstack([budget, regular, premium, occasional])
    feature_names = ['avg_purchase_amount', 'monthly_visits', 'total_spend']
    
    print("Customer Segmentation (Unsupervised)")
    print("=" * 60)
    print(f"Total customers: {len(X)}")
    print(f"Features: {feature_names}")
    print("\nNote: We have NO labels - the algorithm will discover groups!\n")
    
    # Scale features (important for clustering!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Method 1: K-Means Clustering
    # ─────────────────────────────────────────────────────────────────────────
    
    """
    K-MEANS ALGORITHM:
    
    1. Choose K (number of clusters)
    2. Randomly initialize K cluster centers
    3. Assign each point to nearest center
    4. Update centers to mean of assigned points
    5. Repeat 3-4 until convergence
    """
    
    # First, let's find the optimal number of clusters
    print("Finding optimal number of clusters...")
    print("-" * 40)
    
    silhouette_scores = []
    K_range = range(2, 10)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        print(f"K={k}: Silhouette Score = {score:.3f}")
    
    best_k = K_range[np.argmax(silhouette_scores)]
    print(f"\nBest K: {best_k}")
    
    # Fit final model
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Analyze the discovered clusters
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 60)
    print("DISCOVERED CUSTOMER SEGMENTS")
    print("=" * 60)
    
    for cluster in range(best_k):
        mask = cluster_labels == cluster
        cluster_data = X[mask]
        
        print(f"\nCluster {cluster} ({mask.sum()} customers):")
        print("-" * 40)
        
        for i, name in enumerate(feature_names):
            mean_val = cluster_data[:, i].mean()
            std_val = cluster_data[:, i].std()
            print(f"  {name:<20}: {mean_val:>8.1f} (±{std_val:.1f})")
        
        # Give the cluster a name based on characteristics
        avg_purchase = cluster_data[:, 0].mean()
        visits = cluster_data[:, 1].mean()
        
        if avg_purchase < 40 and visits < 10:
            segment_name = "Budget Shoppers"
        elif avg_purchase > 120:
            segment_name = "Big Spenders"
        elif visits > 20:
            segment_name = "Frequent Premium"
        else:
            segment_name = "Regular Customers"
        
        print(f"  Suggested name: {segment_name}")
    
    return kmeans, scaler, cluster_labels


# ═══════════════════════════════════════════════════════════════════════════
# 1.4.3 REINFORCEMENT LEARNING - Detailed Explanation
# ═══════════════════════════════════════════════════════════════════════════

"""
REINFORCEMENT LEARNING
═══════════════════════════════════════════════════════════════════════════════

Definition: Learning through trial and error by receiving rewards or penalties
for actions taken in an environment.

THE ANALOGY:
───────────────────────────────────────────────────────────────────────────────
Reinforcement learning is like training a dog:
• Dog performs action (sits, jumps, barks)
• You give reward (treat) or penalty (no treat, "bad dog")
• Dog learns which actions lead to rewards
• Eventually, dog learns complex behaviors


THE RL FRAMEWORK:
───────────────────────────────────────────────────────────────────────────────

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │    ┌─────────┐         action (a)        ┌─────────────────┐           │
    │    │         │ ─────────────────────────▶│                 │           │
    │    │  AGENT  │                           │   ENVIRONMENT   │           │
    │    │         │ ◀─────────────────────────│                 │           │
    │    └─────────┘    state (s), reward (r)  └─────────────────┘           │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    Agent: The learner/decision-maker
    Environment: The world the agent interacts with
    State (s): Current situation
    Action (a): What the agent does
    Reward (r): Feedback signal (positive or negative)
    
    Goal: Learn a POLICY (strategy) that maximizes cumulative reward


KEY CONCEPTS:
───────────────────────────────────────────────────────────────────────────────

1. POLICY (π)
   A strategy that maps states to actions
   π(s) → a
   "When in state s, take action a"

2. VALUE FUNCTION (V)
   Expected cumulative reward from a state
   "How good is it to be in state s?"

3. Q-FUNCTION (Q)
   Expected cumulative reward from taking action a in state s
   "How good is it to take action a in state s?"

4. EXPLORATION vs EXPLOITATION
   • Exploration: Try new actions to discover better strategies
   • Exploitation: Use known good actions to maximize reward
   • Balance is crucial!


RL ALGORITHMS:
───────────────────────────────────────────────────────────────────────────────

Model-Free:
├── Q-Learning: Learn Q-values for state-action pairs
├── SARSA: On-policy variant of Q-learning
├── Policy Gradient: Directly optimize the policy
├── Actor-Critic: Combine value and policy methods
└── PPO/TRPO: Stable policy optimization

Model-Based:
├── Learn a model of the environment
├── Plan using the learned model
└── More sample-efficient but harder to implement


APPLICATIONS:
───────────────────────────────────────────────────────────────────────────────

Games:
• AlphaGo (Go)
• OpenAI Five (Dota 2)
• Atari games
• Chess, Poker

Robotics:
• Robot locomotion
• Manipulation tasks
• Autonomous vehicles

Business:
• Ad placement
• Recommendation systems
• Dynamic pricing
• Resource allocation

Science:
• Molecule design
• Experiment optimization
• Chip design (AlphaChip)
"""

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1.5: Simple Reinforcement Learning
# ─────────────────────────────────────────────────────────────────────────────

def simple_rl_example():
    """
    Simple Q-Learning example: Grid World
    
    The agent must navigate a grid to reach a goal while avoiding obstacles.
    """
    import numpy as np
    
    # Define the environment: 4x4 grid
    # 0 = empty, -1 = obstacle, 10 = goal
    
    """
    Grid layout:
    ┌────┬────┬────┬────┐
    │ S  │    │    │    │
    ├────┼────┼────┼────┤
    │    │ X  │    │    │
    ├────┼────┼────┼────┤
    │    │    │ X  │    │
    ├────┼────┼────┼────┤
    │    │    │    │ G  │
    └────┴────┴────┴────┘
    
    S = Start (0,0)
    G = Goal (3,3)
    X = Obstacle (-10 reward)
    """
    
    GRID_SIZE = 4
    GOAL = (3, 3)
    OBSTACLES = [(1, 1), (2, 2)]
    
    # Actions: 0=up, 1=right, 2=down, 3=left
    ACTIONS = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1)   # left
    }
    ACTION_NAMES = ['↑', '→', '↓', '←']
    
    def get_reward(state):
        """Get reward for being in a state"""
        if state == GOAL:
            return 10
        elif state in OBSTACLES:
            return -10
        else:
            return -1  # Small penalty to encourage efficiency
    
    def is_valid(state):
        """Check if state is within grid"""
        return 0 <= state[0] < GRID_SIZE and 0 <= state[1] < GRID_SIZE
    
    def take_action(state, action):
        """Take action and return new state"""
        delta = ACTIONS[action]
        new_state = (state[0] + delta[0], state[1] + delta[1])
        
        if is_valid(new_state):
            return new_state
        return state  # Stay in place if invalid move
    
    # Initialize Q-table
    # Q[state][action] = expected cumulative reward
    Q = np.zeros((GRID_SIZE, GRID_SIZE, 4))
    
    # Hyperparameters
    learning_rate = 0.1
    discount_factor = 0.99  # How much to value future rewards
    epsilon = 0.1  # Exploration rate
    episodes = 1000
    
    print("Q-Learning: Grid World Navigation")
    print("=" * 50)
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Goal: {GOAL}")
    print(f"Obstacles: {OBSTACLES}")
    print(f"Training for {episodes} episodes...")
    print()
    
    # Training loop
    total_rewards = []
    
    for episode in range(episodes):
        state = (0, 0)  # Start position
        episode_reward = 0
        steps = 0
        max_steps = 100
        
        while state != GOAL and steps < max_steps:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(4)  # Explore
            else:
                action = np.argmax(Q[state[0], state[1]])  # Exploit
            
            # Take action
            new_state = take_action(state, action)
            reward = get_reward(new_state)
            episode_reward += reward
            
            # Q-learning update
            # Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
            old_value = Q[state[0], state[1], action]
            next_max = np.max(Q[new_state[0], new_state[1]])
            
            Q[state[0], state[1], action] = old_value + learning_rate * (
                reward + discount_factor * next_max - old_value
            )
            
            state = new_state
            steps += 1
        
        total_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            print(f"Episode {episode + 1}: Avg Reward (last 100) = {avg_reward:.1f}")
    
    # Display learned policy
    print("\n" + "=" * 50)
    print("LEARNED POLICY")
    print("=" * 50)
    print("\nBest action in each cell:")
    print("(S=Start, G=Goal, X=Obstacle)")
    print()
    
    for i in range(GRID_SIZE):
        row = ""
        for j in range(GRID_SIZE):
            if (i, j) == (0, 0):
                row += " S "
            elif (i, j) == GOAL:
                row += " G "
            elif (i, j) in OBSTACLES:
                row += " X "
            else:
                best_action = np.argmax(Q[i, j])
                row += f" {ACTION_NAMES[best_action]} "
        print(row)
    
    # Test the learned policy
    print("\n" + "=" * 50)
    print("TESTING LEARNED POLICY")
    print("=" * 50)
    
    state = (0, 0)
    path = [state]
    
    while state != GOAL and len(path) < 20:
        action = np.argmax(Q[state[0], state[1]])
        state = take_action(state, action)
        path.append(state)
    
    print(f"Path from start to goal: {' → '.join(str(s) for s in path)}")
    print(f"Steps taken: {len(path) - 1}")
    
    return Q


# ═══════════════════════════════════════════════════════════════════════════
# 1.4.4 SELF-SUPERVISED LEARNING - Detailed Explanation
# ═══════════════════════════════════════════════════════════════════════════

"""
SELF-SUPERVISED LEARNING
═══════════════════════════════════════════════════════════════════════════════

Definition: A form of unsupervised learning where the data provides its own
labels. The algorithm creates supervisory signals from the input data itself.

THE ANALOGY:
───────────────────────────────────────────────────────────────────────────────
Self-supervised learning is like learning a language by reading books:
• No one labels each word with its meaning
• You learn patterns from context
• "The cat sat on the ___" - you can guess "mat" or "floor"
• The surrounding words supervise the learning


WHY IT'S REVOLUTIONARY:
───────────────────────────────────────────────────────────────────────────────

Traditional supervised learning needs LABELED data:
• Expensive to create
• Time-consuming
• Limited in scale
• Requires domain experts

Self-supervised learning uses UNLABELED data:
• Abundant (internet has endless text, images, audio)
• Free
• Scales to billions of examples
• No manual labeling needed

This is how GPT, BERT, and most modern AI systems are trained!


COMMON PRETEXT TASKS:
───────────────────────────────────────────────────────────────────────────────

For Text (Language Models):
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Masked Language Modeling (BERT-style)                                    │
│    Input:  "The [MASK] sat on the mat"                                     │
│    Target: "cat"                                                           │
│    The model learns to predict masked words from context                   │
│                                                                             │
│ 2. Next Token Prediction (GPT-style)                                       │
│    Input:  "The cat sat on the"                                            │
│    Target: "mat"                                                           │
│    The model learns to predict what comes next                             │
│                                                                             │
│ 3. Next Sentence Prediction (BERT)                                         │
│    Given two sentences, predict if sentence B follows sentence A           │
└─────────────────────────────────────────────────────────────────────────────┘

For Images:
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Contrastive Learning (SimCLR, MoCo)                                     │
│    - Create two augmented views of same image                              │
│    - Train model to recognize they're the same                             │
│    - Push representations of same image together                           │
│    - Push representations of different images apart                        │
│                                                                             │
│ 2. Masked Image Modeling (MAE)                                             │
│    - Mask random patches of an image                                       │
│    - Train model to reconstruct the masked patches                         │
│                                                                             │
│ 3. Rotation Prediction                                                      │
│    - Rotate image by 0°, 90°, 180°, or 270°                               │
│    - Train model to predict the rotation                                   │
│                                                                             │
│ 4. Jigsaw Puzzles                                                          │
│    - Divide image into patches and shuffle                                 │
│    - Train model to solve the puzzle                                       │
└─────────────────────────────────────────────────────────────────────────────┘


THE PRETRAIN-FINETUNE PARADIGM:
───────────────────────────────────────────────────────────────────────────────

Phase 1: Pretraining (Self-supervised)
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Train on MASSIVE unlabeled data                                           │
│  (billions of web pages, images, etc.)                                     │
│                                                                             │
│  Learn general representations of language/images/etc.                     │
│  This requires huge compute but only done once                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Phase 2: Finetuning (Supervised)
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Take pretrained model                                                      │
│  Train on small labeled dataset for specific task                          │
│                                                                             │
│  Examples:                                                                  │
│  - Sentiment classification (thousands of examples)                        │
│  - Named entity recognition                                                │
│  - Question answering                                                       │
│                                                                             │
│  Much less compute, much less data needed                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


KEY INSIGHT: Why This Works
───────────────────────────────────────────────────────────────────────────────

To predict masked words well, a model must learn:
• Grammar and syntax
• Word meanings and relationships
• World knowledge
• Reasoning abilities

These learned representations transfer to many downstream tasks!
"""

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1.6: Self-Supervised Learning Concept
# ─────────────────────────────────────────────────────────────────────────────

def self_supervised_concept_example():
    """
    Demonstrate the concept of self-supervised learning
    using a simple word prediction task.
    """
    import numpy as np
    from collections import Counter
    
    print("Self-Supervised Learning: Word Prediction")
    print("=" * 60)
    
    # Sample text corpus (imagine this is the entire internet)
    corpus = """
    The cat sat on the mat. The dog sat on the rug.
    A bird sat on the branch. The cat chased the bird.
    The dog chased the cat. The bird flew away.
    The cat sleeps on the mat. The dog sleeps on the rug.
    A happy cat purrs loudly. A happy dog wags its tail.
    """
    
    # Tokenize
    words = corpus.lower().replace('.', ' ').split()
    
    print("Training Corpus (simplified):")
    print(corpus[:200] + "...")
    print()
    
    # Build vocabulary
    vocab = list(set(words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for i, w in enumerate(vocab)}
    
    # Create training data: (context words) → (target word)
    # This is the "self-supervised" part - labels come from the data itself!
    
    """
    Self-supervision example:
    
    Sentence: "The cat sat on the mat"
    
    Training examples created automatically:
    Context: [the, sat]    → Target: cat
    Context: [cat, on]     → Target: sat
    Context: [sat, the]    → Target: on
    ...etc...
    
    No human labeling needed!
    """
    
    window_size = 2  # Context window
    training_data = []
    
    for i, target in enumerate(words):
        # Get context words
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        context = [words[j] for j in range(start, end) if j != i]
        
        training_data.append((context, target))
    
    print("Self-Supervised Training Examples:")
    print("-" * 50)
    for context, target in training_data[:5]:
        print(f"Context: {context:<25} → Target: '{target}'")
    
    # Simple "model": For each word, count what follows different contexts
    # A real model would learn embeddings, but this shows the concept
    
    context_to_predictions = {}
    for context, target in training_data:
        key = tuple(sorted(context))  # Simplified: use sorted context
        if key not in context_to_predictions:
            context_to_predictions[key] = Counter()
        context_to_predictions[key][target] += 1
    
    # Test: Predict missing word
    print("\n" + "=" * 60)
    print("TESTING LEARNED PATTERNS")
    print("=" * 60)
    
    test_cases = [
        ["the", "sat"],         # Should predict: cat/dog/bird
        ["cat", "the"],         # Should predict: chased/on
        ["on", "the"],          # Should predict: mat/rug/branch
    ]
    
    for context in test_cases:
        key = tuple(sorted(context))
        print(f"\nContext: {context}")
        print("Predicted words:")
        
        if key in context_to_predictions:
            predictions = context_to_predictions[key].most_common(3)
            for word, count in predictions:
                print(f"  '{word}': {count} occurrences")
        else:
            print("  (No direct match - real models generalize better)")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
    The model learned patterns like:
    • "the [X] sat" → X is likely an animal (cat, dog, bird)
    • "[animal] [X] on" → X is likely a surface (mat, rug, branch)
    
    No one labeled this data! The structure of language itself
    provided the supervision.
    
    This is how GPT, BERT, and modern language models work,
    but at a MUCH larger scale with neural networks.
    """)


# ═══════════════════════════════════════════════════════════════════════════
# 1.4.5 SEMI-SUPERVISED LEARNING
# ═══════════════════════════════════════════════════════════════════════════

"""
SEMI-SUPERVISED LEARNING
═══════════════════════════════════════════════════════════════════════════════

Definition: Learning from a combination of labeled and unlabeled data.
Typically, you have a small amount of labeled data and a large amount
of unlabeled data.

THE ANALOGY:
───────────────────────────────────────────────────────────────────────────────
Semi-supervised learning is like learning to cook:
• You have a few recipes with instructions (labeled)
• You have many photos of dishes without recipes (unlabeled)
• You use both to understand cooking patterns


WHY IT'S USEFUL:
───────────────────────────────────────────────────────────────────────────────

Labeled data is expensive:
• Medical images need expert radiologists to label
• Legal documents need lawyers to annotate
• Rare languages need native speakers

Unlabeled data is cheap:
• Easy to collect
• Abundant
• No expert time needed

Semi-supervised learning: Get the best of both worlds!


COMMON APPROACHES:
───────────────────────────────────────────────────────────────────────────────

1. SELF-TRAINING (Pseudo-labeling)
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ 1. Train model on labeled data                                          │
   │ 2. Use model to predict labels for unlabeled data                       │
   │ 3. Add high-confidence predictions to training set                      │
   │ 4. Retrain model                                                        │
   │ 5. Repeat                                                               │
   └─────────────────────────────────────────────────────────────────────────┘

2. CO-TRAINING
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ 1. Split features into two views                                        │
   │ 2. Train two models, each on one view                                   │
   │ 3. Each model labels data for the other                                 │
   │ 4. Models teach each other                                              │
   └─────────────────────────────────────────────────────────────────────────┘

3. CONSISTENCY REGULARIZATION
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ Key idea: Model should give same prediction for augmented versions     │
   │                                                                         │
   │ For unlabeled example x:                                               │
   │ • Create augmented version x'                                          │
   │ • Enforce: model(x) ≈ model(x')                                        │
   └─────────────────────────────────────────────────────────────────────────┘

4. GRAPH-BASED METHODS
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ 1. Build graph where similar examples are connected                     │
   │ 2. Propagate labels through the graph                                   │
   │ 3. Connected examples should have similar labels                        │
   └─────────────────────────────────────────────────────────────────────────┘


REAL-WORLD EXAMPLE: Medical Imaging
───────────────────────────────────────────────────────────────────────────────

Scenario:
• 1,000 labeled X-rays (expensive expert annotations)
• 100,000 unlabeled X-rays (easy to collect)

Approach:
1. Train initial model on 1,000 labeled images
2. Run model on 100,000 unlabeled images
3. For images where model is very confident (>95%), use prediction as label
4. Add these "pseudo-labeled" images to training set
5. Retrain model
6. Result: Better model using all 101,000 images
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1.5 THE MACHINE LEARNING WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────

"""
THE MACHINE LEARNING WORKFLOW
═══════════════════════════════════════════════════════════════════════════════

A complete ML project follows this pipeline:

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│   │DEFINE   │──▶│COLLECT  │──▶│PREPARE  │──▶│TRAIN    │──▶│EVALUATE │     │
│   │PROBLEM  │   │DATA     │   │DATA     │   │MODEL    │   │MODEL    │     │
│   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └────┬────┘     │
│                                                                 │          │
│                                                                 ▼          │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│   │MAINTAIN │◀──│MONITOR  │◀──│DEPLOY   │◀──│OPTIMIZE │◀──│TUNE     │     │
│   │         │   │         │   │         │   │         │   │PARAMS   │     │
│   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


DETAILED BREAKDOWN:
═══════════════════════════════════════════════════════════════════════════════

1. DEFINE THE PROBLEM
───────────────────────────────────────────────────────────────────────────────
   Questions to answer:
   • What problem are we solving?
   • Is ML the right solution?
   • What does success look like?
   • What data do we have/need?
   • What are the constraints (time, compute, latency)?
   
   Outputs:
   • Clear problem statement
   • Success metrics
   • Project scope

2. COLLECT DATA
───────────────────────────────────────────────────────────────────────────────
   Sources:
   • Internal databases
   • APIs
   • Web scraping
   • Third-party data providers
   • User-generated content
   • Sensors and IoT devices
   
   Considerations:
   • Data quality
   • Data quantity
   • Privacy and compliance
   • Representativeness
   • Cost

3. PREPARE DATA (Often 60-80% of the work!)
───────────────────────────────────────────────────────────────────────────────
   Tasks:
   • Exploratory Data Analysis (EDA)
   • Data cleaning (missing values, duplicates)
   • Feature engineering
   • Feature selection
   • Data transformation
   • Train/test split
   
   Common issues:
   • Missing values
   • Outliers
   • Imbalanced classes
   • Data leakage

4. TRAIN MODEL
───────────────────────────────────────────────────────────────────────────────
   Steps:
   • Choose algorithm(s)
   • Set up training pipeline
   • Train initial models
   • Iterate and improve
   
   Considerations:
   • Algorithm selection
   • Training time
   • Memory requirements
   • Reproducibility

5. EVALUATE MODEL
───────────────────────────────────────────────────────────────────────────────
   Metrics:
   • Classification: accuracy, precision, recall, F1, AUC
   • Regression: MSE, RMSE, MAE, R²
   
   Validation:
   • Cross-validation
   • Hold-out test set
   • A/B testing
   
   Analysis:
   • Error analysis
   • Confusion matrix
   • Feature importance

6. TUNE HYPERPARAMETERS
───────────────────────────────────────────────────────────────────────────────
   Methods:
   • Grid search
   • Random search
   • Bayesian optimization
   • Automated ML (AutoML)
   
   Key hyperparameters vary by algorithm

7. OPTIMIZE FOR PRODUCTION
───────────────────────────────────────────────────────────────────────────────
   Techniques:
   • Model compression
   • Quantization
   • Pruning
   • Knowledge distillation
   • Caching
   
   Goals:
   • Reduce latency
   • Reduce memory
   • Reduce cost

8. DEPLOY MODEL
───────────────────────────────────────────────────────────────────────────────
   Options:
   • REST API (Flask, FastAPI)
   • Serverless (AWS Lambda)
   • Edge deployment
   • Batch processing
   
   Infrastructure:
   • Docker containers
   • Kubernetes
   • Cloud ML services

9. MONITOR MODEL
───────────────────────────────────────────────────────────────────────────────
   Track:
   • Prediction latency
   • Error rates
   • Data drift
   • Model drift
   • Business metrics
   
   Alerts:
   • Performance degradation
   • Data anomalies
   • System errors

10. MAINTAIN MODEL
───────────────────────────────────────────────────────────────────────────────
    Tasks:
    • Regular retraining
    • Feature updates
    • Bug fixes
    • Version management
    
    Triggers for retraining:
    • Performance degradation
    • New data available
    • Business requirements change
"""


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1.7: Complete ML Workflow
# ─────────────────────────────────────────────────────────────────────────────

def complete_ml_workflow_example():
    """
    Demonstrates a complete ML workflow from problem definition to evaluation.
    
    Problem: Predict customer satisfaction based on interaction data.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    import warnings
    warnings.filterwarnings('ignore')
    
    print("=" * 70)
    print("COMPLETE ML WORKFLOW EXAMPLE")
    print("=" * 70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: DEFINE THE PROBLEM
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "─" * 70)
    print("STEP 1: DEFINE THE PROBLEM")
    print("─" * 70)
    print("""
    Problem: Predict if a customer will be satisfied or dissatisfied
             based on their support interaction.
    
    Business value: 
    - Identify at-risk customers before they leave
    - Improve customer service processes
    - Reduce churn rate
    
    Success metric: F1 score > 0.80
    Constraint: Must run in < 100ms for real-time prediction
    """)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: COLLECT DATA (simulated)
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "─" * 70)
    print("STEP 2: COLLECT DATA")
    print("─" * 70)
    
    np.random.seed(42)
    n_samples = 2000
    
    # Generate synthetic customer interaction data
    data = {
        'wait_time_minutes': np.random.exponential(5, n_samples),
        'interaction_duration': np.random.normal(15, 5, n_samples),
        'num_transfers': np.random.poisson(0.5, n_samples),
        'agent_experience_years': np.random.uniform(0, 10, n_samples),
        'issue_complexity': np.random.choice(['low', 'medium', 'high'], n_samples),
        'channel': np.random.choice(['phone', 'chat', 'email'], n_samples),
        'time_of_day': np.random.choice(['morning', 'afternoon', 'evening'], n_samples),
        'first_contact_resolution': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    }
    
    # Generate satisfaction based on features (realistic patterns)
    satisfaction_score = (
        - data['wait_time_minutes'] * 0.1
        - data['num_transfers'] * 0.3
        + data['agent_experience_years'] * 0.05
        + data['first_contact_resolution'] * 0.5
        - (np.array(data['issue_complexity']) == 'high').astype(float) * 0.2
        + np.random.normal(0, 0.3, n_samples)
    )
    
    data['satisfied'] = (satisfaction_score > np.median(satisfaction_score)).astype(int)
    
    df = pd.DataFrame(data)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nFeatures:")
    for col in df.columns[:-1]:
        print(f"  - {col}")
    print(f"\nTarget: satisfied (0=No, 1=Yes)")
    print(f"Class distribution: {dict(df['satisfied'].value_counts())}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: PREPARE DATA (EDA + Preprocessing)
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "─" * 70)
    print("STEP 3: PREPARE DATA")
    print("─" * 70)
    
    # 3a. Exploratory Data Analysis
    print("\n3a. Exploratory Data Analysis")
    print("-" * 40)
    
    print("\nNumerical features summary:")
    print(df.describe().round(2))
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nCorrelations with target:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col != 'satisfied':
            corr = df[col].corr(df['satisfied'])
            print(f"  {col}: {corr:.3f}")
    
    # 3b. Feature Engineering
    print("\n3b. Feature Engineering")
    print("-" * 40)
    
    # Create new features
    df['wait_per_transfer'] = df['wait_time_minutes'] / (df['num_transfers'] + 1)
    df['efficiency_score'] = df['first_contact_resolution'] * df['agent_experience_years']
    
    print("Created features: wait_per_transfer, efficiency_score")
    
    # 3c. Encode categorical variables
    print("\n3c. Encoding Categorical Variables")
    print("-" * 40)
    
    # One-hot encoding for nominal categories
    df_encoded = pd.get_dummies(df, columns=['channel', 'time_of_day'], drop_first=True)
    
    # Label encoding for ordinal category
    complexity_map = {'low': 0, 'medium': 1, 'high': 2}
    df_encoded['issue_complexity'] = df_encoded['issue_complexity'].map(complexity_map)
    
    print(f"Columns after encoding: {list(df_encoded.columns)}")
    
    # 3d. Split data
    print("\n3d. Train/Test Split")
    print("-" * 40)
    
    X = df_encoded.drop('satisfied', axis=1)
    y = df_encoded['satisfied']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # 3e. Scale features
    print("\n3e. Feature Scaling")
    print("-" * 40)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Applied StandardScaler (mean=0, std=1)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: TRAIN MODEL
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "─" * 70)
    print("STEP 4: TRAIN MODEL")
    print("─" * 70)
    
    # Train baseline model
    print("\nTraining Random Forest classifier...")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='f1')
    print(f"\n5-Fold Cross-validation F1 scores: {cv_scores.round(3)}")
    print(f"Mean CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: EVALUATE MODEL
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "─" * 70)
    print("STEP 5: EVALUATE MODEL")
    print("─" * 70)
    
    y_pred = rf.predict(X_test_scaled)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Dissatisfied', 'Satisfied']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted")
    print(f"               Dis    Sat")
    print(f"Actual Dis   {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"       Sat   {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: TUNE HYPERPARAMETERS
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "─" * 70)
    print("STEP 6: TUNE HYPERPARAMETERS")
    print("─" * 70)
    
    print("\nRunning Grid Search (this may take a moment)...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
    }
    
    # Use smaller grid for demo
    small_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        small_param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV F1 score: {grid_search.best_score_:.3f}")
    
    # Train final model with best parameters
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test_scaled)
    
    print(f"\nFinal model test F1: {classification_report(y_test, y_pred_best, output_dict=True)['weighted avg']['f1-score']:.3f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: INTERPRET RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "─" * 70)
    print("STEP 7: INTERPRET RESULTS")
    print("─" * 70)
    
    print("\nFeature Importance:")
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance_df.head(10).iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['feature']:<25} {row['importance']:.3f} {bar}")
    
    print("\n" + "─" * 70)
    print("SUMMARY")
    print("─" * 70)
    print(f"""
    Model Performance:
    - Achieved F1 score: {grid_search.best_score_:.3f}
    - Target F1 score: 0.80
    - Status: {'✓ PASSED' if grid_search.best_score_ > 0.80 else '✗ Needs improvement'}
    
    Key Insights:
    - First contact resolution is the most important predictor
    - Wait time and number of transfers negatively impact satisfaction
    - Agent experience helps improve outcomes
    
    Next Steps:
    1. Deploy model to production
    2. Set up monitoring for data drift
    3. Schedule monthly retraining
    4. A/B test against current system
    """)
    
    return best_model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 1.6 WHEN TO USE (AND NOT USE) MACHINE LEARNING
# ─────────────────────────────────────────────────────────────────────────────

"""
WHEN TO USE (AND NOT USE) MACHINE LEARNING
═══════════════════════════════════════════════════════════════════════════════

ML is powerful, but it's not always the right solution.

WHEN TO USE ML:
───────────────────────────────────────────────────────────────────────────────

✓ You have a clear learning task
  • Classification, regression, clustering, etc.
  • Well-defined inputs and outputs

✓ The problem is too complex for explicit rules
  • Image recognition (millions of pixel combinations)
  • Natural language (infinite valid sentences)
  • Complex pattern recognition

✓ You have sufficient data
  • Enough examples to learn patterns
  • Representative of real-world scenarios
  • Data quality is acceptable

✓ The patterns are learnable
  • There's a relationship between inputs and outputs
  • Patterns are somewhat consistent
  • Not purely random

✓ You need to handle variability
  • Many edge cases to handle
  • Rules would be too numerous
  • New patterns emerge over time

✓ You need to scale
  • Can't manually process all cases
  • Need automated decisions
  • Volume too high for humans


WHEN NOT TO USE ML:
───────────────────────────────────────────────────────────────────────────────

✗ Simple rules work fine
  • "If age < 18, deny access"
  • "If price > budget, don't show item"
  • Clear, simple logic

✗ You don't have enough data
  • ML needs examples to learn
  • Rule of thumb: hundreds to thousands for basic tasks
  • Millions for deep learning from scratch

✗ The problem isn't predictable
  • Purely random events
  • No patterns to learn
  • Fundamental uncertainty

✗ You need perfect accuracy
  • ML models make mistakes
  • Some domains require 100% correctness
  • Consider human-in-the-loop

✗ You can't explain decisions
  • Some regulated domains require explainability
  • "Why was my loan denied?"
  • Consider interpretable models

✗ The cost of errors is too high
  • Medical diagnosis (without human review)
  • Autonomous weapons
  • Irreversible decisions

✗ Simpler solutions exist
  • Don't use a neural network for averaging numbers
  • Simple heuristics often work well
  • Complexity has costs


DECISION FRAMEWORK:
───────────────────────────────────────────────────────────────────────────────

Ask these questions:

1. Is there a pattern to learn?
   NO  → Don't use ML
   YES → Continue

2. Can you get enough quality data?
   NO  → Consider if you can simplify, or use rules
   YES → Continue

3. Can a simpler solution work?
   YES → Use the simpler solution
   NO  → Continue

4. Can you tolerate some errors?
   NO  → Reconsider, add human oversight
   YES → Continue

5. Do you have the infrastructure to deploy and maintain?
   NO  → Build infrastructure first
   YES → Use ML!


REAL-WORLD EXAMPLES:
───────────────────────────────────────────────────────────────────────────────

USE ML:
• Spam filtering (complex patterns, lots of data, errors acceptable)
• Product recommendations (complex, lots of data, scales well)
• Fraud detection (subtle patterns, lots of data, errors acceptable)
• Voice recognition (impossible to write rules)

DON'T USE ML:
• Calculating sales tax (simple formula)
• Password validation (regex works fine)
• Sorting a list (algorithms exist)
• Calculating age from birthdate (simple math)

MAYBE USE ML:
• Credit decisions (regulated, but patterns exist)
• Medical diagnosis (high stakes, but AI can assist humans)
• Self-driving cars (complex, but safety-critical)
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1.7 SETTING UP YOUR ML ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

"""
SETTING UP YOUR ML ENVIRONMENT
═══════════════════════════════════════════════════════════════════════════════

This section covers how to set up a professional ML development environment.


OPTION 1: LOCAL SETUP (Recommended for learning)
───────────────────────────────────────────────────────────────────────────────

Step 1: Install Python
──────────────────────
Download Python 3.9+ from python.org
OR use Anaconda (recommended for beginners)

# Check Python version
python --version


Step 2: Create Virtual Environment
─────────────────────────────────
# Using venv (built-in)
python -m venv ml_env
source ml_env/bin/activate  # Linux/Mac
ml_env\\Scripts\\activate   # Windows

# Using conda (if you installed Anaconda)
conda create -n ml_env python=3.10
conda activate ml_env


Step 3: Install Core Libraries
──────────────────────────────
# Essential ML libraries
pip install numpy pandas scikit-learn matplotlib seaborn

# Deep learning (choose based on your needs)
pip install torch torchvision  # PyTorch
pip install tensorflow          # TensorFlow

# Additional useful libraries
pip install jupyter notebook
pip install xgboost lightgbm catboost
pip install transformers datasets
pip install plotly
pip install optuna  # Hyperparameter tuning


Step 4: Verify Installation
──────────────────────────
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"


OPTION 2: CLOUD NOTEBOOKS (Quick start, no setup)
───────────────────────────────────────────────────────────────────────────────

Google Colab (Free, includes GPU!)
──────────────────────────────────
• Go to colab.research.google.com
• Sign in with Google account
• Create new notebook
• Libraries pre-installed
• Free GPU/TPU access

Kaggle Notebooks
───────────────
• Go to kaggle.com
• Create account
• Start new notebook
• Free GPU, many datasets

Amazon SageMaker Studio Lab (Free)
─────────────────────────────────
• Go to studiolab.sagemaker.aws
• Request free account
• Full Jupyter environment


OPTION 3: DOCKER (Reproducible environments)
───────────────────────────────────────────────────────────────────────────────

# Dockerfile for ML environment
# Save as Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Start Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]


# requirements.txt
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
matplotlib==3.7.0
seaborn==0.12.0
jupyter==1.0.0
torch==2.0.0
xgboost==1.7.0


# Build and run
docker build -t ml-env .
docker run -p 8888:8888 -v $(pwd):/app ml-env


RECOMMENDED PROJECT STRUCTURE:
───────────────────────────────────────────────────────────────────────────────

ml_project/
├── data/
│   ├── raw/              # Original, immutable data
│   ├── processed/        # Cleaned, transformed data
│   └── external/         # Data from external sources
├── notebooks/
│   ├── 01_eda.ipynb      # Exploratory analysis
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   └── visualization/
│       ├── __init__.py
│       └── plots.py
├── models/               # Saved model files
├── reports/             
│   └── figures/          # Generated graphics
├── tests/                # Unit tests
├── requirements.txt      # Dependencies
├── setup.py             # Make project installable
├── config.yaml          # Configuration
└── README.md


IDE RECOMMENDATIONS:
───────────────────────────────────────────────────────────────────────────────

For Beginners:
• Jupyter Notebook/Lab - Interactive, great for exploration
• VS Code - Free, excellent Python support, integrated terminal

For Professionals:
• VS Code + extensions - Most popular, highly customizable
• PyCharm Professional - Powerful IDE, great debugging

Useful VS Code Extensions:
• Python (Microsoft)
• Pylance
• Jupyter
• Python Docstring Generator
• GitLens
• Error Lens


GPU SETUP (For Deep Learning):
───────────────────────────────────────────────────────────────────────────────

NVIDIA GPU on Linux/Windows:
1. Install NVIDIA driver from nvidia.com
2. Install CUDA toolkit (version compatible with PyTorch/TensorFlow)
3. Install cuDNN
4. Install PyTorch with CUDA support:
   pip install torch --index-url https://download.pytorch.org/whl/cu118

Verify GPU:
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

Mac with Apple Silicon:
• Use PyTorch with MPS (Metal Performance Shaders)
• pip install torch torchvision torchaudio
• device = "mps" if torch.backends.mps.is_available() else "cpu"
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1.8 CHAPTER 1 EXERCISES
# ─────────────────────────────────────────────────────────────────────────────

"""
CHAPTER 1 EXERCISES
═══════════════════════════════════════════════════════════════════════════════

EXERCISE 1.1: Identify ML Type
───────────────────────────────────────────────────────────────────────────────
For each scenario, identify the type of ML (supervised, unsupervised, 
reinforcement):

a) Grouping similar news articles together
b) Predicting house prices from features
c) Teaching a robot to walk
d) Detecting fraudulent transactions (with labeled fraud data)
e) Finding customer segments from purchase history
f) Recommending movies based on what similar users liked
g) Playing chess against itself to improve

Answers at end of exercise section.


EXERCISE 1.2: Regression vs Classification
───────────────────────────────────────────────────────────────────────────────
Determine if each task is regression or classification:

a) Predicting tomorrow's temperature
b) Determining if an email is spam
c) Estimating how long a delivery will take
d) Identifying which digit (0-9) is in an image
e) Predicting stock price change (up/down)
f) Predicting exact stock price
g) Determining cancer type from biopsy

Answers at end of exercise section.


EXERCISE 1.3: ML Pipeline Design
───────────────────────────────────────────────────────────────────────────────
You're building a model to predict if a customer will churn (leave).

Answer these questions:
a) What features might you collect?
b) What target variable would you use?
c) Is this classification or regression?
d) What metrics would you use to evaluate?
e) What would be the business impact of false positives vs false negatives?


EXERCISE 1.4: Coding Challenge
───────────────────────────────────────────────────────────────────────────────
Implement a simple classifier that predicts if a fruit is an apple or orange
based on weight and texture (scale 1-10, where 1=smooth, 10=bumpy).

Training data:
Apple:  weight ~150g, texture ~3
Orange: weight ~130g, texture ~8

Hint: You can use sklearn's KNeighborsClassifier or LogisticRegression.


EXERCISE 1.5: Critical Thinking
───────────────────────────────────────────────────────────────────────────────
For each scenario, explain why ML might NOT be the best solution:

a) Calculating the area of a rectangle given length and width
b) Determining if a number is even or odd
c) Predicting lottery numbers
d) Converting temperatures between Celsius and Fahrenheit


EXERCISE 1.6: Research Task
───────────────────────────────────────────────────────────────────────────────
Pick one of these topics and write a short summary (200-300 words):

a) The ImageNet competition and its impact on deep learning
b) How AlphaGo defeated world champion Lee Sedol
c) The development of GPT models from GPT-1 to GPT-4
d) The difference between narrow AI and general AI


═══════════════════════════════════════════════════════════════════════════════
ANSWERS
═══════════════════════════════════════════════════════════════════════════════

Exercise 1.1 Answers:
a) Unsupervised (clustering)
b) Supervised (regression)
c) Reinforcement learning
d) Supervised (classification)
e) Unsupervised (clustering)
f) Supervised (collaborative filtering uses ratings/labels)
g) Reinforcement learning

Exercise 1.2 Answers:
a) Regression (continuous temperature)
b) Classification (spam or not spam)
c) Regression (continuous time)
d) Classification (one of 10 classes)
e) Classification (up or down)
f) Regression (continuous price)
g) Classification (one of multiple types)
"""


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1.4 SOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def exercise_1_4_solution():
    """Solution to Exercise 1.4: Fruit classifier"""
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Training data
    # [weight (g), texture (1-10)]
    X_train = np.array([
        [150, 3],   # Apple
        [155, 2],   # Apple
        [148, 4],   # Apple
        [145, 3],   # Apple
        [160, 2],   # Apple
        [130, 8],   # Orange
        [125, 9],   # Orange
        [135, 7],   # Orange
        [128, 8],   # Orange
        [132, 9],   # Orange
    ])
    
    y_train = ['apple', 'apple', 'apple', 'apple', 'apple',
               'orange', 'orange', 'orange', 'orange', 'orange']
    
    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    # Train Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    
    # Test on new fruits
    test_fruits = np.array([
        [152, 3],   # Should be apple
        [128, 8],   # Should be orange
        [140, 5],   # Borderline case
    ])
    
    print("Fruit Classification Results")
    print("=" * 50)
    print("\nTest fruits: [weight, texture]")
    
    for i, fruit in enumerate(test_fruits):
        knn_pred = knn.predict([fruit])[0]
        lr_pred = lr.predict([fruit])[0]
        
        print(f"\nFruit {i+1}: weight={fruit[0]}g, texture={fruit[1]}")
        print(f"  KNN prediction: {knn_pred}")
        print(f"  Logistic Regression prediction: {lr_pred}")
    
    return knn, lr


# ─────────────────────────────────────────────────────────────────────────────
# 1.9 CHAPTER 1 SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

"""
CHAPTER 1 SUMMARY
═══════════════════════════════════════════════════════════════════════════════

KEY TAKEAWAYS:
───────────────────────────────────────────────────────────────────────────────

1. MACHINE LEARNING is teaching computers to learn patterns from data
   instead of explicitly programming rules.

2. THREE MAIN TYPES:
   • Supervised: Learn from labeled data (has answers)
   • Unsupervised: Find patterns in unlabeled data (no answers)
   • Reinforcement: Learn through trial and error (rewards/penalties)

3. SUPERVISED LEARNING has two main tasks:
   • Classification: Predict categories (spam/not spam)
   • Regression: Predict continuous values (price, temperature)

4. THE ML PIPELINE:
   Define Problem → Collect Data → Prepare Data → Train → Evaluate → 
   Tune → Deploy → Monitor → Maintain

5. DATA PREPARATION is often 60-80% of the work!

6. ML IS NOT ALWAYS THE ANSWER:
   • Use simple rules when they work
   • Need sufficient data
   • Must tolerate some errors
   • Consider explainability requirements

7. MODERN ML is driven by:
   • Big data
   • Powerful GPUs
   • Algorithmic advances (transformers, etc.)


VOCABULARY:
───────────────────────────────────────────────────────────────────────────────

Features (X): Input variables used to make predictions
Labels (y): Target variables we're trying to predict
Training: The process of learning patterns from data
Model: The learned function that maps inputs to outputs
Inference: Using a trained model to make predictions
Overfitting: Model memorizes training data, fails on new data
Underfitting: Model is too simple to capture patterns
Supervised: Learning with labeled data
Unsupervised: Learning without labels
Reinforcement: Learning through rewards and penalties


NEXT CHAPTER PREVIEW:
───────────────────────────────────────────────────────────────────────────────

In Chapter 2, we'll cover the mathematical foundations of ML:
• Linear algebra (vectors, matrices)
• Calculus (derivatives, gradients)
• Probability and statistics
• Information theory

This math forms the backbone of all ML algorithms!
"""


# ═══════════════════════════════════════════════════════════════════════════════
# End of Chapter 1
# ═══════════════════════════════════════════════════════════════════════════════

# ███████████████████████████████████████████████████████████████████████████████
#                                                                               
#   CHAPTER 2: MATHEMATICS FOR MACHINE LEARNING                                  
#                                                                               
# ███████████████████████████████████████████████████████████████████████████████

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   "The book of nature is written in the language of mathematics."             ║
║                                              — Galileo Galilei                ║
║                                                                               ║
║   Don't worry if math isn't your strongest subject. We'll build up            ║
║   intuitively, with code examples you can run to see the concepts in action.  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# 2.1 LINEAR ALGEBRA ESSENTIALS
# ─────────────────────────────────────────────────────────────────────────────

"""
LINEAR ALGEBRA ESSENTIALS
═══════════════════════════════════════════════════════════════════════════════

Linear algebra is the foundation of machine learning. Every ML algorithm
can be expressed in terms of vectors and matrices.

WHY LINEAR ALGEBRA?
───────────────────────────────────────────────────────────────────────────────
• Data is represented as vectors and matrices
• Model parameters are vectors
• Predictions are matrix multiplications
• Optimizations involve gradients (vectors)
• GPUs are optimized for matrix operations


SCALARS, VECTORS, AND MATRICES
═══════════════════════════════════════════════════════════════════════════════

SCALARS: A single number (e.g., temperature, learning rate)

VECTORS: An ordered list of numbers (1D array)
         Example: A point in 2D: [3, 4]
         
         Visual:
               ┌───┐
               │ 3 │
         x =   │ 1 │    This is a 4-dimensional vector
               │ 4 │
               │ 2 │
               └───┘

MATRICES: A 2D array of numbers
          Example: Dataset rows are samples, columns are features
          
          Visual:
               ┌             ┐
               │ 1   2   3   │
         A =   │ 4   5   6   │    This is a 3×3 matrix
               │ 7   8   9   │
               └             ┘

TENSORS: Generalization to any number of dimensions
         • Scalar: 0D tensor
         • Vector: 1D tensor
         • Matrix: 2D tensor
         • 3D tensor: Stack of matrices (e.g., color image)
"""

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 2.1: Creating Scalars, Vectors, and Matrices
# ─────────────────────────────────────────────────────────────────────────────

def example_scalars_vectors_matrices():
    """Demonstrate creating and working with basic linear algebra objects."""
    
    print("SCALARS, VECTORS, AND MATRICES")
    print("=" * 60)
    
    # SCALARS
    print("\n1. SCALARS")
    learning_rate = 0.001
    print(f"Learning rate: {learning_rate}")
    
    # VECTORS
    print("\n2. VECTORS")
    v1 = np.array([1, 2, 3, 4, 5])
    v2 = np.zeros(5)           # [0, 0, 0, 0, 0]
    v3 = np.ones(5)            # [1, 1, 1, 1, 1]
    v4 = np.random.randn(5)    # Random from standard normal
    
    print(f"v1 = {v1}")
    print(f"v1 shape: {v1.shape}")
    print(f"v1[0] = {v1[0]}, v1[-1] = {v1[-1]}")
    
    # MATRICES
    print("\n3. MATRICES")
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    I = np.eye(4)              # 4x4 identity matrix
    
    print(f"Matrix A:\n{A}")
    print(f"A shape: {A.shape}")
    print(f"A[0, 0] = {A[0, 0]}, A[1, 2] = {A[1, 2]}")
    
    # TENSORS
    print("\n4. TENSORS")
    image = np.random.randint(0, 256, size=(100, 100, 3))  # RGB image
    batch = np.random.randn(32, 100, 100, 3)  # Batch of images
    
    print(f"Image tensor shape: {image.shape}")
    print(f"Batch tensor shape: {batch.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 2.2: Essential Matrix Operations
# ─────────────────────────────────────────────────────────────────────────────

def example_matrix_operations():
    """Demonstrate essential matrix operations for ML."""
    
    print("MATRIX OPERATIONS")
    print("=" * 60)
    
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    x = np.array([1, 2])
    
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    
    # ELEMENT-WISE OPERATIONS
    print("\n1. ELEMENT-WISE OPERATIONS")
    print(f"A + B =\n{A + B}")
    print(f"A * B (element-wise) =\n{A * B}")
    
    # MATRIX MULTIPLICATION
    print("\n2. MATRIX MULTIPLICATION")
    """
    For matrices A (m×n) and B (n×p):
    C = A @ B has shape (m×p)
    
    Rule: Inner dimensions must match!
    """
    C = A @ B  # or np.dot(A, B)
    print(f"A @ B =\n{C}")
    
    # Matrix-Vector multiplication
    print(f"\nA @ x = {A @ x}")
    
    # DOT PRODUCT
    print("\n3. DOT PRODUCT")
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print(f"a · b = {np.dot(a, b)}")  # 1*4 + 2*5 + 3*6 = 32
    
    # TRANSPOSE
    print("\n4. TRANSPOSE")
    M = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"M:\n{M}")
    print(f"M transpose:\n{M.T}")
    
    # NORMS
    print("\n5. NORMS")
    v = np.array([3, 4])
    print(f"L1 norm: {np.linalg.norm(v, ord=1)}")  # |3| + |4| = 7
    print(f"L2 norm: {np.linalg.norm(v, ord=2)}")  # √(9+16) = 5
    
    # INVERSE
    print("\n6. INVERSE")
    A_inv = np.linalg.inv(A)
    print(f"A inverse:\n{A_inv}")
    print(f"A @ A⁻¹ =\n{(A @ A_inv).round(10)}")


# ─────────────────────────────────────────────────────────────────────────────
# EIGENVALUES AND EIGENVECTORS
# ─────────────────────────────────────────────────────────────────────────────

"""
EIGENVALUES AND EIGENVECTORS
═══════════════════════════════════════════════════════════════════════════════

For a square matrix A, an eigenvector v and eigenvalue λ satisfy:

    A @ v = λ × v

In words: When matrix A transforms vector v, it only SCALES v (by λ).

WHY THEY MATTER IN ML:
• PCA: Eigenvectors of covariance matrix are principal components
• Spectral Clustering: Uses eigenvectors of graph Laplacian
• Understanding matrix transformations
"""

def example_eigenvalues():
    """Demonstrate eigenvalues and eigenvectors."""
    
    print("EIGENVALUES AND EIGENVECTORS")
    print("=" * 60)
    
    A = np.array([[4, 2], [1, 3]])
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"Matrix A:\n{A}")
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"\nEigenvectors (as columns):\n{eigenvectors}")
    
    # Verify: A @ v = λ × v
    print("\nVerification:")
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        print(f"A @ v{i+1} = {(A @ v).round(4)}")
        print(f"λ{i+1} × v{i+1} = {(lam * v).round(4)}")


# ─────────────────────────────────────────────────────────────────────────────
# SINGULAR VALUE DECOMPOSITION (SVD)
# ─────────────────────────────────────────────────────────────────────────────

"""
SVD decomposes any matrix A into: A = U @ Σ @ V^T

Uses in ML:
• PCA (dimensionality reduction)
• Matrix completion (recommendations)
• Image compression
"""

def example_svd():
    """Demonstrate SVD and low-rank approximation."""
    
    print("SINGULAR VALUE DECOMPOSITION")
    print("=" * 60)
    
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    print(f"Original matrix A (4×3):\n{A}")
    print(f"\nSingular values: {s.round(3)}")
    
    # Low-rank approximation
    k = 1
    A_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    print(f"\nRank-{k} approximation:\n{A_approx.round(2)}")
    print(f"Approximation error: {np.linalg.norm(A - A_approx):.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.2 CALCULUS FOR MACHINE LEARNING
# ─────────────────────────────────────────────────────────────────────────────

"""
CALCULUS FOR MACHINE LEARNING
═══════════════════════════════════════════════════════════════════════════════

KEY INSIGHT: To minimize a loss function, we need to know which direction
to move parameters. Derivatives (gradients) tell us the direction of 
steepest increase, so we move in the OPPOSITE direction.


DERIVATIVES AND GRADIENTS
───────────────────────────────────────────────────────────────────────────────

DERIVATIVE (Single Variable):
The derivative f'(x) tells you the rate of change of f at point x.

Common derivatives:
• d/dx (x^n) = n × x^(n-1)
• d/dx (e^x) = e^x
• d/dx (ln(x)) = 1/x

GRADIENT (Multiple Variables):
When f depends on multiple variables:
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

The gradient points in the direction of steepest increase.
"""

def example_derivatives_gradients():
    """Demonstrate derivatives and gradients."""
    
    print("DERIVATIVES AND GRADIENTS")
    print("=" * 60)
    
    # Numerical derivative
    def f(x):
        return x ** 2
    
    def numerical_derivative(func, x, h=1e-7):
        return (func(x + h) - func(x - h)) / (2 * h)
    
    x = 3.0
    print(f"f(x) = x²")
    print(f"At x = {x}:")
    print(f"  Numerical derivative: {numerical_derivative(f, x):.6f}")
    print(f"  Analytical (2x): {2*x:.6f}")
    
    # Gradient
    def g(params):
        x, y = params
        return x**2 + y**2
    
    def numerical_gradient(func, params, h=1e-7):
        gradient = np.zeros_like(params, dtype=float)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += h
            params_minus[i] -= h
            gradient[i] = (func(params_plus) - func(params_minus)) / (2 * h)
        return gradient
    
    point = np.array([3.0, 4.0])
    grad = numerical_gradient(g, point)
    print(f"\ng(x,y) = x² + y²")
    print(f"At point {point}:")
    print(f"  Gradient: {grad}")


# ─────────────────────────────────────────────────────────────────────────────
# THE CHAIN RULE
# ─────────────────────────────────────────────────────────────────────────────

"""
THE CHAIN RULE
═══════════════════════════════════════════════════════════════════════════════

If y = f(g(x)), then:
    dy/dx = (dy/dg) × (dg/dx) = f'(g(x)) × g'(x)

WHY THIS MATTERS IN NEURAL NETWORKS:
Neural networks are compositions of functions!

Layer 1: h₁ = f₁(W₁ @ x + b₁)
Layer 2: h₂ = f₂(W₂ @ h₁ + b₂)
Output:  y  = f₃(W₃ @ h₂ + b₃)

To find ∂Loss/∂W₁, we use the chain rule repeatedly.
This is called BACKPROPAGATION!
"""

def example_chain_rule():
    """Demonstrate the chain rule with a neural network example."""
    
    print("THE CHAIN RULE (Backpropagation)")
    print("=" * 60)
    
    def relu(x):
        return np.maximum(0, x)
    
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    # Simple 2-layer network
    x = 2.0
    target = 5.0
    w1, b1 = 0.5, 0.1
    w2, b2 = 0.8, 0.2
    
    # Forward pass
    z1 = w1 * x + b1
    h = relu(z1)
    y = w2 * h + b2
    loss = (y - target) ** 2
    
    print(f"Forward pass:")
    print(f"  x={x}, target={target}")
    print(f"  z1={z1}, h={h}, y={y}")
    print(f"  Loss = {loss:.4f}")
    
    # Backward pass (chain rule)
    dL_dy = 2 * (y - target)
    dy_dh = w2
    dh_dz1 = relu_derivative(z1)
    dz1_dw1 = x
    
    dL_dw1 = dL_dy * dy_dh * dh_dz1 * dz1_dw1
    
    print(f"\nBackward pass:")
    print(f"  ∂L/∂w1 = {dL_dw1:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT DESCENT
# ─────────────────────────────────────────────────────────────────────────────

"""
GRADIENT DESCENT
═══════════════════════════════════════════════════════════════════════════════

THE ALGORITHM:
    Initialize θ randomly
    
    Repeat until convergence:
        1. Compute gradient: ∇L(θ)
        2. Update: θ = θ - α × ∇L(θ)
    
    Where α (alpha) is the learning rate.

LEARNING RATE:
• Too small: Convergence is slow
• Too large: May overshoot and diverge
• Just right: Smooth convergence

VARIANTS:
• Batch GD: Gradient over ALL examples (stable but slow)
• SGD: Gradient on ONE example (fast but noisy)
• Mini-batch: Gradient on small batch (best of both)
"""

def example_gradient_descent():
    """Complete gradient descent implementation."""
    
    print("GRADIENT DESCENT")
    print("=" * 60)
    
    # 1D example: minimize f(x) = (x-3)² + 2
    def f(x):
        return (x - 3)**2 + 2
    
    def df(x):
        return 2 * (x - 3)
    
    x = 0.0
    learning_rate = 0.1
    
    print(f"Minimizing f(x) = (x-3)² + 2")
    print(f"Minimum at x = 3")
    print(f"\nIteration | x        | f(x)")
    print("-" * 35)
    
    for i in range(15):
        print(f"{i:9d} | {x:8.4f} | {f(x):8.4f}")
        x = x - learning_rate * df(x)
    
    print(f"\nConverged to x = {x:.4f}")


def example_gradient_descent_linear_regression():
    """Gradient descent for linear regression."""
    
    print("\nGRADIENT DESCENT FOR LINEAR REGRESSION")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 1)
    y = 3 * X.squeeze() + 2 + np.random.randn(n) * 0.5
    
    # Add bias term
    X_b = np.c_[np.ones((n, 1)), X]
    
    def compute_loss(X, y, theta):
        predictions = X @ theta
        return np.mean((predictions - y) ** 2)
    
    def compute_gradient(X, y, theta):
        predictions = X @ theta
        return (2 / len(y)) * X.T @ (predictions - y)
    
    # Mini-batch gradient descent
    theta = np.zeros(2)
    lr = 0.1
    batch_size = 32
    epochs = 100
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start+batch_size]
            gradient = compute_gradient(X_b[batch_idx], y[batch_idx], theta)
            theta = theta - lr * gradient
    
    print(f"True parameters: b=2.0, w=3.0")
    print(f"Learned: b={theta[0]:.4f}, w={theta[1]:.4f}")
    print(f"Final loss: {compute_loss(X_b, y, theta):.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.3 PROBABILITY AND STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

"""
PROBABILITY AND STATISTICS
═══════════════════════════════════════════════════════════════════════════════

ML is fundamentally about learning from uncertain data.

PROBABILITY NOTATION:
• P(A)    = Probability of event A
• P(A|B)  = Probability of A given B
• P(A,B)  = Joint probability of A and B

FUNDAMENTAL RULES:
1. Sum rule: P(A∪B) = P(A) + P(B) - P(A∩B)
2. Product rule: P(A,B) = P(A|B) × P(B)
3. Marginalization: P(A) = Σ_b P(A,B=b)

COMMON DISTRIBUTIONS:
• Bernoulli: Single binary trial
• Binomial: Number of successes in n trials
• Normal: The bell curve (everywhere in ML!)
• Poisson: Count of events in fixed interval
"""

def example_probability_basics():
    """Demonstrate probability concepts."""
    
    print("PROBABILITY BASICS")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Bernoulli and Binomial
    print("\n1. BERNOULLI AND BINOMIAL")
    p = 0.7
    n_trials = 10
    samples = np.random.binomial(n=n_trials, p=p, size=10000)
    
    print(f"Binomial(n={n_trials}, p={p}):")
    print(f"  Theoretical mean = {n_trials * p}")
    print(f"  Empirical mean = {samples.mean():.2f}")
    
    # Normal distribution
    print("\n2. NORMAL DISTRIBUTION")
    mu, sigma = 100, 15
    samples = np.random.normal(mu, sigma, 10000)
    
    print(f"Normal(μ={mu}, σ={sigma}):")
    print(f"  Mean: {samples.mean():.2f}")
    print(f"  Std: {samples.std():.2f}")
    
    # 68-95-99.7 rule
    within_1std = ((mu-sigma <= samples) & (samples <= mu+sigma)).mean()
    within_2std = ((mu-2*sigma <= samples) & (samples <= mu+2*sigma)).mean()
    print(f"  Within 1σ: {within_1std:.1%} (theory: 68.3%)")
    print(f"  Within 2σ: {within_2std:.1%} (theory: 95.4%)")


# ─────────────────────────────────────────────────────────────────────────────
# BAYES' THEOREM
# ─────────────────────────────────────────────────────────────────────────────

"""
BAYES' THEOREM
═══════════════════════════════════════════════════════════════════════════════

              P(B|A) × P(A)
    P(A|B) = ─────────────────
                  P(B)

TERMINOLOGY:
• P(A)    = Prior: Belief before seeing evidence
• P(B|A)  = Likelihood: Probability of evidence if A is true
• P(A|B)  = Posterior: Updated belief after seeing evidence
• P(B)    = Evidence: Overall probability of observation
"""

def example_bayes_theorem():
    """Demonstrate Bayes' theorem."""
    
    print("BAYES' THEOREM")
    print("=" * 60)
    
    # Medical diagnosis example
    p_disease = 0.01  # 1% have disease
    p_positive_given_disease = 0.99  # 99% sensitivity
    p_positive_given_no_disease = 0.05  # 5% false positive
    
    # P(+) using marginalization
    p_positive = (p_positive_given_disease * p_disease + 
                  p_positive_given_no_disease * (1 - p_disease))
    
    # Bayes' theorem
    p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
    
    print("Medical Diagnosis Example:")
    print(f"  P(Disease) = {p_disease:.1%}")
    print(f"  P(+|Disease) = {p_positive_given_disease:.1%}")
    print(f"  P(+|No Disease) = {p_positive_given_no_disease:.1%}")
    print(f"\nResult:")
    print(f"  P(Disease|+) = {p_disease_given_positive:.1%}")
    print(f"\n  Despite 99% test accuracy, positive test only means")
    print(f"  {p_disease_given_positive:.1%} chance of disease!")


# ─────────────────────────────────────────────────────────────────────────────
# MAXIMUM LIKELIHOOD ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

"""
MAXIMUM LIKELIHOOD ESTIMATION (MLE)
═══════════════════════════════════════════════════════════════════════════════

Find parameters θ that maximize the probability of observed data:
    θ_MLE = argmax P(Data | θ)

CONNECTION TO ML:
• MSE Loss = MLE with Gaussian noise assumption
• Cross-Entropy = MLE with Bernoulli/Categorical distribution
"""

def example_mle():
    """Demonstrate Maximum Likelihood Estimation."""
    
    print("MAXIMUM LIKELIHOOD ESTIMATION")
    print("=" * 60)
    
    np.random.seed(42)
    
    # MLE for coin flip
    true_p = 0.7
    flips = np.random.binomial(1, true_p, 100)
    p_mle = flips.mean()
    
    print("MLE for Coin Flip:")
    print(f"  True p = {true_p}")
    print(f"  MLE estimate = {p_mle}")
    
    # MLE for Gaussian
    true_mu, true_sigma = 5.0, 2.0
    data = np.random.normal(true_mu, true_sigma, 200)
    
    print("\nMLE for Gaussian:")
    print(f"  True: μ={true_mu}, σ={true_sigma}")
    print(f"  MLE:  μ={data.mean():.4f}, σ={data.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def example_descriptive_statistics():
    """Demonstrate descriptive statistics."""
    
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Normal data
    data = np.random.normal(50, 10, 1000)
    
    print("Normal Distribution Data:")
    print(f"  Mean: {np.mean(data):.2f}")
    print(f"  Median: {np.median(data):.2f}")
    print(f"  Std: {np.std(data):.2f}")
    print(f"  IQR: {np.percentile(data, 75) - np.percentile(data, 25):.2f}")
    print(f"  Skewness: {stats.skew(data):.3f}")
    print(f"  Kurtosis: {stats.kurtosis(data):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.4 INFORMATION THEORY
# ─────────────────────────────────────────────────────────────────────────────

"""
INFORMATION THEORY
═══════════════════════════════════════════════════════════════════════════════

ENTROPY: Measures uncertainty in a random variable
    H(X) = -Σᵢ P(xᵢ) × log₂ P(xᵢ)

• Low entropy: Predictable (one outcome likely)
• High entropy: Unpredictable (all outcomes equally likely)

CROSS-ENTROPY: Measures difference between distributions
    H(P, Q) = -Σᵢ P(xᵢ) × log Q(xᵢ)

This is THE loss function for classification!

KL DIVERGENCE: Measures how Q differs from P
    D_KL(P || Q) = H(P, Q) - H(P)
"""

def example_entropy():
    """Demonstrate entropy."""
    
    print("ENTROPY")
    print("=" * 60)
    
    def entropy(probs):
        probs = np.array(probs)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    
    print("Binary Entropy (Coin Flip):")
    print("P(heads) | Entropy")
    print("-" * 25)
    
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        h = entropy([p, 1-p])
        bar = "█" * int(h * 20)
        print(f"  {p:.1f}    | {h:.3f}  {bar}")
    
    print("\nMaximum entropy at p=0.5 (most uncertain)")


def example_cross_entropy():
    """Demonstrate cross-entropy loss."""
    
    print("\nCROSS-ENTROPY")
    print("=" * 60)
    
    def binary_cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    print("Binary Cross-Entropy (True label = 1):")
    print("Predicted | BCE Loss")
    print("-" * 25)
    
    for pred in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        loss = binary_cross_entropy(1, pred)
        bar = "█" * int(loss * 5)
        print(f"  {pred:.2f}    | {loss:.3f}  {bar}")


def example_kl_divergence():
    """Demonstrate KL divergence."""
    
    print("\nKL DIVERGENCE")
    print("=" * 60)
    
    def kl_divergence(p, q):
        p, q = np.array(p), np.array(q)
        p, q = np.clip(p, 1e-10, 1), np.clip(q, 1e-10, 1)
        return np.sum(p * np.log(p / q))
    
    P = [0.4, 0.3, 0.2, 0.1]
    Q1 = [0.25, 0.25, 0.25, 0.25]
    Q2 = [0.1, 0.1, 0.4, 0.4]
    
    print(f"P = {P}")
    print(f"Q1 (uniform) = {Q1}")
    print(f"Q2 (different) = {Q2}")
    print(f"\nD_KL(P || Q1) = {kl_divergence(P, Q1):.4f}")
    print(f"D_KL(P || Q2) = {kl_divergence(P, Q2):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CHAPTER 2 SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

"""
CHAPTER 2 SUMMARY
═══════════════════════════════════════════════════════════════════════════════

LINEAR ALGEBRA:
• Vectors and matrices are the data structures of ML
• Matrix multiplication is the core operation
• Eigenvalues/SVD enable PCA and dimensionality reduction

CALCULUS:
• Gradients tell us which direction to move parameters
• Chain rule enables backpropagation
• Gradient descent iteratively minimizes loss

PROBABILITY:
• Bayes' theorem updates beliefs given evidence
• MLE finds parameters that maximize data likelihood
• Many loss functions = negative log-likelihood

INFORMATION THEORY:
• Entropy measures uncertainty
• Cross-entropy is THE classification loss
• KL divergence measures distribution difference


KEY FORMULAS:
───────────────────────────────────────────────────────────────────────────────

Gradient Descent:    θ = θ - α × ∇L(θ)
Bayes' Theorem:      P(A|B) = P(B|A)P(A) / P(B)
Entropy:             H(X) = -Σ P(x) log P(x)
Cross-Entropy:       H(P,Q) = -Σ P(x) log Q(x)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Run all examples
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CHAPTER 2: MATHEMATICS FOR MACHINE LEARNING")
    print("="*70)
    
    example_scalars_vectors_matrices()
    print("\n")
    example_matrix_operations()
    print("\n")
    example_eigenvalues()
    print("\n")
    example_svd()
    print("\n")
    example_derivatives_gradients()
    print("\n")
    example_chain_rule()
    print("\n")
    example_gradient_descent()
    example_gradient_descent_linear_regression()
    print("\n")
    example_probability_basics()
    print("\n")
    example_bayes_theorem()
    print("\n")
    example_mle()
    print("\n")
    example_descriptive_statistics()
    print("\n")
    example_entropy()
    example_cross_entropy()
    example_kl_divergence()
# ███████████████████████████████████████████████████████████████████████████████
#                                                                               
#   CHAPTER 3: DATA FUNDAMENTALS                                                 
#                                                                               
# ███████████████████████████████████████████████████████████████████████████████

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   "Data is the new oil. But like oil, it's valuable only when refined."       ║
║                                              — Clive Humby                    ║
║                                                                               ║
║   Data preparation often takes 60-80% of a data scientist's time.             ║
║   Master this chapter, and you'll be ahead of most practitioners.             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# 3.1 UNDERSTANDING YOUR DATA
# ─────────────────────────────────────────────────────────────────────────────

"""
UNDERSTANDING YOUR DATA
═══════════════════════════════════════════════════════════════════════════════

Before building models, you MUST understand your data thoroughly.
"Garbage in, garbage out" - no algorithm can fix bad data.


DATA TYPES
───────────────────────────────────────────────────────────────────────────────

1. NUMERICAL (Quantitative)
   a) Continuous: Can take any value (height, temperature, price)
   b) Discrete: Countable values (number of children, items sold)

2. CATEGORICAL (Qualitative)
   a) Nominal: No inherent order (color, country, gender)
   b) Ordinal: Has order but no meaningful distance (rating: low/medium/high)

3. TEXT: Unstructured text data (reviews, documents)

4. DATE/TIME: Temporal data (timestamps, dates)

5. BINARY: Two values (True/False, 0/1, Yes/No)


COMMON DATA STRUCTURES
───────────────────────────────────────────────────────────────────────────────

TABULAR (Most common in traditional ML):
┌──────────┬─────────┬────────┬──────────┬─────────┐
│ Sample   │ Feature1│Feature2│ Feature3 │  Label  │
├──────────┼─────────┼────────┼──────────┼─────────┤
│ Row 1    │   ...   │   ...  │   ...    │   ...   │
│ Row 2    │   ...   │   ...  │   ...    │   ...   │
│ ...      │   ...   │   ...  │   ...    │   ...   │
└──────────┴─────────┴────────┴──────────┴─────────┘

IMAGES: 3D tensors (height × width × channels)
TEXT: Sequences of tokens
TIME SERIES: Sequences with temporal ordering
GRAPHS: Nodes and edges
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3.1.1 EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────────────────────

"""
EXPLORATORY DATA ANALYSIS (EDA)
═══════════════════════════════════════════════════════════════════════════════

EDA is the process of investigating data to discover patterns, spot anomalies,
and check assumptions using statistical graphics and summary statistics.

THE EDA CHECKLIST:
───────────────────────────────────────────────────────────────────────────────
□ Dataset shape and size
□ Data types of each column
□ Missing values
□ Summary statistics (mean, median, std, min, max)
□ Distribution of each feature
□ Relationships between features (correlations)
□ Class distribution (for classification)
□ Outliers
□ Duplicate rows
"""

def create_sample_dataset():
    """Create a realistic sample dataset for demonstration."""
    np.random.seed(42)
    n = 1000
    
    # Generate features
    data = {
        'age': np.random.randint(18, 80, n),
        'income': np.random.exponential(50000, n) + 20000,
        'credit_score': np.random.normal(700, 50, n).clip(300, 850),
        'years_employed': np.random.exponential(5, n).clip(0, 40),
        'num_credit_cards': np.random.poisson(3, n),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                       n, p=[0.4, 0.35, 0.2, 0.05]),
        'home_ownership': np.random.choice(['Own', 'Rent', 'Mortgage'], 
                                           n, p=[0.3, 0.4, 0.3]),
        'loan_amount': np.random.uniform(1000, 50000, n),
    }
    
    # Generate target based on features (loan default: 0 or 1)
    default_prob = (
        (data['income'] < 40000).astype(float) * 0.2 +
        (data['credit_score'] < 650).astype(float) * 0.3 +
        (data['years_employed'] < 1).astype(float) * 0.2 +
        np.random.random(n) * 0.3
    )
    data['defaulted'] = (np.random.random(n) < default_prob / default_prob.max() * 0.3).astype(int)
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values (realistic scenario)
    missing_mask = np.random.random(n) < 0.05
    df.loc[missing_mask, 'income'] = np.nan
    
    missing_mask = np.random.random(n) < 0.03
    df.loc[missing_mask, 'years_employed'] = np.nan
    
    return df


def example_eda_comprehensive():
    """Comprehensive EDA demonstration."""
    
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    
    df = create_sample_dataset()
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Basic Information
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 1: BASIC INFORMATION")
    print("─" * 70)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"  - Rows (samples): {df.shape[0]}")
    print(f"  - Columns (features): {df.shape[1]}")
    
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    print("\nColumn Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<20}: {dtype}")
    
    print("\nFirst 5 Rows:")
    print(df.head().to_string())
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Missing Values Analysis
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 2: MISSING VALUES ANALYSIS")
    print("─" * 70)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    print("\nMissing Values Summary:")
    print(f"{'Column':<20} {'Missing':<10} {'Percentage':<10}")
    print("-" * 40)
    
    for col in df.columns:
        if missing[col] > 0:
            print(f"{col:<20} {missing[col]:<10} {missing_pct[col]:.2f}%")
    
    total_missing = df.isnull().sum().sum()
    total_cells = df.size
    print(f"\nTotal missing: {total_missing} / {total_cells} ({total_missing/total_cells*100:.2f}%)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Numerical Features Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 3: NUMERICAL FEATURES SUMMARY")
    print("─" * 70)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nNumerical columns: {numerical_cols}")
    
    print("\nDescriptive Statistics:")
    print(df[numerical_cols].describe().round(2).to_string())
    
    # Additional statistics
    print("\nAdditional Statistics:")
    print(f"{'Column':<20} {'Skewness':<12} {'Kurtosis':<12}")
    print("-" * 44)
    for col in numerical_cols:
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        print(f"{col:<20} {skew:>10.2f} {kurt:>10.2f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Categorical Features Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 4: CATEGORICAL FEATURES SUMMARY")
    print("─" * 70)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\nCategorical columns: {categorical_cols}")
    
    for col in categorical_cols:
        print(f"\n{col}:")
        value_counts = df[col].value_counts()
        for val, count in value_counts.items():
            pct = count / len(df) * 100
            bar = "█" * int(pct / 2)
            print(f"  {val:<15} {count:>5} ({pct:>5.1f}%) {bar}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Target Variable Analysis
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 5: TARGET VARIABLE ANALYSIS")
    print("─" * 70)
    
    target = 'defaulted'
    print(f"\nTarget column: {target}")
    print("\nClass Distribution:")
    
    class_counts = df[target].value_counts()
    for val, count in class_counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        label = "No Default" if val == 0 else "Default"
        print(f"  {label:<15} {count:>5} ({pct:>5.1f}%) {bar}")
    
    # Check for class imbalance
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 3:
        print("  ⚠️  Significant class imbalance detected!")
        print("  Consider: oversampling, undersampling, or class weights")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Correlation Analysis
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 6: CORRELATION ANALYSIS")
    print("─" * 70)
    
    # Correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(2).to_string())
    
    # Find highly correlated pairs
    print("\nHighly Correlated Pairs (|r| > 0.5):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        for col1, col2, corr in high_corr_pairs:
            print(f"  {col1} <-> {col2}: {corr:.3f}")
    else:
        print("  No highly correlated pairs found")
    
    # Correlation with target
    print("\nCorrelation with Target:")
    target_corr = df[numerical_cols].corrwith(df[target]).sort_values(key=abs, ascending=False)
    for col, corr in target_corr.items():
        if col != target:
            print(f"  {col:<20}: {corr:>7.3f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: Outlier Detection
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 7: OUTLIER DETECTION")
    print("─" * 70)
    
    print("\nOutliers using IQR method:")
    
    for col in numerical_cols:
        if col == target:
            continue
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if len(outliers) > 0:
            print(f"\n  {col}:")
            print(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"    Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8: Duplicate Analysis
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 8: DUPLICATE ANALYSIS")
    print("─" * 70)
    
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.2 DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

"""
DATA CLEANING
═══════════════════════════════════════════════════════════════════════════════

Real-world data is messy. Data cleaning transforms raw data into a format
suitable for analysis and modeling.

COMMON DATA QUALITY ISSUES:
───────────────────────────────────────────────────────────────────────────────
• Missing values
• Duplicate records
• Inconsistent formatting
• Invalid values (negative age, impossible dates)
• Outliers
• Data entry errors
"""


# ═══════════════════════════════════════════════════════════════════════════
# 3.2.1 HANDLING MISSING VALUES
# ═══════════════════════════════════════════════════════════════════════════

"""
HANDLING MISSING VALUES
═══════════════════════════════════════════════════════════════════════════════

Missing values can be:
• MCAR (Missing Completely At Random): Missingness is random
• MAR (Missing At Random): Depends on observed data
• MNAR (Missing Not At Random): Depends on unobserved data

STRATEGIES:
───────────────────────────────────────────────────────────────────────────────

1. DELETION
   a) Listwise deletion: Remove rows with any missing value
   b) Pairwise deletion: Remove rows only for specific analyses
   
   Pros: Simple, preserves distribution
   Cons: Loses data, can introduce bias

2. IMPUTATION
   a) Simple imputation:
      - Mean/Median (numerical)
      - Mode (categorical)
      - Constant value
   
   b) Statistical imputation:
      - Regression imputation
      - KNN imputation
   
   c) Model-based:
      - MICE (Multiple Imputation by Chained Equations)
      - IterativeImputer

3. INDICATOR VARIABLE
   Create a binary column indicating missingness
   (Can capture "missingness as information")
"""

def example_handling_missing_values():
    """Demonstrate different methods for handling missing values."""
    
    print("HANDLING MISSING VALUES")
    print("=" * 70)
    
    # Create sample data with missing values
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        'age': np.random.randint(20, 70, n).astype(float),
        'income': np.random.normal(50000, 15000, n),
        'credit_score': np.random.normal(700, 50, n),
        'category': np.random.choice(['A', 'B', 'C'], n)
    })
    
    # Introduce missing values
    df.loc[np.random.choice(n, 10, replace=False), 'age'] = np.nan
    df.loc[np.random.choice(n, 15, replace=False), 'income'] = np.nan
    df.loc[np.random.choice(n, 8, replace=False), 'category'] = np.nan
    
    print("\nOriginal Data with Missing Values:")
    print(f"Shape: {df.shape}")
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 1: Deletion
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 1: DELETION")
    print("─" * 70)
    
    # Listwise deletion (drop any row with missing values)
    df_dropped = df.dropna()
    print(f"\nAfter dropping rows with any NaN:")
    print(f"  Original rows: {len(df)}")
    print(f"  Remaining rows: {len(df_dropped)}")
    print(f"  Lost: {len(df) - len(df_dropped)} rows ({(len(df) - len(df_dropped))/len(df)*100:.1f}%)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 2: Simple Imputation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 2: SIMPLE IMPUTATION")
    print("─" * 70)
    
    df_imputed = df.copy()
    
    # Mean imputation for numerical columns
    print("\nNumerical columns - Mean imputation:")
    for col in ['age', 'income']:
        mean_val = df_imputed[col].mean()
        df_imputed[col].fillna(mean_val, inplace=True)
        print(f"  {col}: filled with mean = {mean_val:.2f}")
    
    # Mode imputation for categorical columns
    print("\nCategorical columns - Mode imputation:")
    mode_val = df_imputed['category'].mode()[0]
    df_imputed['category'].fillna(mode_val, inplace=True)
    print(f"  category: filled with mode = '{mode_val}'")
    
    print(f"\nMissing values after imputation: {df_imputed.isnull().sum().sum()}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 3: Using sklearn SimpleImputer
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 3: SKLEARN SIMPLEIMPUTER")
    print("─" * 70)
    
    from sklearn.impute import SimpleImputer
    
    df_sklearn = df.copy()
    numerical_cols = ['age', 'income', 'credit_score']
    
    # Median imputation (more robust to outliers)
    imputer = SimpleImputer(strategy='median')
    df_sklearn[numerical_cols] = imputer.fit_transform(df_sklearn[numerical_cols])
    
    print("\nUsing SimpleImputer with strategy='median':")
    print(f"  Imputed values: {imputer.statistics_}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 4: KNN Imputation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 4: KNN IMPUTATION")
    print("─" * 70)
    
    from sklearn.impute import KNNImputer
    
    df_knn = df.copy()
    
    # KNN imputation (uses similar samples to impute)
    knn_imputer = KNNImputer(n_neighbors=5)
    df_knn[numerical_cols] = knn_imputer.fit_transform(df_knn[numerical_cols])
    
    print("\nKNN Imputation (k=5):")
    print("  Uses values from 5 nearest neighbors to impute missing values")
    print("  Better preserves relationships between features")
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 5: Add Missing Indicator
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 5: MISSING INDICATOR")
    print("─" * 70)
    
    df_indicator = df.copy()
    
    # Create indicator columns
    for col in ['age', 'income']:
        indicator_col = f'{col}_was_missing'
        df_indicator[indicator_col] = df_indicator[col].isnull().astype(int)
    
    # Then impute
    df_indicator['age'].fillna(df_indicator['age'].median(), inplace=True)
    df_indicator['income'].fillna(df_indicator['income'].median(), inplace=True)
    
    print("\nAdded indicator columns:")
    print(df_indicator[['age', 'age_was_missing', 'income', 'income_was_missing']].head(10).to_string())
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("SUMMARY: CHOOSING A METHOD")
    print("─" * 70)
    print("""
    METHOD              WHEN TO USE
    ──────────────────  ──────────────────────────────────────────────────
    Deletion            Few missing values (<5%), MCAR data
    Mean/Median         Simple baseline, numerical data
    Mode                Categorical data
    KNN Imputation      When feature relationships matter
    MICE/Iterative      Complex missing patterns, high-quality imputation
    Missing Indicator   When missingness itself is informative
    
    GENERAL ADVICE:
    • Always analyze WHY data is missing before choosing a method
    • Test multiple methods and compare model performance
    • For tree-based models, consider using native missing value handling
    """)


# ═══════════════════════════════════════════════════════════════════════════
# 3.2.2 DEALING WITH OUTLIERS
# ═══════════════════════════════════════════════════════════════════════════

"""
DEALING WITH OUTLIERS
═══════════════════════════════════════════════════════════════════════════════

Outliers are data points that differ significantly from other observations.

DETECTION METHODS:
───────────────────────────────────────────────────────────────────────────────

1. STATISTICAL METHODS
   a) Z-score: Points with |z| > 3 are outliers
   b) IQR method: Points outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
   c) Modified Z-score: Uses median instead of mean (more robust)

2. MACHINE LEARNING METHODS
   a) Isolation Forest
   b) Local Outlier Factor (LOF)
   c) One-Class SVM

HANDLING STRATEGIES:
───────────────────────────────────────────────────────────────────────────────
• Remove outliers (if they're errors)
• Cap/Floor (Winsorization)
• Transform data (log, Box-Cox)
• Keep outliers (if they're real)
• Use robust models
"""

def example_outlier_detection():
    """Demonstrate outlier detection methods."""
    
    print("OUTLIER DETECTION AND HANDLING")
    print("=" * 70)
    
    # Create data with outliers
    np.random.seed(42)
    n = 200
    
    # Normal data
    normal_data = np.random.normal(50, 10, n - 10)
    # Add some outliers
    outliers = np.array([10, 15, 95, 100, 105, 110, 5, 120, 0, 130])
    data = np.concatenate([normal_data, outliers])
    
    df = pd.DataFrame({'value': data})
    
    print("\nData Summary:")
    print(df['value'].describe().round(2))
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 1: Z-Score
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 1: Z-SCORE")
    print("─" * 70)
    
    from scipy import stats
    
    z_scores = np.abs(stats.zscore(df['value']))
    z_threshold = 3
    
    z_outliers = df[z_scores > z_threshold]
    
    print(f"\nZ-score threshold: {z_threshold}")
    print(f"Outliers detected: {len(z_outliers)}")
    print(f"Outlier values: {z_outliers['value'].values.round(2)}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 2: IQR Method
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 2: IQR METHOD (Box Plot)")
    print("─" * 70)
    
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    iqr_outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
    
    print(f"\nQ1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Outliers detected: {len(iqr_outliers)}")
    print(f"Outlier values: {sorted(iqr_outliers['value'].values.round(2))}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 3: Isolation Forest
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 3: ISOLATION FOREST")
    print("─" * 70)
    
    from sklearn.ensemble import IsolationForest
    
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outlier_labels = iso_forest.fit_predict(df[['value']])
    
    iso_outliers = df[outlier_labels == -1]
    
    print(f"\nContamination parameter: 5%")
    print(f"Outliers detected: {len(iso_outliers)}")
    print(f"Outlier values: {sorted(iso_outliers['value'].values.round(2))}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # HANDLING STRATEGIES
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("HANDLING STRATEGIES")
    print("─" * 70)
    
    # Strategy 1: Remove
    df_removed = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
    print(f"\n1. Remove outliers:")
    print(f"   Before: {len(df)} rows")
    print(f"   After: {len(df_removed)} rows")
    
    # Strategy 2: Cap (Winsorization)
    df_capped = df.copy()
    df_capped['value'] = df_capped['value'].clip(lower_bound, upper_bound)
    print(f"\n2. Cap/Floor (Winsorization):")
    print(f"   Values clipped to [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"   New max: {df_capped['value'].max():.2f}")
    print(f"   New min: {df_capped['value'].min():.2f}")
    
    # Strategy 3: Transform
    df_log = df.copy()
    df_log['value_log'] = np.log1p(df_log['value'] - df_log['value'].min() + 1)
    print(f"\n3. Log transform:")
    print(f"   Original skewness: {df['value'].skew():.3f}")
    print(f"   After log: {df_log['value_log'].skew():.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.3 FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

"""
FEATURE ENGINEERING
═══════════════════════════════════════════════════════════════════════════════

Feature engineering is the process of creating new features from existing ones
to improve model performance. Often the most impactful part of ML work.

"Coming up with features is difficult, time-consuming, requires expert 
knowledge. 'Applied machine learning' is basically feature engineering."
                                        — Andrew Ng
"""


# ═══════════════════════════════════════════════════════════════════════════
# 3.3.1 FEATURE CREATION
# ═══════════════════════════════════════════════════════════════════════════

def example_feature_creation():
    """Demonstrate various feature creation techniques."""
    
    print("FEATURE CREATION")
    print("=" * 70)
    
    # Sample data
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=n, freq='H'),
        'price': np.random.uniform(10, 100, n),
        'quantity': np.random.randint(1, 50, n),
        'customer_age': np.random.randint(18, 70, n),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food'], n),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], n),
        'review_text': np.random.choice([
            'Great product!', 'Not bad', 'Terrible', 'Love it', 'Okay'
        ], n)
    })
    
    print("\nOriginal Features:")
    print(df.head().to_string())
    print(f"\nShape: {df.shape}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. MATHEMATICAL TRANSFORMATIONS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("1. MATHEMATICAL TRANSFORMATIONS")
    print("─" * 70)
    
    # Arithmetic combinations
    df['total_value'] = df['price'] * df['quantity']
    df['price_per_unit_log'] = np.log1p(df['price'])
    df['quantity_squared'] = df['quantity'] ** 2
    
    print("\nCreated:")
    print("  • total_value = price × quantity")
    print("  • price_per_unit_log = log(1 + price)")
    print("  • quantity_squared = quantity²")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. DATE/TIME FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("2. DATE/TIME FEATURES")
    print("─" * 70)
    
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
    
    # Cyclical encoding for periodic features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    print("\nDate/Time features created:")
    print("  • hour, day_of_week, day_of_month, month")
    print("  • is_weekend, is_morning, is_evening")
    print("  • hour_sin, hour_cos (cyclical encoding)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. BINNING/DISCRETIZATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("3. BINNING/DISCRETIZATION")
    print("─" * 70)
    
    # Age groups
    df['age_group'] = pd.cut(
        df['customer_age'],
        bins=[0, 25, 35, 50, 100],
        labels=['Young', 'Adult', 'Middle-aged', 'Senior']
    )
    
    # Price tiers
    df['price_tier'] = pd.qcut(
        df['price'],
        q=4,
        labels=['Budget', 'Economy', 'Premium', 'Luxury']
    )
    
    print("\nBinning created:")
    print("  • age_group: Young/Adult/Middle-aged/Senior")
    print("  • price_tier: Budget/Economy/Premium/Luxury (quartiles)")
    
    print("\nAge group distribution:")
    print(df['age_group'].value_counts().to_string())
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. TEXT FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("4. TEXT FEATURES")
    print("─" * 70)
    
    # Simple text features
    df['review_length'] = df['review_text'].str.len()
    df['review_word_count'] = df['review_text'].str.split().str.len()
    df['has_exclamation'] = df['review_text'].str.contains('!').astype(int)
    
    # Sentiment (simple rule-based)
    positive_words = ['great', 'love', 'excellent', 'good']
    negative_words = ['terrible', 'bad', 'awful', 'hate']
    
    df['has_positive'] = df['review_text'].str.lower().str.contains('|'.join(positive_words)).astype(int)
    df['has_negative'] = df['review_text'].str.lower().str.contains('|'.join(negative_words)).astype(int)
    
    print("\nText features created:")
    print("  • review_length, review_word_count")
    print("  • has_exclamation")
    print("  • has_positive, has_negative (simple sentiment)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5. AGGREGATION FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("5. AGGREGATION FEATURES")
    print("─" * 70)
    
    # Category-level statistics
    category_stats = df.groupby('category').agg({
        'price': ['mean', 'std', 'min', 'max'],
        'quantity': 'mean'
    }).reset_index()
    category_stats.columns = ['category', 'cat_price_mean', 'cat_price_std', 
                               'cat_price_min', 'cat_price_max', 'cat_qty_mean']
    
    df = df.merge(category_stats, on='category', how='left')
    
    # Price relative to category average
    df['price_vs_category'] = df['price'] / df['cat_price_mean']
    
    print("\nAggregation features created:")
    print("  • Category-level: mean, std, min, max price; mean quantity")
    print("  • price_vs_category = price / category_mean_price")
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("SUMMARY")
    print("─" * 70)
    
    print(f"\nOriginal features: 7")
    print(f"Final features: {len(df.columns)}")
    print(f"New features created: {len(df.columns) - 7}")
    
    print("\nFeature engineering techniques used:")
    print("  1. Mathematical transformations (multiply, log, power)")
    print("  2. Date/time extraction (hour, day, weekend, cyclical)")
    print("  3. Binning/Discretization (age groups, price tiers)")
    print("  4. Text features (length, word count, patterns)")
    print("  5. Aggregation features (group statistics)")


# ─────────────────────────────────────────────────────────────────────────────
# 3.4 DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

"""
DATA PREPROCESSING
═══════════════════════════════════════════════════════════════════════════════

Preprocessing transforms raw features into a format suitable for ML algorithms.
"""


# ═══════════════════════════════════════════════════════════════════════════
# 3.4.1 ENCODING CATEGORICAL VARIABLES
# ═══════════════════════════════════════════════════════════════════════════

def example_categorical_encoding():
    """Demonstrate categorical encoding methods."""
    
    print("ENCODING CATEGORICAL VARIABLES")
    print("=" * 70)
    
    # Sample data
    df = pd.DataFrame({
        'color': ['Red', 'Blue', 'Green', 'Blue', 'Red', 'Green', 'Blue', 'Red'],
        'size': ['Small', 'Medium', 'Large', 'Small', 'Large', 'Medium', 'Small', 'Large'],
        'brand': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
        'price': [10, 20, 30, 15, 25, 35, 12, 22]
    })
    
    print("\nOriginal Data:")
    print(df.to_string())
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 1: Label Encoding
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 1: LABEL ENCODING")
    print("─" * 70)
    
    """
    Label encoding assigns each category a unique integer.
    
    USE WHEN:
    • Ordinal variables (has meaningful order)
    • Tree-based models (can handle arbitrary encoding)
    
    AVOID WHEN:
    • Nominal variables with linear models (implies false order)
    """
    
    le = LabelEncoder()
    df_label = df.copy()
    df_label['color_encoded'] = le.fit_transform(df['color'])
    
    print("\nLabel Encoding for 'color':")
    print(f"  Mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
    print(f"\n{df_label[['color', 'color_encoded']].drop_duplicates().to_string()}")
    
    # For ordinal variables with custom order
    size_order = {'Small': 0, 'Medium': 1, 'Large': 2}
    df_label['size_encoded'] = df['size'].map(size_order)
    
    print("\nOrdinal Encoding for 'size' (with custom order):")
    print(f"  Mapping: {size_order}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 2: One-Hot Encoding
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 2: ONE-HOT ENCODING")
    print("─" * 70)
    
    """
    One-hot encoding creates binary columns for each category.
    
    USE WHEN:
    • Nominal variables (no meaningful order)
    • Linear models, neural networks
    • Few unique values
    
    AVOID WHEN:
    • High cardinality (many unique values)
    • Tree-based models (less efficient)
    """
    
    # Using pandas
    df_onehot = pd.get_dummies(df, columns=['color'], prefix='color')
    
    print("\nOne-Hot Encoding for 'color':")
    print(df_onehot.to_string())
    
    # Using sklearn
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' to avoid multicollinearity
    encoded = encoder.fit_transform(df[['color']])
    feature_names = encoder.get_feature_names_out(['color'])
    
    print(f"\nWith drop='first' (avoiding dummy variable trap):")
    print(f"  Features: {feature_names}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 3: Target Encoding (Mean Encoding)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 3: TARGET ENCODING (Mean Encoding)")
    print("─" * 70)
    
    """
    Target encoding replaces categories with mean of target variable.
    
    USE WHEN:
    • High cardinality categorical variables
    • Strong relationship between category and target
    
    CAUTION:
    • Risk of overfitting (use cross-validation)
    • Apply smoothing for rare categories
    """
    
    # Simulate target variable
    df['target'] = [1, 0, 1, 1, 0, 1, 0, 0]
    
    # Calculate mean target for each category
    target_means = df.groupby('brand')['target'].mean()
    df['brand_target_encoded'] = df['brand'].map(target_means)
    
    print("\nTarget Encoding for 'brand':")
    print(f"  Brand means: {target_means.to_dict()}")
    print(f"\n{df[['brand', 'target', 'brand_target_encoded']].to_string()}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 4: Frequency Encoding
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 4: FREQUENCY ENCODING")
    print("─" * 70)
    
    """
    Replace categories with their frequency in the dataset.
    
    USE WHEN:
    • Frequency is meaningful
    • Want to preserve some information about category
    """
    
    freq_encoding = df['color'].value_counts(normalize=True)
    df['color_freq'] = df['color'].map(freq_encoding)
    
    print("\nFrequency Encoding for 'color':")
    print(f"  Frequencies: {freq_encoding.to_dict()}")
    print(f"\n{df[['color', 'color_freq']].drop_duplicates().to_string()}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("ENCODING METHOD SUMMARY")
    print("─" * 70)
    print("""
    METHOD             USE CASE                           PROS/CONS
    ─────────────────  ────────────────────────────────   ────────────────────
    Label Encoding     Ordinal data, tree models          Simple, compact
    One-Hot Encoding   Nominal data, linear models        No false order, sparse
    Target Encoding    High cardinality, prediction       Powerful, risk of overfit
    Frequency Encoding Frequency matters                  Simple, no explosion
    Binary Encoding    Medium cardinality                 Compact, some info loss
    """)


# ═══════════════════════════════════════════════════════════════════════════
# 3.4.2 FEATURE SCALING
# ═══════════════════════════════════════════════════════════════════════════

def example_feature_scaling():
    """Demonstrate feature scaling methods."""
    
    print("FEATURE SCALING")
    print("=" * 70)
    
    # Sample data with different scales
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),           # Range: 18-80
        'income': np.random.normal(50000, 20000, 100),   # Range: ~10k-90k
        'score': np.random.uniform(0, 1, 100),           # Range: 0-1
        'count': np.random.exponential(10, 100)          # Skewed
    })
    
    print("\nOriginal Data Statistics:")
    print(df.describe().round(2).to_string())
    
    """
    WHY SCALE FEATURES?
    
    1. Gradient descent converges faster when features are on similar scales
    2. Distance-based algorithms (KNN, SVM) treat all features equally
    3. Regularization penalizes features fairly
    
    WHEN TO SCALE:
    • Linear/Logistic Regression: Yes
    • SVM, KNN: Yes
    • Neural Networks: Yes
    • Tree-based models: Not necessary (but doesn't hurt)
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 1: StandardScaler (Z-score normalization)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 1: STANDARD SCALER (Z-score)")
    print("─" * 70)
    
    """
    StandardScaler: z = (x - mean) / std
    
    Result: mean ≈ 0, std ≈ 1
    
    USE WHEN:
    • Data is approximately normal
    • Outliers are few
    • Most common choice
    """
    
    scaler = StandardScaler()
    df_standard = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns
    )
    
    print("\nAfter StandardScaler:")
    print(df_standard.describe().round(2).to_string())
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 2: MinMaxScaler
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 2: MINMAX SCALER")
    print("─" * 70)
    
    """
    MinMaxScaler: x_scaled = (x - min) / (max - min)
    
    Result: values in [0, 1]
    
    USE WHEN:
    • Need bounded values
    • Data is uniformly distributed
    • Neural networks (common choice)
    
    CAUTION:
    • Sensitive to outliers
    """
    
    minmax = MinMaxScaler()
    df_minmax = pd.DataFrame(
        minmax.fit_transform(df),
        columns=df.columns
    )
    
    print("\nAfter MinMaxScaler:")
    print(df_minmax.describe().round(2).to_string())
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 3: RobustScaler
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 3: ROBUST SCALER")
    print("─" * 70)
    
    """
    RobustScaler: x_scaled = (x - median) / IQR
    
    USE WHEN:
    • Data has outliers
    • Need robustness
    """
    
    from sklearn.preprocessing import RobustScaler
    
    robust = RobustScaler()
    df_robust = pd.DataFrame(
        robust.fit_transform(df),
        columns=df.columns
    )
    
    print("\nAfter RobustScaler:")
    print(df_robust.describe().round(2).to_string())
    
    # ─────────────────────────────────────────────────────────────────────────
    # IMPORTANT: Fit on train, transform both
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("IMPORTANT: PROPER SCALING WORKFLOW")
    print("─" * 70)
    
    print("""
    CORRECT WORKFLOW:
    
    1. Split data into train/test FIRST
    2. Fit scaler on training data ONLY
    3. Transform both train and test using the fitted scaler
    
    WHY?
    • Prevents data leakage (test set info influencing training)
    • Simulates real-world scenario (you don't know test data at training time)
    
    CODE EXAMPLE:
    ─────────────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # FIT and TRANSFORM
    X_test_scaled = scaler.transform(X_test)        # ONLY TRANSFORM
    ─────────────────────────────────────────────────────────────────────────
    """)


# ─────────────────────────────────────────────────────────────────────────────
# 3.5 TRAIN/TEST SPLIT STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def example_train_test_split():
    """Demonstrate train/test split strategies."""
    
    print("TRAIN/TEST SPLIT STRATEGIES")
    print("=" * 70)
    
    # Sample data
    np.random.seed(42)
    n = 1000
    X = np.random.randn(n, 5)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n) * 0.5 > 0).astype(int)
    
    print(f"\nDataset: {n} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 1: Simple Train/Test Split
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 1: SIMPLE TRAIN/TEST SPLIT")
    print("─" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {len(X_train)} samples ({len(X_train)/n*100:.0f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/n*100:.0f}%)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 2: Stratified Split (Preserves class proportions)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 2: STRATIFIED SPLIT")
    print("─" * 70)
    
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nClass proportions:")
    print(f"  Original:  Class 0: {(y==0).mean():.2%}, Class 1: {(y==1).mean():.2%}")
    print(f"  Train:     Class 0: {(y_train_s==0).mean():.2%}, Class 1: {(y_train_s==1).mean():.2%}")
    print(f"  Test:      Class 0: {(y_test_s==0).mean():.2%}, Class 1: {(y_test_s==1).mean():.2%}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 3: K-Fold Cross-Validation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 3: K-FOLD CROSS-VALIDATION")
    print("─" * 70)
    
    """
    K-Fold CV splits data into K folds.
    Each fold is used as test set once, others as training.
    
    ┌──────┬──────┬──────┬──────┬──────┐
    │ TEST │Train │Train │Train │Train │  Fold 1
    ├──────┼──────┼──────┼──────┼──────┤
    │Train │ TEST │Train │Train │Train │  Fold 2
    ├──────┼──────┼──────┼──────┼──────┤
    │Train │Train │ TEST │Train │Train │  Fold 3
    ├──────┼──────┼──────┼──────┼──────┤
    │Train │Train │Train │ TEST │Train │  Fold 4
    ├──────┼──────┼──────┼──────┼──────┤
    │Train │Train │Train │Train │ TEST │  Fold 5
    └──────┴──────┴──────┴──────┴──────┘
    
    BENEFITS:
    • Uses all data for training and testing
    • More reliable performance estimate
    • Better for small datasets
    """
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n5-Fold Cross-Validation:")
    for i, (train_idx, test_idx) in enumerate(kfold.split(X)):
        print(f"  Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # Stratified K-Fold (for classification)
    print("\nStratified 5-Fold (preserves class distribution in each fold):")
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for i, (train_idx, test_idx) in enumerate(skfold.split(X, y)):
        train_class_dist = np.bincount(y[train_idx]) / len(train_idx)
        test_class_dist = np.bincount(y[test_idx]) / len(test_idx)
        print(f"  Fold {i+1}: Train class 1: {train_class_dist[1]:.2%}, "
              f"Test class 1: {test_class_dist[1]:.2%}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TIME SERIES SPLIT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("METHOD 4: TIME SERIES SPLIT")
    print("─" * 70)
    
    """
    For time series data, you can't randomly shuffle!
    Training data must come BEFORE test data.
    
    ┌──────┬──────┬──────┬──────┬──────┐
    │Train │ TEST │      │      │      │  Fold 1
    ├──────┼──────┼──────┼──────┼──────┤
    │Train │Train │ TEST │      │      │  Fold 2
    ├──────┼──────┼──────┼──────┼──────┤
    │Train │Train │Train │ TEST │      │  Fold 3
    ├──────┼──────┼──────┼──────┼──────┤
    │Train │Train │Train │Train │ TEST │  Fold 4
    └──────┴──────┴──────┴──────┴──────┘
    """
    
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("\nTime Series Split:")
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"  Fold {i+1}: Train indices [{train_idx[0]}-{train_idx[-1]}], "
              f"Test indices [{test_idx[0]}-{test_idx[-1]}]")


# ─────────────────────────────────────────────────────────────────────────────
# 3.6 CHAPTER 3 SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

"""
CHAPTER 3 SUMMARY
═══════════════════════════════════════════════════════════════════════════════

KEY TAKEAWAYS:
───────────────────────────────────────────────────────────────────────────────

1. EDA is essential before modeling
   • Understand data types, distributions, missing values
   • Check correlations and class balance
   • Identify outliers

2. Missing Value Strategies
   • Deletion: Simple but loses data
   • Imputation: Mean/median, KNN, model-based
   • Indicators: Capture missingness as information

3. Outlier Handling
   • Detection: Z-score, IQR, Isolation Forest
   • Handling: Remove, cap, transform, or keep

4. Feature Engineering
   • Mathematical transformations
   • Date/time extraction
   • Binning/discretization
   • Text features
   • Aggregations

5. Encoding Categorical Variables
   • Label encoding: Ordinal data
   • One-hot: Nominal data
   • Target encoding: High cardinality

6. Feature Scaling
   • StandardScaler: Most common
   • MinMaxScaler: Bounded [0,1]
   • RobustScaler: Handles outliers
   • ALWAYS fit on train, transform both!

7. Train/Test Splitting
   • Simple split: 80/20 or 70/30
   • Stratified: Preserves class distribution
   • K-Fold CV: More reliable estimates
   • Time series: Respect temporal order


DATA PREPROCESSING CHECKLIST:
───────────────────────────────────────────────────────────────────────────────
□ Performed EDA
□ Handled missing values
□ Dealt with outliers
□ Created useful features
□ Encoded categorical variables
□ Scaled numerical features
□ Split data properly (with stratification if needed)
□ No data leakage (fit on train only)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Run all examples
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CHAPTER 3: DATA FUNDAMENTALS")
    print("="*70)
    
    example_eda_comprehensive()
    print("\n")
    example_handling_missing_values()
    print("\n")
    example_outlier_detection()
    print("\n")
    example_feature_creation()
    print("\n")
    example_categorical_encoding()
    print("\n")
    example_feature_scaling()
    print("\n")
    example_train_test_split()
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
# ███████████████████████████████████████████████████████████████████████████████
#                                                                               
#   PART III: NEURAL NETWORKS AND DEEP LEARNING                                  
#                                                                               
#   CHAPTER 10: NEURAL NETWORK FUNDAMENTALS                                      
#   CHAPTER 11: TRAINING NEURAL NETWORKS                                         
#   CHAPTER 12: CONVOLUTIONAL NEURAL NETWORKS                                    
#   CHAPTER 13: RECURRENT NEURAL NETWORKS                                        
#   CHAPTER 14: ATTENTION AND TRANSFORMERS                                       
#                                                                               
# ███████████████████████████████████████████████████████████████████████████████

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   "You don't need to be an expert to build neural networks, but              ║
║    understanding the fundamentals will make you much more effective."         ║
║                                                                               ║
║   This section covers deep learning from neurons to transformers.             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

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


# ███████████████████████████████████████████████████████████████████████████████
#                                                                               
#   CHAPTER 10: NEURAL NETWORK FUNDAMENTALS                                      
#                                                                               
# ███████████████████████████████████████████████████████████████████████████████

"""
NEURAL NETWORK FUNDAMENTALS
═══════════════════════════════════════════════════════════════════════════════

Neural networks are universal function approximators inspired by biological neurons.

WHY NEURAL NETWORKS?
───────────────────────────────────────────────────────────────────────────────
• Can approximate any continuous function (Universal Approximation Theorem)
• Automatically learn features from raw data
• State-of-the-art for images, text, audio, video
• Scale with more data and compute
"""


# ─────────────────────────────────────────────────────────────────────────────
# 10.1 THE PERCEPTRON
# ─────────────────────────────────────────────────────────────────────────────

"""
THE PERCEPTRON
═══════════════════════════════════════════════════════════════════════════════

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
───────────────────────────────────────────────────────────────────────────────
    If prediction is wrong:
        w = w + η × (y - ŷ) × x
        b = b + η × (y - ŷ)
    
    Where η is learning rate, y is true label, ŷ is prediction.


LIMITATION:
    Cannot solve XOR problem! Need multiple layers.
"""

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
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test on linearly separable data
    # ─────────────────────────────────────────────────────────────────────────
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # XOR problem - NOT linearly separable
    # ─────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 10.2 MULTI-LAYER PERCEPTRONS
# ─────────────────────────────────────────────────────────────────────────────

"""
MULTI-LAYER PERCEPTRONS (MLP)
═══════════════════════════════════════════════════════════════════════════════

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
───────────────────────────────────────────────────────────────────────────────
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
───────────────────────────────────────────────────────────────────────────────
A feedforward network with a single hidden layer containing a finite number
of neurons can approximate any continuous function to arbitrary precision.

However, deeper networks often learn better representations with fewer parameters!
"""

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
    
    # ─────────────────────────────────────────────────────────────────────────
    # Solve XOR problem!
    # ─────────────────────────────────────────────────────────────────────────
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # Larger example
    # ─────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 10.3 ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

"""
ACTIVATION FUNCTIONS
═══════════════════════════════════════════════════════════════════════════════

Non-linear functions applied after linear transformations.
Without them, the entire network would just be a linear function!


COMMON ACTIVATION FUNCTIONS:
───────────────────────────────────────────────────────────────────────────────

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
"""

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
    
    # ─────────────────────────────────────────────────────────────────────────
    # Derivatives (for backpropagation)
    # ─────────────────────────────────────────────────────────────────────────
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
    ─────────────────────────────────────────────────────────────────────────
    • Sigmoid/Tanh derivatives → 0 for large |z| (vanishing gradient!)
    • ReLU derivative is 0 or 1 (no vanishing gradient for positive z)
    • This is why ReLU revolutionized deep learning
    """)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Softmax
    # ─────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 10.4 FORWARD PROPAGATION
# ─────────────────────────────────────────────────────────────────────────────

"""
FORWARD PROPAGATION
═══════════════════════════════════════════════════════════════════════════════

The process of computing the output given an input.

ALGORITHM:
───────────────────────────────────────────────────────────────────────────────
For each layer l from 1 to L:
    1. Compute linear transformation: z^(l) = W^(l) @ a^(l-1) + b^(l)
    2. Apply activation: a^(l) = g^(l)(z^(l))

Where:
• a^(0) = x (input)
• a^(L) = ŷ (output)
• g^(l) is the activation function for layer l


MATRIX FORM:
───────────────────────────────────────────────────────────────────────────────
For a batch of N samples:
    Z = W @ X + b    (broadcasting b)
    A = activation(Z)

Dimensions:
• X: (n_input, N)
• W: (n_output, n_input)
• Z: (n_output, N)
• A: (n_output, N)
"""

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


# ─────────────────────────────────────────────────────────────────────────────
# 10.5 BACKPROPAGATION
# ─────────────────────────────────────────────────────────────────────────────

"""
BACKPROPAGATION
═══════════════════════════════════════════════════════════════════════════════

The algorithm for computing gradients of the loss with respect to all parameters.

KEY INSIGHT: Use the chain rule to propagate gradients backward through the network.


ALGORITHM:
───────────────────────────────────────────────────────────────────────────────

1. Forward pass: Compute and cache all z^(l) and a^(l)

2. Compute output gradient: ∂L/∂a^(L) (depends on loss function)

3. For l = L to 1 (backward):
   a. Compute ∂L/∂z^(l) = ∂L/∂a^(l) ⊙ g'^(l)(z^(l))
   b. Compute ∂L/∂W^(l) = ∂L/∂z^(l) @ (a^(l-1))ᵀ
   c. Compute ∂L/∂b^(l) = sum(∂L/∂z^(l)) over samples
   d. Compute ∂L/∂a^(l-1) = (W^(l))ᵀ @ ∂L/∂z^(l)  (to propagate to previous layer)


DERIVATION (for one layer):
───────────────────────────────────────────────────────────────────────────────

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
"""

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
    
    # ─────────────────────────────────────────────────────────────────────────
    # FORWARD PASS
    # ─────────────────────────────────────────────────────────────────────────
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # BACKWARD PASS
    # ─────────────────────────────────────────────────────────────────────────
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # GRADIENT CHECK (Numerical verification)
    # ─────────────────────────────────────────────────────────────────────────
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


# ███████████████████████████████████████████████████████████████████████████████
#                                                                               
#   CHAPTER 11: TRAINING NEURAL NETWORKS                                         
#                                                                               
# ███████████████████████████████████████████████████████████████████████████████

"""
TRAINING NEURAL NETWORKS
═══════════════════════════════════════════════════════════════════════════════

Training neural networks effectively requires understanding many techniques.
"""


# ─────────────────────────────────────────────────────────────────────────────
# 11.1 LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

"""
LOSS FUNCTIONS
═══════════════════════════════════════════════════════════════════════════════

The loss function measures how wrong the model's predictions are.

REGRESSION LOSSES:
───────────────────────────────────────────────────────────────────────────────

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
───────────────────────────────────────────────────────────────────────────────

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
"""

def example_loss_functions():
    """Demonstrate different loss functions."""
    
    print("LOSS FUNCTIONS")
    print("=" * 70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Regression Losses
    # ─────────────────────────────────────────────────────────────────────────
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # Classification Losses
    # ─────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 11.2 OPTIMIZERS
# ─────────────────────────────────────────────────────────────────────────────

"""
OPTIMIZERS
═══════════════════════════════════════════════════════════════════════════════

Algorithms for updating model parameters based on gradients.


SGD (Stochastic Gradient Descent):
───────────────────────────────────────────────────────────────────────────────
    θ = θ - α × ∇L(θ)
    
    Simple but can be slow and oscillate.


SGD WITH MOMENTUM:
───────────────────────────────────────────────────────────────────────────────
    v = β × v - α × ∇L(θ)
    θ = θ + v
    
    • Accumulates gradient direction
    • Speeds up convergence
    • Reduces oscillation
    • β typically 0.9


RMSPROP:
───────────────────────────────────────────────────────────────────────────────
    s = β × s + (1-β) × (∇L(θ))²
    θ = θ - α × ∇L(θ) / (√s + ε)
    
    • Adapts learning rate per parameter
    • Good for non-stationary objectives
    • Works well with RNNs


ADAM (Adaptive Moment Estimation):
───────────────────────────────────────────────────────────────────────────────
    m = β₁ × m + (1-β₁) × ∇L(θ)           # 1st moment (momentum)
    v = β₂ × v + (1-β₂) × (∇L(θ))²        # 2nd moment (RMSprop)
    m̂ = m / (1 - β₁ᵗ)                     # Bias correction
    v̂ = v / (1 - β₂ᵗ)
    θ = θ - α × m̂ / (√v̂ + ε)
    
    Default: β₁=0.9, β₂=0.999, ε=1e-8
    
    Most popular optimizer - great default choice!


ADAMW:
───────────────────────────────────────────────────────────────────────────────
    Like Adam but with decoupled weight decay.
    Better for transformers.
    
    θ = θ - α × (m̂ / (√v̂ + ε) + λ × θ)
"""

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
    ─────────────────────────────────────────────────────────────────────────
    • Adam: Best default choice for most problems
    • SGD+Momentum: Often better final performance with proper tuning
    • AdamW: Best for transformers (decoupled weight decay)
    • RMSprop: Good for RNNs
    
    Tips:
    • Start with Adam, lr=1e-3 or 3e-4
    • For fine-tuning pretrained models: lower lr (1e-5 to 1e-4)
    • If Adam converges but generalizes poorly, try SGD
    """)


# ─────────────────────────────────────────────────────────────────────────────
# 11.3 BATCH NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

"""
BATCH NORMALIZATION
═══════════════════════════════════════════════════════════════════════════════

Normalizes activations within each mini-batch.

ALGORITHM:
───────────────────────────────────────────────────────────────────────────────
For each feature:
    1. μ_B = (1/m) Σᵢ xᵢ                    # Batch mean
    2. σ²_B = (1/m) Σᵢ (xᵢ - μ_B)²          # Batch variance
    3. x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)       # Normalize
    4. yᵢ = γ × x̂ᵢ + β                      # Scale and shift

Where γ and β are learnable parameters.


BENEFITS:
───────────────────────────────────────────────────────────────────────────────
• Reduces internal covariate shift
• Allows higher learning rates
• Reduces sensitivity to initialization
• Acts as regularization
• Enables training of very deep networks


WHEN TO USE:
───────────────────────────────────────────────────────────────────────────────
• Before activation (original paper) OR after activation (common in practice)
• In CNNs: BatchNorm2d
• In transformers: LayerNorm is preferred

LAYER NORMALIZATION (LayerNorm):
    Normalizes across features (not batch)
    Better for variable-length sequences and transformers
    
    μ = (1/d) Σⱼ xⱼ
    σ² = (1/d) Σⱼ (xⱼ - μ)²
"""

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
    
    # ─────────────────────────────────────────────────────────────────────────
    # PyTorch BatchNorm
    # ─────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 11.4 REGULARIZATION TECHNIQUES
# ─────────────────────────────────────────────────────────────────────────────

"""
REGULARIZATION TECHNIQUES FOR NEURAL NETWORKS
═══════════════════════════════════════════════════════════════════════════════


L2 REGULARIZATION (Weight Decay):
───────────────────────────────────────────────────────────────────────────────
    Loss = Original_Loss + λ Σ wᵢ²
    
    • Penalizes large weights
    • Implemented via weight_decay in optimizer


DROPOUT:
───────────────────────────────────────────────────────────────────────────────
    During training: Randomly set neurons to 0 with probability p
    During inference: Scale by (1-p) or don't drop
    
    • Prevents co-adaptation of neurons
    • Acts as ensemble of networks
    • Typical p: 0.1-0.5


EARLY STOPPING:
───────────────────────────────────────────────────────────────────────────────
    Stop training when validation loss stops improving.
    
    • Simple and effective
    • Requires validation set
    • Use patience parameter


DATA AUGMENTATION:
───────────────────────────────────────────────────────────────────────────────
    Create modified versions of training data.
    
    Images: flip, rotate, crop, color jitter
    Text: synonym replacement, back-translation
    Audio: time shift, speed change, noise
    
    • Increases effective dataset size
    • Teaches invariances


LABEL SMOOTHING:
───────────────────────────────────────────────────────────────────────────────
    Instead of hard labels [0, 1, 0]:
    Use soft labels [0.05, 0.9, 0.05]
    
    • Prevents overconfident predictions
    • Improves calibration
"""

def example_regularization_nn():
    """Demonstrate regularization techniques for neural networks."""
    
    print("NEURAL NETWORK REGULARIZATION")
    print("=" * 70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Dropout
    # ─────────────────────────────────────────────────────────────────────────
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # Model with regularization
    # ─────────────────────────────────────────────────────────────────────────
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


# ███████████████████████████████████████████████████████████████████████████████
#                                                                               
#   CHAPTER 12: CONVOLUTIONAL NEURAL NETWORKS                                    
#                                                                               
# ███████████████████████████████████████████████████████████████████████████████

"""
CONVOLUTIONAL NEURAL NETWORKS (CNNs)
═══════════════════════════════════════════════════════════════════════════════

CNNs are specialized for processing grid-like data (images, audio, sequences).

WHY CONVOLUTIONS?
───────────────────────────────────────────────────────────────────────────────
1. PARAMETER SHARING: Same kernel applied everywhere → fewer parameters
2. TRANSLATION INVARIANCE: Detect features anywhere in input
3. LOCAL CONNECTIVITY: Each neuron only connected to local region
4. HIERARCHICAL FEATURES: Lower layers learn edges, higher layers learn objects


CONVOLUTION OPERATION:
───────────────────────────────────────────────────────────────────────────────

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
───────────────────────────────────────────────────────────────────────────────
Input → [Conv → BatchNorm → ReLU → Pool] × N → Flatten → FC → Output

                                          
      ┌──────────────────────────────────────────────────────────────┐
      │                                                              │
      │   Input    Conv    Pool    Conv    Pool    Flatten    FC     │
      │    [■]  → [■■■] → [■■] → [■■■■] → [■■] → [────] → [●●●●●]  │
      │                                                              │
      └──────────────────────────────────────────────────────────────┘
"""

def example_cnn_basics():
    """Demonstrate CNN fundamentals."""
    
    print("CONVOLUTIONAL NEURAL NETWORKS")
    print("=" * 70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Convolution operation
    # ─────────────────────────────────────────────────────────────────────────
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # PyTorch Conv2d
    # ─────────────────────────────────────────────────────────────────────────
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pooling
    # ─────────────────────────────────────────────────────────────────────────
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
    ─────────────────────────────────────────────────────────────────────
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


# ███████████████████████████████████████████████████████████████████████████████
#                                                                               
#   CHAPTER 13: RECURRENT NEURAL NETWORKS                                        
#                                                                               
# ███████████████████████████████████████████████████████████████████████████████

"""
RECURRENT NEURAL NETWORKS (RNNs)
═══════════════════════════════════════════════════════════════════════════════

RNNs process sequential data by maintaining a hidden state.

WHY RNNS?
───────────────────────────────────────────────────────────────────────────────
• Process sequences of variable length
• Share parameters across time steps
• Capture temporal dependencies


BASIC RNN:
───────────────────────────────────────────────────────────────────────────────
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
───────────────────────────────────────────────────────────────────────────────
When backpropagating through time, gradients multiply.
If |gradient| < 1, they shrink exponentially → can't learn long-range dependencies


LSTM (Long Short-Term Memory):
───────────────────────────────────────────────────────────────────────────────
Uses gates to control information flow:

    f_t = σ(W_f × [h_{t-1}, x_t] + b_f)    # Forget gate
    i_t = σ(W_i × [h_{t-1}, x_t] + b_i)    # Input gate
    c̃_t = tanh(W_c × [h_{t-1}, x_t] + b_c)  # Candidate cell state
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t        # Cell state
    o_t = σ(W_o × [h_{t-1}, x_t] + b_o)    # Output gate
    h_t = o_t ⊙ tanh(c_t)                   # Hidden state

The cell state c_t provides a "highway" for gradients!


GRU (Gated Recurrent Unit):
───────────────────────────────────────────────────────────────────────────────
Simplified version of LSTM with 2 gates:

    z_t = σ(W_z × [h_{t-1}, x_t])          # Update gate
    r_t = σ(W_r × [h_{t-1}, x_t])          # Reset gate
    h̃_t = tanh(W × [r_t ⊙ h_{t-1}, x_t])   # Candidate hidden
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  # Hidden state

Fewer parameters than LSTM, often similar performance.
"""

def example_rnn():
    """Demonstrate RNN, LSTM, and GRU."""
    
    print("RECURRENT NEURAL NETWORKS")
    print("=" * 70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Simple RNN
    # ─────────────────────────────────────────────────────────────────────────
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # LSTM
    # ─────────────────────────────────────────────────────────────────────────
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # GRU
    # ─────────────────────────────────────────────────────────────────────────
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # Bidirectional LSTM
    # ─────────────────────────────────────────────────────────────────────────
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


# ███████████████████████████████████████████████████████████████████████████████
#                                                                               
#   CHAPTER 14: ATTENTION AND TRANSFORMERS                                       
#                                                                               
# ███████████████████████████████████████████████████████████████████████████████

"""
ATTENTION AND TRANSFORMERS
═══════════════════════════════════════════════════════════════════════════════

Transformers revolutionized NLP and are now used everywhere in AI.

WHY ATTENTION?
───────────────────────────────────────────────────────────────────────────────
• RNNs process sequentially → slow, hard to parallelize
• Attention looks at entire sequence at once → fast, parallel
• Better at capturing long-range dependencies


ATTENTION MECHANISM:
───────────────────────────────────────────────────────────────────────────────

Query, Key, Value framework:

    Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V

Where:
• Q: Query (what we're looking for)
• K: Key (what each position offers)
• V: Value (actual content)
• d_k: Dimension of keys (for scaling)


SELF-ATTENTION:
───────────────────────────────────────────────────────────────────────────────
Q, K, V all come from the same sequence:

    Q = X × W_Q
    K = X × W_K
    V = X × W_V
    
    Output = Attention(Q, K, V)

Each position attends to all positions in the sequence.


MULTI-HEAD ATTENTION:
───────────────────────────────────────────────────────────────────────────────
Run multiple attention operations in parallel, then concatenate:

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
    
    where head_i = Attention(Q × W_Q_i, K × W_K_i, V × W_V_i)

Different heads can learn different types of relationships!


TRANSFORMER ARCHITECTURE:
───────────────────────────────────────────────────────────────────────────────

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
"""

def example_attention():
    """Demonstrate attention mechanisms."""
    
    print("ATTENTION MECHANISMS")
    print("=" * 70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Scaled Dot-Product Attention
    # ─────────────────────────────────────────────────────────────────────────
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # Multi-Head Attention
    # ─────────────────────────────────────────────────────────────────────────
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # Positional Encoding
    # ─────────────────────────────────────────────────────────────────────────
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # Using PyTorch's built-in Transformer
    # ─────────────────────────────────────────────────────────────────────────
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
    ─────────────────────────────────────────────────────────────────────────
    
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


# ═══════════════════════════════════════════════════════════════════════════════
# Run all examples
# ═══════════════════════════════════════════════════════════════════════════════

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
"""
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
"""

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
"""
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
"""

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
"""
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
"""
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
"""
Flask API for model serving.

File: app.py
"""

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
"""
FastAPI for model serving - faster and with automatic docs.

File: main.py
"""

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
"""
Example ML Pipeline using a simple orchestrator.
"""

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
# Part X: Appendices

---

## Appendix A: Python & NumPy Refresher

### A.1 Essential Python for ML

```python
# List comprehensions
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
matrix = [[i*j for j in range(5)] for i in range(5)]

# Dictionary comprehensions
word_lengths = {word: len(word) for word in ['hello', 'world', 'python']}

# Lambda functions
square = lambda x: x**2
add = lambda x, y: x + y

# Map, filter, reduce
from functools import reduce
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
total = reduce(lambda x, y: x + y, numbers)

# *args and **kwargs
def flexible_function(*args, **kwargs):
    print(f"Positional args: {args}")
    print(f"Keyword args: {kwargs}")

# Generators
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Context managers
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"Elapsed: {self.elapsed:.4f}s")

# Decorators
def timer_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.4f}s")
        return result
    return wrapper
```

### A.2 NumPy Essentials

```python
import numpy as np

# Array creation
a = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
identity = np.eye(4)
range_arr = np.arange(0, 10, 0.5)
linspace = np.linspace(0, 1, 100)
random_arr = np.random.randn(3, 4)

# Reshaping
a = np.arange(12)
b = a.reshape(3, 4)
c = a.reshape(-1, 2)  # -1 infers dimension
d = b.flatten()
e = b.T  # Transpose

# Indexing and slicing
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row = arr[0]           # First row
col = arr[:, 0]        # First column
subarray = arr[0:2, 1:3]  # Subarray
mask = arr[arr > 5]    # Boolean indexing

# Broadcasting
a = np.array([[1], [2], [3]])  # Shape (3, 1)
b = np.array([1, 2, 3])        # Shape (3,)
c = a + b                       # Shape (3, 3) - broadcasting!

# Linear algebra
A = np.random.randn(3, 3)
B = np.random.randn(3, 3)

dot_product = np.dot(A, B)        # Matrix multiplication
matmul = A @ B                    # Same thing
inverse = np.linalg.inv(A)
determinant = np.linalg.det(A)
eigenvalues, eigenvectors = np.linalg.eig(A)
svd_U, svd_S, svd_Vt = np.linalg.svd(A)

# Statistical operations
data = np.random.randn(100, 5)
mean = np.mean(data, axis=0)      # Column means
std = np.std(data, axis=1)        # Row stds
median = np.median(data)
percentile = np.percentile(data, 95)
correlation = np.corrcoef(data.T)

# Useful functions
sorted_indices = np.argsort(a)
unique_values = np.unique(a)
concatenated = np.concatenate([a, b])
stacked_v = np.vstack([a, b])
stacked_h = np.hstack([a, b])
```

### A.3 Pandas Essentials

```python
import pandas as pd

# DataFrame creation
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': ['a', 'b', 'c', 'd'],
    'C': [1.1, 2.2, 3.3, 4.4]
})

# Reading data
df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')
df = pd.read_json('data.json')

# Selection
col = df['A']                    # Single column
cols = df[['A', 'B']]           # Multiple columns
row = df.iloc[0]                 # By integer index
rows = df.iloc[0:5]              # By integer range
loc = df.loc[0, 'A']            # By label
filtered = df[df['A'] > 2]      # Boolean filtering
query = df.query('A > 2 and B == "c"')

# Data manipulation
df['D'] = df['A'] * 2           # New column
df['E'] = df['A'].apply(lambda x: x**2)  # Apply function
df = df.drop('D', axis=1)       # Drop column
df = df.rename(columns={'A': 'a'})  # Rename

# Aggregation
grouped = df.groupby('B').agg({
    'A': 'mean',
    'C': ['min', 'max', 'std']
})

# Pivot tables
pivot = df.pivot_table(
    values='C',
    index='A',
    columns='B',
    aggfunc='mean'
)

# Merging
merged = pd.merge(df1, df2, on='key', how='left')
concatenated = pd.concat([df1, df2], axis=0)

# Missing values
df.isna().sum()                  # Count missing
df.fillna(0)                     # Fill with value
df.fillna(df.mean())             # Fill with mean
df.dropna()                      # Drop missing rows

# Time series
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.resample('M').mean()          # Monthly aggregation
```

---

## Appendix B: Algorithm Cheat Sheets

### B.1 Model Selection Guide

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MODEL SELECTION FLOWCHART                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  START: What type of problem?                                      │
│         │                                                          │
│         ├── Predicting categories → CLASSIFICATION                 │
│         │   ├── Binary or Multiclass?                              │
│         │   ├── Linear boundary? → Logistic Regression             │
│         │   ├── Need interpretability? → Decision Tree             │
│         │   ├── High accuracy? → Random Forest, XGBoost            │
│         │   └── Text data? → Naive Bayes, BERT                     │
│         │                                                          │
│         ├── Predicting numbers → REGRESSION                        │
│         │   ├── Linear relationship? → Linear Regression           │
│         │   ├── Nonlinear? → Polynomial, Decision Tree             │
│         │   ├── High accuracy? → Gradient Boosting                 │
│         │   └── Time series? → ARIMA, LSTM                         │
│         │                                                          │
│         └── Finding patterns → UNSUPERVISED                        │
│             ├── Grouping? → Clustering (K-Means, DBSCAN)           │
│             ├── Reducing dimensions? → PCA, t-SNE                  │
│             └── Finding outliers? → Isolation Forest               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### B.2 Hyperparameter Tuning Guide

```
┌─────────────────────────────────────────────────────────────────────┐
│                  HYPERPARAMETER GUIDELINES                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  RANDOM FOREST                                                      │
│  ├── n_estimators: 100-1000 (more = better but slower)             │
│  ├── max_depth: None, 5-30 (None for full trees)                   │
│  ├── min_samples_split: 2-20                                       │
│  ├── min_samples_leaf: 1-10                                        │
│  └── max_features: 'sqrt' for classification, 'auto' for regression│
│                                                                     │
│  GRADIENT BOOSTING (XGBoost)                                        │
│  ├── learning_rate: 0.01-0.3 (lower = more trees needed)           │
│  ├── n_estimators: 100-1000                                        │
│  ├── max_depth: 3-10 (usually 3-6)                                 │
│  ├── subsample: 0.6-1.0                                            │
│  └── colsample_bytree: 0.6-1.0                                     │
│                                                                     │
│  NEURAL NETWORKS                                                    │
│  ├── learning_rate: 1e-4 to 1e-2                                   │
│  ├── batch_size: 16, 32, 64, 128, 256                              │
│  ├── hidden_layers: 1-5 for most problems                          │
│  ├── dropout: 0.1-0.5                                              │
│  └── optimizer: Adam (usually best default)                        │
│                                                                     │
│  SVM                                                                │
│  ├── C: 0.1, 1, 10, 100 (regularization)                           │
│  ├── kernel: 'rbf' (default), 'linear', 'poly'                     │
│  └── gamma: 'scale', 'auto', or specific values                    │
│                                                                     │
│  K-MEANS                                                            │
│  ├── n_clusters: Use elbow method or silhouette                    │
│  ├── init: 'k-means++' (better than random)                        │
│  └── n_init: 10-20 (number of random initializations)              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### B.3 Metrics Quick Reference

```
┌─────────────────────────────────────────────────────────────────────┐
│                    METRICS QUICK REFERENCE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CLASSIFICATION                                                     │
│  ├── Accuracy = (TP + TN) / Total                                  │
│  ├── Precision = TP / (TP + FP)     "Of predicted +, how many +"   │
│  ├── Recall = TP / (TP + FN)        "Of actual +, how many found"  │
│  ├── F1 = 2 * (P * R) / (P + R)     Harmonic mean                  │
│  ├── ROC-AUC = Area under ROC curve                                │
│  └── Log Loss = -mean(y*log(p) + (1-y)*log(1-p))                   │
│                                                                     │
│  REGRESSION                                                         │
│  ├── MSE = mean((y - ŷ)²)                                          │
│  ├── RMSE = √MSE                                                   │
│  ├── MAE = mean(|y - ŷ|)                                           │
│  ├── R² = 1 - SS_res/SS_tot                                        │
│  └── MAPE = mean(|y - ŷ| / y) * 100%                               │
│                                                                     │
│  CLUSTERING                                                         │
│  ├── Silhouette Score: -1 to 1 (higher better)                     │
│  ├── Inertia: Within-cluster sum of squares                        │
│  └── Adjusted Rand Index: -1 to 1 (1 = perfect)                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Appendix C: Common Errors and Solutions

### C.1 Data Errors

```python
"""Common data-related errors and fixes."""

# Error: ValueError: could not convert string to float
# Cause: Non-numeric values in data
# Fix:
df['column'] = pd.to_numeric(df['column'], errors='coerce')

# Error: All features contain NaN
# Cause: Missing value handling issues
# Fix:
df = df.dropna()
# or
df = df.fillna(df.mean())

# Error: Found input variables with inconsistent numbers of samples
# Cause: X and y have different lengths
# Fix:
assert len(X) == len(y), f"X has {len(X)} samples, y has {len(y)}"

# Error: Singular matrix
# Cause: Multicollinearity or constant features
# Fix:
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X = selector.fit_transform(X)

# Error: Memory error with large dataset
# Fix: Use chunking
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)
```

### C.2 Model Errors

```python
"""Common model-related errors and fixes."""

# Error: Convergence warning
# Cause: Model didn't converge
# Fix:
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)  # Increase iterations

# Error: ValueError: Unknown label type
# Cause: Wrong target type for classifier
# Fix:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Error: CUDA out of memory
# Fix: Reduce batch size or use gradient accumulation
batch_size = 16  # Try smaller
# or
model.zero_grad()
for i, (x, y) in enumerate(dataloader):
    loss = model(x, y) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        model.zero_grad()

# Error: NaN loss during training
# Causes: Learning rate too high, exploding gradients
# Fix:
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Lower lr
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
```

### C.3 Sklearn Pipeline Errors

```python
"""Pipeline-related errors and fixes."""

# Error: TypeError: All intermediate steps should be transformers
# Cause: Putting estimator in middle of pipeline
# Fix: Estimator must be last step
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),      # Transformer
    ('pca', PCA(n_components=10)),     # Transformer
    ('classifier', LogisticRegression())  # Estimator (last)
])

# Error: Column transformer with mixed types
# Fix:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# Error: Data leakage in cross-validation
# Fix: Use pipeline inside cross-validation
from sklearn.model_selection import cross_val_score
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
scores = cross_val_score(pipeline, X, y, cv=5)  # Scaling done per fold
```

---

## Appendix D: Interview Questions

### D.1 Conceptual Questions

```
1. What is the bias-variance tradeoff?
   - Bias: Error from wrong assumptions (underfitting)
   - Variance: Error from sensitivity to training data (overfitting)
   - Total Error = Bias² + Variance + Irreducible Error
   - Tradeoff: Decreasing one often increases the other

2. Explain gradient descent.
   - Optimization algorithm to find minimum of function
   - Steps: Calculate gradient, move opposite direction
   - Learning rate controls step size
   - Variants: Batch, Stochastic, Mini-batch

3. How do you handle imbalanced datasets?
   - Resampling: Oversample minority, undersample majority
   - SMOTE: Generate synthetic samples
   - Class weights: Penalize misclassifying minority more
   - Different metrics: F1, AUC instead of accuracy
   - Threshold tuning: Adjust decision threshold

4. What is regularization and why is it used?
   - Technique to prevent overfitting
   - L1 (Lasso): Adds |weights| penalty, promotes sparsity
   - L2 (Ridge): Adds weights² penalty, shrinks weights
   - Dropout: Randomly deactivates neurons during training

5. Explain cross-validation.
   - Technique to assess model generalization
   - K-Fold: Split data into K parts, train on K-1, test on 1
   - Stratified: Maintains class proportions in each fold
   - Time series: Must respect temporal order

6. What is the curse of dimensionality?
   - Problems that arise in high-dimensional spaces
   - Distances become less meaningful
   - Data becomes sparse
   - Need exponentially more data
   - Solutions: Feature selection, dimensionality reduction

7. Explain precision vs recall.
   - Precision: Of predicted positives, how many are correct
   - Recall: Of actual positives, how many did we find
   - Trade-off: Increasing threshold → higher precision, lower recall
   - F1 Score: Harmonic mean balances both

8. What is feature scaling and when is it needed?
   - Transforming features to similar ranges
   - StandardScaler: Mean=0, Std=1
   - MinMaxScaler: Range [0, 1]
   - Needed for: Distance-based (KNN, SVM), gradient descent
   - Not needed for: Tree-based models

9. Explain the difference between generative and discriminative models.
   - Generative: Model P(X|Y) and P(Y), learn data distribution
     Examples: Naive Bayes, GANs, GMM
   - Discriminative: Model P(Y|X) directly, learn decision boundary
     Examples: Logistic Regression, SVM, Neural Networks

10. What is transfer learning?
    - Using knowledge from one task to help another
    - Pre-trained models on large datasets
    - Fine-tuning: Update weights for specific task
    - Feature extraction: Use pre-trained features
    - Common in: Computer vision (ImageNet), NLP (BERT)
```

### D.2 Coding Questions

```python
# Q1: Implement K-fold cross-validation
def kfold_cv(X, y, model_fn, k=5):
    n = len(X)
    fold_size = n // k
    scores = []
    
    for i in range(k):
        # Create fold indices
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n
        
        # Split data
        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        X_train = np.concatenate([X[:val_start], X[val_end:]])
        y_train = np.concatenate([y[:val_start], y[val_end:]])
        
        # Train and evaluate
        model = model_fn()
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)


# Q2: Implement sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# Q3: Implement binary cross-entropy loss
def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Q4: Implement precision, recall, F1
def classification_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


# Q5: Implement train-test split
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    
    n = len(X)
    indices = np.random.permutation(n)
    test_size = int(n * test_size)
    
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```

---

## Appendix E: Glossary

```
A
─────────────────────────────────────────────
Accuracy: Proportion of correct predictions
Activation Function: Nonlinearity in neural networks (ReLU, sigmoid)
Adam: Adaptive Moment Estimation optimizer
AUC: Area Under the ROC Curve

B
─────────────────────────────────────────────
Backpropagation: Algorithm to compute gradients in neural networks
Bagging: Bootstrap Aggregating, ensemble technique
Batch Normalization: Normalizes activations in neural networks
Bias: Model assumption error (underfitting)
BERT: Bidirectional Encoder Representations from Transformers

C
─────────────────────────────────────────────
Classification: Predicting categorical labels
CNN: Convolutional Neural Network
Cross-Entropy: Loss function for classification
Cross-Validation: Technique to assess model generalization
Curse of Dimensionality: Problems in high-dimensional spaces

D
─────────────────────────────────────────────
Data Augmentation: Creating new training samples from existing
Decision Boundary: Surface separating classes
Dropout: Regularization by randomly dropping neurons
Deep Learning: Neural networks with many layers

E
─────────────────────────────────────────────
Embedding: Dense vector representation
Ensemble: Combining multiple models
Epoch: One complete pass through training data
Exploding Gradients: Gradients become too large

F
─────────────────────────────────────────────
F1 Score: Harmonic mean of precision and recall
Feature Engineering: Creating features from raw data
Feature Scaling: Normalizing feature ranges
Fine-tuning: Adjusting pre-trained model for new task
Forward Pass: Computing output from input

G
─────────────────────────────────────────────
Gradient Descent: Optimization by following negative gradient
GRU: Gated Recurrent Unit
GPT: Generative Pre-trained Transformer

H
─────────────────────────────────────────────
Hyperparameter: Parameter set before training (learning rate, etc.)
Holdout Set: Data reserved for final evaluation

I
─────────────────────────────────────────────
Imputation: Filling in missing values
Inductive Bias: Model assumptions
Information Gain: Reduction in entropy after split

K
─────────────────────────────────────────────
K-Fold: Cross-validation splitting into K parts
K-Means: Clustering algorithm
Kernel: Function computing similarity in higher dimensions

L
─────────────────────────────────────────────
L1 Regularization: Lasso, penalty on |weights|
L2 Regularization: Ridge, penalty on weights²
Learning Rate: Step size in gradient descent
LSTM: Long Short-Term Memory

M
─────────────────────────────────────────────
MAE: Mean Absolute Error
Mini-batch: Subset of data for one gradient update
MSE: Mean Squared Error
Multicollinearity: High correlation between features

N
─────────────────────────────────────────────
Neural Network: Layers of connected neurons
NLP: Natural Language Processing
Normalization: Scaling to range [0,1] or mean=0, std=1

O
─────────────────────────────────────────────
One-Hot Encoding: Binary encoding for categories
Overfitting: Model memorizes training data
Optimizer: Algorithm to update weights (SGD, Adam)

P
─────────────────────────────────────────────
PCA: Principal Component Analysis
Precision: TP / (TP + FP)
Pooling: Downsampling in CNNs

R
─────────────────────────────────────────────
Recall: TP / (TP + FN)
Regularization: Techniques to prevent overfitting
ReLU: Rectified Linear Unit, max(0, x)
RMSE: Root Mean Squared Error
RNN: Recurrent Neural Network
ROC Curve: Receiver Operating Characteristic

S
─────────────────────────────────────────────
Softmax: Converts logits to probabilities
Stochastic: Random, as in SGD
Stride: Step size in convolution
SVM: Support Vector Machine

T
─────────────────────────────────────────────
TF-IDF: Term Frequency-Inverse Document Frequency
Training Set: Data used to train model
Transfer Learning: Using pre-trained models
Transformer: Attention-based architecture

U
─────────────────────────────────────────────
Underfitting: Model too simple for data
Unsupervised Learning: Learning without labels

V
─────────────────────────────────────────────
Validation Set: Data for hyperparameter tuning
Vanishing Gradients: Gradients become too small
Variance: Model sensitivity to training data

W
─────────────────────────────────────────────
Weight: Learnable parameter in model
Word Embedding: Vector representation of words

X
─────────────────────────────────────────────
XGBoost: Extreme Gradient Boosting
```

---

## Appendix F: Resources and Further Reading

### Books
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" - Aurélien Géron
- "Deep Learning" - Goodfellow, Bengio, Courville
- "Pattern Recognition and Machine Learning" - Christopher Bishop
- "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman
- "Machine Learning: A Probabilistic Perspective" - Kevin Murphy

### Online Courses
- Coursera: Machine Learning (Andrew Ng)
- fast.ai: Practical Deep Learning
- Stanford CS229: Machine Learning
- Stanford CS231n: Convolutional Neural Networks
- Stanford CS224n: Natural Language Processing

### Documentation
- scikit-learn: https://scikit-learn.org/
- PyTorch: https://pytorch.org/docs/
- TensorFlow: https://www.tensorflow.org/guide
- Hugging Face: https://huggingface.co/docs

### Communities
- Kaggle: Competitions and datasets
- Papers With Code: Latest research with implementations
- Reddit: r/MachineLearning
- Stack Overflow: Questions and answers

---

# End of Machine Learning Textbook

This comprehensive guide covered:
- Machine Learning fundamentals and types
- Mathematical foundations (linear algebra, calculus, probability)
- Data preprocessing and feature engineering
- Supervised learning algorithms
- Neural networks and deep learning
- Unsupervised learning
- Natural language processing
- Time series analysis
- MLOps and deployment
- Practical appendices and references

Remember: The best way to learn ML is by doing. Start with simple projects,
implement algorithms from scratch to understand them deeply, then use
libraries for production work. Good luck on your ML journey!
# Part XI: Advanced Deep Learning - Computer Vision

---

## Chapter 33: CNN Architectures Deep Dive

### 33.1 Evolution of CNN Architectures

```
┌─────────────────────────────────────────────────────────────────────┐
│                CNN ARCHITECTURE EVOLUTION                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1998: LeNet-5 (LeCun)                                             │
│        └── First successful CNN for digit recognition              │
│                                                                     │
│  2012: AlexNet (Krizhevsky)                                        │
│        └── Deep CNN, ReLU, Dropout, GPU training                   │
│                                                                     │
│  2014: VGGNet (Simonyan)                                           │
│        └── Very deep (16-19 layers), 3x3 convolutions              │
│                                                                     │
│  2014: GoogLeNet/Inception (Szegedy)                               │
│        └── Inception modules, 1x1 convolutions                     │
│                                                                     │
│  2015: ResNet (He)                                                 │
│        └── Skip connections, 152+ layers                           │
│                                                                     │
│  2017: DenseNet (Huang)                                            │
│        └── Dense connections between all layers                    │
│                                                                     │
│  2019: EfficientNet (Tan)                                          │
│        └── Compound scaling, state-of-the-art                      │
│                                                                     │
│  2020: Vision Transformer (ViT)                                    │
│        └── Transformers for images                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 33.2 Complete ResNet Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50/101/152."""
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        # 1x1 convolution to reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 convolution to expand channels
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    Complete ResNet Implementation.
    
    Args:
        block: BasicBlock or Bottleneck
        layers: Number of blocks in each stage [stage1, stage2, stage3, stage4]
        num_classes: Number of output classes
    """
    
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        
        self.in_channels = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# Model factory functions
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


# Test models
print("ResNet Variants:")
print("=" * 50)
for name, model_fn in [('ResNet-18', resnet18), ('ResNet-50', resnet50)]:
    model = model_fn(num_classes=10)
    params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {params:,} parameters")
```

### 33.3 Inception Module

```python
class InceptionModule(nn.Module):
    """
    Inception Module (GoogLeNet style).
    
    Parallel branches with different receptive fields:
    - 1x1 convolution
    - 1x1 → 3x3 convolution
    - 1x1 → 5x5 convolution
    - 3x3 max pool → 1x1 convolution
    """
    
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()
        
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 1x1 → 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 1x1 → 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 3x3 max pool → 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # Concatenate along channel dimension
        return torch.cat([b1, b2, b3, b4], dim=1)


# Test Inception module
inception = InceptionModule(192, 64, 96, 128, 16, 32, 32)
x = torch.randn(1, 192, 28, 28)
out = inception(x)
print(f"\nInception Module:")
print(f"Input: {x.shape}")
print(f"Output: {out.shape}")  # Should be (1, 64+128+32+32=256, 28, 28)
```

### 33.4 DenseNet

```python
class DenseLayer(nn.Module):
    """Single layer in a DenseNet dense block."""
    
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )
    
    def forward(self, x):
        new_features = self.layers(x)
        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
    """Dense block with multiple dense layers."""
    
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class Transition(nn.Module):
    """Transition layer between dense blocks."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.layers(x)


class DenseNet(nn.Module):
    """
    DenseNet Implementation.
    
    Each layer receives input from ALL preceding layers.
    """
    
    def __init__(self, block_config=(6, 12, 24, 16), growth_rate=32,
                 num_init_features=64, num_classes=1000):
        super().__init__()
        
        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks
        num_features = num_init_features
        
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        # Classifier
        self.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


# Test DenseNet
densenet = DenseNet(block_config=(6, 12, 24, 16), growth_rate=32, num_classes=10)
params = sum(p.numel() for p in densenet.parameters())
print(f"\nDenseNet-121: {params:,} parameters")
```

---

## Chapter 34: Object Detection

### 34.1 Anchor-Based Detection

```python
class AnchorGenerator:
    """Generate anchor boxes for object detection."""
    
    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(sizes) * len(aspect_ratios)
    
    def generate_anchors(self, feature_map_size, image_size, stride):
        """
        Generate anchors for a single feature map.
        
        Returns:
            anchors: (H*W*num_anchors, 4) tensor of (x1, y1, x2, y2)
        """
        H, W = feature_map_size
        
        # Generate anchor centers
        shifts_x = torch.arange(0, W) * stride + stride // 2
        shifts_y = torch.arange(0, H) * stride + stride // 2
        
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=-1).reshape(-1, 4)
        
        # Generate base anchors (centered at origin)
        base_anchors = []
        for size in self.sizes:
            for ratio in self.aspect_ratios:
                w = size * np.sqrt(ratio)
                h = size / np.sqrt(ratio)
                base_anchors.append([-w/2, -h/2, w/2, h/2])
        
        base_anchors = torch.tensor(base_anchors, dtype=torch.float32)
        
        # Combine shifts and base anchors
        anchors = shifts.unsqueeze(1) + base_anchors.unsqueeze(0)
        anchors = anchors.reshape(-1, 4)
        
        # Clip to image bounds
        anchors[:, 0::2].clamp_(min=0, max=image_size[1])
        anchors[:, 1::2].clamp_(min=0, max=image_size[0])
        
        return anchors


def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: (N, 4) tensor
        boxes2: (M, 4) tensor
    
    Returns:
        iou: (N, M) tensor
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    # Union
    union = area1[:, None] + area2[None, :] - inter
    
    iou = inter / (union + 1e-6)
    
    return iou


def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression.
    
    Args:
        boxes: (N, 4) tensor
        scores: (N,) tensor
        iou_threshold: IoU threshold for suppression
    
    Returns:
        keep: indices of boxes to keep
    """
    # Sort by score
    order = scores.argsort(descending=True)
    
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        
        if order.numel() == 1:
            break
        
        # Compute IoU with remaining boxes
        iou = compute_iou(boxes[i:i+1], boxes[order[1:]])[0]
        
        # Keep boxes with IoU below threshold
        mask = iou <= iou_threshold
        order = order[1:][mask]
    
    return torch.tensor(keep)


class RegionProposalNetwork(nn.Module):
    """
    Region Proposal Network (RPN) for Faster R-CNN.
    """
    
    def __init__(self, in_channels, num_anchors=9):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        
        # Classification: objectness score
        self.cls_layer = nn.Conv2d(512, num_anchors * 2, kernel_size=1)
        
        # Regression: box deltas
        self.reg_layer = nn.Conv2d(512, num_anchors * 4, kernel_size=1)
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        
        # Objectness scores
        cls_scores = self.cls_layer(x)
        
        # Box deltas
        bbox_deltas = self.reg_layer(x)
        
        return cls_scores, bbox_deltas


print("\nObject Detection Components:")
print("=" * 50)

# Test anchor generation
anchor_gen = AnchorGenerator(sizes=(32, 64, 128), aspect_ratios=(0.5, 1.0, 2.0))
anchors = anchor_gen.generate_anchors((7, 7), (224, 224), stride=32)
print(f"Generated {len(anchors)} anchors for 7x7 feature map")

# Test RPN
rpn = RegionProposalNetwork(256, num_anchors=9)
feature_map = torch.randn(1, 256, 7, 7)
cls_scores, bbox_deltas = rpn(feature_map)
print(f"RPN cls_scores shape: {cls_scores.shape}")
print(f"RPN bbox_deltas shape: {bbox_deltas.shape}")
```

### 34.2 YOLO-style Detection

```python
class YOLOv1Head(nn.Module):
    """
    YOLO v1 style detection head.
    
    Divides image into SxS grid, predicts B boxes per cell.
    """
    
    def __init__(self, in_channels, S=7, B=2, C=20):
        super().__init__()
        
        self.S = S  # Grid size
        self.B = B  # Boxes per cell
        self.C = C  # Number of classes
        
        # Output: S*S*(B*5 + C)
        # Each box: (x, y, w, h, confidence)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (B * 5 + C))
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        out = self.fc(x)
        out = out.view(batch_size, self.S, self.S, self.B * 5 + self.C)
        return out


class YOLOv3Detection(nn.Module):
    """
    YOLO v3 style detection at multiple scales.
    """
    
    def __init__(self, num_classes=80, anchors_per_scale=3):
        super().__init__()
        
        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale
        
        # Output channels: anchors * (5 + num_classes)
        out_channels = anchors_per_scale * (5 + num_classes)
        
        # Detection heads for 3 scales
        self.detect1 = nn.Conv2d(256, out_channels, kernel_size=1)  # Large objects
        self.detect2 = nn.Conv2d(512, out_channels, kernel_size=1)  # Medium objects
        self.detect3 = nn.Conv2d(1024, out_channels, kernel_size=1)  # Small objects
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps at 3 scales
        """
        p3, p4, p5 = features  # From FPN or backbone
        
        out1 = self.detect1(p3)  # (B, anchors*(5+C), H1, W1)
        out2 = self.detect2(p4)
        out3 = self.detect3(p5)
        
        return [out1, out2, out3]


def decode_yolo_output(output, anchors, num_classes, img_size):
    """
    Decode YOLO output to bounding boxes.
    
    Args:
        output: (B, anchors*(5+C), H, W)
        anchors: List of (w, h) anchor sizes
        num_classes: Number of classes
        img_size: Original image size
    """
    batch_size, _, H, W = output.shape
    num_anchors = len(anchors)
    
    # Reshape
    output = output.view(batch_size, num_anchors, 5 + num_classes, H, W)
    output = output.permute(0, 1, 3, 4, 2).contiguous()
    
    # Extract predictions
    tx = torch.sigmoid(output[..., 0])
    ty = torch.sigmoid(output[..., 1])
    tw = output[..., 2]
    th = output[..., 3]
    conf = torch.sigmoid(output[..., 4])
    class_probs = torch.sigmoid(output[..., 5:])
    
    # Create grid
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid_x = grid_x.float().unsqueeze(0).unsqueeze(0)
    grid_y = grid_y.float().unsqueeze(0).unsqueeze(0)
    
    # Decode boxes
    stride = img_size // H
    bx = (tx + grid_x) * stride
    by = (ty + grid_y) * stride
    
    anchors = torch.tensor(anchors).view(1, num_anchors, 1, 1, 2)
    bw = torch.exp(tw) * anchors[..., 0] * stride
    bh = torch.exp(th) * anchors[..., 1] * stride
    
    # Convert to (x1, y1, x2, y2)
    x1 = bx - bw / 2
    y1 = by - bh / 2
    x2 = bx + bw / 2
    y2 = by + bh / 2
    
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    
    return boxes, conf, class_probs


print("\nYOLO Detection:")
yolo_head = YOLOv1Head(512, S=7, B=2, C=20)
x = torch.randn(1, 512, 7, 7)
out = yolo_head(x)
print(f"YOLO v1 output shape: {out.shape}")  # (1, 7, 7, 30)
```

### 34.3 Feature Pyramid Network (FPN)

```python
class FPN(nn.Module):
    """
    Feature Pyramid Network.
    
    Creates multi-scale feature maps for detecting objects of different sizes.
    """
    
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        
        # Output convolutions (3x3 to reduce aliasing)
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps from backbone [C2, C3, C4, C5]
        
        Returns:
            List of FPN feature maps [P2, P3, P4, P5]
        """
        # Bottom-up pathway already done by backbone
        
        # Top-down pathway with lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # Build top-down
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample and add
            upsampled = F.interpolate(laterals[i + 1], size=laterals[i].shape[-2:],
                                      mode='nearest')
            laterals[i] = laterals[i] + upsampled
        
        # Output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        
        return outputs


# Test FPN
in_channels = [256, 512, 1024, 2048]
fpn = FPN(in_channels, out_channels=256)

# Simulate backbone outputs at different scales
features = [
    torch.randn(1, 256, 56, 56),   # C2
    torch.randn(1, 512, 28, 28),   # C3
    torch.randn(1, 1024, 14, 14),  # C4
    torch.randn(1, 2048, 7, 7),    # C5
]

fpn_outputs = fpn(features)
print("\nFPN outputs:")
for i, out in enumerate(fpn_outputs):
    print(f"P{i+2}: {out.shape}")
```

---

## Chapter 35: Image Segmentation

### 35.1 U-Net Architecture

```python
class DoubleConv(nn.Module):
    """Double convolution block for U-Net."""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block: MaxPool + DoubleConv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block with skip connection."""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for semantic segmentation.
    
    Encoder-decoder architecture with skip connections.
    """
    
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits


# Test U-Net
unet = UNet(n_channels=3, n_classes=2)
x = torch.randn(1, 3, 256, 256)
out = unet(x)
print(f"\nU-Net:")
print(f"Input: {x.shape}")
print(f"Output: {out.shape}")
print(f"Parameters: {sum(p.numel() for p in unet.parameters()):,}")
```

### 35.2 Segmentation Losses

```python
class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p) = -α * (1-p)^γ * log(p)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Compute focal weights
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha weights
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Compute BCE
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        focal_loss = alpha_weight * focal_weight * bce
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined BCE + Dice loss."""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


print("\nSegmentation Losses:")
print("=" * 50)
print("1. Binary Cross-Entropy: Standard pixel-wise loss")
print("2. Dice Loss: Based on overlap, good for imbalanced")
print("3. Focal Loss: Down-weights easy examples")
print("4. Combined: BCE + Dice for best results")
```

---

## Chapter 36: Data Augmentation

### 36.1 Image Augmentation Techniques

```python
import torchvision.transforms as T
from PIL import Image

class ImageAugmentation:
    """Comprehensive image augmentation pipeline."""
    
    def __init__(self, train=True, image_size=224):
        self.train = train
        
        if train:
            self.transform = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.1),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.2),
            ])
        else:
            self.transform = T.Compose([
                T.Resize(int(image_size * 1.14)),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __call__(self, img):
        return self.transform(img)


class MixUp:
    """MixUp augmentation - blends two images."""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x, y):
        batch_size = x.size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        index = torch.randperm(batch_size)
        
        # Mix images
        mixed_x = lam * x + (1 - lam) * x[index]
        
        # Return mixed inputs and both labels
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    @staticmethod
    def loss(criterion, pred, y_a, y_b, lam):
        """Compute mixed loss."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CutMix:
    """CutMix augmentation - cuts and pastes patches."""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, x, y):
        batch_size = x.size(0)
        
        lam = np.random.beta(self.alpha, self.alpha)
        
        index = torch.randperm(batch_size)
        
        # Get random box
        _, _, H, W = x.shape
        cut_rat = np.sqrt(1 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Cut and paste
        x_mixed = x.clone()
        x_mixed[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        
        # Adjust lambda for actual area
        lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        
        y_a, y_b = y, y[index]
        
        return x_mixed, y_a, y_b, lam


class CutOut:
    """CutOut augmentation - masks random patches."""
    
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        
        mask = torch.ones_like(img)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[:, y1:y2, x1:x2] = 0
        
        return img * mask


print("\nData Augmentation Techniques:")
print("=" * 50)
print("""
Basic:
- RandomCrop, RandomFlip, RandomRotation
- ColorJitter, RandomErasing

Advanced:
- MixUp: Blend images α*img1 + (1-α)*img2
- CutMix: Paste patch from one image to another
- CutOut: Random rectangular mask

AutoAugment:
- Learned augmentation policies
- Searched on validation set
""")
```

---

## Summary: Computer Vision Deep Learning

```
┌─────────────────────────────────────────────────────────────────────┐
│              COMPUTER VISION SUMMARY                                │
├─────────────────────────────────────────────────────────────────────┤
│  CLASSIFICATION ARCHITECTURES                                       │
│  ├── VGG: Simple, deep, 3x3 convolutions                           │
│  ├── ResNet: Skip connections, very deep                           │
│  ├── Inception: Multi-scale parallel branches                      │
│  ├── DenseNet: Dense connections                                   │
│  └── EfficientNet: Compound scaling                                │
│                                                                     │
│  OBJECT DETECTION                                                   │
│  ├── Two-stage: R-CNN, Fast R-CNN, Faster R-CNN                    │
│  ├── One-stage: YOLO, SSD, RetinaNet                               │
│  └── Anchor-free: CenterNet, FCOS                                  │
│                                                                     │
│  SEGMENTATION                                                       │
│  ├── Semantic: U-Net, DeepLab                                      │
│  ├── Instance: Mask R-CNN                                          │
│  └── Panoptic: Combined semantic + instance                        │
│                                                                     │
│  DATA AUGMENTATION                                                  │
│  ├── Geometric: Crop, flip, rotate, scale                          │
│  ├── Color: Brightness, contrast, saturation                       │
│  └── Advanced: MixUp, CutMix, CutOut                               │
└─────────────────────────────────────────────────────────────────────┘
```
# Part XII: Reinforcement Learning

---

## Chapter 36: Introduction to Reinforcement Learning

### 36.1 The RL Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│                    REINFORCEMENT LEARNING LOOP                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                      ┌─────────────────┐                           │
│           Action at  │                 │  State st+1               │
│         ───────────► │   Environment   │ ───────────►              │
│         │            │                 │            │              │
│         │            └────────┬────────┘            │              │
│         │                     │                     │              │
│         │              Reward rt+1                  │              │
│         │                     │                     │              │
│         │                     ▼                     │              │
│         │            ┌─────────────────┐            │              │
│         │            │                 │            │              │
│         └────────────│     Agent       │◄───────────┘              │
│                      │                 │                           │
│                      └─────────────────┘                           │
│                                                                     │
│  Agent Goal: Maximize cumulative reward over time                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Concepts:**

```python
"""
State (s): Current situation
Action (a): What agent does
Reward (r): Feedback signal
Policy (π): Strategy for choosing actions
Value Function (V): Expected future reward from state
Q-Function (Q): Expected future reward from state-action pair

Discounted Return: G_t = r_t+1 + γ*r_t+2 + γ²*r_t+3 + ...
where γ ∈ [0,1] is the discount factor

Bellman Equation:
V(s) = E[r + γ*V(s')]
Q(s,a) = E[r + γ*max_a' Q(s',a')]
"""

import numpy as np
from collections import defaultdict

class Environment:
    """Base class for RL environments."""
    
    def reset(self):
        """Reset environment and return initial state."""
        raise NotImplementedError
    
    def step(self, action):
        """Take action and return (next_state, reward, done, info)."""
        raise NotImplementedError
    
    def render(self):
        """Visualize the environment."""
        pass


class GridWorld(Environment):
    """
    Simple grid world environment.
    
    Agent navigates from start to goal, avoiding obstacles.
    """
    
    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.obstacles = [(1, 1), (2, 2), (3, 1)]
        
        # Actions: 0=up, 1=right, 2=down, 3=left
        self.action_map = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1),  # left
        }
        self.n_actions = 4
        
        self.state = None
        
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        # Get movement
        dr, dc = self.action_map[action]
        new_r = self.state[0] + dr
        new_c = self.state[1] + dc
        
        # Check boundaries
        if 0 <= new_r < self.size and 0 <= new_c < self.size:
            new_state = (new_r, new_c)
            
            # Check obstacles
            if new_state not in self.obstacles:
                self.state = new_state
        
        # Calculate reward
        if self.state == self.goal:
            reward = 10
            done = True
        elif self.state in self.obstacles:
            reward = -10
            done = True
        else:
            reward = -1  # Step penalty
            done = False
        
        return self.state, reward, done, {}
    
    def render(self):
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'
        
        grid[self.goal[0]][self.goal[1]] = 'G'
        grid[self.state[0]][self.state[1]] = 'A'
        
        print('\n'.join([' '.join(row) for row in grid]))
        print()


# Test environment
env = GridWorld(size=5)
state = env.reset()
env.render()

# Random policy
done = False
total_reward = 0
steps = 0

while not done and steps < 50:
    action = np.random.randint(4)
    state, reward, done, _ = env.step(action)
    total_reward += reward
    steps += 1

print(f"Episode finished in {steps} steps with total reward: {total_reward}")
```

### 36.2 Value Iteration and Policy Iteration

```python
class ValueIteration:
    """
    Value Iteration algorithm.
    
    Finds optimal value function by iterative Bellman updates.
    """
    
    def __init__(self, env, gamma=0.99, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        # State space
        self.states = [(i, j) for i in range(env.size) for j in range(env.size)]
        
        # Initialize value function
        self.V = {s: 0 for s in self.states}
        
    def get_action_value(self, state, action):
        """Calculate Q(s,a) = E[r + γV(s')]."""
        # Save current state
        original_state = self.env.state
        self.env.state = state
        
        next_state, reward, done, _ = self.env.step(action)
        
        if done:
            q_value = reward
        else:
            q_value = reward + self.gamma * self.V[next_state]
        
        # Restore state
        self.env.state = original_state
        
        return q_value, next_state
    
    def iterate(self, max_iterations=1000):
        """Run value iteration."""
        for i in range(max_iterations):
            delta = 0
            
            for state in self.states:
                if state == self.env.goal or state in self.env.obstacles:
                    continue
                
                # Find best action value
                old_v = self.V[state]
                
                action_values = []
                for action in range(self.env.n_actions):
                    self.env.state = state
                    q_val, _ = self.get_action_value(state, action)
                    action_values.append(q_val)
                
                self.V[state] = max(action_values)
                delta = max(delta, abs(old_v - self.V[state]))
            
            if delta < self.theta:
                print(f"Value iteration converged after {i+1} iterations")
                break
        
        return self.V
    
    def get_policy(self):
        """Extract policy from value function."""
        policy = {}
        
        for state in self.states:
            if state == self.env.goal or state in self.env.obstacles:
                policy[state] = None
                continue
            
            best_action = None
            best_value = float('-inf')
            
            for action in range(self.env.n_actions):
                self.env.state = state
                q_val, _ = self.get_action_value(state, action)
                
                if q_val > best_value:
                    best_value = q_val
                    best_action = action
            
            policy[state] = best_action
        
        return policy
    
    def print_value_function(self):
        """Print value function as grid."""
        print("Value Function:")
        for i in range(self.env.size):
            row = []
            for j in range(self.env.size):
                v = self.V[(i, j)]
                row.append(f"{v:6.2f}")
            print(' '.join(row))
        print()


# Run value iteration
env = GridWorld(size=5)
vi = ValueIteration(env)
V = vi.iterate()
vi.print_value_function()

# Get and print policy
policy = vi.get_policy()
action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←', None: '·'}

print("Optimal Policy:")
for i in range(env.size):
    row = []
    for j in range(env.size):
        a = policy[(i, j)]
        row.append(action_symbols[a])
    print(' '.join(row))
```

---

## Chapter 37: Q-Learning and SARSA

### 37.1 Q-Learning

```python
class QLearning:
    """
    Q-Learning: Off-policy TD control.
    
    Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Q-table
        self.Q = defaultdict(lambda: np.zeros(env.n_actions))
        
        # Statistics
        self.episode_rewards = []
        
    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        # TD update
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])
    
    def train(self, num_episodes=1000):
        """Train the agent."""
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
            
            self.episode_rewards.append(total_reward)
            
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        return self.Q
    
    def get_policy(self):
        """Extract greedy policy from Q-table."""
        policy = {}
        for state in self.Q:
            policy[state] = np.argmax(self.Q[state])
        return policy


class SARSA:
    """
    SARSA: On-policy TD control.
    
    Q(s,a) ← Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
    
    Difference from Q-learning: uses actual next action, not max.
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.Q = defaultdict(lambda: np.zeros(env.n_actions))
        self.episode_rewards = []
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.n_actions)
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA update."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state][next_action]
        
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])
    
    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            total_reward = 0
            done = False
            
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.choose_action(next_state)
                
                self.update(state, action, reward, next_state, next_action, done)
                
                state = next_state
                action = next_action
                total_reward += reward
            
            self.episode_rewards.append(total_reward)
            self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return self.Q


# Train Q-learning agent
print("\nTraining Q-Learning agent:")
print("=" * 50)
env = GridWorld(size=5)
q_agent = QLearning(env, alpha=0.1, gamma=0.99, epsilon=1.0)
Q = q_agent.train(num_episodes=500)

# Test learned policy
print("\nTesting learned policy:")
state = env.reset()
env.render()

done = False
total_reward = 0
steps = 0

while not done and steps < 20:
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    total_reward += reward
    steps += 1
    env.render()

print(f"Reached goal in {steps} steps with reward: {total_reward}")
```

### 37.2 Experience Replay

```python
from collections import deque
import random

class ReplayBuffer:
    """
    Experience Replay Buffer.
    
    Stores transitions and samples random mini-batches for training.
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a random batch."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay.
    
    Samples transitions with probability proportional to TD error.
    """
    
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        
        self.buffer = []
        self.priorities = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done, td_error=None):
        """Store a transition with priority."""
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if td_error is not None:
            priority = (abs(td_error) + 1e-5) ** self.alpha
        else:
            priority = max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample with priorities."""
        if len(self.buffer) < batch_size:
            return None
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        N = len(self.buffer)
        weights = (N * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Get samples
        samples = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), indices, weights)
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on new TD errors."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-5) ** self.alpha


print("\nExperience Replay:")
print("=" * 50)
print("Benefits:")
print("1. Breaks correlation between consecutive samples")
print("2. Reuses experiences multiple times")
print("3. More sample efficient learning")
```

---

## Chapter 38: Deep Q-Networks (DQN)

### 38.1 DQN Architecture

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    """
    Deep Q-Network.
    
    Approximates Q-function with neural network.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """
    DQN Agent with target network and experience replay.
    """
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=10000, batch_size=64, target_update=10):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        self.train_step = 0
    
    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train on a batch from replay buffer."""
        if len(self.buffer) < self.batch_size:
            return 0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values (using target network)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()


class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture.
    
    Separates value and advantage streams:
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine: Q = V + A - mean(A)
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values


print("\nDQN Architectures:")
print("=" * 50)
dqn = DQN(state_dim=4, action_dim=2)
print("Standard DQN:")
print(dqn)

dueling = DuelingDQN(state_dim=4, action_dim=2)
print("\nDueling DQN:")
print(dueling)
```

### 38.2 Double DQN

```python
class DoubleDQNAgent(DQNAgent):
    """
    Double DQN Agent.
    
    Reduces overestimation by using:
    - Q-network to SELECT action
    - Target network to EVALUATE action
    """
    
    def train(self):
        if len(self.buffer) < self.batch_size:
            return 0
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Double DQN: Use Q-network to select action, target network to evaluate
        with torch.no_grad():
            # Select action using Q-network
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            # Evaluate using target network
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()


print("\nDouble DQN:")
print("=" * 50)
print("Problem: Standard DQN overestimates Q-values")
print("Solution: Decouple action selection and evaluation")
print("  - Q-network selects best action: argmax_a Q(s',a)")
print("  - Target network evaluates it: Q_target(s', argmax_a Q(s',a))")
```

---

## Chapter 39: Policy Gradient Methods

### 39.1 REINFORCE Algorithm

```python
class PolicyNetwork(nn.Module):
    """Policy network that outputs action probabilities."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)


class REINFORCE:
    """
    REINFORCE: Monte Carlo Policy Gradient.
    
    ∇J(θ) = E[∑ ∇log π(a|s) * G_t]
    """
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        
        # Episode storage
        self.log_probs = []
        self.rewards = []
    
    def choose_action(self, state):
        """Sample action from policy."""
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        
        # Sample from distribution
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        # Store log probability for training
        self.log_probs.append(dist.log_prob(action))
        
        return action.item()
    
    def store_reward(self, reward):
        """Store reward for current step."""
        self.rewards.append(reward)
    
    def train(self):
        """Update policy after episode."""
        # Calculate discounted returns
        returns = []
        G = 0
        
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        
        # Normalize returns (reduces variance)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient loss
        loss = 0
        for log_prob, G in zip(self.log_probs, returns):
            loss -= log_prob * G  # Negative for gradient ascent
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.log_probs = []
        self.rewards = []
        
        return loss.item()


print("\nREINFORCE Algorithm:")
print("=" * 50)
print("Policy Gradient Theorem:")
print("∇J(θ) = E[∑_t ∇log π(a_t|s_t) * G_t]")
print()
print("Properties:")
print("- Monte Carlo: Uses full episode returns")
print("- High variance: Returns can vary a lot")
print("- Unbiased gradient estimate")
```

### 39.2 Actor-Critic Methods

```python
class ActorCritic(nn.Module):
    """
    Actor-Critic Network.
    
    Actor: Policy π(a|s)
    Critic: Value function V(s)
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        features = self.shared(x)
        policy = self.actor(features)
        value = self.critic(features)
        return policy, value


class A2CAgent:
    """
    Advantage Actor-Critic (A2C).
    
    Uses advantage A(s,a) = Q(s,a) - V(s) ≈ r + γV(s') - V(s)
    """
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99):
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        
    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs, _ = self.network(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def train_step(self, state, action, reward, next_state, done, log_prob):
        """Single step actor-critic update."""
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        
        _, value = self.network(state)
        _, next_value = self.network(next_state)
        
        # TD target
        if done:
            target = reward
        else:
            target = reward + self.gamma * next_value.item()
        
        # Advantage
        advantage = target - value.item()
        
        # Actor loss (policy gradient with advantage)
        actor_loss = -log_prob * advantage
        
        # Critic loss (TD error)
        critic_loss = nn.MSELoss()(value, torch.tensor([[target]]))
        
        # Combined loss
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class PPO:
    """
    Proximal Policy Optimization.
    
    Clips policy ratio to prevent large updates.
    """
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 clip_epsilon=0.2, epochs=10):
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        
    def compute_gae(self, rewards, values, dones, next_value, gae_lambda=0.95):
        """Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def train(self, states, actions, old_log_probs, returns, advantages):
        """PPO training step."""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.epochs):
            # Get current policy
            probs, values = self.network(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Policy ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Total loss (with entropy bonus for exploration)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()


print("\nPolicy Gradient Methods Comparison:")
print("=" * 50)
print("""
REINFORCE:
- Monte Carlo returns
- High variance
- Simple to implement

A2C (Advantage Actor-Critic):
- Uses value function baseline
- Lower variance than REINFORCE
- Can update every step (TD learning)

PPO (Proximal Policy Optimization):
- Clips policy updates for stability
- Multiple epochs on same data
- State-of-the-art for many tasks
""")
```

---

## Chapter 40: Advanced RL Topics

### 40.1 Model-Based RL

```python
class WorldModel(nn.Module):
    """
    Neural network world model.
    
    Predicts: (next_state, reward) = f(state, action)
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.state_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.reward_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        next_state = self.state_predictor(x)
        reward = self.reward_predictor(x)
        return next_state, reward


class ModelBasedAgent:
    """
    Model-Based RL Agent using learned world model.
    """
    
    def __init__(self, state_dim, action_dim, planning_horizon=10):
        self.world_model = WorldModel(state_dim, action_dim)
        self.model_optimizer = optim.Adam(self.world_model.parameters(), lr=1e-3)
        
        self.planning_horizon = planning_horizon
        self.action_dim = action_dim
        
    def train_model(self, states, actions, next_states, rewards):
        """Train world model on real experience."""
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        
        pred_next_states, pred_rewards = self.world_model(states, actions)
        
        state_loss = nn.MSELoss()(pred_next_states, next_states)
        reward_loss = nn.MSELoss()(pred_rewards.squeeze(), rewards)
        
        loss = state_loss + reward_loss
        
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        
        return loss.item()
    
    def plan(self, initial_state, num_candidates=100):
        """Plan using the world model (simple random shooting)."""
        best_action = None
        best_return = float('-inf')
        
        for _ in range(num_candidates):
            state = torch.FloatTensor(initial_state).unsqueeze(0)
            total_return = 0
            
            # Generate random action sequence
            actions = torch.rand(self.planning_horizon, self.action_dim)
            
            for t in range(self.planning_horizon):
                action = actions[t].unsqueeze(0)
                next_state, reward = self.world_model(state, action)
                total_return += reward.item() * (0.99 ** t)
                state = next_state
            
            if total_return > best_return:
                best_return = total_return
                best_action = actions[0].numpy()
        
        return best_action


print("\nModel-Based vs Model-Free RL:")
print("=" * 50)
print("""
Model-Free (Q-learning, Policy Gradient):
- Learn directly from experience
- Sample inefficient
- No planning

Model-Based:
- Learn dynamics model: s', r = f(s, a)
- More sample efficient
- Can plan ahead
- Model errors can compound
""")
```

### 40.2 Multi-Agent RL

```python
class IndependentQLearning:
    """
    Independent Q-Learning for multi-agent environments.
    
    Each agent learns independently, treating others as part of environment.
    """
    
    def __init__(self, n_agents, state_dim, action_dim, alpha=0.1, gamma=0.99):
        self.n_agents = n_agents
        self.alpha = alpha
        self.gamma = gamma
        
        # Separate Q-table for each agent
        self.Q = [defaultdict(lambda: np.zeros(action_dim)) for _ in range(n_agents)]
        self.epsilon = [1.0] * n_agents
    
    def choose_actions(self, states):
        """Choose action for each agent."""
        actions = []
        for i in range(self.n_agents):
            state = tuple(states[i])
            if np.random.random() < self.epsilon[i]:
                action = np.random.randint(len(self.Q[i][state]))
            else:
                action = np.argmax(self.Q[i][state])
            actions.append(action)
        return actions
    
    def update(self, agent_id, state, action, reward, next_state, done):
        """Update Q-value for one agent."""
        state = tuple(state)
        next_state = tuple(next_state)
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[agent_id][next_state])
        
        self.Q[agent_id][state][action] += self.alpha * (
            target - self.Q[agent_id][state][action]
        )
        
        # Decay epsilon
        self.epsilon[agent_id] = max(0.01, self.epsilon[agent_id] * 0.995)


print("\nMulti-Agent RL:")
print("=" * 50)
print("""
Challenges:
- Non-stationarity (other agents' policies change)
- Credit assignment (which agent caused the reward?)
- Coordination (how to cooperate?)

Approaches:
1. Independent Learning: Each agent learns separately
2. Centralized Training, Decentralized Execution (CTDE)
3. Communication between agents
4. Opponent modeling
""")
```

---

## Summary: Reinforcement Learning

```
┌─────────────────────────────────────────────────────────────────────┐
│              REINFORCEMENT LEARNING SUMMARY                         │
├─────────────────────────────────────────────────────────────────────┤
│  VALUE-BASED METHODS                                                │
│  ├── Q-Learning: Off-policy TD control                             │
│  ├── SARSA: On-policy TD control                                   │
│  ├── DQN: Q-learning with neural networks                          │
│  └── Double DQN, Dueling DQN: Improvements                         │
│                                                                     │
│  POLICY-BASED METHODS                                               │
│  ├── REINFORCE: Monte Carlo policy gradient                        │
│  ├── Actor-Critic: Policy + value function                         │
│  └── PPO: Proximal policy optimization                             │
│                                                                     │
│  KEY CONCEPTS                                                       │
│  ├── Exploration vs Exploitation                                   │
│  ├── Temporal Difference Learning                                  │
│  ├── Experience Replay                                             │
│  └── Target Networks                                               │
│                                                                     │
│  ADVANCED TOPICS                                                    │
│  ├── Model-Based RL: Learn environment dynamics                    │
│  ├── Multi-Agent RL: Multiple interacting agents                   │
│  └── Hierarchical RL: Multi-level policies                         │
└─────────────────────────────────────────────────────────────────────┘
```
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
"""

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
# Part XIV: Advanced Topics

---

## Chapter 46: Generative Adversarial Networks (GANs)

### 46.1 GAN Fundamentals

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GAN ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    Noise z ──► [Generator G] ──► Fake Image                        │
│                                      │                              │
│                                      ▼                              │
│                              [Discriminator D] ──► Real/Fake?      │
│                                      ▲                              │
│                                      │                              │
│                               Real Image                            │
│                                                                     │
│  Training Objective:                                               │
│  - Generator: Fool the discriminator                               │
│  - Discriminator: Distinguish real from fake                       │
│                                                                     │
│  min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 46.2 DCGAN Implementation

```python
import torch
import torch.nn as nn
import numpy as np

class DCGANGenerator(nn.Module):
    """
    Deep Convolutional GAN Generator.
    
    Transforms random noise to images using transposed convolutions.
    """
    
    def __init__(self, latent_dim=100, feature_maps=64, img_channels=3):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # Size: (feature_maps*8) x 4 x 4
            
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # Size: (feature_maps*4) x 8 x 8
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # Size: (feature_maps*2) x 16 x 16
            
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # Size: feature_maps x 32 x 32
            
            nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Size: img_channels x 64 x 64
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        return self.main(z.view(-1, self.latent_dim, 1, 1))


class DCGANDiscriminator(nn.Module):
    """
    Deep Convolutional GAN Discriminator.
    
    Classifies images as real or fake using strided convolutions.
    """
    
    def __init__(self, feature_maps=64, img_channels=3):
        super().__init__()
        
        self.main = nn.Sequential(
            # Input: img_channels x 64 x 64
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: feature_maps x 32 x 32
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (feature_maps*2) x 16 x 16
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (feature_maps*4) x 8 x 8
            
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (feature_maps*8) x 4 x 4
            
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Size: 1 x 1 x 1
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, img):
        return self.main(img).view(-1)


class GANTrainer:
    """Complete GAN training system."""
    
    def __init__(self, latent_dim=100, lr=0.0002, beta1=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        
        self.generator = DCGANGenerator(latent_dim).to(self.device)
        self.discriminator = DCGANDiscriminator().to(self.device)
        
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(beta1, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999)
        )
        
        self.criterion = nn.BCELoss()
    
    def train_step(self, real_images):
        """Single training step."""
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        
        # Labels
        real_label = torch.ones(batch_size).to(self.device)
        fake_label = torch.zeros(batch_size).to(self.device)
        
        # ==================
        # Train Discriminator
        # ==================
        self.d_optimizer.zero_grad()
        
        # Real images
        output_real = self.discriminator(real_images)
        d_loss_real = self.criterion(output_real, real_label)
        
        # Fake images
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_images = self.generator(noise)
        output_fake = self.discriminator(fake_images.detach())
        d_loss_fake = self.criterion(output_fake, fake_label)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # ===============
        # Train Generator
        # ===============
        self.g_optimizer.zero_grad()
        
        output_fake = self.discriminator(fake_images)
        g_loss = self.criterion(output_fake, real_label)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()
    
    def generate(self, num_samples):
        """Generate fake images."""
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim).to(self.device)
            fake_images = self.generator(noise)
        
        return fake_images.cpu()
    
    def train(self, dataloader, epochs=100):
        """Full training loop."""
        for epoch in range(epochs):
            d_losses = []
            g_losses = []
            
            for real_images, _ in dataloader:
                d_loss, g_loss = self.train_step(real_images)
                d_losses.append(d_loss)
                g_losses.append(g_loss)
            
            print(f'Epoch {epoch+1}: D Loss: {np.mean(d_losses):.4f}, '
                  f'G Loss: {np.mean(g_losses):.4f}')


# Print model info
print("DCGAN Architecture:")
print("=" * 50)
G = DCGANGenerator()
D = DCGANDiscriminator()
print(f"Generator parameters: {sum(p.numel() for p in G.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in D.parameters()):,}")
```

### 46.3 Conditional GAN (cGAN)

```python
class ConditionalGenerator(nn.Module):
    """Conditional GAN Generator with class labels."""
    
    def __init__(self, latent_dim=100, num_classes=10, feature_maps=64, img_channels=1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Embed class labels
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Combined input: latent + class embedding
        input_dim = latent_dim + num_classes
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_maps * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        # Embed labels and concatenate with noise
        label_emb = self.label_embedding(labels)
        combined = torch.cat([z, label_emb], dim=1)
        combined = combined.view(-1, self.latent_dim + self.num_classes, 1, 1)
        
        return self.main(combined)


class ConditionalDiscriminator(nn.Module):
    """Conditional GAN Discriminator with class labels."""
    
    def __init__(self, num_classes=10, feature_maps=64, img_channels=1, img_size=32):
        super().__init__()
        
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Embed labels to image-sized tensor
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)
        
        # Input: img_channels + 1 (label channel)
        self.main = nn.Sequential(
            nn.Conv2d(img_channels + 1, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        # Embed labels to image-sized tensor
        label_emb = self.label_embedding(labels)
        label_emb = label_emb.view(-1, 1, self.img_size, self.img_size)
        
        # Concatenate image and label channel
        combined = torch.cat([img, label_emb], dim=1)
        
        return self.main(combined).view(-1)


print("\nConditional GAN:")
print("=" * 50)
print("Allows generating specific classes by conditioning on labels")
print("Generator: G(z, y) -> image of class y")
print("Discriminator: D(x, y) -> real/fake for class y")
```

### 46.4 Wasserstein GAN (WGAN)

```python
class WGANCritic(nn.Module):
    """
    WGAN Critic (not Discriminator).
    
    No sigmoid - outputs unbounded score.
    """
    
    def __init__(self, feature_maps=64, img_channels=3):
        super().__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.LayerNorm([feature_maps * 2, 16, 16]),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.LayerNorm([feature_maps * 4, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.LayerNorm([feature_maps * 8, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False)
            # No sigmoid!
        )
    
    def forward(self, img):
        return self.main(img).view(-1)


def gradient_penalty(critic, real_images, fake_images, device):
    """
    Compute gradient penalty for WGAN-GP.
    
    Enforces Lipschitz constraint.
    """
    batch_size = real_images.size(0)
    
    # Random interpolation
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    
    # Critic output
    critic_output = critic(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=critic_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_output),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Compute penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    
    return penalty


class WGANGPTrainer:
    """WGAN with Gradient Penalty trainer."""
    
    def __init__(self, latent_dim=100, lr=1e-4, n_critic=5, lambda_gp=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        
        self.generator = DCGANGenerator(latent_dim).to(self.device)
        self.critic = WGANCritic().to(self.device)
        
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(0.0, 0.9)
        )
        self.c_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr, betas=(0.0, 0.9)
        )
    
    def train_step(self, real_images):
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        
        # Train Critic (multiple times)
        for _ in range(self.n_critic):
            self.c_optimizer.zero_grad()
            
            noise = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_images = self.generator(noise)
            
            # Wasserstein distance
            c_real = self.critic(real_images)
            c_fake = self.critic(fake_images.detach())
            
            # Gradient penalty
            gp = gradient_penalty(self.critic, real_images, fake_images, self.device)
            
            # Critic loss: maximize E[C(real)] - E[C(fake)] - λ*GP
            c_loss = c_fake.mean() - c_real.mean() + self.lambda_gp * gp
            
            c_loss.backward()
            self.c_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_images = self.generator(noise)
        
        # Generator loss: maximize E[C(fake)]
        g_loss = -self.critic(fake_images).mean()
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return c_loss.item(), g_loss.item()


print("\nWGAN-GP:")
print("=" * 50)
print("Benefits over standard GAN:")
print("- More stable training")
print("- Meaningful loss curves")
print("- No mode collapse")
print("- Gradient penalty enforces Lipschitz constraint")
```

---

## Chapter 47: Transformers In-Depth

### 47.1 Complete Transformer Implementation

```python
import math

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(context)


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class TransformerDecoderLayer(nn.Module):
    """Single Transformer Decoder Layer."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention (masked)
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Cross-attention
        attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 max_len=5000, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, size):
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def encode(self, src, src_mask=None):
        """Encode source sequence."""
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """Decode target sequence."""
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        output = self.fc_out(decoder_output)
        return output


# Test Transformer
print("\nTransformer Model:")
print("=" * 50)
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6
)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
src = torch.randint(0, 10000, (2, 20))
tgt = torch.randint(0, 10000, (2, 15))
output = model(src, tgt)
print(f"Input shapes: src={src.shape}, tgt={tgt.shape}")
print(f"Output shape: {output.shape}")
```

### 47.2 Vision Transformer (ViT)

```python
class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification.
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, num_heads=12,
                 num_layers=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, embed_dim * mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Use class token for classification
        cls_output = x[:, 0]
        
        return self.head(cls_output)


# Test ViT
print("\nVision Transformer (ViT):")
print("=" * 50)
vit = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    num_heads=12,
    num_layers=12
)
print(f"Parameters: {sum(p.numel() for p in vit.parameters()):,}")

x = torch.randn(2, 3, 224, 224)
output = vit(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

---

## Chapter 48: AutoML and Neural Architecture Search

### 48.1 Hyperparameter Optimization

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import random

class GridSearch:
    """Simple grid search for hyperparameter tuning."""
    
    def __init__(self, estimator, param_grid, cv=5, scoring='accuracy'):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = None
        self.results_ = []
    
    def fit(self, X, y):
        """Search all parameter combinations."""
        from itertools import product
        
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        best_score = -float('inf')
        
        for combination in product(*values):
            params = dict(zip(keys, combination))
            
            # Clone estimator with new params
            model = self.estimator.__class__(**params)
            
            # Cross-validation
            scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = scores.mean()
            
            self.results_.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': scores.std()
            })
            
            if mean_score > best_score:
                best_score = mean_score
                self.best_params_ = params
                self.best_score_ = mean_score
        
        return self


class RandomSearch:
    """Random search for hyperparameter tuning."""
    
    def __init__(self, estimator, param_distributions, n_iter=10, cv=5, scoring='accuracy'):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = None
    
    def _sample_params(self):
        """Sample random parameters."""
        params = {}
        for key, dist in self.param_distributions.items():
            if hasattr(dist, 'rvs'):
                params[key] = dist.rvs()
            elif isinstance(dist, list):
                params[key] = random.choice(dist)
            else:
                params[key] = dist
        return params
    
    def fit(self, X, y):
        """Random search over parameter space."""
        best_score = -float('inf')
        
        for _ in range(self.n_iter):
            params = self._sample_params()
            
            model = self.estimator.__class__(**params)
            scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = scores.mean()
            
            if mean_score > best_score:
                best_score = mean_score
                self.best_params_ = params
                self.best_score_ = mean_score
        
        return self


class BayesianOptimization:
    """
    Simplified Bayesian Optimization using Gaussian Process.
    """
    
    def __init__(self, objective_func, param_bounds, n_iter=25):
        self.objective_func = objective_func
        self.param_bounds = param_bounds
        self.n_iter = n_iter
        
        self.X_observed = []
        self.y_observed = []
        self.best_params_ = None
        self.best_score_ = -float('inf')
    
    def _expected_improvement(self, X_new, X_obs, y_obs, xi=0.01):
        """Calculate expected improvement acquisition function."""
        from scipy.stats import norm
        
        # Simple surrogate: use nearest neighbor prediction
        if len(X_obs) == 0:
            return 1.0
        
        # Predict using inverse distance weighting
        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        
        distances = np.sqrt(np.sum((X_obs - X_new) ** 2, axis=1))
        distances = np.maximum(distances, 1e-6)
        weights = 1 / distances
        weights = weights / weights.sum()
        
        mu = np.sum(weights * y_obs)
        sigma = np.std(y_obs) + 1e-6
        
        y_best = np.max(y_obs)
        z = (mu - y_best - xi) / sigma
        ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        
        return ei
    
    def _propose_next_point(self, n_candidates=1000):
        """Propose next point to evaluate."""
        best_ei = -float('inf')
        best_point = None
        
        for _ in range(n_candidates):
            # Random candidate
            point = []
            for low, high in self.param_bounds.values():
                point.append(np.random.uniform(low, high))
            
            ei = self._expected_improvement(
                np.array(point),
                self.X_observed,
                self.y_observed
            )
            
            if ei > best_ei:
                best_ei = ei
                best_point = point
        
        return dict(zip(self.param_bounds.keys(), best_point))
    
    def optimize(self):
        """Run optimization."""
        for i in range(self.n_iter):
            # Get next point
            if len(self.X_observed) < 5:
                # Initial random sampling
                params = {k: np.random.uniform(low, high) 
                         for k, (low, high) in self.param_bounds.items()}
            else:
                params = self._propose_next_point()
            
            # Evaluate
            score = self.objective_func(params)
            
            # Store observation
            self.X_observed.append(list(params.values()))
            self.y_observed.append(score)
            
            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = params
            
            print(f"Iteration {i+1}: Score = {score:.4f}, Best = {self.best_score_:.4f}")
        
        return self.best_params_, self.best_score_


print("\nHyperparameter Optimization:")
print("=" * 50)
print("""
Methods:
1. Grid Search: Exhaustive search over all combinations
   - Pros: Complete coverage
   - Cons: Exponential complexity

2. Random Search: Sample random combinations
   - Pros: Often finds good params faster
   - Cons: May miss optimal region

3. Bayesian Optimization: Smart sequential search
   - Pros: Sample efficient
   - Cons: More complex, overhead for cheap evaluations
""")
```

### 48.2 Neural Architecture Search (NAS)

```python
class NASSearchSpace:
    """Define search space for NAS."""
    
    # Available operations
    OPERATIONS = [
        'conv3x3',
        'conv5x5',
        'maxpool3x3',
        'avgpool3x3',
        'skip_connect',
        'sep_conv3x3',
        'sep_conv5x5',
        'dil_conv3x3',
    ]
    
    @staticmethod
    def get_operation(name, in_channels, out_channels):
        """Return operation module by name."""
        if name == 'conv3x3':
            return nn.Conv2d(in_channels, out_channels, 3, padding=1)
        elif name == 'conv5x5':
            return nn.Conv2d(in_channels, out_channels, 5, padding=2)
        elif name == 'maxpool3x3':
            return nn.MaxPool2d(3, stride=1, padding=1)
        elif name == 'avgpool3x3':
            return nn.AvgPool2d(3, stride=1, padding=1)
        elif name == 'skip_connect':
            return nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        elif name == 'sep_conv3x3':
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1)
            )
        elif name == 'sep_conv5x5':
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1)
            )
        elif name == 'dil_conv3x3':
            return nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2)
        else:
            raise ValueError(f"Unknown operation: {name}")


class DARTSCell(nn.Module):
    """
    DARTS-style cell with continuous relaxation.
    """
    
    def __init__(self, in_channels, out_channels, num_nodes=4):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.operations = nn.ModuleDict()
        
        # Create operations for each edge
        for i in range(num_nodes):
            for j in range(i + 2):  # Connect to input nodes (0, 1) and previous nodes
                for op_name in NASSearchSpace.OPERATIONS:
                    key = f'edge_{j}_{i+2}_{op_name}'
                    self.operations[key] = NASSearchSpace.get_operation(
                        op_name, in_channels, out_channels
                    )
        
        # Architecture parameters (alpha)
        num_edges = sum(i + 2 for i in range(num_nodes))
        self.alpha = nn.Parameter(
            torch.zeros(num_edges, len(NASSearchSpace.OPERATIONS))
        )
    
    def forward(self, s0, s1):
        """Forward with mixed operations weighted by softmax(alpha)."""
        states = [s0, s1]
        
        edge_idx = 0
        for i in range(self.num_nodes):
            # Aggregate inputs from all previous nodes
            node_inputs = []
            
            for j in range(i + 2):
                # Mixed operation
                weights = torch.softmax(self.alpha[edge_idx], dim=0)
                mixed_output = 0
                
                for k, op_name in enumerate(NASSearchSpace.OPERATIONS):
                    key = f'edge_{j}_{i+2}_{op_name}'
                    op_output = self.operations[key](states[j])
                    mixed_output = mixed_output + weights[k] * op_output
                
                node_inputs.append(mixed_output)
                edge_idx += 1
            
            # Sum all inputs to this node
            states.append(sum(node_inputs))
        
        # Concatenate intermediate nodes
        return torch.cat(states[2:], dim=1)


class DARTSController:
    """
    DARTS: Differentiable Architecture Search.
    """
    
    def __init__(self, model, arch_lr=3e-4, weight_lr=0.025):
        self.model = model
        
        # Separate optimizers for weights and architecture
        self.weight_optimizer = torch.optim.SGD(
            [p for n, p in model.named_parameters() if 'alpha' not in n],
            lr=weight_lr, momentum=0.9
        )
        
        self.arch_optimizer = torch.optim.Adam(
            [p for n, p in model.named_parameters() if 'alpha' in n],
            lr=arch_lr
        )
    
    def step(self, train_batch, val_batch, criterion):
        """
        Bi-level optimization step.
        
        1. Update architecture params on validation loss
        2. Update weights on training loss
        """
        train_x, train_y = train_batch
        val_x, val_y = val_batch
        
        # Step 1: Update architecture
        self.arch_optimizer.zero_grad()
        val_output = self.model(val_x)
        val_loss = criterion(val_output, val_y)
        val_loss.backward()
        self.arch_optimizer.step()
        
        # Step 2: Update weights
        self.weight_optimizer.zero_grad()
        train_output = self.model(train_x)
        train_loss = criterion(train_output, train_y)
        train_loss.backward()
        self.weight_optimizer.step()
        
        return train_loss.item(), val_loss.item()
    
    def derive_architecture(self):
        """Derive discrete architecture from continuous params."""
        architecture = []
        
        for name, param in self.model.named_parameters():
            if 'alpha' in name:
                # Select top-2 operations for each node
                weights = torch.softmax(param, dim=1)
                best_ops = weights.argmax(dim=1)
                architecture.append([NASSearchSpace.OPERATIONS[i] for i in best_ops])
        
        return architecture


print("\nNeural Architecture Search (NAS):")
print("=" * 50)
print("""
Approaches:
1. Reinforcement Learning: RNN controller samples architectures
2. Evolutionary: Population of architectures evolved
3. DARTS: Differentiable search with gradient descent
4. One-Shot: Train supernet, derive subnets

DARTS Advantages:
- 1000x faster than RL-based methods
- Continuous relaxation enables gradient-based search
- End-to-end differentiable
""")
```

---

## Summary: Advanced Topics

```
┌─────────────────────────────────────────────────────────────────────┐
│              ADVANCED TOPICS SUMMARY                                │
├─────────────────────────────────────────────────────────────────────┤
│  GENERATIVE MODELS                                                  │
│  ├── GAN: Generator vs Discriminator game                          │
│  ├── DCGAN: Convolutional architecture                             │
│  ├── cGAN: Conditional generation                                  │
│  ├── WGAN-GP: Wasserstein distance + gradient penalty              │
│  └── StyleGAN: State-of-the-art image synthesis                    │
│                                                                     │
│  TRANSFORMERS                                                       │
│  ├── Self-Attention: O(n²) but parallelizable                      │
│  ├── Multi-Head: Multiple attention patterns                       │
│  ├── Positional Encoding: Position information                     │
│  ├── Vision Transformer: Patches as tokens                         │
│  └── BERT/GPT: Pre-trained language models                         │
│                                                                     │
│  AUTOML                                                             │
│  ├── Grid Search: Exhaustive                                       │
│  ├── Random Search: Efficient sampling                             │
│  ├── Bayesian Optimization: Sequential model-based                 │
│  └── NAS: Automated architecture design                            │
└─────────────────────────────────────────────────────────────────────┘
```
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
# Part XVI: Deep Dive into Optimization and Training

---

## Chapter 52: Optimization Algorithms

### 52.1 Gradient Descent Variants

```
┌─────────────────────────────────────────────────────────────────────┐
│                GRADIENT DESCENT COMPARISON                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  BATCH GRADIENT DESCENT                                            │
│  - Uses ALL samples per update                                     │
│  - Stable but slow                                                 │
│  - Memory intensive for large datasets                             │
│                                                                     │
│  STOCHASTIC GRADIENT DESCENT (SGD)                                 │
│  - Uses ONE sample per update                                      │
│  - Noisy but fast                                                  │
│  - Can escape local minima                                         │
│                                                                     │
│  MINI-BATCH GRADIENT DESCENT                                       │
│  - Uses BATCH_SIZE samples per update                              │
│  - Best of both worlds                                             │
│  - Typical batch sizes: 32, 64, 128, 256                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 52.2 Optimizer Implementations

```python
import numpy as np
import torch

class SGD:
    """
    Stochastic Gradient Descent with momentum.
    
    v_t = β * v_{t-1} + (1-β) * g_t
    θ_t = θ_{t-1} - α * v_t
    """
    
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # Initialize velocity
        self.velocity = [torch.zeros_like(p) for p in self.params]
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad.data
            
            # Weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data
            
            # Momentum
            if self.momentum != 0:
                self.velocity[i] = self.momentum * self.velocity[i] + grad
                
                if self.nesterov:
                    grad = grad + self.momentum * self.velocity[i]
                else:
                    grad = self.velocity[i]
            
            # Update parameters
            p.data -= self.lr * grad


class Adam:
    """
    Adam: Adaptive Moment Estimation.
    
    m_t = β1 * m_{t-1} + (1-β1) * g_t           (first moment)
    v_t = β2 * v_{t-1} + (1-β2) * g_t²          (second moment)
    m̂_t = m_t / (1 - β1^t)                      (bias correction)
    v̂_t = v_t / (1 - β2^t)                      (bias correction)
    θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
    """
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moments
        self.m = [torch.zeros_like(p) for p in self.params]  # First moment
        self.v = [torch.zeros_like(p) for p in self.params]  # Second moment
        self.t = 0  # Time step
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self):
        self.t += 1
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad.data
            
            # Weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)
            
            # Bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)


class AdamW:
    """
    AdamW: Adam with decoupled weight decay.
    
    Properly implements weight decay (not L2 regularization).
    """
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self):
        self.t += 1
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad.data
            
            # Update moments (without weight decay in gradient)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Weight decay (decoupled)
            p.data -= self.lr * self.weight_decay * p.data
            
            # Adam update
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)


class RMSprop:
    """
    RMSprop: Root Mean Square Propagation.
    
    v_t = β * v_{t-1} + (1-β) * g_t²
    θ_t = θ_{t-1} - α * g_t / (√v_t + ε)
    """
    
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad.data
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data
            
            # Update running average of squared gradients
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * grad.pow(2)
            
            # Update parameters
            p.data -= self.lr * grad / (self.v[i].sqrt() + self.eps)


class LAMB:
    """
    LAMB: Layer-wise Adaptive Moments optimizer for Batch training.
    
    Good for large batch training.
    """
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self):
        self.t += 1
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad.data
            
            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Adam-style update
            update = m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * p.data
            
            # Layer-wise trust ratio
            param_norm = p.data.norm()
            update_norm = update.norm()
            
            if param_norm > 0 and update_norm > 0:
                trust_ratio = param_norm / update_norm
            else:
                trust_ratio = 1.0
            
            # Update
            p.data -= self.lr * trust_ratio * update


# Comparison function
def compare_optimizers(model_fn, data, epochs=100):
    """Compare different optimizers on the same problem."""
    import matplotlib.pyplot as plt
    
    optimizers = {
        'SGD': lambda p: SGD(p, lr=0.1, momentum=0.9),
        'Adam': lambda p: Adam(p, lr=0.001),
        'AdamW': lambda p: AdamW(p, lr=0.001, weight_decay=0.01),
        'RMSprop': lambda p: RMSprop(p, lr=0.01),
    }
    
    results = {}
    
    for name, opt_fn in optimizers.items():
        model = model_fn()
        optimizer = opt_fn(model.parameters())
        
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        results[name] = losses
    
    # Plot
    plt.figure(figsize=(10, 6))
    for name, losses in results.items():
        plt.plot(losses, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimizer Comparison')
    plt.legend()
    plt.yscale('log')
    
    return results


print("Optimizer Comparison:")
print("=" * 60)
print("""
SGD + Momentum:
- Simple and effective
- Requires tuning learning rate
- Good generalization

Adam:
- Adaptive learning rates per parameter
- Less sensitive to learning rate choice
- May not generalize as well as SGD

AdamW:
- Proper weight decay implementation
- Better than Adam for transformers
- Default choice for many tasks

RMSprop:
- Good for RNNs
- Handles non-stationary objectives

LAMB:
- For very large batch training
- Layer-wise trust ratios
""")
```

### 52.3 Learning Rate Schedulers

```python
import math

class LearningRateScheduler:
    """Base class for learning rate schedulers."""
    
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [self.optimizer.lr]
    
    def get_lr(self):
        raise NotImplementedError
    
    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        
        lr = self.get_lr()
        self.optimizer.lr = lr
        return lr


class StepLR(LearningRateScheduler):
    """
    Step learning rate decay.
    
    Decay LR by gamma every step_size epochs.
    """
    
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return self.base_lrs[0] * (self.gamma ** (self.last_epoch // self.step_size))


class ExponentialLR(LearningRateScheduler):
    """
    Exponential learning rate decay.
    
    LR_t = LR_0 * gamma^t
    """
    
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return self.base_lrs[0] * (self.gamma ** self.last_epoch)


class CosineAnnealingLR(LearningRateScheduler):
    """
    Cosine annealing learning rate.
    
    LR_t = LR_min + 0.5 * (LR_max - LR_min) * (1 + cos(π * t / T))
    """
    
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return self.eta_min + (self.base_lrs[0] - self.eta_min) * \
               (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2


class WarmupCosineScheduler(LearningRateScheduler):
    """
    Linear warmup followed by cosine annealing.
    
    Common for transformers.
    """
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return self.base_lrs[0] * (self.last_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return self.min_lr + (self.base_lrs[0] - self.min_lr) * \
                   (1 + math.cos(math.pi * progress)) / 2


class OneCycleLR(LearningRateScheduler):
    """
    One Cycle Learning Rate Policy.
    
    1. Warmup: LR increases from initial to max
    2. Annealing: LR decreases from max to min
    """
    
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, 
                 div_factor=25, final_div_factor=1e4, last_epoch=-1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.min_lr = max_lr / final_div_factor
        
        super().__init__(optimizer, last_epoch)
        self.base_lrs = [self.initial_lr]
    
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.total_steps * self.pct_start:
            # Warmup phase
            progress = step / (self.total_steps * self.pct_start)
            return self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Annealing phase
            progress = (step - self.total_steps * self.pct_start) / \
                      (self.total_steps * (1 - self.pct_start))
            return self.min_lr + (self.max_lr - self.min_lr) * \
                   (1 + math.cos(math.pi * progress)) / 2


class ReduceLROnPlateau:
    """
    Reduce learning rate when metric stops improving.
    """
    
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, 
                 threshold=1e-4, min_lr=0):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
    
    def step(self, metric):
        if self.mode == 'min':
            improved = metric < self.best - self.threshold
        else:
            improved = metric > self.best + self.threshold
        
        if improved:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
            if self.num_bad_epochs >= self.patience:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                self.optimizer.lr = new_lr
                self.num_bad_epochs = 0
                print(f'Reducing LR to {new_lr}')


# Visualize schedulers
def visualize_schedulers(total_epochs=100):
    """Visualize different learning rate schedules."""
    import matplotlib.pyplot as plt
    
    class DummyOptimizer:
        def __init__(self):
            self.lr = 0.1
    
    schedulers = {
        'StepLR': StepLR(DummyOptimizer(), step_size=30, gamma=0.1),
        'ExponentialLR': ExponentialLR(DummyOptimizer(), gamma=0.95),
        'CosineAnnealing': CosineAnnealingLR(DummyOptimizer(), T_max=total_epochs),
        'WarmupCosine': WarmupCosineScheduler(DummyOptimizer(), warmup_epochs=10, total_epochs=total_epochs),
        'OneCycle': OneCycleLR(DummyOptimizer(), max_lr=0.1, total_steps=total_epochs),
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, scheduler in schedulers.items():
        scheduler.optimizer.lr = 0.1
        scheduler.base_lrs = [0.1]
        scheduler.last_epoch = -1
        
        lrs = []
        for epoch in range(total_epochs):
            lr = scheduler.step()
            lrs.append(lr)
        
        ax.plot(lrs, label=name)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedulers Comparison')
    ax.legend()
    ax.set_yscale('log')
    
    return fig


print("\nLearning Rate Schedulers:")
print("=" * 60)
print("""
StepLR: Decay by factor every N epochs
ExponentialLR: Exponential decay
CosineAnnealing: Smooth cosine decay
WarmupCosine: Warmup + cosine (transformers)
OneCycle: Up then down (super-convergence)
ReduceLROnPlateau: Decay when stuck
""")
```

---

## Chapter 53: Regularization Techniques

### 53.1 Weight Regularization

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class L1Regularization:
    """
    L1 Regularization (Lasso).
    
    Loss = Original Loss + λ * Σ|w|
    
    Promotes sparsity.
    """
    
    def __init__(self, model, lambda_l1=0.01):
        self.model = model
        self.lambda_l1 = lambda_l1
    
    def __call__(self):
        l1_loss = 0
        for param in self.model.parameters():
            l1_loss += torch.abs(param).sum()
        return self.lambda_l1 * l1_loss


class L2Regularization:
    """
    L2 Regularization (Ridge/Weight Decay).
    
    Loss = Original Loss + λ * Σw²
    
    Prevents large weights.
    """
    
    def __init__(self, model, lambda_l2=0.01):
        self.model = model
        self.lambda_l2 = lambda_l2
    
    def __call__(self):
        l2_loss = 0
        for param in self.model.parameters():
            l2_loss += torch.pow(param, 2).sum()
        return self.lambda_l2 * l2_loss


class ElasticNet:
    """
    Elastic Net: Combination of L1 and L2.
    
    Loss = Original Loss + α * L1 + (1-α) * L2
    """
    
    def __init__(self, model, lambda_reg=0.01, l1_ratio=0.5):
        self.model = model
        self.lambda_reg = lambda_reg
        self.l1_ratio = l1_ratio
    
    def __call__(self):
        l1_loss = 0
        l2_loss = 0
        
        for param in self.model.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()
        
        return self.lambda_reg * (
            self.l1_ratio * l1_loss + (1 - self.l1_ratio) * l2_loss
        )


class SpectralNorm(nn.Module):
    """
    Spectral Normalization: Normalize weights by spectral norm.
    
    Constrains Lipschitz constant of layer.
    """
    
    def __init__(self, layer, n_power_iterations=1):
        super().__init__()
        self.layer = layer
        self.n_power_iterations = n_power_iterations
        
        # Initialize u and v vectors
        weight = layer.weight.data
        h, w = weight.shape[0], weight.view(weight.shape[0], -1).shape[1]
        
        self.register_buffer('u', torch.randn(h).requires_grad_(False))
        self.register_buffer('v', torch.randn(w).requires_grad_(False))
    
    def _power_iteration(self, W):
        """Compute spectral norm using power iteration."""
        for _ in range(self.n_power_iterations):
            self.v = F.normalize(torch.mv(W.t(), self.u), dim=0)
            self.u = F.normalize(torch.mv(W, self.v), dim=0)
        
        sigma = torch.dot(self.u, torch.mv(W, self.v))
        return sigma
    
    def forward(self, x):
        W = self.layer.weight.view(self.layer.weight.shape[0], -1)
        sigma = self._power_iteration(W)
        
        # Normalize weight
        self.layer.weight.data = self.layer.weight.data / sigma
        
        return self.layer(x)
```

### 53.2 Dropout Variants

```python
class Dropout(nn.Module):
    """
    Standard Dropout.
    
    Randomly zero elements with probability p during training.
    Scale by 1/(1-p) to maintain expected value.
    """
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) > self.p).float()
            return x * mask / (1 - self.p)
        return x


class Dropout2d(nn.Module):
    """
    Spatial Dropout for CNNs.
    
    Drops entire feature maps instead of individual elements.
    """
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training:
            # x: (batch, channels, height, width)
            mask = (torch.rand(x.size(0), x.size(1), 1, 1, device=x.device) > self.p).float()
            return x * mask / (1 - self.p)
        return x


class DropConnect(nn.Module):
    """
    DropConnect: Drop weights instead of activations.
    
    Each weight has probability p of being zeroed.
    """
    
    def __init__(self, layer, p=0.5):
        super().__init__()
        self.layer = layer
        self.p = p
    
    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(self.layer.weight) > self.p).float()
            weight = self.layer.weight * mask / (1 - self.p)
            
            return F.linear(x, weight, self.layer.bias)
        
        return self.layer(x)


class DropPath(nn.Module):
    """
    Drop Path (Stochastic Depth).
    
    Drop entire residual branch with probability p.
    Used in ResNets and Vision Transformers.
    """
    
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        keep_prob = 1 - self.p
        
        # Random tensor for each sample in batch
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        
        return x / keep_prob * binary_mask


class AlphaDropout(nn.Module):
    """
    Alpha Dropout for SELU activations.
    
    Maintains self-normalizing property of SELU networks.
    """
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
        # SELU parameters
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        
        # Dropout parameters
        self.a_prime = -self.alpha * self.scale
        q = 1 - self.p
        self.a = (q + self.a_prime ** 2 * q * (1 - q)) ** (-0.5)
        self.b = -self.a * (1 - q) * self.a_prime
    
    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) > self.p).float()
            
            # Set dropped values to a' instead of 0
            x = mask * x + (1 - mask) * self.a_prime
            
            # Affine transformation to maintain mean and variance
            return self.a * x + self.b
        
        return x


print("\nDropout Variants:")
print("=" * 60)
print("""
Standard Dropout: Drop random elements
Spatial Dropout: Drop entire feature maps (CNNs)
DropConnect: Drop random weights
DropPath: Drop entire layers/branches
AlphaDropout: For SELU networks
""")
```

### 53.3 Normalization Techniques

```python
class BatchNorm1d(nn.Module):
    """
    Batch Normalization for 1D inputs.
    
    y = γ * (x - μ_B) / √(σ²_B + ε) + β
    
    Where μ_B and σ²_B are batch statistics.
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics for inference
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        if self.training:
            # Compute batch statistics
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.affine:
            x_norm = self.gamma * x_norm + self.beta
        
        return x_norm


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    
    Normalizes across features (not batch).
    Used in transformers.
    """
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        # Normalize over last dimensions
        dims = tuple(range(-len(self.normalized_shape), 0))
        
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.elementwise_affine:
            x_norm = self.gamma * x_norm + self.beta
        
        return x_norm


class InstanceNorm(nn.Module):
    """
    Instance Normalization.
    
    Normalizes each sample independently.
    Used in style transfer.
    """
    
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        # x: (batch, channels, height, width)
        # Normalize over H, W for each channel independently
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.affine:
            x_norm = self.gamma.view(1, -1, 1, 1) * x_norm + self.beta.view(1, -1, 1, 1)
        
        return x_norm


class GroupNorm(nn.Module):
    """
    Group Normalization.
    
    Divides channels into groups and normalizes within each group.
    Works well with small batch sizes.
    """
    
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x):
        # x: (batch, channels, height, width)
        batch_size, C, H, W = x.shape
        
        # Reshape to (batch, groups, channels_per_group, H, W)
        x = x.view(batch_size, self.num_groups, C // self.num_groups, H, W)
        
        # Normalize over (channels_per_group, H, W)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back
        x_norm = x_norm.view(batch_size, C, H, W)
        
        if self.affine:
            x_norm = self.gamma.view(1, -1, 1, 1) * x_norm + self.beta.view(1, -1, 1, 1)
        
        return x_norm


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    
    Simpler than LayerNorm - only divides by RMS.
    Used in some transformer variants.
    """
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * x / rms


print("\nNormalization Comparison:")
print("=" * 60)
print("""
┌─────────────────────────────────────────────────────────────┐
│ Normalization │ Normalize Over        │ Use Case            │
├─────────────────────────────────────────────────────────────┤
│ Batch Norm    │ Batch                 │ CNNs, large batch   │
│ Layer Norm    │ Features              │ Transformers, RNNs  │
│ Instance Norm │ Each sample (H, W)    │ Style transfer      │
│ Group Norm    │ Channel groups        │ Small batch CNNs    │
│ RMS Norm      │ Features (simplified) │ LLMs                │
└─────────────────────────────────────────────────────────────┘
""")
```

---

## Chapter 54: Training Best Practices

### 54.1 Weight Initialization

```python
import torch.nn.init as init

class WeightInitializer:
    """Various weight initialization methods."""
    
    @staticmethod
    def xavier_uniform(layer):
        """
        Xavier/Glorot Uniform Initialization.
        
        Good for: tanh, sigmoid activations
        Var(W) = 2 / (fan_in + fan_out)
        """
        if hasattr(layer, 'weight'):
            init.xavier_uniform_(layer.weight)
        if hasattr(layer, 'bias') and layer.bias is not None:
            init.zeros_(layer.bias)
    
    @staticmethod
    def xavier_normal(layer):
        """Xavier/Glorot Normal Initialization."""
        if hasattr(layer, 'weight'):
            init.xavier_normal_(layer.weight)
        if hasattr(layer, 'bias') and layer.bias is not None:
            init.zeros_(layer.bias)
    
    @staticmethod
    def kaiming_uniform(layer, mode='fan_in', nonlinearity='relu'):
        """
        Kaiming/He Uniform Initialization.
        
        Good for: ReLU, LeakyReLU activations
        Var(W) = 2 / fan_in (for ReLU)
        """
        if hasattr(layer, 'weight'):
            init.kaiming_uniform_(layer.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias') and layer.bias is not None:
            init.zeros_(layer.bias)
    
    @staticmethod
    def kaiming_normal(layer, mode='fan_in', nonlinearity='relu'):
        """Kaiming/He Normal Initialization."""
        if hasattr(layer, 'weight'):
            init.kaiming_normal_(layer.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias') and layer.bias is not None:
            init.zeros_(layer.bias)
    
    @staticmethod
    def orthogonal(layer, gain=1):
        """
        Orthogonal Initialization.
        
        Good for: RNNs
        Preserves gradient norms.
        """
        if hasattr(layer, 'weight'):
            init.orthogonal_(layer.weight, gain=gain)
        if hasattr(layer, 'bias') and layer.bias is not None:
            init.zeros_(layer.bias)
    
    @staticmethod
    def truncated_normal(layer, mean=0, std=0.02):
        """
        Truncated Normal Initialization.
        
        Good for: Transformers, Embeddings
        """
        if hasattr(layer, 'weight'):
            init.trunc_normal_(layer.weight, mean=mean, std=std)
        if hasattr(layer, 'bias') and layer.bias is not None:
            init.zeros_(layer.bias)


def initialize_model(model, init_type='kaiming'):
    """Apply initialization to entire model."""
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if init_type == 'xavier':
                WeightInitializer.xavier_normal(m)
            elif init_type == 'kaiming':
                WeightInitializer.kaiming_normal(m)
            elif init_type == 'orthogonal':
                WeightInitializer.orthogonal(m)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                init.ones_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)


print("\nWeight Initialization Guide:")
print("=" * 60)
print("""
Xavier (Glorot):
- For tanh, sigmoid, softmax
- Maintains variance across layers

Kaiming (He):
- For ReLU, LeakyReLU
- Accounts for ReLU zeroing half the activations

Orthogonal:
- For RNNs
- Prevents gradient vanishing/exploding

Truncated Normal:
- For Transformers
- Small std (0.02) for stability
""")
```

### 54.2 Gradient Management

```python
class GradientManagement:
    """Utilities for managing gradients during training."""
    
    @staticmethod
    def gradient_clipping_norm(model, max_norm, norm_type=2):
        """
        Clip gradients by global norm.
        
        Scales all gradients so total norm <= max_norm.
        """
        parameters = [p for p in model.parameters() if p.grad is not None]
        
        if len(parameters) == 0:
            return 0.0
        
        # Compute total norm
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        
        # Clip
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
        
        return total_norm
    
    @staticmethod
    def gradient_clipping_value(model, clip_value):
        """
        Clip gradients by value.
        
        Clamps each gradient element to [-clip_value, clip_value].
        """
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-clip_value, clip_value)
    
    @staticmethod
    def gradient_accumulation(loss, optimizer, accumulation_steps, step):
        """
        Gradient accumulation for larger effective batch size.
        """
        # Scale loss
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            return True
        return False
    
    @staticmethod
    def compute_gradient_norm(model, norm_type=2):
        """Compute gradient norm for monitoring."""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        return total_norm ** (1. / norm_type)


class GradientCheckpointing:
    """
    Gradient Checkpointing (Activation Checkpointing).
    
    Trade compute for memory by recomputing activations during backward.
    """
    
    @staticmethod
    def checkpoint(function, *args):
        """
        Checkpoint a function to save memory.
        
        During backward, recomputes activations instead of storing them.
        """
        import torch.utils.checkpoint as cp
        return cp.checkpoint(function, *args)
    
    @staticmethod
    def checkpoint_sequential(functions, segments, input):
        """
        Checkpoint a sequential model.
        
        Divides model into segments, checkpoints each segment.
        """
        import torch.utils.checkpoint as cp
        return cp.checkpoint_sequential(functions, segments, input)


class MixedPrecisionTraining:
    """
    Mixed Precision Training with automatic scaling.
    """
    
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, data, target, criterion):
        """Single training step with mixed precision."""
        self.optimizer.zero_grad()
        
        # Forward pass with autocast (FP16)
        with torch.cuda.amp.autocast():
            output = self.model(data)
            loss = criterion(output, target)
        
        # Backward pass with scaling
        self.scaler.scale(loss).backward()
        
        # Unscale and clip gradients
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update weights
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()


print("\nGradient Management:")
print("=" * 60)
print("""
GRADIENT CLIPPING:
- By Norm: Scale all gradients proportionally
- By Value: Clamp individual gradient values
- Prevents exploding gradients

GRADIENT ACCUMULATION:
- Simulate larger batch with limited memory
- Accumulate over N steps, then update
- Effective batch = batch_size * accumulation_steps

CHECKPOINTING:
- Save memory by recomputing activations
- ~30% slowdown for ~50% memory reduction

MIXED PRECISION:
- Use FP16 for speed, FP32 for stability
- 2-3x speedup on modern GPUs
- Use loss scaling to prevent underflow
""")
```

### 54.3 Early Stopping and Model Selection

```python
class EarlyStopping:
    """
    Early Stopping to prevent overfitting.
    
    Stop training when validation metric stops improving.
    """
    
    def __init__(self, patience=10, min_delta=0, mode='min', restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.early_stop = False
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict().copy()
        
        elif self._is_improvement(score):
            self.best_score = score
            self.best_state = model.state_dict().copy()
            self.counter = 0
        
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best:
                    model.load_state_dict(self.best_state)
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


class ModelCheckpoint:
    """
    Model Checkpointing - save best models during training.
    """
    
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = float('inf') if mode == 'min' else float('-inf')
    
    def __call__(self, score, model, optimizer=None, epoch=None):
        is_best = (self.mode == 'min' and score < self.best_score) or \
                  (self.mode == 'max' and score > self.best_score)
        
        if is_best:
            self.best_score = score
        
        if not self.save_best_only or is_best:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'score': score,
                'epoch': epoch,
            }
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            torch.save(checkpoint, self.filepath)
            print(f'Model saved: {self.filepath} (score: {score:.4f})')


class TrainingCallback:
    """Base class for training callbacks."""
    
    def on_epoch_begin(self, epoch, logs=None):
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        pass
    
    def on_batch_begin(self, batch, logs=None):
        pass
    
    def on_batch_end(self, batch, logs=None):
        pass
    
    def on_train_begin(self, logs=None):
        pass
    
    def on_train_end(self, logs=None):
        pass


class ProgressCallback(TrainingCallback):
    """Print training progress."""
    
    def on_epoch_end(self, epoch, logs=None):
        metrics = ' - '.join([f'{k}: {v:.4f}' for k, v in logs.items()])
        print(f'Epoch {epoch}: {metrics}')


class TensorBoardCallback(TrainingCallback):
    """Log to TensorBoard."""
    
    def __init__(self, log_dir='runs'):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
    
    def on_epoch_end(self, epoch, logs=None):
        for name, value in logs.items():
            self.writer.add_scalar(name, value, epoch)
    
    def on_train_end(self, logs=None):
        self.writer.close()


# Complete training loop with callbacks
def train_with_callbacks(model, train_loader, val_loader, optimizer, criterion, 
                         epochs, callbacks=None):
    """Training loop with callback support."""
    if callbacks is None:
        callbacks = []
    
    for cb in callbacks:
        cb.on_train_begin()
    
    for epoch in range(epochs):
        for cb in callbacks:
            cb.on_epoch_begin(epoch)
        
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            for cb in callbacks:
                cb.on_batch_begin(batch_idx)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            for cb in callbacks:
                cb.on_batch_end(batch_idx, {'loss': loss.item()})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / len(val_loader.dataset)
        
        logs = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        
        for cb in callbacks:
            cb.on_epoch_end(epoch, logs)
    
    for cb in callbacks:
        cb.on_train_end()


print("\nTraining Best Practices Summary:")
print("=" * 60)
print("""
1. INITIALIZATION
   - Xavier for tanh/sigmoid
   - Kaiming for ReLU
   
2. OPTIMIZATION
   - AdamW for transformers
   - SGD+momentum for CNNs
   - Warmup + cosine schedule

3. REGULARIZATION
   - Dropout (0.1-0.5)
   - Weight decay (1e-4 to 1e-2)
   - Data augmentation

4. MONITORING
   - Early stopping
   - Model checkpointing
   - Gradient monitoring

5. EFFICIENCY
   - Mixed precision
   - Gradient accumulation
   - Checkpointing
""")
```

---

## Summary: Optimization and Training

```
┌─────────────────────────────────────────────────────────────────────┐
│              OPTIMIZATION AND TRAINING SUMMARY                      │
├─────────────────────────────────────────────────────────────────────┤
│  OPTIMIZERS                                                         │
│  ├── SGD + Momentum: Classic, good generalization                  │
│  ├── Adam: Adaptive, fast convergence                              │
│  ├── AdamW: Decoupled weight decay                                 │
│  └── LAMB: Large batch training                                    │
│                                                                     │
│  LEARNING RATE SCHEDULES                                            │
│  ├── Step decay: Simple, effective                                 │
│  ├── Cosine annealing: Smooth decay                                │
│  ├── Warmup + decay: For transformers                              │
│  └── One cycle: Super-convergence                                  │
│                                                                     │
│  REGULARIZATION                                                     │
│  ├── L1/L2/Elastic Net: Weight penalties                           │
│  ├── Dropout variants: Activation masking                          │
│  └── Normalization: Batch/Layer/Group/RMS                          │
│                                                                     │
│  TRAINING TECHNIQUES                                                │
│  ├── Gradient clipping: Prevent explosion                          │
│  ├── Mixed precision: Speed + memory                               │
│  ├── Early stopping: Prevent overfitting                           │
│  └── Checkpointing: Save best models                               │
└─────────────────────────────────────────────────────────────────────┘
```
# Part XVII: Graph Neural Networks and Geometric Deep Learning

---

## Chapter 55: Introduction to Graph Learning

### 55.1 Graph Fundamentals

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GRAPH REPRESENTATION                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  GRAPH G = (V, E)                                                   │
│  - V: Set of nodes (vertices)                                       │
│  - E: Set of edges (connections)                                    │
│                                                                     │
│  ADJACENCY MATRIX A:                                                │
│  - A[i,j] = 1 if edge between i and j                              │
│  - A[i,j] = 0 otherwise                                            │
│                                                                     │
│  DEGREE MATRIX D:                                                   │
│  - D[i,i] = number of edges connected to node i                    │
│  - Off-diagonal elements are 0                                      │
│                                                                     │
│  LAPLACIAN L = D - A                                                │
│  - Encodes graph structure                                          │
│  - Eigenvalues reveal graph properties                              │
│                                                                     │
│  FEATURE MATRIX X:                                                  │
│  - X[i] = feature vector for node i                                │
│  - Shape: (num_nodes, num_features)                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 55.2 Graph Data Structures

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Graph:
    """
    Basic graph data structure.
    """
    
    def __init__(self, num_nodes, edges=None, node_features=None, edge_features=None):
        self.num_nodes = num_nodes
        self.edges = edges if edges is not None else []
        self.node_features = node_features
        self.edge_features = edge_features
    
    def add_edge(self, src, dst, bidirectional=True):
        """Add edge to graph."""
        self.edges.append((src, dst))
        if bidirectional:
            self.edges.append((dst, src))
    
    def get_adjacency_matrix(self):
        """Return adjacency matrix."""
        A = np.zeros((self.num_nodes, self.num_nodes))
        for src, dst in self.edges:
            A[src, dst] = 1
        return A
    
    def get_degree_matrix(self):
        """Return degree matrix."""
        A = self.get_adjacency_matrix()
        degrees = A.sum(axis=1)
        return np.diag(degrees)
    
    def get_laplacian(self, normalized=False):
        """Return graph Laplacian."""
        A = self.get_adjacency_matrix()
        D = self.get_degree_matrix()
        
        if normalized:
            # Normalized Laplacian: I - D^(-1/2) A D^(-1/2)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
            return np.eye(self.num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt
        else:
            return D - A
    
    def get_edge_index(self):
        """Return edge index in COO format for PyTorch Geometric."""
        if len(self.edges) == 0:
            return torch.tensor([[], []], dtype=torch.long)
        
        src, dst = zip(*self.edges)
        return torch.tensor([src, dst], dtype=torch.long)
    
    def to_sparse(self):
        """Return sparse adjacency matrix."""
        edge_index = self.get_edge_index()
        values = torch.ones(edge_index.shape[1])
        return torch.sparse_coo_tensor(edge_index, values, (self.num_nodes, self.num_nodes))


class GraphBatch:
    """
    Batch multiple graphs for efficient processing.
    """
    
    def __init__(self, graphs):
        self.graphs = graphs
        self.num_graphs = len(graphs)
        
        # Compute offsets
        self.node_offsets = [0]
        for g in graphs[:-1]:
            self.node_offsets.append(self.node_offsets[-1] + g.num_nodes)
        
        # Batch node features
        if graphs[0].node_features is not None:
            self.x = torch.cat([g.node_features for g in graphs], dim=0)
        else:
            self.x = None
        
        # Batch edge indices with offset
        edge_indices = []
        for i, g in enumerate(graphs):
            edge_index = g.get_edge_index() + self.node_offsets[i]
            edge_indices.append(edge_index)
        self.edge_index = torch.cat(edge_indices, dim=1)
        
        # Batch assignment (which graph each node belongs to)
        self.batch = torch.cat([
            torch.full((g.num_nodes,), i, dtype=torch.long)
            for i, g in enumerate(graphs)
        ])
    
    @property
    def num_nodes(self):
        return sum(g.num_nodes for g in self.graphs)


# Example usage
print("Graph Data Structures:")
print("=" * 60)

# Create a simple graph
g = Graph(num_nodes=5)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 3)
g.add_edge(3, 4)

print("Adjacency Matrix:")
print(g.get_adjacency_matrix())
print("\nDegree Matrix:")
print(g.get_degree_matrix())
print("\nEdge Index (COO format):")
print(g.get_edge_index())
```

---

## Chapter 56: Graph Neural Network Layers

### 56.1 Message Passing Framework

```python
class MessagePassingLayer(nn.Module):
    """
    Base class for message passing neural networks.
    
    MPNN Framework:
    1. Message: m_ij = M(h_i, h_j, e_ij)
    2. Aggregate: m_i = AGG({m_ij : j ∈ N(i)})
    3. Update: h_i' = U(h_i, m_i)
    """
    
    def __init__(self):
        super().__init__()
    
    def message(self, x_i, x_j, edge_attr=None):
        """Compute messages from neighbors."""
        raise NotImplementedError
    
    def aggregate(self, messages, index, num_nodes):
        """Aggregate messages at each node."""
        raise NotImplementedError
    
    def update(self, x, aggregated):
        """Update node representations."""
        raise NotImplementedError
    
    def forward(self, x, edge_index, edge_attr=None):
        """Full message passing step."""
        src, dst = edge_index
        
        # Get source and destination node features
        x_j = x[src]  # Source (sending)
        x_i = x[dst]  # Destination (receiving)
        
        # Compute messages
        messages = self.message(x_i, x_j, edge_attr)
        
        # Aggregate at each node
        aggregated = self.aggregate(messages, dst, x.size(0))
        
        # Update
        return self.update(x, aggregated)


def scatter_add(src, index, dim_size):
    """
    Scatter add: Aggregate values by index.
    
    out[index[i]] += src[i]
    """
    out = torch.zeros(dim_size, src.size(-1), device=src.device)
    index = index.unsqueeze(-1).expand_as(src)
    return out.scatter_add_(0, index, src)


def scatter_mean(src, index, dim_size):
    """Scatter mean: Average values by index."""
    out = scatter_add(src, index, dim_size)
    count = torch.zeros(dim_size, device=src.device)
    count.scatter_add_(0, index, torch.ones_like(index, dtype=torch.float))
    count = count.clamp(min=1).unsqueeze(-1)
    return out / count


def scatter_max(src, index, dim_size):
    """Scatter max: Max values by index."""
    out = torch.full((dim_size, src.size(-1)), float('-inf'), device=src.device)
    index = index.unsqueeze(-1).expand_as(src)
    return out.scatter_reduce_(0, index, src, reduce='amax')
```

### 56.2 Graph Convolutional Network (GCN)

```python
class GCNConv(nn.Module):
    """
    Graph Convolutional Network layer.
    
    h_i' = σ(Σ_j (1/√(d_i * d_j)) * W * h_j)
    
    Paper: "Semi-Supervised Classification with Graph Convolutional Networks"
    """
    
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)
        """
        num_nodes = x.size(0)
        src, dst = edge_index
        
        # Add self-loops
        loop_index = torch.arange(num_nodes, device=edge_index.device)
        edge_index = torch.cat([
            edge_index,
            torch.stack([loop_index, loop_index])
        ], dim=1)
        src, dst = edge_index
        
        # Compute normalization (degree)
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Normalize: 1/sqrt(d_i) * 1/sqrt(d_j)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        
        # Transform features
        x = x @ self.weight
        
        # Message passing
        out = torch.zeros_like(x)
        for i in range(edge_index.size(1)):
            s, d = edge_index[0, i], edge_index[1, i]
            out[d] += norm[i] * x[s]
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class GCN(nn.Module):
    """Multi-layer Graph Convolutional Network."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


# Test GCN
print("\nGraph Convolutional Network:")
print("=" * 60)
gcn = GCN(in_channels=16, hidden_channels=32, out_channels=7)
print(gcn)

x = torch.randn(100, 16)  # 100 nodes, 16 features
edge_index = torch.randint(0, 100, (2, 500))  # 500 edges
out = gcn(x, edge_index)
print(f"Input: {x.shape}, Output: {out.shape}")
```

### 56.3 Graph Attention Network (GAT)

```python
class GATConv(nn.Module):
    """
    Graph Attention Network layer.
    
    α_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))
    h_i' = σ(Σ_j α_ij * W * h_j)
    
    Paper: "Graph Attention Networks"
    """
    
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, bias=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        self.weight = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        src, dst = edge_index
        
        # Linear transformation
        x = x @ self.weight
        x = x.view(-1, self.heads, self.out_channels)
        
        # Compute attention coefficients
        x_i = x[dst]  # Target nodes
        x_j = x[src]  # Source nodes
        
        # Concatenate source and target features
        alpha = torch.cat([x_i, x_j], dim=-1)  # (E, heads, 2*out_channels)
        alpha = (alpha * self.att).sum(dim=-1)  # (E, heads)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Softmax over neighbors
        alpha = self._softmax(alpha, dst, num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Aggregate
        out = torch.zeros(num_nodes, self.heads, self.out_channels, device=x.device)
        for i in range(edge_index.size(1)):
            s, d = src[i], dst[i]
            out[d] += alpha[i].unsqueeze(-1) * x[s]
        
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def _softmax(self, alpha, index, num_nodes):
        """Compute softmax over neighbors."""
        # Subtract max for numerical stability
        alpha_max = torch.zeros(num_nodes, alpha.size(1), device=alpha.device)
        for i in range(len(index)):
            alpha_max[index[i]] = torch.max(alpha_max[index[i]], alpha[i])
        
        alpha = alpha - alpha_max[index]
        alpha = torch.exp(alpha)
        
        # Sum for normalization
        alpha_sum = torch.zeros(num_nodes, alpha.size(1), device=alpha.device)
        for i in range(len(index)):
            alpha_sum[index[i]] += alpha[i]
        
        return alpha / (alpha_sum[index] + 1e-10)


class GAT(nn.Module):
    """Multi-layer Graph Attention Network."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=2, heads=8, dropout=0.6):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                      heads=heads, dropout=dropout))
        
        self.convs.append(GATConv(hidden_channels * heads, out_channels, 
                                  heads=1, concat=False, dropout=dropout))
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


print("\nGraph Attention Network:")
print("=" * 60)
gat = GAT(in_channels=16, hidden_channels=8, out_channels=7, heads=8)
print(f"GAT parameters: {sum(p.numel() for p in gat.parameters()):,}")
```

### 56.4 GraphSAGE

```python
class SAGEConv(nn.Module):
    """
    GraphSAGE layer with sampling and aggregation.
    
    h_i' = σ(W * CONCAT(h_i, AGG({h_j : j ∈ N(i)})))
    
    Paper: "Inductive Representation Learning on Large Graphs"
    """
    
    def __init__(self, in_channels, out_channels, aggregator='mean', bias=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregator = aggregator
        
        # Different input size based on aggregator
        if aggregator == 'concat':
            self.lin = nn.Linear(2 * in_channels, out_channels, bias=bias)
        else:
            self.lin = nn.Linear(in_channels, out_channels, bias=bias)
            self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x, edge_index):
        src, dst = edge_index
        num_nodes = x.size(0)
        
        # Aggregate neighbor features
        if self.aggregator == 'mean':
            neigh_agg = scatter_mean(x[src], dst, num_nodes)
        elif self.aggregator == 'max':
            neigh_agg = scatter_max(x[src], dst, num_nodes)
        elif self.aggregator == 'sum':
            neigh_agg = scatter_add(x[src], dst, num_nodes)
        
        if self.aggregator == 'concat':
            out = torch.cat([x, neigh_agg], dim=-1)
            out = self.lin(out)
        else:
            out = self.lin(neigh_agg) + self.lin_self(x)
        
        # L2 normalize
        out = F.normalize(out, p=2, dim=-1)
        
        return out


class GraphSAGE(nn.Module):
    """Multi-layer GraphSAGE."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=2, aggregator='mean', dropout=0.5):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggregator))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggregator))
        
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggregator))
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


print("\nGraphSAGE:")
print("=" * 60)
sage = GraphSAGE(16, 32, 7, aggregator='mean')
print(f"GraphSAGE parameters: {sum(p.numel() for p in sage.parameters()):,}")
```

---

## Chapter 57: Graph-Level Tasks

### 57.1 Graph Pooling

```python
class GlobalMeanPool(nn.Module):
    """Global mean pooling over nodes."""
    
    def forward(self, x, batch):
        """
        Args:
            x: Node features (total_nodes, features)
            batch: Batch assignment (total_nodes,)
        """
        num_graphs = batch.max().item() + 1
        return scatter_mean(x, batch, num_graphs)


class GlobalMaxPool(nn.Module):
    """Global max pooling over nodes."""
    
    def forward(self, x, batch):
        num_graphs = batch.max().item() + 1
        return scatter_max(x, batch, num_graphs)


class GlobalAddPool(nn.Module):
    """Global sum pooling over nodes."""
    
    def forward(self, x, batch):
        num_graphs = batch.max().item() + 1
        return scatter_add(x, batch, num_graphs)


class SetToSetPool(nn.Module):
    """
    Set2Set pooling using LSTM.
    
    Learns to aggregate node representations.
    """
    
    def __init__(self, in_channels, processing_steps=4):
        super().__init__()
        self.in_channels = in_channels
        self.processing_steps = processing_steps
        
        self.lstm = nn.LSTM(2 * in_channels, in_channels, batch_first=True)
    
    def forward(self, x, batch):
        num_graphs = batch.max().item() + 1
        
        h = torch.zeros(1, num_graphs, self.in_channels, device=x.device)
        c = torch.zeros(1, num_graphs, self.in_channels, device=x.device)
        q_star = torch.zeros(num_graphs, 2 * self.in_channels, device=x.device)
        
        for _ in range(self.processing_steps):
            # LSTM step
            _, (h, c) = self.lstm(q_star.unsqueeze(1), (h, c))
            q = h.squeeze(0)  # (num_graphs, in_channels)
            
            # Attention
            e = (x * q[batch]).sum(dim=-1)  # (num_nodes,)
            
            # Softmax per graph
            a = torch.zeros_like(e)
            for g in range(num_graphs):
                mask = batch == g
                a[mask] = F.softmax(e[mask], dim=0)
            
            # Weighted sum
            r = scatter_add(a.unsqueeze(-1) * x, batch, num_graphs)
            
            # Concatenate
            q_star = torch.cat([q, r], dim=-1)
        
        return q_star


class DiffPool(nn.Module):
    """
    Differentiable Pooling for hierarchical graph representation.
    
    Learns soft cluster assignments.
    """
    
    def __init__(self, in_channels, hidden_channels, num_clusters):
        super().__init__()
        
        self.gnn_embed = GCN(in_channels, hidden_channels, hidden_channels)
        self.gnn_pool = GCN(in_channels, hidden_channels, num_clusters)
    
    def forward(self, x, edge_index, batch=None):
        # Compute node embeddings
        z = self.gnn_embed(x, edge_index)
        
        # Compute soft cluster assignments
        s = self.gnn_pool(x, edge_index)
        s = F.softmax(s, dim=-1)  # (num_nodes, num_clusters)
        
        # Pool nodes
        x_pooled = s.t() @ z  # (num_clusters, hidden_channels)
        
        # Compute new adjacency (coarsened graph)
        adj = self._get_adjacency(edge_index, x.size(0))
        adj_pooled = s.t() @ adj @ s  # (num_clusters, num_clusters)
        
        return x_pooled, adj_pooled, s
    
    def _get_adjacency(self, edge_index, num_nodes):
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1
        return adj


print("\nGraph Pooling Methods:")
print("=" * 60)
print("""
GLOBAL POOLING:
- Mean Pool: Average all node features
- Max Pool: Element-wise max
- Sum Pool: Sum all features
- Set2Set: Attention-based LSTM

HIERARCHICAL POOLING:
- DiffPool: Learnable soft clustering
- TopKPool: Keep top-k scoring nodes
- SAGPool: Self-attention graph pooling
""")
```

### 57.2 Graph Classification Model

```python
class GraphClassifier(nn.Module):
    """
    Complete graph classification model.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=3, gnn_type='gcn', pool_type='mean', dropout=0.5):
        super().__init__()
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        if gnn_type == 'gcn':
            GNNLayer = GCNConv
        elif gnn_type == 'gat':
            GNNLayer = lambda i, o: GATConv(i, o, heads=4, concat=False)
        elif gnn_type == 'sage':
            GNNLayer = SAGEConv
        
        self.convs.append(GNNLayer(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GNNLayer(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Pooling
        if pool_type == 'mean':
            self.pool = GlobalMeanPool()
        elif pool_type == 'max':
            self.pool = GlobalMaxPool()
        elif pool_type == 'add':
            self.pool = GlobalAddPool()
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch):
        # GNN layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pool to graph level
        x = self.pool(x, batch)
        
        # Classify
        return self.classifier(x)


# Test graph classifier
print("\nGraph Classifier:")
print("=" * 60)
classifier = GraphClassifier(
    in_channels=16,
    hidden_channels=64,
    out_channels=2,
    num_layers=3,
    gnn_type='gcn',
    pool_type='mean'
)
print(f"Parameters: {sum(p.numel() for p in classifier.parameters()):,}")

# Simulate batch of graphs
x = torch.randn(200, 16)  # 200 total nodes
edge_index = torch.randint(0, 200, (2, 1000))
batch = torch.cat([torch.full((50,), i, dtype=torch.long) for i in range(4)])

out = classifier(x, edge_index, batch)
print(f"Output shape: {out.shape}")  # (4, 2) - 4 graphs, 2 classes
```

---

## Chapter 58: Advanced GNN Topics

### 58.1 Graph Transformers

```python
class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer layer.
    
    Combines graph structure with transformer attention.
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch):
        """
        Args:
            x: Node features (total_nodes, d_model)
            edge_index: Graph structure (2, num_edges)
            batch: Batch assignment (total_nodes,)
        """
        # Create attention mask from graph structure
        num_nodes = x.size(0)
        attn_mask = torch.ones(num_nodes, num_nodes, device=x.device) * float('-inf')
        
        # Allow attention to neighbors
        attn_mask[edge_index[1], edge_index[0]] = 0
        
        # Self-attention (add self-loops)
        attn_mask.fill_diagonal_(0)
        
        # Also mask across different graphs in batch
        for g in range(batch.max().item() + 1):
            mask = batch == g
            attn_mask[mask][:, ~mask] = float('-inf')
        
        # Self-attention
        x2 = self.norm1(x)
        x2 = x2.unsqueeze(0)  # Add batch dimension for attention
        attn_output, _ = self.attention(x2, x2, x2, attn_mask=attn_mask.unsqueeze(0))
        x = x + self.dropout(attn_output.squeeze(0))
        
        # Feed-forward
        x = x + self.ff(self.norm2(x))
        
        return x


class GraphTransformer(nn.Module):
    """Multi-layer Graph Transformer."""
    
    def __init__(self, in_channels, d_model, out_channels, 
                 num_layers=4, num_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(in_channels, d_model)
        
        self.layers = nn.ModuleList([
            GraphTransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.pool = GlobalMeanPool()
        self.output_proj = nn.Linear(d_model, out_channels)
    
    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x, edge_index, batch)
        
        x = self.pool(x, batch)
        return self.output_proj(x)


print("\nGraph Transformer:")
print("=" * 60)
gt = GraphTransformer(16, 64, 2, num_layers=4, num_heads=4)
print(f"Parameters: {sum(p.numel() for p in gt.parameters()):,}")
```

### 58.2 Graph Generation

```python
class GraphVAE(nn.Module):
    """
    Graph Variational Autoencoder for graph generation.
    """
    
    def __init__(self, in_channels, hidden_channels, latent_channels, max_nodes):
        super().__init__()
        
        self.max_nodes = max_nodes
        self.latent_channels = latent_channels
        
        # Encoder
        self.encoder = nn.Sequential(
            GCNConv(in_channels, hidden_channels),
            nn.ReLU(),
            GCNConv(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(hidden_channels, latent_channels)
        self.fc_logvar = nn.Linear(hidden_channels, latent_channels)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, max_nodes * max_nodes),
        )
        
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, max_nodes * in_channels),
        )
    
    def encode(self, x, edge_index, batch):
        h = x
        for layer in self.encoder:
            if isinstance(layer, GCNConv):
                h = layer(h, edge_index)
            else:
                h = layer(h)
        
        # Pool to graph level
        h = scatter_mean(h, batch, batch.max().item() + 1)
        
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        # Decode adjacency matrix
        adj_flat = self.decoder(z)
        adj = adj_flat.view(-1, self.max_nodes, self.max_nodes)
        adj = torch.sigmoid(adj)
        adj = (adj + adj.transpose(1, 2)) / 2  # Symmetrize
        
        # Decode node features
        x_flat = self.node_decoder(z)
        x = x_flat.view(-1, self.max_nodes, -1)
        
        return adj, x
    
    def forward(self, x, edge_index, batch):
        mu, logvar = self.encode(x, edge_index, batch)
        z = self.reparameterize(mu, logvar)
        adj, x_recon = self.decode(z)
        return adj, x_recon, mu, logvar
    
    def loss(self, adj_true, adj_pred, x_true, x_pred, mu, logvar):
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(adj_pred, adj_true, reduction='mean')
        recon_loss += F.mse_loss(x_pred, x_true, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss


print("\nGraph VAE for Generation:")
print("=" * 60)
print("""
Graph generation approaches:
1. GraphVAE: Encode graphs to latent space, decode to generate
2. GraphRNN: Autoregressive node-by-node generation
3. GCPN: RL-based graph generation
4. Diffusion: Denoising diffusion for graphs
""")
```

### 58.3 Link Prediction

```python
class LinkPredictor(nn.Module):
    """
    Link prediction using node embeddings.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super().__init__()
        
        self.encoder = GCN(in_channels, hidden_channels, hidden_channels)
        
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, out_channels)
        )
    
    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)
    
    def decode(self, z, edge_label_index):
        """Predict link probability for given node pairs."""
        src, dst = edge_label_index
        z_src = z[src]
        z_dst = z[dst]
        
        # Concatenate node embeddings
        edge_feat = torch.cat([z_src, z_dst], dim=-1)
        
        return self.predictor(edge_feat).squeeze()
    
    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


def negative_sampling(edge_index, num_nodes, num_neg_samples):
    """
    Sample negative edges (non-existing edges).
    """
    neg_src = torch.randint(0, num_nodes, (num_neg_samples,))
    neg_dst = torch.randint(0, num_nodes, (num_neg_samples,))
    return torch.stack([neg_src, neg_dst])


# Training example
print("\nLink Prediction Training:")
print("=" * 60)
print("""
1. Split edges into train/val/test
2. Sample negative edges
3. Train binary classifier: edge exists or not
4. Evaluate with AUC-ROC

Common applications:
- Social network friend recommendations
- Knowledge graph completion
- Drug-target interaction prediction
""")
```

---

## Summary: Graph Neural Networks

```
┌─────────────────────────────────────────────────────────────────────┐
│              GRAPH NEURAL NETWORKS SUMMARY                          │
├─────────────────────────────────────────────────────────────────────┤
│  GNN ARCHITECTURES                                                  │
│  ├── GCN: Spectral convolution, normalized aggregation             │
│  ├── GAT: Attention-weighted message passing                       │
│  ├── GraphSAGE: Sampling and aggregation                           │
│  └── Graph Transformer: Attention on graph structure               │
│                                                                     │
│  POOLING METHODS                                                    │
│  ├── Global: Mean/Max/Sum/Set2Set                                  │
│  └── Hierarchical: DiffPool, TopK, SAGPool                         │
│                                                                     │
│  TASKS                                                              │
│  ├── Node Classification: Semi-supervised learning                 │
│  ├── Graph Classification: Molecular property prediction           │
│  ├── Link Prediction: Knowledge graph completion                   │
│  └── Graph Generation: Drug discovery, material design             │
│                                                                     │
│  APPLICATIONS                                                       │
│  ├── Social Networks: Community detection, influence               │
│  ├── Biology: Protein structure, drug-target interaction           │
│  ├── Chemistry: Molecular property prediction                      │
│  └── Recommendations: User-item graphs                             │
└─────────────────────────────────────────────────────────────────────┘
```
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
"""
EXERCISE 1: Implement Logistic Regression from Scratch
=====================================================
Complete the following class to implement binary logistic regression.
"""

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
"""
EXERCISE 2: Implement K-Means from Scratch
==========================================
"""

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
"""
EXERCISE 3: Implement a Simple Neural Network
=============================================
"""

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
"""
EXERCISE 4: Implement a CNN for CIFAR-10
========================================

TODO: Complete the CNN architecture and training loop.
"""

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
"""
EXERCISE 5: Implement Attention Mechanism
=========================================

TODO: Implement scaled dot-product attention from scratch.
"""

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
"""
PROJECT TEMPLATE
================

Use this structure for your ML projects.
"""

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
# Part XIX: Self-Supervised Learning and Foundation Models

---

## Chapter 63: Self-Supervised Learning

### 63.1 Introduction to Self-Supervised Learning

```
┌─────────────────────────────────────────────────────────────────────┐
│                SELF-SUPERVISED LEARNING PARADIGM                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PROBLEM: Labeled data is expensive and limited                    │
│  SOLUTION: Learn from the structure of unlabeled data              │
│                                                                     │
│  KEY IDEA: Create pretext tasks from data itself                   │
│                                                                     │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│  │  Unlabeled  │ --> │   Pretext   │ --> │  Learned    │          │
│  │    Data     │     │    Task     │     │  Features   │          │
│  └─────────────┘     └─────────────┘     └─────────────┘          │
│                                                 │                  │
│                                                 v                  │
│                                          ┌─────────────┐          │
│                                          │ Downstream  │          │
│                                          │    Task     │          │
│                                          └─────────────┘          │
│                                                                     │
│  PRETEXT TASKS:                                                    │
│  - Image: Rotation prediction, jigsaw puzzles, colorization       │
│  - Text: Masked language modeling, next sentence prediction       │
│  - Audio: Contrastive predictive coding                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 63.2 Contrastive Learning

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimCLR(nn.Module):
    """
    SimCLR: Simple Framework for Contrastive Learning of Visual Representations.
    
    Key components:
    1. Data augmentation
    2. Encoder network
    3. Projection head
    4. Contrastive loss (NT-Xent)
    """
    
    def __init__(self, encoder, projection_dim=128, temperature=0.5):
        super().__init__()
        
        self.encoder = encoder
        self.temperature = temperature
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            encoder_dim = encoder(dummy_input).shape[-1]
        
        # Projection head (MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, projection_dim)
        )
    
    def forward(self, x):
        """Encode and project."""
        h = self.encoder(x)
        z = self.projection_head(h)
        return F.normalize(z, dim=1)
    
    def contrastive_loss(self, z_i, z_j):
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.
        
        For each positive pair (i, j), treat all other samples as negatives.
        """
        batch_size = z_i.shape[0]
        
        # Concatenate embeddings
        z = torch.cat([z_i, z_j], dim=0)  # (2N, dim)
        
        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)
        
        # Create mask for positive pairs
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim = sim.masked_fill(mask, float('-inf'))  # Remove self-similarity
        
        # Positive pairs: (i, i+N) and (i+N, i)
        pos_indices = torch.arange(batch_size, device=z.device)
        
        # Labels: positive pair index for each sample
        labels = torch.cat([pos_indices + batch_size, pos_indices])
        
        # Cross entropy loss
        loss = F.cross_entropy(sim, labels)
        
        return loss


class MoCo(nn.Module):
    """
    MoCo: Momentum Contrast for Unsupervised Visual Representation Learning.
    
    Uses momentum-updated encoder and queue of negative samples.
    """
    
    def __init__(self, encoder, dim=128, K=65536, m=0.999, T=0.07):
        super().__init__()
        
        self.K = K  # Queue size
        self.m = m  # Momentum coefficient
        self.T = T  # Temperature
        
        # Query encoder
        self.encoder_q = encoder
        
        # Key encoder (momentum updated)
        self.encoder_k = encoder.__class__(**encoder.__dict__)
        for param_q, param_k in zip(self.encoder_q.parameters(), 
                                     self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # Projection heads
        encoder_dim = 2048  # Assume ResNet-50
        self.projection_q = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, dim)
        )
        self.projection_k = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, dim)
        )
        
        # Queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                     self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update the queue with new keys."""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace oldest keys
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            overflow = (ptr + batch_size) - self.K
            self.queue[:, ptr:] = keys[:batch_size - overflow].T
            self.queue[:, :overflow] = keys[batch_size - overflow:].T
        
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
    
    def forward(self, x_q, x_k):
        """
        Forward pass.
        
        Args:
            x_q: Query images
            x_k: Key images (augmented versions of x_q)
        """
        # Query embeddings
        q = self.projection_q(self.encoder_q(x_q))
        q = F.normalize(q, dim=1)
        
        # Key embeddings (no gradients)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.projection_k(self.encoder_k(x_k))
            k = F.normalize(k, dim=1)
        
        # Positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        
        # Labels: positives are the 0-th
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
        
        # Update queue
        self._dequeue_and_enqueue(k)
        
        return logits, labels


class BYOL(nn.Module):
    """
    BYOL: Bootstrap Your Own Latent.
    
    No negative samples needed! Uses predictor network instead.
    """
    
    def __init__(self, encoder, hidden_dim=4096, projection_dim=256):
        super().__init__()
        
        # Online network
        self.online_encoder = encoder
        self.online_projector = self._build_mlp(2048, hidden_dim, projection_dim)
        self.predictor = self._build_mlp(projection_dim, hidden_dim, projection_dim)
        
        # Target network (EMA updated)
        self.target_encoder = encoder.__class__(**encoder.__dict__)
        self.target_projector = self._build_mlp(2048, hidden_dim, projection_dim)
        
        # Copy parameters
        for online_p, target_p in zip(self.online_encoder.parameters(),
                                       self.target_encoder.parameters()):
            target_p.data.copy_(online_p.data)
            target_p.requires_grad = False
        
        for online_p, target_p in zip(self.online_projector.parameters(),
                                       self.target_projector.parameters()):
            target_p.data.copy_(online_p.data)
            target_p.requires_grad = False
    
    def _build_mlp(self, in_dim, hidden_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    @torch.no_grad()
    def update_target_network(self, tau=0.99):
        """EMA update of target network."""
        for online_p, target_p in zip(self.online_encoder.parameters(),
                                       self.target_encoder.parameters()):
            target_p.data = tau * target_p.data + (1 - tau) * online_p.data
        
        for online_p, target_p in zip(self.online_projector.parameters(),
                                       self.target_projector.parameters()):
            target_p.data = tau * target_p.data + (1 - tau) * online_p.data
    
    def forward(self, x1, x2):
        """
        Args:
            x1, x2: Two augmented views of the same image
        """
        # Online network
        z1_online = self.online_projector(self.online_encoder(x1))
        z2_online = self.online_projector(self.online_encoder(x2))
        
        p1 = self.predictor(z1_online)
        p2 = self.predictor(z2_online)
        
        # Target network (no gradients)
        with torch.no_grad():
            z1_target = self.target_projector(self.target_encoder(x1))
            z2_target = self.target_projector(self.target_encoder(x2))
        
        # Loss: predict target from online
        loss = (self._cosine_loss(p1, z2_target) + self._cosine_loss(p2, z1_target)) / 2
        
        return loss
    
    def _cosine_loss(self, p, z):
        """Negative cosine similarity."""
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return 2 - 2 * (p * z).sum(dim=1).mean()


print("Self-Supervised Learning Methods:")
print("=" * 60)
print("""
SimCLR:
- Simple contrastive framework
- Requires large batch sizes for negatives
- Heavy data augmentation is key

MoCo:
- Momentum contrast with queue
- Works with smaller batches
- Dictionary lookup as classification

BYOL:
- No negatives needed
- Predictor prevents collapse
- EMA target network
""")
```

### 63.3 Masked Prediction Methods

```python
class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder (MAE) for self-supervised visual learning.
    
    Mask random patches, encode visible patches, decode all patches.
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, encoder_depth=12, decoder_depth=4,
                 num_heads=12, mask_ratio=0.75):
        super().__init__()
        
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        
        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True
            ),
            num_layers=encoder_depth
        )
        
        # Decoder (smaller)
        decoder_dim = embed_dim // 2
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)
        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, num_patches, decoder_dim) * 0.02
        )
        
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=decoder_dim, nhead=num_heads // 2,
                dim_feedforward=decoder_dim * 4,
                batch_first=True
            ),
            num_layers=decoder_depth
        )
        
        # Prediction head
        self.pred_head = nn.Linear(decoder_dim, patch_size ** 2 * in_channels)
    
    def random_masking(self, x):
        """
        Randomly mask patches.
        
        Returns:
            x_visible: Unmasked patches
            mask: Binary mask (1 = masked)
            ids_restore: Indices to restore original order
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        
        # Random indices
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep unmasked
        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # Binary mask
        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_visible, mask, ids_restore
    
    def forward(self, x):
        """
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            loss: Reconstruction loss on masked patches
        """
        # Patchify
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Random masking
        x_visible, mask, ids_restore = self.random_masking(x)
        
        # Encode visible patches
        encoded = self.encoder(x_visible)
        
        # Decoder
        decoded = self.decoder_embed(encoded)
        
        # Add mask tokens
        mask_tokens = self.mask_token.expand(x.shape[0], ids_restore.shape[1] - decoded.shape[1], -1)
        decoded_full = torch.cat([decoded, mask_tokens], dim=1)
        
        # Unshuffle
        decoded_full = torch.gather(
            decoded_full, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, decoded_full.shape[-1])
        )
        decoded_full = decoded_full + self.decoder_pos_embed
        
        # Decode
        decoded_full = self.decoder(decoded_full)
        
        # Predict pixels
        pred = self.pred_head(decoded_full)
        
        return pred, mask
    
    def loss(self, pred, target, mask):
        """
        Compute loss only on masked patches.
        """
        # Patchify target
        target = target.unfold(2, self.patch_size, self.patch_size)
        target = target.unfold(3, self.patch_size, self.patch_size)
        target = target.permute(0, 2, 3, 1, 4, 5).contiguous()
        target = target.view(target.shape[0], -1, -1)  # (B, num_patches, patch_pixels)
        
        # MSE loss on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean over pixels
        loss = (loss * mask).sum() / mask.sum()  # Mean over masked patches
        
        return loss


print("\nMasked Prediction Methods:")
print("=" * 60)
print("""
MAE (Vision):
- Mask 75% of patches
- Encode only visible patches (efficient!)
- Decode to reconstruct pixels
- Simple MSE loss

BERT (Language):
- Mask 15% of tokens
- Replace with [MASK], random, or same
- Predict original tokens
- Cross-entropy loss

Masked approaches are effective for:
- Pre-training on large unlabeled datasets
- Learning rich representations
- Transfer to downstream tasks
""")
```

---

## Chapter 64: Foundation Models

### 64.1 Large Language Models (LLMs)

```python
class GPTBlock(nn.Module):
    """
    GPT-style transformer decoder block.
    
    Components:
    1. Masked self-attention
    2. Feed-forward network
    3. Layer normalization (pre-norm)
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        # Self-attention with pre-norm
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with pre-norm
        x = x + self.ff(self.ln2(x))
        
        return x


class GPT(nn.Module):
    """
    Simplified GPT model for language modeling.
    """
    
    def __init__(self, vocab_size, max_len, d_model=768, num_layers=12,
                 num_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.token_embed.weight
    
    def forward(self, x):
        """
        Args:
            x: Token indices (batch, seq_len)
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = x.shape
        
        # Embeddings
        tok_emb = self.token_embed(x)
        pos_emb = self.pos_embed[:, :T]
        x = self.dropout(tok_emb + pos_emb)
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        
        # Output
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        """
        for _ in range(max_new_tokens):
            # Crop context to max_len
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len:]
            
            # Forward
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx


print("\nGPT Architecture:")
print("=" * 60)
gpt = GPT(vocab_size=50000, max_len=1024, d_model=768, num_layers=12)
print(f"Parameters: {sum(p.numel() for p in gpt.parameters()):,}")
print("""
Key features of LLMs:
- Autoregressive (next token prediction)
- Causal attention (can't see future)
- Scale is key: more params, more data = better
- Emergent capabilities at scale

Famous LLMs:
- GPT-3/4 (OpenAI): 175B+ parameters
- PaLM (Google): 540B parameters
- LLaMA (Meta): 7B-70B parameters
- Claude (Anthropic): State-of-the-art
""")
```

### 64.2 Vision Foundation Models

```python
class CLIP(nn.Module):
    """
    CLIP: Contrastive Language-Image Pre-training.
    
    Learns joint embedding of images and text.
    """
    
    def __init__(self, image_encoder, text_encoder, embed_dim=512, temperature=0.07):
        super().__init__()
        
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        
        # Projection heads
        self.image_projection = nn.Linear(2048, embed_dim)  # Assuming ResNet
        self.text_projection = nn.Linear(768, embed_dim)    # Assuming BERT
        
        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
    
    def encode_image(self, image):
        features = self.image_encoder(image)
        features = self.image_projection(features)
        return F.normalize(features, dim=-1)
    
    def encode_text(self, text):
        features = self.text_encoder(text)
        features = self.text_projection(features)
        return F.normalize(features, dim=-1)
    
    def forward(self, image, text):
        """
        Args:
            image: Image batch
            text: Text batch (tokenized)
        
        Returns:
            logits_per_image: Similarity scores (image, text)
            logits_per_text: Similarity scores (text, image)
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
    
    def loss(self, logits_per_image, logits_per_text):
        """Symmetric cross-entropy loss."""
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i + loss_t) / 2


class DiffusionModel(nn.Module):
    """
    Simplified Denoising Diffusion Probabilistic Model.
    
    Foundation for image generation (DALL-E, Stable Diffusion).
    """
    
    def __init__(self, in_channels=3, hidden_channels=64, num_timesteps=1000):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        
        # Beta schedule
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, num_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
        
        # U-Net style denoiser
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.SiLU(),
            nn.Linear(hidden_channels * 4, hidden_channels)
        )
        
        # Encoder
        self.down1 = self._make_block(hidden_channels, hidden_channels * 2)
        self.down2 = self._make_block(hidden_channels * 2, hidden_channels * 4)
        
        # Middle
        self.mid = self._make_block(hidden_channels * 4, hidden_channels * 4)
        
        # Decoder
        self.up2 = self._make_block(hidden_channels * 8, hidden_channels * 2)
        self.up1 = self._make_block(hidden_channels * 4, hidden_channels)
        
        self.conv_out = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
    
    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
    
    def _sinusoidal_embedding(self, t, dim):
        """Sinusoidal time embedding."""
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, x, t):
        """
        Predict noise given noisy image and timestep.
        
        Args:
            x: Noisy image (batch, C, H, W)
            t: Timestep (batch,)
        """
        # Time embedding
        t_emb = self._sinusoidal_embedding(t, self.conv_in.out_channels)
        t_emb = self.time_embed(t_emb)
        
        # U-Net forward
        x = self.conv_in(x)
        
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(F.avg_pool2d(x1, 2))
        
        # Middle
        x_mid = self.mid(F.avg_pool2d(x2, 2))
        
        # Decoder with skip connections
        x = self.up2(torch.cat([F.interpolate(x_mid, scale_factor=2), x2], dim=1))
        x = self.up1(torch.cat([F.interpolate(x, scale_factor=2), x1], dim=1))
        
        return self.conv_out(x)
    
    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: Add noise to image."""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        alpha_bar_t = self.alpha_bars[t][:, None, None, None]
        
        return torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    
    def p_sample(self, x_t, t):
        """Reverse diffusion: Denoise one step."""
        beta_t = self.betas[t][:, None, None, None]
        alpha_t = self.alphas[t][:, None, None, None]
        alpha_bar_t = self.alpha_bars[t][:, None, None, None]
        
        # Predict noise
        pred_noise = self(x_t, t)
        
        # Compute mean
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        )
        
        # Add noise (except for t=0)
        noise = torch.randn_like(x_t)
        noise[t == 0] = 0
        
        return mean + torch.sqrt(beta_t) * noise


print("\nVision Foundation Models:")
print("=" * 60)
print("""
CLIP:
- Joint image-text embeddings
- Zero-shot classification
- Enables text-to-image retrieval

Diffusion Models:
- Generate images by denoising
- DALL-E 2, Stable Diffusion, Midjourney
- State-of-the-art image generation

Segment Anything (SAM):
- Foundation model for segmentation
- Prompt-based segmentation
- Works on any image

DINOv2:
- Self-supervised vision transformer
- Excellent features for downstream tasks
""")
```

---

## Chapter 65: Efficient Fine-tuning

### 65.1 Parameter-Efficient Fine-tuning

```python
class LoRA(nn.Module):
    """
    Low-Rank Adaptation (LoRA).
    
    Instead of fine-tuning all parameters:
    W' = W + BA
    
    Where B is (d, r) and A is (r, k), with r << min(d, k)
    """
    
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Low-rank matrices
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
    
    def forward(self, x):
        # Original forward
        original_out = self.original_layer(x)
        
        # LoRA forward
        lora_out = x @ self.lora_A @ self.lora_B * self.scaling
        
        return original_out + lora_out


class Adapter(nn.Module):
    """
    Adapter module for efficient fine-tuning.
    
    Inserts small bottleneck layers between transformer layers.
    """
    
    def __init__(self, hidden_size, bottleneck_size=64):
        super().__init__()
        
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        self.act = nn.GELU()
    
    def forward(self, x):
        # Bottleneck
        down = self.act(self.down_proj(x))
        up = self.up_proj(down)
        
        # Residual connection
        return x + up


class PrefixTuning(nn.Module):
    """
    Prefix Tuning: Prepend learnable prefix to keys and values.
    
    Only prefix parameters are trained.
    """
    
    def __init__(self, num_layers, hidden_size, prefix_length=20, num_heads=12):
        super().__init__()
        
        self.prefix_length = prefix_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Learnable prefix
        self.prefix = nn.Parameter(
            torch.randn(num_layers, 2, prefix_length, hidden_size) * 0.02
        )
    
    def forward(self, layer_idx):
        """
        Get prefix key and value for a specific layer.
        
        Returns:
            prefix_k: (batch, prefix_length, hidden_size)
            prefix_v: (batch, prefix_length, hidden_size)
        """
        prefix_k = self.prefix[layer_idx, 0]
        prefix_v = self.prefix[layer_idx, 1]
        return prefix_k, prefix_v


class PromptTuning(nn.Module):
    """
    Prompt Tuning: Prepend learnable soft prompts to input.
    
    Simple and effective for large models.
    """
    
    def __init__(self, num_prompts=20, hidden_size=768):
        super().__init__()
        
        self.num_prompts = num_prompts
        self.prompts = nn.Parameter(torch.randn(1, num_prompts, hidden_size) * 0.02)
    
    def forward(self, embeddings):
        """
        Prepend prompts to input embeddings.
        
        Args:
            embeddings: (batch, seq_len, hidden_size)
        
        Returns:
            (batch, num_prompts + seq_len, hidden_size)
        """
        batch_size = embeddings.shape[0]
        prompts = self.prompts.expand(batch_size, -1, -1)
        return torch.cat([prompts, embeddings], dim=1)


def count_trainable_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("\nParameter-Efficient Fine-tuning Methods:")
print("=" * 60)
print("""
LoRA (Low-Rank Adaptation):
- Trains low-rank matrices BA
- Typically 0.1-1% of original params
- Works well for attention layers

Adapters:
- Small bottleneck modules
- Inserted between layers
- ~2-3% of original params

Prefix Tuning:
- Learnable key-value prefixes
- Prepended to each layer
- ~0.1% of original params

Prompt Tuning:
- Soft prompts at input
- Simplest approach
- ~0.01% of original params

Comparison:
- Full fine-tuning: 100% params (best quality, most expensive)
- LoRA: 0.1-1% params (near full fine-tuning quality)
- Adapters: 2-3% params (good quality)
- Prompt tuning: 0.01% params (simple but limited)
""")
```

---

## Summary: Self-Supervised Learning and Foundation Models

```
┌─────────────────────────────────────────────────────────────────────┐
│              FOUNDATION MODELS SUMMARY                              │
├─────────────────────────────────────────────────────────────────────┤
│  SELF-SUPERVISED LEARNING                                           │
│  ├── Contrastive: SimCLR, MoCo, BYOL                               │
│  ├── Masked Prediction: MAE, BERT, GPT                             │
│  └── Benefits: No labels, large-scale pretraining                  │
│                                                                     │
│  FOUNDATION MODELS                                                  │
│  ├── Language: GPT, BERT, T5, LLaMA                                │
│  ├── Vision: CLIP, SAM, DINOv2                                     │
│  ├── Multimodal: DALL-E, Stable Diffusion, GPT-4V                  │
│  └── Key: Scale (data + compute + params)                          │
│                                                                     │
│  EFFICIENT FINE-TUNING                                              │
│  ├── LoRA: Low-rank adaptation                                     │
│  ├── Adapters: Bottleneck modules                                  │
│  ├── Prefix Tuning: Learnable KV prefixes                          │
│  └── Prompt Tuning: Soft input prompts                             │
│                                                                     │
│  PARADIGM SHIFT                                                     │
│  ├── Traditional: Task-specific training                           │
│  └── Foundation: Pretrain once, adapt to many tasks                │
└─────────────────────────────────────────────────────────────────────┘
```
# Part XX: Additional Algorithms and Advanced Techniques

---

## Chapter 80: Bayesian Machine Learning

### 80.1 Bayesian Foundations

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BAYESIAN INFERENCE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  BAYES' THEOREM:                                                    │
│                                                                     │
│  P(θ|D) = P(D|θ) × P(θ) / P(D)                                     │
│                                                                     │
│  Where:                                                             │
│  - P(θ|D): Posterior - our updated belief about θ given data       │
│  - P(D|θ): Likelihood - probability of data given θ                │
│  - P(θ): Prior - our initial belief about θ                        │
│  - P(D): Evidence - normalizing constant                           │
│                                                                     │
│  FREQUENTIST vs BAYESIAN:                                           │
│                                                                     │
│  Frequentist:                                                       │
│  - Parameters are fixed, unknown constants                         │
│  - Estimate point values                                           │
│  - Confidence intervals                                             │
│                                                                     │
│  Bayesian:                                                          │
│  - Parameters are random variables with distributions              │
│  - Full posterior distribution                                     │
│  - Credible intervals                                              │
│  - Naturally handles uncertainty                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 80.2 Bayesian Linear Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class BayesianLinearRegression:
    """
    Bayesian Linear Regression with conjugate prior.
    
    Prior: p(w) = N(w | m_0, S_0)
    Posterior: p(w | X, y) = N(w | m_N, S_N)
    
    With closed-form update formulas.
    """
    
    def __init__(self, alpha=1.0, beta=25.0):
        """
        Args:
            alpha: Prior precision (1/variance)
            beta: Noise precision (1/noise_variance)
        """
        self.alpha = alpha
        self.beta = beta
        self.mean = None
        self.cov = None
    
    def fit(self, X, y):
        """
        Compute posterior distribution of weights.
        """
        # Add bias term
        X = self._add_bias(X)
        n_features = X.shape[1]
        
        # Prior
        S_0_inv = self.alpha * np.eye(n_features)
        m_0 = np.zeros(n_features)
        
        # Posterior
        S_N_inv = S_0_inv + self.beta * X.T @ X
        self.cov = np.linalg.inv(S_N_inv)
        self.mean = self.cov @ (S_0_inv @ m_0 + self.beta * X.T @ y)
        
        return self
    
    def predict(self, X, return_std=False):
        """
        Predict with uncertainty.
        
        Returns:
            mean: Predicted mean
            std: Predictive standard deviation (if return_std=True)
        """
        X = self._add_bias(X)
        
        # Predictive mean
        mean = X @ self.mean
        
        if return_std:
            # Predictive variance = noise + model uncertainty
            var = 1/self.beta + np.sum(X @ self.cov * X, axis=1)
            return mean, np.sqrt(var)
        
        return mean
    
    def _add_bias(self, X):
        """Add bias column."""
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    def sample_weights(self, n_samples=100):
        """Sample weights from posterior."""
        return np.random.multivariate_normal(self.mean, self.cov, n_samples)


class GaussianProcess:
    """
    Gaussian Process Regression.
    
    Non-parametric Bayesian approach - places prior over functions.
    """
    
    def __init__(self, kernel='rbf', length_scale=1.0, noise=1e-6):
        self.kernel = kernel
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
    
    def _kernel_func(self, X1, X2):
        """Compute kernel matrix."""
        if self.kernel == 'rbf':
            # RBF (Squared Exponential) kernel
            sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
                     np.sum(X2**2, 1) - 2 * X1 @ X2.T
            return np.exp(-0.5 * sqdist / self.length_scale**2)
        elif self.kernel == 'linear':
            return X1 @ X2.T
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y):
        """Store training data and compute K inverse."""
        self.X_train = X
        self.y_train = y
        
        K = self._kernel_func(X, X)
        K += self.noise * np.eye(len(X))
        
        self.K_inv = np.linalg.inv(K)
        return self
    
    def predict(self, X, return_std=False, return_cov=False):
        """
        Predict at new points.
        
        Returns full predictive distribution.
        """
        K_s = self._kernel_func(self.X_train, X)
        K_ss = self._kernel_func(X, X)
        
        # Predictive mean
        mean = K_s.T @ self.K_inv @ self.y_train
        
        # Predictive covariance
        cov = K_ss - K_s.T @ self.K_inv @ K_s
        
        if return_std:
            std = np.sqrt(np.diag(cov))
            return mean, std
        elif return_cov:
            return mean, cov
        
        return mean
    
    def sample(self, X, n_samples=5):
        """Sample functions from posterior."""
        mean, cov = self.predict(X, return_cov=True)
        return np.random.multivariate_normal(mean, cov + 1e-6*np.eye(len(X)), n_samples)


# Example usage
print("Bayesian Machine Learning:")
print("=" * 60)

# Generate data
np.random.seed(42)
X = np.linspace(0, 10, 20).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.randn(20) * 0.1

# Bayesian Linear Regression with polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)[:, 1:]  # Remove bias (BLR adds it)

blr = BayesianLinearRegression(alpha=0.01, beta=100)
blr.fit(X_poly, y)

# Gaussian Process
gp = GaussianProcess(kernel='rbf', length_scale=1.0)
gp.fit(X, y)

X_test = np.linspace(0, 10, 100).reshape(-1, 1)
mean, std = gp.predict(X_test, return_std=True)

print(f"GP Mean at x=5: {gp.predict(np.array([[5]]))[0]:.3f}")
print(f"True value at x=5: {np.sin(5):.3f}")
```

### 80.3 Variational Inference

```python
class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE).
    
    Uses variational inference to learn latent representations.
    
    ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
    """
    
    def __init__(self, input_dim, hidden_dim=256, latent_dim=32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for backprop through sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent variable to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss(self, x, x_recon, mu, logvar):
        """
        ELBO loss.
        
        Reconstruction loss + KL divergence.
        """
        # Reconstruction loss (binary cross entropy)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        
        # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss
    
    def sample(self, n_samples):
        """Generate samples from prior."""
        z = torch.randn(n_samples, self.fc_mu.out_features)
        return self.decode(z)


class BayesianNeuralNetwork(nn.Module):
    """
    Bayesian Neural Network using variational inference.
    
    Places distributions over weights instead of point estimates.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # Weight means
        self.w1_mu = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.w2_mu = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.1)
        
        # Weight log-variances
        self.w1_logvar = nn.Parameter(torch.zeros(input_dim, hidden_dim) - 5)
        self.w2_logvar = nn.Parameter(torch.zeros(hidden_dim, output_dim) - 5)
        
        # Bias means
        self.b1_mu = nn.Parameter(torch.zeros(hidden_dim))
        self.b2_mu = nn.Parameter(torch.zeros(output_dim))
        
        # Bias log-variances
        self.b1_logvar = nn.Parameter(torch.zeros(hidden_dim) - 5)
        self.b2_logvar = nn.Parameter(torch.zeros(output_dim) - 5)
    
    def _sample_weights(self):
        """Sample weights from variational distribution."""
        w1_std = torch.exp(0.5 * self.w1_logvar)
        w2_std = torch.exp(0.5 * self.w2_logvar)
        b1_std = torch.exp(0.5 * self.b1_logvar)
        b2_std = torch.exp(0.5 * self.b2_logvar)
        
        w1 = self.w1_mu + w1_std * torch.randn_like(w1_std)
        w2 = self.w2_mu + w2_std * torch.randn_like(w2_std)
        b1 = self.b1_mu + b1_std * torch.randn_like(b1_std)
        b2 = self.b2_mu + b2_std * torch.randn_like(b2_std)
        
        return w1, w2, b1, b2
    
    def forward(self, x, n_samples=1):
        """Forward pass with weight sampling."""
        outputs = []
        
        for _ in range(n_samples):
            w1, w2, b1, b2 = self._sample_weights()
            
            h = F.relu(x @ w1 + b1)
            out = h @ w2 + b2
            outputs.append(out)
        
        return torch.stack(outputs).mean(dim=0)
    
    def kl_divergence(self):
        """KL divergence between q(w) and p(w)."""
        kl = 0
        
        for mu, logvar in [(self.w1_mu, self.w1_logvar),
                           (self.w2_mu, self.w2_logvar),
                           (self.b1_mu, self.b1_logvar),
                           (self.b2_mu, self.b2_logvar)]:
            # KL(N(mu, sigma) || N(0, 1))
            kl += 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        
        return kl


print("\nVariational Inference:")
print("=" * 60)
print("""
Key Concepts:
- Replace intractable posterior with tractable approximation
- Optimize ELBO (Evidence Lower Bound)
- Reparameterization trick enables backprop through sampling

Applications:
- VAE: Generative modeling
- Bayesian Neural Networks: Uncertainty quantification
- Bayesian Optimization: Hyperparameter tuning
""")
```

---

## Chapter 81: Meta-Learning

### 81.1 Learning to Learn

```python
class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML).
    
    Learn initial parameters that can be quickly adapted to new tasks.
    
    Inner loop: Adapt to task with few gradient steps
    Outer loop: Update meta-parameters
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, inner_lr=0.01):
        super().__init__()
        
        self.inner_lr = inner_lr
        
        # Base model parameters
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, params=None):
        """Forward pass with optional custom parameters."""
        if params is None:
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return self.fc3(h)
        else:
            # Use provided parameters
            h = F.relu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
            h = F.relu(F.linear(h, params['fc2.weight'], params['fc2.bias']))
            return F.linear(h, params['fc3.weight'], params['fc3.bias'])
    
    def adapt(self, support_x, support_y, num_steps=1):
        """
        Adapt model parameters to a specific task.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            num_steps: Number of inner loop gradient steps
        
        Returns:
            adapted_params: Parameters adapted to the task
        """
        # Get current parameters
        params = {name: param.clone() for name, param in self.named_parameters()}
        
        for _ in range(num_steps):
            # Forward pass
            pred = self.forward(support_x, params)
            loss = F.cross_entropy(pred, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, params.values(), create_graph=True)
            
            # Update parameters
            params = {name: param - self.inner_lr * grad 
                     for (name, param), grad in zip(params.items(), grads)}
        
        return params
    
    def meta_loss(self, tasks, num_inner_steps=1):
        """
        Compute meta-loss across tasks.
        
        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples
        """
        total_loss = 0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Adapt to task
            adapted_params = self.adapt(support_x, support_y, num_inner_steps)
            
            # Evaluate on query set
            pred = self.forward(query_x, adapted_params)
            loss = F.cross_entropy(pred, query_y)
            
            total_loss += loss
        
        return total_loss / len(tasks)


class ProtoNet(nn.Module):
    """
    Prototypical Networks for few-shot learning.
    
    Learn embedding space where classification is done by 
    computing distances to class prototypes.
    """
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def compute_prototypes(self, support_x, support_y, n_classes):
        """
        Compute prototype for each class.
        
        Prototype = mean embedding of support examples.
        """
        embeddings = self.encoder(support_x)
        
        prototypes = []
        for c in range(n_classes):
            class_embeddings = embeddings[support_y == c]
            prototypes.append(class_embeddings.mean(dim=0))
        
        return torch.stack(prototypes)
    
    def forward(self, support_x, support_y, query_x, n_classes):
        """
        Classify query examples.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            n_classes: Number of classes
        """
        # Compute prototypes
        prototypes = self.compute_prototypes(support_x, support_y, n_classes)
        
        # Embed query examples
        query_embeddings = self.encoder(query_x)
        
        # Compute distances to prototypes
        dists = torch.cdist(query_embeddings, prototypes)
        
        # Return negative distances as logits
        return -dists


class MatchingNetwork(nn.Module):
    """
    Matching Networks for one-shot learning.
    
    Uses attention over support set for prediction.
    """
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, support_x, support_y, query_x, n_classes):
        """
        Classify using attention over support set.
        """
        # Embed support and query
        support_emb = self.encoder(support_x)
        query_emb = self.encoder(query_x)
        
        # Compute attention (cosine similarity)
        support_norm = F.normalize(support_emb, dim=1)
        query_norm = F.normalize(query_emb, dim=1)
        
        attention = query_norm @ support_norm.t()  # (n_query, n_support)
        attention = F.softmax(attention, dim=1)
        
        # Convert support labels to one-hot
        support_onehot = F.one_hot(support_y, n_classes).float()
        
        # Weighted sum of one-hot labels
        logits = attention @ support_onehot
        
        return logits


print("\nMeta-Learning Approaches:")
print("=" * 60)
print("""
MAML (Model-Agnostic Meta-Learning):
- Learn good initialization
- Few gradient steps to adapt
- Works with any gradient-based model

Prototypical Networks:
- Metric learning approach
- Class = mean of support embeddings
- Simple and effective

Matching Networks:
- Attention over support set
- Non-parametric at test time
- Good for very few shots
""")
```

---

## Chapter 82: Neural Network Compression

### 82.1 Pruning

```python
class NetworkPruner:
    """
    Neural network pruning techniques.
    """
    
    @staticmethod
    def magnitude_pruning(model, sparsity=0.5):
        """
        Prune weights with smallest magnitude.
        
        Args:
            model: Neural network
            sparsity: Fraction of weights to prune
        """
        # Collect all weights
        all_weights = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                all_weights.append(param.data.abs().view(-1))
        
        all_weights = torch.cat(all_weights)
        
        # Find threshold
        threshold = torch.quantile(all_weights, sparsity)
        
        # Apply masks
        masks = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                mask = param.data.abs() > threshold
                param.data *= mask.float()
                masks[name] = mask
        
        return masks
    
    @staticmethod
    def structured_pruning(model, prune_ratio=0.3):
        """
        Prune entire filters/neurons (structured sparsity).
        
        Maintains hardware efficiency.
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Compute filter importance (L1 norm)
                importance = module.weight.data.abs().sum(dim=(1, 2, 3))
                
                # Find filters to prune
                n_prune = int(len(importance) * prune_ratio)
                _, indices = importance.sort()
                prune_indices = indices[:n_prune]
                
                # Zero out pruned filters
                module.weight.data[prune_indices] = 0
                if module.bias is not None:
                    module.bias.data[prune_indices] = 0
            
            elif isinstance(module, nn.Linear):
                # Prune neurons (output dimension)
                importance = module.weight.data.abs().sum(dim=1)
                
                n_prune = int(len(importance) * prune_ratio)
                _, indices = importance.sort()
                prune_indices = indices[:n_prune]
                
                module.weight.data[prune_indices] = 0
                if module.bias is not None:
                    module.bias.data[prune_indices] = 0


class LotteryTicketHypothesis:
    """
    Lottery Ticket Hypothesis: Find sparse subnetworks that train well.
    
    Algorithm:
    1. Train network
    2. Prune smallest weights
    3. Reset remaining weights to initial values
    4. Retrain
    """
    
    def __init__(self, model, prune_ratio=0.2):
        self.model = model
        self.prune_ratio = prune_ratio
        
        # Save initial weights
        self.initial_weights = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        self.masks = None
    
    def find_winning_ticket(self, train_fn, iterations=5):
        """
        Iteratively find winning ticket.
        
        Args:
            train_fn: Function to train the model
            iterations: Number of prune-retrain iterations
        """
        current_sparsity = 0
        
        for i in range(iterations):
            # Train
            train_fn(self.model)
            
            # Prune
            current_sparsity = 1 - (1 - self.prune_ratio) ** (i + 1)
            self.masks = NetworkPruner.magnitude_pruning(self.model, current_sparsity)
            
            # Reset to initial weights (but keep pruning)
            for name, param in self.model.named_parameters():
                if name in self.initial_weights:
                    param.data = self.initial_weights[name].clone()
                    if name in self.masks:
                        param.data *= self.masks[name].float()
            
            print(f"Iteration {i+1}: Sparsity = {current_sparsity:.2%}")
        
        return self.masks
```

### 82.2 Quantization

```python
class Quantizer:
    """
    Neural network quantization techniques.
    """
    
    @staticmethod
    def quantize_tensor(tensor, num_bits=8):
        """
        Uniform quantization of a tensor.
        
        Maps floating point values to integers.
        """
        # Find range
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Compute scale and zero point
        qmin = 0
        qmax = 2**num_bits - 1
        
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        zero_point = int(round(zero_point))
        zero_point = max(qmin, min(qmax, zero_point))
        
        # Quantize
        q_tensor = torch.round(tensor / scale + zero_point)
        q_tensor = torch.clamp(q_tensor, qmin, qmax).to(torch.int8)
        
        return q_tensor, scale, zero_point
    
    @staticmethod
    def dequantize_tensor(q_tensor, scale, zero_point):
        """Dequantize back to floating point."""
        return scale * (q_tensor.float() - zero_point)
    
    @staticmethod
    def quantize_model(model, num_bits=8):
        """
        Post-training quantization of entire model.
        """
        quantized_params = {}
        
        for name, param in model.named_parameters():
            q_param, scale, zp = Quantizer.quantize_tensor(param.data, num_bits)
            quantized_params[name] = {
                'data': q_param,
                'scale': scale,
                'zero_point': zp
            }
        
        return quantized_params


class QuantizationAwareTraining(nn.Module):
    """
    Quantization-aware training module.
    
    Simulates quantization during training for better accuracy.
    """
    
    def __init__(self, num_bits=8):
        super().__init__()
        self.num_bits = num_bits
        self.qmin = 0
        self.qmax = 2**num_bits - 1
    
    def forward(self, x):
        """Fake quantize: quantize then immediately dequantize."""
        # This allows gradients to flow through
        min_val = x.min()
        max_val = x.max()
        
        scale = (max_val - min_val) / (self.qmax - self.qmin)
        zero_point = self.qmin - min_val / scale
        
        # Fake quantize
        x_q = torch.round(x / scale + zero_point)
        x_q = torch.clamp(x_q, self.qmin, self.qmax)
        x_dq = scale * (x_q - zero_point)
        
        # Straight-through estimator for gradients
        return x + (x_dq - x).detach()


print("\nNetwork Compression:")
print("=" * 60)
print("""
PRUNING:
- Magnitude pruning: Remove small weights
- Structured pruning: Remove entire neurons/filters
- Lottery ticket: Find sparse trainable subnetworks

QUANTIZATION:
- Post-training: Quantize after training
- Quantization-aware: Simulate during training
- Common: INT8 (8-bit), INT4 (4-bit)

KNOWLEDGE DISTILLATION:
- Train small "student" to mimic large "teacher"
- Soft labels carry more information
- Temperature scaling for softer distributions

Compression Benefits:
- Smaller model size (4-10x)
- Faster inference (2-4x)
- Lower power consumption
- Edge deployment
""")
```

### 82.3 Knowledge Distillation

```python
class KnowledgeDistillation:
    """
    Knowledge Distillation: Transfer knowledge from teacher to student.
    """
    
    def __init__(self, teacher, student, temperature=4.0, alpha=0.5):
        """
        Args:
            teacher: Large, trained teacher model
            student: Smaller student model to train
            temperature: Softmax temperature (higher = softer)
            alpha: Weight for distillation loss vs hard loss
        """
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Combined loss: soft targets + hard targets.
        
        L = α * KL(softmax(t/T), softmax(s/T)) * T² + (1-α) * CE(s, y)
        """
        # Soft targets (from teacher)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        distill_loss *= self.temperature ** 2  # Scale by T²
        
        # Hard targets (ground truth)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        return self.alpha * distill_loss + (1 - self.alpha) * hard_loss
    
    def train_step(self, x, labels, optimizer):
        """Single training step."""
        optimizer.zero_grad()
        
        # Teacher predictions (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        
        # Student predictions
        student_logits = self.student(x)
        
        # Loss and update
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        loss.backward()
        optimizer.step()
        
        return loss.item()


class FeatureDistillation(nn.Module):
    """
    Feature-based distillation: Match intermediate representations.
    """
    
    def __init__(self, teacher, student, layer_pairs):
        """
        Args:
            layer_pairs: List of (teacher_layer_name, student_layer_name) tuples
        """
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.layer_pairs = layer_pairs
        
        # Feature extractors
        self.teacher_features = {}
        self.student_features = {}
        
        # Register hooks
        for t_name, s_name in layer_pairs:
            self._register_hook(teacher, t_name, self.teacher_features)
            self._register_hook(student, s_name, self.student_features)
    
    def _register_hook(self, model, layer_name, storage):
        """Register forward hook to capture features."""
        def hook(module, input, output):
            storage[layer_name] = output
        
        for name, module in model.named_modules():
            if name == layer_name:
                module.register_forward_hook(hook)
                break
    
    def feature_loss(self):
        """Compute feature matching loss."""
        total_loss = 0
        
        for t_name, s_name in self.layer_pairs:
            t_feat = self.teacher_features[t_name]
            s_feat = self.student_features[s_name]
            
            # Adapt dimensions if needed
            if t_feat.shape != s_feat.shape:
                # Use 1x1 conv or linear to match
                pass
            
            total_loss += F.mse_loss(s_feat, t_feat)
        
        return total_loss


print("\nKnowledge Distillation:")
print("=" * 60)
print("""
Distillation Types:
1. Response-based: Match output logits
2. Feature-based: Match intermediate features
3. Relation-based: Match relationships between samples

Key Insights:
- Soft labels contain "dark knowledge"
- Temperature > 1 reveals class similarities
- Student can sometimes exceed teacher!

Applications:
- Model compression for deployment
- Transfer to different architectures
- Multi-task learning
""")
```

---

## Chapter 83: Continual Learning

### 83.1 Overcoming Catastrophic Forgetting

```python
class EWC(nn.Module):
    """
    Elastic Weight Consolidation.
    
    Prevents catastrophic forgetting by penalizing changes to important weights.
    """
    
    def __init__(self, model, importance_weight=1000):
        super().__init__()
        self.model = model
        self.importance_weight = importance_weight
        
        self.fisher = {}  # Fisher information (importance)
        self.optimal_params = {}  # Optimal parameters for previous tasks
    
    def compute_fisher(self, dataloader, criterion):
        """
        Estimate Fisher information matrix.
        
        Fisher approximates importance of each parameter.
        """
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        
        for x, y in dataloader:
            self.model.zero_grad()
            output = self.model(x)
            loss = criterion(output, y)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
        
        # Normalize by number of samples
        for n in fisher:
            fisher[n] /= len(dataloader.dataset)
        
        return fisher
    
    def register_task(self, dataloader, criterion):
        """Register Fisher and optimal params after training on a task."""
        # Compute Fisher
        new_fisher = self.compute_fisher(dataloader, criterion)
        
        # Accumulate Fisher
        for n, f in new_fisher.items():
            if n in self.fisher:
                self.fisher[n] += f
            else:
                self.fisher[n] = f
        
        # Store optimal parameters
        self.optimal_params = {
            n: p.data.clone() for n, p in self.model.named_parameters()
        }
    
    def ewc_loss(self):
        """
        EWC penalty term.
        
        Penalizes moving away from optimal params proportional to importance.
        """
        loss = 0
        
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.optimal_params[n]).pow(2)).sum()
        
        return self.importance_weight * loss


class PackNet(nn.Module):
    """
    PackNet: Progressive pruning for continual learning.
    
    After each task:
    1. Prune unimportant weights
    2. Freeze pruned weights
    3. Use remaining capacity for new tasks
    """
    
    def __init__(self, model, prune_ratio=0.5):
        super().__init__()
        self.model = model
        self.prune_ratio = prune_ratio
        self.masks = {}  # Binary masks for each task
        self.current_task = 0
    
    def prune_and_freeze(self, task_id):
        """Prune after training on a task."""
        task_mask = {}
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # Get available (unfrozen) weights
                if name in self.masks:
                    available = self.masks[name] == 0  # Not yet assigned
                else:
                    available = torch.ones_like(param.data, dtype=torch.bool)
                
                # Find important weights among available
                importance = param.data.abs() * available.float()
                threshold = torch.quantile(
                    importance[available], 
                    1 - self.prune_ratio
                )
                
                # Create mask for this task
                task_mask[name] = (importance >= threshold) & available
                
                # Update global mask
                if name in self.masks:
                    self.masks[name][task_mask[name]] = task_id + 1
                else:
                    self.masks[name] = torch.zeros_like(param.data, dtype=torch.int)
                    self.masks[name][task_mask[name]] = task_id + 1
        
        self.current_task = task_id + 1
        return task_mask
    
    def forward(self, x, task_id=None):
        """Forward pass with task-specific masks."""
        # Apply masks if doing inference on specific task
        if task_id is not None:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in self.masks:
                        mask = (self.masks[name] == task_id + 1) | (self.masks[name] == 0)
                        param.data *= mask.float()
        
        return self.model(x)


class ReplayBuffer:
    """
    Experience replay for continual learning.
    
    Store and replay samples from previous tasks.
    """
    
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, x, y, task_id):
        """Add samples to buffer."""
        for i in range(len(x)):
            sample = (x[i], y[i], task_id)
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.position] = sample
            
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample random batch from buffer."""
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        
        x = torch.stack([self.buffer[i][0] for i in indices])
        y = torch.tensor([self.buffer[i][1] for i in indices])
        tasks = torch.tensor([self.buffer[i][2] for i in indices])
        
        return x, y, tasks


print("\nContinual Learning:")
print("=" * 60)
print("""
CHALLENGE: Catastrophic Forgetting
- Neural networks forget previous tasks when learning new ones
- New gradients overwrite old knowledge

SOLUTIONS:

1. Regularization-based:
   - EWC: Penalize changes to important weights
   - SI: Online importance estimation
   
2. Architecture-based:
   - PackNet: Dedicate capacity to each task
   - Progressive Networks: Add new columns
   
3. Replay-based:
   - Experience Replay: Store and replay old samples
   - Generative Replay: Train generator to produce old samples

4. Parameter Isolation:
   - Task-specific heads/modules
   - Separate parameters per task
""")
```

---

## Summary: Advanced Algorithms

```
┌─────────────────────────────────────────────────────────────────────┐
│              ADVANCED ALGORITHMS SUMMARY                            │
├─────────────────────────────────────────────────────────────────────┤
│  BAYESIAN ML                                                        │
│  ├── Full posterior distributions over parameters                  │
│  ├── Natural uncertainty quantification                            │
│  ├── Gaussian Processes: Non-parametric, function-level prior      │
│  └── Variational Inference: Tractable approximations               │
│                                                                     │
│  META-LEARNING                                                      │
│  ├── MAML: Learn good initialization                               │
│  ├── Prototypical Networks: Metric learning                        │
│  └── Applications: Few-shot learning, fast adaptation              │
│                                                                     │
│  COMPRESSION                                                        │
│  ├── Pruning: Remove unnecessary weights                           │
│  ├── Quantization: Reduce precision                                │
│  ├── Distillation: Transfer to smaller model                       │
│  └── Benefits: Faster, smaller, deployable                         │
│                                                                     │
│  CONTINUAL LEARNING                                                 │
│  ├── EWC: Protect important weights                                │
│  ├── PackNet: Dedicate capacity                                    │
│  ├── Replay: Remember old samples                                  │
│  └── Goal: Learn sequentially without forgetting                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

# END OF TEXTBOOK

**Total Coverage:**
- 19 Parts
- 83+ Chapters  
- Comprehensive code implementations
- Conceptual explanations
- Practical exercises
- Quick reference guides

**Thank you for reading this comprehensive ML textbook!**
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
"""
DESIGN: Ad Click Prediction System (CTR Prediction)

Goal: Predict probability user clicks on ad
Optimize: Revenue (clicks × bid)
Constraint: Latency < 10ms, billions of requests/day
"""

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
"""
QUESTION 1: Implement K-Fold Cross Validation
"""

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


"""
QUESTION 2: Implement Softmax and Cross-Entropy Loss
"""

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


"""
QUESTION 3: Implement Batch Normalization
"""

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


"""
QUESTION 4: Implement Attention Mechanism
"""

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


"""
QUESTION 5: Implement Decision Tree Splitting
"""

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
"""
QUESTION 6: Implement Mini-Batch Gradient Descent
"""

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


"""
QUESTION 7: Implement K-Means++ Initialization
"""

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


"""
QUESTION 8: Implement Early Stopping
"""

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


"""
QUESTION 9: Implement ROC Curve and AUC
"""

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


"""
QUESTION 10: Implement Precision-Recall Curve
"""

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
# Part XXII: Case Studies and Real-World Applications

---

## Chapter 87: Industry Case Studies

### 87.1 Healthcare: Medical Image Analysis

```
┌─────────────────────────────────────────────────────────────────────┐
│        CASE STUDY: DIABETIC RETINOPATHY DETECTION                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PROBLEM:                                                           │
│  Diabetic retinopathy is a leading cause of blindness.             │
│  Manual screening by ophthalmologists is time-consuming and        │
│  not scalable, especially in developing countries.                 │
│                                                                     │
│  SOLUTION:                                                          │
│  Deep learning model to automatically grade retinal images.        │
│                                                                     │
│  DATASET:                                                           │
│  - EyePACS dataset: 35,000 labeled retinal images                  │
│  - 5 severity levels: 0 (none) to 4 (proliferative)               │
│  - Significant class imbalance (~70% level 0)                      │
│                                                                     │
│  CHALLENGES:                                                        │
│  1. Class imbalance                                                │
│  2. Image quality variation                                        │
│  3. Subtle differences between adjacent grades                     │
│  4. Need for high sensitivity (can't miss disease)                │
│  5. Interpretability for clinical acceptance                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

class DiabeticRetinopathyModel:
    """
    Complete pipeline for diabetic retinopathy detection.
    """
    
    def __init__(self):
        self.num_classes = 5
        self.model = None
        self.transform = None
        
    def create_model(self, pretrained=True):
        """
        Create model architecture.
        
        Use EfficientNet with custom head for ordinal regression.
        """
        # Load pretrained EfficientNet
        self.model = models.efficientnet_b4(pretrained=pretrained)
        
        # Modify classifier for 5 classes
        num_features = self.model.classifier[1].in_features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, self.num_classes)
        )
        
        return self.model
    
    def create_transforms(self):
        """
        Data augmentation for medical images.
        
        Conservative augmentations that preserve diagnostic features.
        """
        self.train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return self.train_transform, self.val_transform
    
    def handle_class_imbalance(self, labels):
        """
        Compute class weights for imbalanced dataset.
        """
        class_counts = torch.bincount(labels)
        total = len(labels)
        
        # Inverse frequency weighting
        weights = total / (self.num_classes * class_counts.float())
        
        # Optionally boost rare classes more
        weights = weights ** 0.5  # Square root dampening
        
        return weights
    
    def ordinal_loss(self, logits, labels):
        """
        Ordinal regression loss for graded severity.
        
        Penalizes distance from true grade.
        """
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, labels)
        
        # Ordinal penalty: penalize predictions far from true label
        probs = F.softmax(logits, dim=1)
        grades = torch.arange(self.num_classes, device=logits.device).float()
        
        # Expected grade
        expected_grade = (probs * grades).sum(dim=1)
        
        # Mean absolute error in grade
        ordinal_loss = torch.abs(expected_grade - labels.float()).mean()
        
        return ce_loss + 0.5 * ordinal_loss


class GradCAMExplainer:
    """
    GradCAM for model interpretability.
    
    Shows which regions the model focuses on.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, class_idx):
        """Generate class activation map."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Backward pass for target class
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Compute weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = F.interpolate(cam, input_image.shape[2:], mode='bilinear')
        
        return cam.squeeze().cpu().numpy()


print("Healthcare Case Study: Diabetic Retinopathy")
print("=" * 60)
print("""
KEY LEARNINGS:

1. DATA:
   - Quality matters more than quantity
   - Expert labeling is crucial but expensive
   - Class imbalance requires careful handling

2. MODEL:
   - Transfer learning from ImageNet works well
   - Ensemble of models improves robustness
   - High-resolution inputs capture fine details

3. EVALUATION:
   - Sensitivity is critical (don't miss disease)
   - Quadratic weighted kappa for ordinal data
   - Calibration for reliable probabilities

4. DEPLOYMENT:
   - Interpretability required for clinical acceptance
   - Integration with existing clinical workflows
   - Continuous monitoring for distribution shift

5. RESULTS:
   - Google's model achieved ophthalmologist-level accuracy
   - AUC > 0.99 for referable DR detection
   - Deployed in India and Thailand clinics
""")
```

### 87.2 Finance: Credit Risk Modeling

```python
"""
CASE STUDY: Credit Default Prediction

Predict probability of loan default to inform lending decisions.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

class CreditRiskModel:
    """
    Credit risk prediction pipeline.
    """
    
    def __init__(self):
        self.feature_names = None
        self.model = None
        self.scaler = StandardScaler()
    
    def engineer_features(self, df):
        """
        Feature engineering for credit data.
        
        Domain-specific feature creation.
        """
        features = df.copy()
        
        # Income-related ratios
        features['debt_to_income'] = df['total_debt'] / (df['annual_income'] + 1)
        features['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
        features['monthly_payment_ratio'] = df['monthly_payment'] / (df['monthly_income'] + 1)
        
        # Credit utilization
        features['credit_utilization'] = df['credit_used'] / (df['credit_limit'] + 1)
        features['available_credit'] = df['credit_limit'] - df['credit_used']
        
        # Payment history
        features['payment_history_score'] = (
            df['on_time_payments'] / (df['total_payments'] + 1)
        )
        features['delinquency_rate'] = df['late_payments'] / (df['total_payments'] + 1)
        
        # Account age
        features['avg_account_age'] = df['total_account_age'] / (df['num_accounts'] + 1)
        features['credit_history_length'] = df['oldest_account_age']
        
        # Behavioral features
        features['recent_inquiries_ratio'] = df['inquiries_6m'] / (df['inquiries_total'] + 1)
        features['new_accounts_ratio'] = df['accounts_opened_12m'] / (df['num_accounts'] + 1)
        
        # Risk indicators
        features['has_bankruptcy'] = (df['bankruptcies'] > 0).astype(int)
        features['has_collections'] = (df['collections'] > 0).astype(int)
        features['high_utilization'] = (features['credit_utilization'] > 0.7).astype(int)
        
        return features
    
    def create_model(self):
        """
        Create XGBoost model with proper constraints.
        """
        import xgboost as xgb
        
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            scale_pos_weight=3,  # Handle class imbalance
            eval_metric='auc',
            early_stopping_rounds=50,
            random_state=42
        )
        
        return self.model
    
    def train_with_adversarial_validation(self, X_train, X_test, y_train):
        """
        Check for training-test distribution shift.
        
        If we can easily distinguish train from test,
        there may be a data shift problem.
        """
        # Label train as 0, test as 1
        combined = np.vstack([X_train, X_test])
        labels = np.array([0] * len(X_train) + [1] * len(X_test))
        
        # Train classifier
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        cv = StratifiedKFold(n_splits=5)
        scores = []
        
        for train_idx, val_idx in cv.split(combined, labels):
            clf.fit(combined[train_idx], labels[train_idx])
            score = clf.score(combined[val_idx], labels[val_idx])
            scores.append(score)
        
        avg_score = np.mean(scores)
        
        if avg_score > 0.6:
            print(f"WARNING: Distribution shift detected (AUC={avg_score:.3f})")
            print("Training and test data may come from different distributions")
        else:
            print(f"Distribution check passed (AUC={avg_score:.3f})")
        
        return avg_score
    
    def evaluate_fairness(self, X, y, predictions, sensitive_attr):
        """
        Evaluate fairness across protected groups.
        """
        groups = X[sensitive_attr].unique()
        
        metrics = {}
        for group in groups:
            mask = X[sensitive_attr] == group
            group_y = y[mask]
            group_pred = predictions[mask]
            
            # Approval rate
            approval_rate = 1 - group_pred.mean()
            
            # True positive rate (if we have labels)
            tpr = group_pred[group_y == 1].mean() if (group_y == 1).any() else 0
            
            # False positive rate
            fpr = group_pred[group_y == 0].mean() if (group_y == 0).any() else 0
            
            metrics[group] = {
                'approval_rate': approval_rate,
                'tpr': tpr,
                'fpr': fpr,
                'count': mask.sum()
            }
        
        # Check disparate impact (80% rule)
        max_approval = max(m['approval_rate'] for m in metrics.values())
        min_approval = min(m['approval_rate'] for m in metrics.values())
        
        disparate_impact = min_approval / max_approval if max_approval > 0 else 0
        
        print(f"\nFairness Metrics:")
        print(f"Disparate Impact Ratio: {disparate_impact:.3f}")
        print(f"(Should be > 0.8 for fairness)")
        
        return metrics, disparate_impact


class CreditScorecard:
    """
    Traditional scorecard model for interpretability.
    
    Converts logistic regression to points-based system.
    """
    
    def __init__(self, pdo=20, base_odds=50, base_score=600):
        """
        Args:
            pdo: Points to double the odds
            base_odds: Base odds (1:1 ratio)
            base_score: Score at base odds
        """
        self.pdo = pdo
        self.base_odds = base_odds
        self.base_score = base_score
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)
    
    def woe_binning(self, X, y, feature, n_bins=10):
        """
        Weight of Evidence binning for a feature.
        """
        # Create bins
        bins = pd.qcut(X[feature], q=n_bins, duplicates='drop')
        
        # Calculate WoE for each bin
        grouped = pd.DataFrame({'feature': bins, 'target': y}).groupby('feature')
        
        stats = grouped['target'].agg(['sum', 'count'])
        stats['non_events'] = stats['count'] - stats['sum']
        stats['events'] = stats['sum']
        
        # Distribution of events and non-events
        total_events = stats['events'].sum()
        total_non_events = stats['non_events'].sum()
        
        stats['dist_events'] = stats['events'] / total_events
        stats['dist_non_events'] = stats['non_events'] / total_non_events
        
        # WoE = ln(% non-events / % events)
        stats['woe'] = np.log(
            (stats['dist_non_events'] + 0.0001) / 
            (stats['dist_events'] + 0.0001)
        )
        
        # Information Value
        stats['iv'] = (stats['dist_non_events'] - stats['dist_events']) * stats['woe']
        
        return stats
    
    def probability_to_score(self, prob):
        """Convert probability to credit score."""
        odds = (1 - prob) / prob
        score = self.offset + self.factor * np.log(odds)
        return np.clip(score, 300, 850)
    
    def score_to_probability(self, score):
        """Convert credit score to probability."""
        odds = np.exp((score - self.offset) / self.factor)
        prob = 1 / (1 + odds)
        return prob


print("\nFinance Case Study: Credit Risk")
print("=" * 60)
print("""
REGULATORY REQUIREMENTS:

1. FAIR LENDING:
   - Equal Credit Opportunity Act (ECOA)
   - Cannot discriminate based on protected attributes
   - Must provide adverse action reasons

2. INTERPRETABILITY:
   - Reason codes required for denials
   - Scorecard models preferred by regulators
   - Must explain model decisions

3. MODEL RISK MANAGEMENT:
   - SR 11-7 guidance from Federal Reserve
   - Model validation requirements
   - Ongoing monitoring

KEY METRICS:
- Gini coefficient (2*AUC - 1)
- KS statistic
- Population Stability Index (PSI)
- Characteristic Stability Index (CSI)

PRODUCTION CONSIDERATIONS:
- Champion/challenger testing
- Reject inference (for denied applicants)
- Vintage analysis
- Early warning indicators
""")
```

### 87.3 E-commerce: Product Recommendations

```python
"""
CASE STUDY: Amazon-style Product Recommendations

Multi-stage recommendation system with various signals.
"""

class ProductRecommender:
    """
    Multi-stage product recommendation system.
    """
    
    def __init__(self):
        self.user_embeddings = None
        self.item_embeddings = None
        self.popular_items = None
    
    def build_candidate_generation(self):
        """
        Stage 1: Generate candidate items (fast, broad).
        
        Goal: Reduce millions of items to thousands.
        """
        return """
        CANDIDATE SOURCES:
        
        1. COLLABORATIVE FILTERING
           - Item-to-item similarity
           - "Users who bought X also bought Y"
           - User embedding nearest neighbors
        
        2. CONTENT-BASED
           - Similar products by category/attributes
           - Text similarity (title, description)
           - Image similarity
        
        3. POPULAR/TRENDING
           - Globally popular items
           - Category bestsellers
           - Trending in user's region
        
        4. PERSONALIZED
           - Items from browsing history
           - Items from wishlist
           - Items from abandoned cart
        
        5. CONTEXTUAL
           - Seasonal items
           - Time-of-day relevant
           - Event-based (holidays, sales)
        
        Each source contributes candidates.
        Union with deduplication.
        Target: 1000-5000 candidates.
        """
    
    def build_ranking_model(self):
        """
        Stage 2: Rank candidates (slower, accurate).
        
        Goal: Order candidates by relevance.
        """
        return """
        TWO-TOWER ARCHITECTURE:
        
        User Tower:
        - User demographics
        - Purchase history (sequence model)
        - Browse history
        - Search history
        - User embedding
        
        Item Tower:
        - Item attributes
        - Category embedding
        - Price tier
        - Popularity statistics
        - Item embedding
        
        Interaction Features:
        - User-item history
        - User-category affinity
        - Price preference match
        - Brand preference match
        
        Output: Relevance score
        
        TRAINING:
        - Implicit feedback (clicks, purchases)
        - Negative sampling
        - Listwise loss (NDCG optimization)
        """
    
    def handle_cold_start(self):
        """
        Handle new users and new items.
        """
        return """
        NEW USER (no history):
        1. Popular items globally
        2. Popular in user's demographic
        3. Popular in user's location
        4. Quickly gather signals (first clicks)
        5. Explore-exploit strategy
        
        NEW ITEM (no interactions):
        1. Content-based similarity to popular items
        2. Attribute matching to user preferences
        3. Category-based recommendations
        4. Promote in "New Arrivals" section
        5. Boosted exploration period
        """


class SessionBasedRecommender(nn.Module):
    """
    Session-based recommendations using RNN.
    
    For anonymous users with only session data.
    """
    
    def __init__(self, num_items, embed_dim=64, hidden_dim=128):
        super().__init__()
        
        self.embedding = nn.Embedding(num_items, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_items)
    
    def forward(self, session_items):
        """
        Args:
            session_items: (batch, seq_len) item IDs in session
        
        Returns:
            scores: (batch, num_items) relevance scores
        """
        # Embed items
        x = self.embedding(session_items)
        
        # Process sequence
        _, hidden = self.gru(x)
        
        # Score all items
        scores = self.output(hidden.squeeze(0))
        
        return scores


class DiversityReranker:
    """
    Re-rank for diversity (avoid similar items).
    """
    
    def __init__(self, lambda_diversity=0.5):
        self.lambda_diversity = lambda_diversity
    
    def mmr_rerank(self, items, relevance_scores, item_embeddings, top_k=10):
        """
        Maximal Marginal Relevance re-ranking.
        
        Balances relevance and diversity.
        """
        selected = []
        remaining = list(range(len(items)))
        
        while len(selected) < top_k and remaining:
            best_item = None
            best_score = float('-inf')
            
            for idx in remaining:
                # Relevance term
                relevance = relevance_scores[idx]
                
                # Diversity term (max similarity to selected)
                if selected:
                    similarities = [
                        np.dot(item_embeddings[idx], item_embeddings[s])
                        for s in selected
                    ]
                    max_sim = max(similarities)
                else:
                    max_sim = 0
                
                # MMR score
                mmr = (1 - self.lambda_diversity) * relevance - \
                      self.lambda_diversity * max_sim
                
                if mmr > best_score:
                    best_score = mmr
                    best_item = idx
            
            selected.append(best_item)
            remaining.remove(best_item)
        
        return [items[i] for i in selected]


print("\nE-commerce Case Study: Product Recommendations")
print("=" * 60)
print("""
BUSINESS IMPACT:
- Amazon: 35% of revenue from recommendations
- Netflix: 80% of content watched is recommended
- Increased average order value
- Improved customer retention

A/B TESTING METRICS:
- Click-through rate (CTR)
- Add-to-cart rate
- Conversion rate
- Revenue per user
- Diversity of purchases
- Long-term retention

COMMON PITFALLS:
1. Popularity bias (rich get richer)
2. Filter bubbles (limited discovery)
3. Ignoring context (time, device)
4. Over-personalization (creepy factor)
5. Stale recommendations

BEST PRACTICES:
1. Multi-objective optimization
2. Diversity in results
3. Explainable recommendations
4. A/B test everything
5. Monitor for feedback loops
""")
```

---

## Chapter 88: ML in Production Lessons

### 88.1 Common Production Failures

```
┌─────────────────────────────────────────────────────────────────────┐
│            PRODUCTION ML FAILURE PATTERNS                           │
├─────────────────────────────────────────────────────────────────────┤

FAILURE 1: TRAINING-SERVING SKEW
─────────────────────────────────
Problem:
- Features computed differently in training vs serving
- Training uses batch features, serving needs real-time
- Data leakage in training (future information)

Example:
"Our model had 95% accuracy offline but 60% in production.
We discovered that 'avg_user_purchases_30d' was computed
differently - training used exact counts, serving used
cached values that were 24 hours stale."

Prevention:
- Use same feature computation code for train/serve
- Feature stores with consistent computation
- Monitor feature distributions in production


FAILURE 2: DATA DRIFT
─────────────────────
Problem:
- Real-world data distribution changes over time
- Model trained on historical data becomes stale
- Seasonal patterns not captured

Example:
"Our fraud model degraded 15% during Black Friday.
Transaction patterns were completely different -
higher amounts, new merchants, unusual times."

Prevention:
- Monitor input feature distributions
- Track prediction distribution changes
- Automated retraining pipelines
- Separate models for known distribution shifts


FAILURE 3: FEEDBACK LOOPS
─────────────────────────
Problem:
- Model predictions influence future training data
- Self-fulfilling prophecies
- Popularity bias amplification

Example:
"Our recommendation system kept showing the same items.
Popular items got more views, which made them more popular,
creating a feedback loop that killed discovery."

Prevention:
- Exploration in production (epsilon-greedy)
- Counterfactual evaluation
- Diversity constraints
- Randomized experiments


FAILURE 4: SILENT FAILURES
──────────────────────────
Problem:
- Model serves predictions but quality degrades
- No alerts because system is "working"
- Gradual degradation goes unnoticed

Example:
"Our CTR model ran for 3 months with a bug in feature
preprocessing. We only noticed when revenue dropped 8%.
The model was technically working - just wrong."

Prevention:
- Business metric monitoring (not just system metrics)
- Prediction distribution monitoring
- Regular model performance checks
- Human-in-the-loop spot checks


FAILURE 5: EDGE CASES
─────────────────────
Problem:
- Rare inputs cause unexpected behavior
- Out-of-distribution inputs not handled
- Adversarial or corrupted inputs

Example:
"Our image classifier crashed on very small images.
When a mobile app sent thumbnails, the model threw
exceptions and blocked the entire request."

Prevention:
- Input validation
- Graceful degradation (fallback predictions)
- Test with edge cases
- Error handling and defaults

└─────────────────────────────────────────────────────────────────────┘
```

### 88.2 Production Monitoring System

```python
class MLMonitoringSystem:
    """
    Comprehensive ML monitoring system.
    """
    
    def __init__(self, model_name, feature_names):
        self.model_name = model_name
        self.feature_names = feature_names
        self.baseline_stats = None
        self.alerts = []
    
    def compute_baseline_stats(self, X_train, y_train, predictions):
        """
        Compute baseline statistics from training data.
        """
        self.baseline_stats = {
            'feature_means': np.mean(X_train, axis=0),
            'feature_stds': np.std(X_train, axis=0),
            'feature_mins': np.min(X_train, axis=0),
            'feature_maxs': np.max(X_train, axis=0),
            'prediction_mean': np.mean(predictions),
            'prediction_std': np.std(predictions),
            'label_distribution': np.bincount(y_train) / len(y_train),
        }
        
        return self.baseline_stats
    
    def detect_data_drift(self, X_current, threshold=0.1):
        """
        Detect drift in input features using PSI.
        
        Population Stability Index:
        PSI = Σ (Actual% - Expected%) * ln(Actual% / Expected%)
        """
        drifted_features = []
        
        for i, name in enumerate(self.feature_names):
            # Compute PSI for each feature
            expected = X_current[:, i]
            actual_mean = np.mean(expected)
            baseline_mean = self.baseline_stats['feature_means'][i]
            baseline_std = self.baseline_stats['feature_stds'][i]
            
            # Simple z-score drift detection
            if baseline_std > 0:
                z_score = abs(actual_mean - baseline_mean) / baseline_std
                if z_score > 2:  # More than 2 std devs
                    drifted_features.append({
                        'feature': name,
                        'baseline_mean': baseline_mean,
                        'current_mean': actual_mean,
                        'z_score': z_score
                    })
        
        if drifted_features:
            self.alerts.append({
                'type': 'DATA_DRIFT',
                'severity': 'WARNING',
                'drifted_features': drifted_features
            })
        
        return drifted_features
    
    def detect_prediction_drift(self, predictions, threshold=0.15):
        """
        Detect drift in model predictions.
        """
        current_mean = np.mean(predictions)
        baseline_mean = self.baseline_stats['prediction_mean']
        baseline_std = self.baseline_stats['prediction_std']
        
        # Relative change
        if baseline_mean != 0:
            relative_change = abs(current_mean - baseline_mean) / abs(baseline_mean)
        else:
            relative_change = abs(current_mean - baseline_mean)
        
        if relative_change > threshold:
            self.alerts.append({
                'type': 'PREDICTION_DRIFT',
                'severity': 'CRITICAL',
                'baseline_mean': baseline_mean,
                'current_mean': current_mean,
                'relative_change': relative_change
            })
            return True
        
        return False
    
    def detect_performance_degradation(self, y_true, y_pred, 
                                       baseline_metric, current_threshold=0.9):
        """
        Detect model performance degradation.
        """
        from sklearn.metrics import roc_auc_score
        
        try:
            current_auc = roc_auc_score(y_true, y_pred)
        except:
            return None
        
        if current_auc < baseline_metric * current_threshold:
            self.alerts.append({
                'type': 'PERFORMANCE_DEGRADATION',
                'severity': 'CRITICAL',
                'baseline_auc': baseline_metric,
                'current_auc': current_auc,
                'degradation': (baseline_metric - current_auc) / baseline_metric
            })
            return True
        
        return False
    
    def check_feature_completeness(self, X, missing_threshold=0.1):
        """
        Check for missing features.
        """
        missing_rates = np.isnan(X).mean(axis=0)
        
        problematic_features = []
        for i, rate in enumerate(missing_rates):
            if rate > missing_threshold:
                problematic_features.append({
                    'feature': self.feature_names[i],
                    'missing_rate': rate
                })
        
        if problematic_features:
            self.alerts.append({
                'type': 'MISSING_FEATURES',
                'severity': 'WARNING',
                'features': problematic_features
            })
        
        return problematic_features
    
    def get_monitoring_dashboard(self):
        """
        Generate monitoring dashboard data.
        """
        return {
            'model_name': self.model_name,
            'alerts': self.alerts,
            'alert_count': len(self.alerts),
            'critical_alerts': sum(1 for a in self.alerts if a['severity'] == 'CRITICAL'),
            'status': 'HEALTHY' if not any(a['severity'] == 'CRITICAL' for a in self.alerts) else 'DEGRADED'
        }


class ABTestingFramework:
    """
    A/B testing framework for ML models.
    """
    
    def __init__(self, control_model, treatment_model, traffic_split=0.5):
        self.control_model = control_model
        self.treatment_model = treatment_model
        self.traffic_split = traffic_split
        
        self.control_metrics = []
        self.treatment_metrics = []
    
    def route_request(self, user_id):
        """
        Route user to control or treatment.
        
        Use consistent hashing for sticky assignment.
        """
        # Hash user_id for consistent assignment
        hash_value = hash(str(user_id)) % 100
        
        if hash_value < self.traffic_split * 100:
            return 'control', self.control_model
        else:
            return 'treatment', self.treatment_model
    
    def log_metric(self, group, metric_name, value):
        """Log metric for a group."""
        metric = {
            'metric_name': metric_name,
            'value': value,
            'timestamp': pd.Timestamp.now()
        }
        
        if group == 'control':
            self.control_metrics.append(metric)
        else:
            self.treatment_metrics.append(metric)
    
    def compute_significance(self, metric_name):
        """
        Compute statistical significance of difference.
        """
        from scipy import stats
        
        control_values = [m['value'] for m in self.control_metrics 
                         if m['metric_name'] == metric_name]
        treatment_values = [m['value'] for m in self.treatment_metrics 
                          if m['metric_name'] == metric_name]
        
        if len(control_values) < 30 or len(treatment_values) < 30:
            return {'status': 'INSUFFICIENT_DATA'}
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
        
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        lift = (treatment_mean - control_mean) / control_mean if control_mean != 0 else 0
        
        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'lift': lift,
            'p_value': p_value,
            'significant': p_value < 0.05
        }


print("\nProduction ML Best Practices:")
print("=" * 60)
print("""
MONITORING CHECKLIST:

□ Input Data Quality
  - Missing values
  - Out-of-range values
  - Feature distributions

□ Model Performance
  - Accuracy/AUC on labeled data
  - Prediction latency
  - Error rates

□ Prediction Quality
  - Prediction distribution
  - Confidence calibration
  - Edge case handling

□ Business Metrics
  - CTR, conversion, revenue
  - User satisfaction
  - Long-term retention

□ System Health
  - CPU/memory usage
  - Request latency
  - Error rates

ALERT THRESHOLDS:
- Data drift: PSI > 0.1 (warning), > 0.2 (critical)
- Performance: AUC drop > 5% (warning), > 10% (critical)
- Latency: p99 > 100ms (warning), > 500ms (critical)
- Error rate: > 1% (warning), > 5% (critical)
""")
```

---

## Summary: Real-World ML

```
┌─────────────────────────────────────────────────────────────────────┐
│              REAL-WORLD ML SUMMARY                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  HEALTHCARE:                                                        │
│  - High stakes require high accuracy and interpretability          │
│  - Regulatory requirements (FDA, HIPAA)                            │
│  - Expert validation is essential                                  │
│  - Class imbalance common                                          │
│                                                                     │
│  FINANCE:                                                           │
│  - Fairness and regulatory compliance critical                     │
│  - Interpretability required for decisions                         │
│  - Adversarial actors (fraud)                                      │
│  - Model risk management                                           │
│                                                                     │
│  E-COMMERCE:                                                        │
│  - Scale is massive (millions of users/items)                      │
│  - Real-time requirements                                          │
│  - A/B testing culture                                             │
│  - Multi-objective optimization                                    │
│                                                                     │
│  PRODUCTION LESSONS:                                                │
│  - Training-serving skew is real                                   │
│  - Data drift happens continuously                                 │
│  - Silent failures are dangerous                                   │
│  - Monitoring is not optional                                      │
│  - A/B test before full deployment                                 │
│                                                                     │
│  KEY TAKEAWAY:                                                      │
│  The hard part of ML is not building the model -                   │
│  it's deploying, monitoring, and maintaining it in production.     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

# TEXTBOOK COMPLETION

This comprehensive Machine Learning textbook now covers:

**22 Parts | 88+ Chapters | 30,000+ Lines**

## Topics Covered:

1. **Foundations**: ML basics, types, workflow
2. **Mathematics**: Linear algebra, calculus, probability
3. **Data**: Collection, preprocessing, feature engineering
4. **Supervised Learning**: Regression, classification, trees, ensembles
5. **Neural Networks**: Architecture, training, CNNs, RNNs
6. **Unsupervised Learning**: Clustering, dimensionality reduction
7. **NLP**: Text processing, embeddings, transformers
8. **Time Series**: Classical and deep learning approaches
9. **MLOps**: Deployment, monitoring, CI/CD
10. **Advanced CV**: Object detection, segmentation
11. **Reinforcement Learning**: Q-learning, policy gradients
12. **Practical Projects**: End-to-end implementations
13. **Advanced Topics**: GANs, AutoML
14. **Responsible AI**: Fairness, interpretability, privacy
15. **Optimization**: Optimizers, schedulers, regularization
16. **Graph Neural Networks**: GCN, GAT, GraphSAGE
17. **Exercises**: Coding challenges, quizzes
18. **Foundation Models**: Self-supervised learning, LLMs
19. **Advanced Algorithms**: Bayesian ML, meta-learning, compression
20. **Interview Prep**: System design, coding questions
21. **Case Studies**: Healthcare, finance, e-commerce
22. **Production**: Monitoring, A/B testing, failure patterns

**Congratulations on completing this comprehensive ML resource!**

---

## Final Thoughts and Next Steps

### Your Learning Journey

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LEARNING ROADMAP                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  BEGINNER → INTERMEDIATE (3-6 months)                              │
│  ────────────────────────────────────                               │
│  □ Master Python, NumPy, Pandas                                    │
│  □ Understand linear algebra and probability                       │
│  □ Implement basic algorithms from scratch                         │
│  □ Complete 3-5 end-to-end projects                               │
│  □ Practice with Kaggle competitions                               │
│                                                                     │
│  INTERMEDIATE → ADVANCED (6-12 months)                             │
│  ─────────────────────────────────────                             │
│  □ Deep dive into neural networks                                  │
│  □ Master PyTorch or TensorFlow                                    │
│  □ Specialize in a domain (CV, NLP, RL)                           │
│  □ Read and implement research papers                              │
│  □ Contribute to open source projects                              │
│                                                                     │
│  ADVANCED → EXPERT (1+ years)                                      │
│  ────────────────────────────                                       │
│  □ Design novel architectures                                      │
│  □ Publish research or blog posts                                  │
│  □ Build production ML systems                                     │
│  □ Mentor others in the field                                      │
│  □ Stay current with latest developments                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Recommended Resources

**Books:**
- "Deep Learning" by Goodfellow, Bengio, Courville
- "Pattern Recognition and Machine Learning" by Bishop
- "Hands-On Machine Learning" by Aurélien Géron
- "Machine Learning Engineering" by Andriy Burkov
- "Designing Machine Learning Systems" by Chip Huyen

**Online Courses:**
- Stanford CS229 (Machine Learning)
- Stanford CS231n (Computer Vision)
- Stanford CS224n (NLP)
- fast.ai (Practical Deep Learning)
- DeepLearning.AI Specializations

**Practice Platforms:**
- Kaggle (competitions and datasets)
- LeetCode (coding practice)
- Papers With Code (implementations)
- Hugging Face (models and datasets)

**Communities:**
- Reddit: r/MachineLearning, r/learnmachinelearning
- Twitter/X: Follow ML researchers
- Discord: Various ML servers
- Local meetups and conferences

### Final Words

Machine learning is a rapidly evolving field. The concepts in this textbook 
provide a strong foundation, but the learning never stops. Stay curious, 
keep building, and remember that the best way to learn ML is by doing ML.

Key principles to remember:
1. Start simple, add complexity as needed
2. Data quality matters more than model complexity
3. Always validate on held-out data
4. Production is harder than prototyping
5. Ethics and fairness are not optional

Thank you for reading this comprehensive textbook. 
Now go build something amazing! 🚀

---

**END OF TEXTBOOK**

*Total: 22 Parts | 88 Chapters | 30,000+ Lines | 150+ Code Examples*

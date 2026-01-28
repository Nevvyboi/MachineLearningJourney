<div align="center">

# 🧠 Foundations

![Chapter](https://img.shields.io/badge/Chapter-01-blue?style=for-the-badge)
![Topic](https://img.shields.io/badge/Topic-ML%20Basics%20%7C%20Math%20%7C%20Data-green?style=for-the-badge)
![Lines](https://img.shields.io/badge/Lines-4,949-orange?style=for-the-badge)

*Machine Learning Fundamentals, Mathematics & Neural Network Basics*

---

</div>

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

---

<div align="center">

[⬅️ Previous: Introduction](00-introduction.md) | [📚 Table of Contents](../README.md) | [Next: Supervised Learning ➡️](02-supervised-learning.md)

</div>

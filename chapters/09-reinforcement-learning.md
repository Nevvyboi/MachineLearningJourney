<div align="center">

# 🎮 Reinforcement Learning

![Chapter](https://img.shields.io/badge/Chapter-09-blue?style=for-the-badge)
![Topic](https://img.shields.io/badge/Topic-RL%20%7C%20Deep%20RL-green?style=for-the-badge)

*Q-Learning, DQN, Policy Gradients & PPO*

---

</div>

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


---

<div align="center">

[⬅️ Previous: Computer Vision](08-computer-vision.md) | [📚 Table of Contents](../README.md) | [Next: Practical Projects ➡️](10-projects.md)

</div>

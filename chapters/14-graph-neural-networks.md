<div align="center">

# 🔗 Graph Neural Networks

![Chapter](https://img.shields.io/badge/Chapter-14-blue?style=for-the-badge)
![Topic](https://img.shields.io/badge/Topic-GNN%20%7C%20Graphs-green?style=for-the-badge)

*GCN, GAT, GraphSAGE & Geometric Deep Learning*

---

</div>

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


---

<div align="center">

[⬅️ Previous: Optimization Deep Dive](13-optimization.md) | [📚 Table of Contents](../README.md) | [Next: Exercises & Quizzes ➡️](15-exercises.md)

</div>

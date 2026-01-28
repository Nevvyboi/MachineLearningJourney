<div align="center">

# 🌐 Foundation Models

![Chapter](https://img.shields.io/badge/Chapter-16-blue?style=for-the-badge)
![Topic](https://img.shields.io/badge/Topic-LLMs%20%7C%20CLIP-green?style=for-the-badge)

*Self-Supervised Learning, Large Language Models & Vision-Language*

---

</div>

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


---

<div align="center">

[⬅️ Previous: Exercises & Quizzes](15-exercises.md) | [📚 Table of Contents](../README.md) | [Next: Advanced Algorithms ➡️](17-advanced-algorithms.md)

</div>

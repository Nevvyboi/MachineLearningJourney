<div align="center">

# ⚡ Advanced Topics

![Chapter](https://img.shields.io/badge/Chapter-11-blue?style=for-the-badge)
![Topic](https://img.shields.io/badge/Topic-GANs%20%7C%20Transformers-green?style=for-the-badge)

*Generative Models, Attention Mechanisms & AutoML*

---

</div>

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


---

<div align="center">

[⬅️ Previous: Practical Projects](10-projects.md) | [📚 Table of Contents](../README.md) | [Next: Responsible AI ➡️](12-responsible-ai.md)

</div>

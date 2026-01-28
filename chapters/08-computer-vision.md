<div align="center">

# 🖼️ Computer Vision

![Chapter](https://img.shields.io/badge/Chapter-08-blue?style=for-the-badge)
![Topic](https://img.shields.io/badge/Topic-CNNs%20%7C%20Detection-green?style=for-the-badge)

*ResNet, YOLO, U-Net & Image Segmentation*

---

</div>

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


---

<div align="center">

[⬅️ Previous: Appendices](07-appendices.md) | [📚 Table of Contents](../README.md) | [Next: Reinforcement Learning ➡️](09-reinforcement-learning.md)

</div>

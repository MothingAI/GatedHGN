# GatedHGN: New Energy Vehicle Risk Assessment

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Accuracy](https://img.shields.io/badge/Accuracy-97.88%25-brightgreen)
![F1-Macro](https://img.shields.io/badge/F1--Macro-98.11%25-brightgreen)

**A Gated Metapath Attention Heterogeneous Graph Neural Network for Risk Assessment**

[📖 Paper](#) • [🚀 Quick Start](#quick-start) • [📚 Documentation](#documentation) • [💡 Examples](#examples)

</div>

---

## 🎯 Overview

**GatedHGN** (Gated Heterogeneous Graph Network) is a state-of-the-art heterogeneous graph neural network that leverages **gated metapath attention mechanisms** for risk assessment in new energy vehicle enterprises.

### Key Features

- ⭐ **Novel Gated Mechanism**: Feature-level dynamic attention weights (256-dim) vs traditional node-level weights
- 🎯 **Exceptional Performance**: 97.88% accuracy, 98.11% F1-Macro
- 🔀 **Multi-Scale Attention**: Feature-level, node-level, and path-level attention mechanisms
- 📊 **Heterogeneous Graph**: Supports multiple node and edge types
- ⚡ **Production Ready**: Pre-trained model with 1.79M parameters
- 🔬 **Well Documented**: Comprehensive documentation and examples

---

## 📊 Performance

### Overall Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| **Accuracy** | **97.88%** | Classification accuracy |
| **F1-Macro** | **98.11%** | Macro-averaged F1 score |
| **F1-Weighted** | **97.88%** | Weighted F1 score |
| **Loss** | **0.0652** | Cross-entropy loss |

### Per-Class F1 Scores

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| General (0) | 99.35% | 97.45% | **98.39%** | 157 |
| Excellent (1) | 96.03% | 99.18% | **97.58%** | 122 |
| Good (2) | 97.87% | 97.05% | **97.46%** | 237 |
| Poor (3) | 98.08% | 100.00% | **99.03%** | 51 |

✅ **All classes achieve F1 > 97%!**

---

## 🏗️ Architecture

```
Input Heterogeneous Graph
│
├─ Company Nodes (567)
│  └─ Features: [567, 10]
│
└─ Industry Nodes (45)
   └─ Features: [45, 23]
│
↓
┌─────────────────────────────────────────┐
│  1. Feature Attention Layer             │
│     ├─ CompanyFeatureAttention          │
│     └─ IndustryFeatureAttention         │
└─────────────────────────────────────────┘
│
↓
┌─────────────────────────────────────────┐
│  2. Cross-Attention Layer               │
│     └─ CompanyIndustryAttention         │
└─────────────────────────────────────────┘
│
↓
┌─────────────────────────────────────────┐
│  3. Metapath Message Passing (3 paths)  │
│     ├─ Direct: company → company        │
│     ├─ Industry Mediated                │
│     │  company → industry → company     │
│     └─ Supply Chain                     │
│        company → industry →             │
│                   industry → company     │
└─────────────────────────────────────────┘
│
↓
┌─────────────────────────────────────────┐
│  4. Gated Metapath Attention ⭐         │
│     └─ Dynamic aggregation with         │
│        feature-level gating             │
└─────────────────────────────────────────┘
│
↓
┌─────────────────────────────────────────┐
│  5. Classifier                          │
│     └─ Output: 4 risk categories        │
└─────────────────────────────────────────┘
│
↓
Risk Predictions [567, 4]
```

---

## 💡 Key Innovation: Gated Metapath Attention

### Traditional Attention vs Gated Mechanism

```python
# Traditional Attention (Node-level, 3 scalar weights)
attention_weights = [w1, w2, w3]  # Fixed weights
output = w1 * emb1 + w2 * emb2 + w3 * emb3

# Gated Mechanism (Feature-level, 256-dim vector)
gate_values = sigmoid(gate_network(concat_embeddings))  # [N, 256]
output = sum(embedding_i * gate_values)  # Fine-grained control
```

### Advantages

- ✅ **Fine-grained Control**: 256-dimensional gate values vs traditional 3 weights
- ✅ **Dynamic Adaptation**: Self-adaptive based on input
- ✅ **Global Information**: Considers all metapaths simultaneously
- ✅ **Better Performance**: +10.58% improvement over baseline

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gatedhgn.git
cd gatedhgn

# Create conda environment
conda create -n gatedhgn python=3.8
conda activate gatedhgn

# Install dependencies
pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.6.1
pip install -r requirements.txt
```

### Quick Prediction

```python
import torch
from hetero_gnn_model_v3 import create_model

# Load data
data = torch.load('data/hetero_graph.pt')

# Create model
model = create_model(
    data=data,
    hidden_channels=256,
    num_layers=3,
    num_heads=4,
    dropout=0.4
)

# Load pre-trained weights
checkpoint = torch.load('best_model_v3.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    output = model(data.x_dict, data.edge_index_dict)
    predictions = output.argmax(dim=1)
    probabilities = torch.softmax(output, dim=1)

print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")
```

### Run Demo

```bash
# Quick start script
python quick_start.py
```

**Output**:
```
================================================================================
HeteroGNN V3.0 - 快速开始
================================================================================

性能指标:
  准确率: 97.88%
  F1-Macro: 98.11%
  所有类别F1 > 97%

使用设备: cuda
正在加载数据: data/hetero_graph.pt
✓ 数据加载成功
  公司节点: 567
  行业节点: 45

✓ 模型加载成功
  训练轮数: 198
  准确率: 97.88%

前10个预测结果:
节点ID       预测类别       置信度
----------------------------------------
0          0          0.9961
1          2          0.9923
...
```

---

## 📁 Project Structure

```
gatedhgn/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── .gitignore                   # Git ignore file
├── LICENSE                      # MIT License
│
├── hetero_gnn_model_v3.py      # Core model implementation
├── quick_start.py              # Quick start script
├── train.py                    # Training script (optional)
│
├── best_model_v3.pt            # Pre-trained model (20MB)
├── results_v3.json             # Training results
│
├── data/
│   ├── hetero_graph.pt         # Graph data (3.3MB)
│   └── data_introduction.md    # Data documentation
│
├── docs/
│   ├── best_model.md           # Model documentation
│   ├── V3_README.md            # V3 details
│   ├── V3_TRAINING_REPORT.md   # Training report
│   ├── GATED_METAPATH_REPORT.md  # Technical analysis
│   └── GRAPHSMOTE_ANALYSIS.md  # GraphSMOTE study
│
└── examples/
    ├── basic_usage.py          # Basic usage example
    ├── custom_training.py      # Custom training example
    └── visualization.py        # Visualization example
```

---

## 📚 Documentation

### Core Documents

- **[Data Introduction](data/data_introduction.md)** - Detailed dataset documentation
- **[Model Documentation](best_model.md)** - Complete model guide
- **[V3 Training Report](docs/V3_TRAINING_REPORT.md)** - 200-epoch training analysis
- **[Gated Metapath Report](docs/GATED_METAPATH_REPORT.md)** - Technical deep dive
- **[GraphSMOTE Analysis](docs/GRAPHSMOTE_ANALYSIS.md)** - Why we don't use GraphSMOTE

### Model Components

| Component | Description | Location |
|-----------|-------------|----------|
| CompanyFeatureAttention | Self-attention for company features | `hetero_gnn_model_v3.py:20` |
| IndustryFeatureAttention | Self-attention for industry features | `hetero_gnn_model_v3.py:53` |
| CompanyIndustryAttention | Cross-attention between companies and industries | `hetero_gnn_model_v3.py:86` |
| GatedMetapathAttention | ⭐ Core innovation: gated metapath aggregation | `hetero_gnn_model_v3.py:122` |
| MetapathAggregator | Metapath-specific message passing | `hetero_gnn_model_v3.py:252` |
| HeteroGNNRiskModel | Main model class | `hetero_gnn_model_v3.py:324` |

---

## 💡 Examples

### Example 1: Load and Predict

```python
import torch
from hetero_gnn_model_v3 import create_model

# Load model
data = torch.load('data/hetero_graph.pt')
model = create_model(data, hidden_channels=256, num_layers=3)
checkpoint = torch.load('best_model_v3.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}

with torch.no_grad():
    output = model(x_dict, edge_index_dict)
    predictions = output.argmax(dim=1)

print(f"Predictions: {predictions}")
```

### Example 2: Analyze Gate Values

```python
# Forward pass
with torch.no_grad():
    output = model(x_dict, edge_index_dict)

# Get gate statistics
gate_stats = model.get_metapath_attention_weights()

print(f"Gate Statistics:")
print(f"  Mean: {gate_stats['mean']:.4f}")
print(f"  Std: {gate_stats['std']:.4f}")
print(f"  Range: [{gate_stats['min']:.4f}, {gate_stats['max']:.4f}]")
```

### Example 3: Custom Training

```python
from hetero_gnn_model_v3 import create_model
import torch.nn.functional as F

# Create model
model = create_model(data, hidden_channels=256, num_layers=3, num_heads=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# Training loop
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    output = model(data.x_dict, data.edge_index_dict)
    loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
```

---

## 📊 Dataset

### Graph Structure

```
Heterogeneous Graph (HeteroData)

Node Types:
  ├── company (567 nodes)
  │   └─ Features: [567, 10]
  └── industry (45 nodes)
      └─ Features: [45, 23]

Edge Types:
  ├── company → company (spillover): 164,351 edges
  ├── company → industry (belongs_to): 945 edges
  ├── industry → company (contains): 945 edges
  └── industry → industry (supply_chain): 83 edges

Total edges: 166,324
```

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| General | 157 | 27.7% |
| Excellent | 122 | 21.5% |
| Good | 237 | 41.8% |
| Poor | 51 | 9.0% |

**Imbalance Ratio**: 4.65:1 (mild imbalance)

### Metapaths

1. **Direct**: `company → company`
2. **Industry Mediated**: `company → industry → company`
3. **Supply Chain**: `company → industry → industry → company`

---

## 🔧 Configuration

### Model Architecture

```python
# Model parameters
hidden_channels = 256      # Hidden layer dimension
num_layers = 3             # Number of metapath aggregator layers
num_heads = 4              # Number of attention heads
dropout = 0.4              # Dropout rate

# Input dimensions
company_channels = 10      # Company feature dimension
industry_channels = 23     # Industry feature dimension
num_classes = 4            # Number of risk categories
```

### Training Configuration

```python
# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005,
    weight_decay=1e-4
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=20
)

# Training parameters
num_epochs = 200
early_stopping_patience = 50
gradient_clip = 1.0

# Random seeds
torch.manual_seed(42)
numpy.random.seed(42)
```

---

## 📈 Performance Comparison

### Version Comparison

| Version | Accuracy | F1-Macro | Features |
|---------|----------|----------|----------|
| V1.0 | 84.48% | 83.58% | Metapath bug |
| V2.0 | ~87.30% | ~86.80% | Fixed implementation |
| **V3.0** | **97.88%** | **98.11%** | **Gated mechanism** |

### Comparison with Paper Baseline

| Model | Accuracy | Improvement |
|-------|----------|-------------|
| Paper Baseline | 82.18% | - |
| **GatedHGN (V3.0)** | **97.88%** | **+15.70%** |

### Ablation Study

| Configuration | Accuracy | Epochs |
|---------------|----------|--------|
| V3.0 (no GraphSMOTE, 100 epochs) | 92.06% | 100 |
| **V3.0 (no GraphSMOTE, 200 epochs)** | **97.88%** | **200** |
| V3.0 (+GraphSMOTE, 73 epochs) | 76.30% | 73 |

**Key Findings**:
- ✅ Gated mechanism improves performance by +10.58%
- ✅ Sufficient training (200 epochs) is crucial
- ✅ GraphSMOTE is not needed for mild imbalance (4.65:1)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
isort .

# Type checking
mypy hetero_gnn_model_v3.py
```

---

## 📝 Citation

If you use GatedHGN in your research, please cite:

```bibtex
@software{gatedhgn2025,
  title={GatedHGN: A Gated Metapath Attention Heterogeneous Graph Neural Network for Risk Assessment},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/gatedhgn},
  version={3.0}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- PyTorch Geometric team for the excellent graph neural network library
- The original research paper for the baseline methodology
- Open-source community for various tools and libraries

---

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [https://github.com/yourusername/gatedhgn](https://github.com/yourusername/gatedhgn)
- **Issues**: [https://github.com/yourusername/gatedhgn/issues](https://github.com/yourusername/gatedhgn/issues)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star ⭐

<div align="center">

**Made with ❤️ for Graph Neural Network Research**

[⬆ Back to Top](#gatedhgn-new-energy-vehicle-risk-assessment)

</div>

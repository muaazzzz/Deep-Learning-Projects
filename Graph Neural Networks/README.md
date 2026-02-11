# PA6: Graph Neural Networks

This assignment explores Graph Neural Networks (GNNs) for learning on structured graph data, covering applications from maze navigation to molecular property prediction and traffic forecasting.

## 📋 Assignment Structure

### Part 1: GNN for Maze Navigation (75 Points)
**File**: `26100117_Part1.ipynb` / `26100117_Part1.py`

Using Graph Neural Networks to learn navigation policies in mazes.

#### Task 1: Maze Visualization & Optimal Paths
**Concepts**:
- **Graph Representation**: Maze as graph (cells = nodes, adjacency = edges)
- **Dijkstra's Algorithm**: Compute optimal paths
- **Path Visualization**: Display optimal routes

#### Task 2: Supervised Policy Learning with GraphSAGE (40 Points)
**Concepts**:
- **GraphSAGE**: Sample and aggregate neighborhood features
- **Message Passing**: Nodes exchange information with neighbors
- **Policy Learning**: Predict next move from current state
- **Imitation Learning**: Train GNN to mimic optimal policy

**Architecture**:
```
Input: Maze Graph + Current Position
GraphSAGE Layers:
  - Neighborhood sampling
  - Feature aggregation (mean, max, LSTM)
  - Multi-layer message passing
Output: Action probabilities (Up, Down, Left, Right)
```

**Training**:
- Generate optimal paths using Dijkstra
- Extract (state, action) pairs
- Train GNN classifier to predict actions
- Supervised learning with cross-entropy loss

#### Task 3: Greedy Rollout & Evaluation (30 Points)
**Concepts**:
- **Greedy Policy**: Always take highest probability action
- **Rollout**: Execute policy from start to goal
- **Performance Metrics**:
  - Success rate (reaches goal)
  - Path length vs. optimal
  - Generalization to new mazes

**Results**:
- GNN learns effective navigation
- Generalizes to unseen maze configurations
- Near-optimal paths in most cases

### Part 2: Graph Convolutional Networks (60 Points)
**File**: `26100117_Part2.ipynb` / `26100117_Part2.py`

Spectral graph convolutions and graph classification.

#### Task 1: Spectral Filtering on Custom Graphs (30 Points)
**Concepts**:
- **Graph Laplacian**: L = D - A (degree - adjacency)
- **Spectral Decomposition**: L = UΛU^T
- **Graph Filters**: f(L) applied to node features
- **Frequency Analysis**: Low-pass, high-pass, band-pass filters

**Test Graphs**:
1. Small connected graph
2. Disconnected components
3. Dense graph

**Filters Applied**:
- Identity filter
- Low-pass filter (smooth features)
- High-pass filter (detect boundaries)

#### Task 2: GCN with Laplacian Regularization (30 Points)
**Concepts**:
- **Graph Convolutional Layer**: H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
- **Laplacian Regularization**: Penalize feature variation across edges
- **MUTAG Dataset**: Molecular graphs classification (mutagenicity)

**Architecture**:
```
Input: Node Features X, Adjacency A
GCN Layer 1: Graph Conv → ReLU → Dropout
GCN Layer 2: Graph Conv → ReLU → Dropout
GCN Layer 3: Graph Conv
Global Pooling: Graph-level representation
Classifier: MLP → Output (2 classes)
```

**Loss Function**:
```
Total Loss = Classification Loss + λ * Laplacian Regularization
where: Laplacian Reg = Σ ||h_i - h_j||² for edges (i,j)
```

**MUTAG Dataset**:
- 188 molecular graphs
- Node features: Atom types (7 types)
- Edge features: Bond types
- Binary classification: Mutagenic or not

**Results**:
- **Final Training Accuracy**: 73.05% (Epoch 100)
- **Best Test Accuracy**: 84.21% (Epoch 100)
- Regularization improves generalization
- Prevents overfitting on small dataset

### Part 3: Spatio-Temporal GCN for Traffic Prediction (Variable Points)
**File**: `26100117_Part3.ipynb` / `26100117_Part3.py`

Combining spatial and temporal convolutions for time-series prediction on graphs.

**Concepts**:
- **Spatial Convolution**: GCN for spatial dependencies (road network)
- **Temporal Convolution**: 1D CNN or RNN for time series
- **ST-GCN**: Combine both for spatio-temporal modeling

**Architecture**:
```
Input: Graph (10×10 grid) + Time series features

Spatio-Temporal Block:
  Temporal Conv (1D) → ReLU
  Graph Conv (Spatial) → ReLU
  Temporal Conv (1D) → Layer Norm

Output: Predicted traffic at future timesteps
```

**Dataset**:
- 10×10 grid graph (100 nodes)
- Traffic measurements over time
- Spatial correlations (neighboring cells)
- Temporal patterns (rush hours, daily cycles)

**Task**:
- Predict future traffic from historical data
- Learn both spatial and temporal patterns
- Evaluate on held-out test set

## 📊 Results

### Part 1: Maze Navigation
- **GraphSAGE** successfully learns navigation policy
- Achieves high success rate on test mazes
- Path length close to optimal (within 10-20%)
- Generalizes to different maze sizes

### Part 2: Graph Classification
**Spectral Filtering**:
- Low-pass filters smooth node features
- High-pass filters detect graph structure changes
- Frequency analysis reveals graph properties

**GCN on MUTAG**:
- **Best Test Accuracy**: 84.21% (at Epoch 100)
- **Best Train Accuracy**: 73.05% (at Epoch 100)
- Laplacian regularization helps prevent overfitting
- Small dataset benefits from regularization

**Training Curve**:
```
Epoch 0:   Train Acc: 56.10%, Test Acc: 47.37%
Epoch 50:  Train Acc: 68.29%, Test Acc: 78.95%
Epoch 100: Train Acc: 73.05%, Test Acc: 84.21%
```

**Understanding the Results**:
The test accuracy exceeding train accuracy can occur with small datasets like MUTAG (188 graphs). This is due to:
- Random train/test split variations
- Regularization effects (Laplacian regularization)
- Small dataset statistics
This is normal and indicates good generalization rather than overfitting.

### Part 3: Traffic Prediction
- ST-GCN captures spatio-temporal patterns
- Outperforms spatial-only or temporal-only models
- Predictions align with daily traffic cycles
- Useful for real-world traffic forecasting

## 🚀 How to Run

### Prerequisites
```bash
# Core dependencies
pip install torch torchvision
pip install torch-geometric  # PyTorch Geometric for GNNs
pip install networkx  # Graph algorithms and visualization
pip install numpy matplotlib seaborn
```

### Installation
```bash
# PyTorch Geometric (version compatible with your PyTorch)
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

# Other dependencies
pip install -r requirements.txt
```

### Running the Assignments

1. **Part 1 - Maze Navigation**:
```bash
jupyter notebook 26100117_Part1.ipynb
```
Or:
```bash
python 26100117_Part1.py
```

2. **Part 2 - GCN & Spectral Methods**:
```bash
jupyter notebook 26100117_Part2.ipynb
```

3. **Part 3 - Traffic Prediction**:
```bash
jupyter notebook 26100117_Part3.ipynb
```

### Expected Outputs

**Part 1**:
- Maze visualizations with optimal paths
- Trained GraphSAGE model
- Rollout demonstrations
- Success rate and path length statistics

**Part 2**:
- Spectral filter visualizations
- Training/test accuracy curves
- Loss curves (classification + regularization)
- Learned node embeddings

**Part 3**:
- Traffic prediction plots
- Spatial heatmaps of traffic
- Temporal evolution visualizations
- MAE/RMSE metrics

## 🧪 Key Learnings

### Graph Neural Networks:
- ✅ Handle irregular, non-Euclidean data
- ✅ Inductive learning (generalize to new graphs)
- ✅ Interpretable via message passing
- 💡 Aggregation function matters (mean, max, attention)

### GraphSAGE:
- ✅ Scales to large graphs via sampling
- ✅ Inductive (doesn't need full graph at training)
- ✅ Flexible aggregators

### GCN:
- ✅ Spectral foundation (principled approach)
- ✅ Efficient matrix operations
- ❌ Transductive (needs full graph)
- 💡 Normalization crucial for training

### Spatio-Temporal Models:
- ✅ Capture complex dependencies
- ✅ Applicable to many domains (traffic, weather, social networks)
- 💡 Combining spatial and temporal is powerful

## 📁 Files Description

| File | Description |
|------|-------------|
| `26100117_Part1.ipynb` | Maze navigation with GraphSAGE |
| `26100117_Part1.py` | Part 1 script |
| `26100117_Part2.ipynb` | GCN and spectral methods |
| `26100117_Part2.py` | Part 2 script |
| `26100117_Part3.ipynb` | Traffic prediction ST-GCN |
| `26100117_Part3.py` | Part 3 script |

## 🎯 Applications

**GNNs in General**:
- Social network analysis
- Recommendation systems
- Drug discovery (molecular graphs)
- Knowledge graphs
- Program analysis (code graphs)

**GraphSAGE**:
- Large-scale graph learning
- Dynamic graphs
- Inductive tasks

**GCN**:
- Node classification
- Graph classification
- Link prediction

**ST-GCN**:
- Traffic forecasting
- Weather prediction
- Epidemic modeling
- Video analysis (skeleton graphs)

## 🔗 Dependencies

- **PyTorch**: Core deep learning
- **PyTorch Geometric**: GNN library with many layers and datasets
- **NetworkX**: Graph creation and algorithms
- **NumPy, Matplotlib**: Numerical computing and visualization
- **Scikit-learn**: Evaluation metrics

## 📚 References

- [GraphSAGE](https://arxiv.org/abs/1706.02216) - Inductive Representation Learning on Large Graphs
- [GCN](https://arxiv.org/abs/1609.02907) - Semi-Supervised Classification with Graph Convolutional Networks
- [Spectral Graph Theory](https://arxiv.org/abs/1312.6203) - Graph Signal Processing
- [ST-GCN](https://arxiv.org/abs/1801.07455) - Spatial Temporal Graph Convolutional Networks

## 💡 Tips for Running

- **MUTAG Dataset**: Automatically downloaded by PyTorch Geometric
- **Maze Generation**: Random mazes generated in code
- **GPU**: Recommended for Part 3, optional for Parts 1-2
- **Memory**: Graph operations can be memory-intensive for large graphs

---

**Training Time**: 
- Part 1: ~10-30 minutes
- Part 2: ~5-10 minutes (small dataset)
- Part 3: ~30-60 minutes depending on dataset size

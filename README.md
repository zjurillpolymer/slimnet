# SLIMNet

Scaling Law-Informed Machine Learning for predicting polymer properties.

Implementation of the paper: [Scaling law-informed machine learning for predicting thermal and electrical properties of polymers: A physics-based approach](https://doi.org/10.1016/j.commatsci.2025.113887) (Xu et al., Computational Materials Science, 2025)

## Results

### QM9 pre-training (normalized MSE)

| Encoder | Test loss |
|:---|---:|
| GATConv (2 layers) | 0.257 |
| **SchNet (6 layers)** | **0.062** |

SchNet with 3D coordinates significantly outperforms GAT on QM9 property prediction.

### PI1070 polymer property prediction (normalized MSE)

| Encoder → SLIMNet | Test loss |
|:---|---:|
| GATConv (frozen) | 0.386 |
| GATConv (fine-tune, lr=1e-5) | 0.374 |
| **SchNet (fine-tune, lr=1e-5)** | **0.057** |

SchNet-based SLIMNet achieves test loss of **0.057** (normalized MSE), a ~7× improvement over the GAT baseline, demonstrating the value of 3D molecular geometry in polymer property prediction.

## Approach

SLIMNet combines a graph neural network (GNN) with a physics-informed decoder built on polymer scaling laws:

```
QM9 pretraining → GNN → V_monomer → SLIMNet (scaling law + MLP) → polymer properties
```

- **Base Model**: GATConv or **SchNet** GNN pretrained on QM9 to predict monomer properties (HOMO, LUMO, dipole, polarizability)
- **Disordered phase**: Scaling law prediction `ϕ_disordered = α · β^γ`
- **Ordered phase**: Neural network prediction from order parameters
- **Final prediction**: `φ = ϕ_disordered + ϕ_ordered`

## Project Structure

```
base_model_molecule_encoder/
├── download_qm9.py              QM9 dataset download & exploration
├── gnn_model.py                 GAT model definition (for import)
├── monomer_predict_GNN.py       GAT QM9 pre-training script
├── Schnet_model_monomer.py      SchNet model + training script
└── best_schnet.pt               Pretrained SchNet weights

decoder/
├── data_loader.py               PI1070 polymer dataset loader
├── convert_molecule_graph.py    RDKit → PyG graph conversion
├── slimnet.py                   SLIMNet training & evaluation
└── plot_results.py              Training curves & pred-vs-true plots
```

## Data

- **QM9**: 134k small molecules for GNN pretraining (downloaded automatically by PyG)
- **PI1070**: 1077 amorphous polymers from RadonPy with MD-computed properties

## Usage

```bash
# 1. Download QM9 and pre-train GNN
python base_model_molecule_encoder/monomer_predict_GNN.py

# 2. Train SLIMNet on polymer data
python decoder/slimnet.py
```

## Reference

```
Xu et al., "Scaling law-informed machine learning for predicting thermal and
electrical properties of polymers: A physics-based approach",
Computational Materials Science 253 (2025) 113887.
```

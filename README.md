# SLIMNet

Scaling Law-Informed Machine Learning for predicting polymer properties.

Implementation of the paper: [Scaling law-informed machine learning for predicting thermal and electrical properties of polymers: A physics-based approach](https://doi.org/10.1016/j.commatsci.2025.113887) (Xu et al., Computational Materials Science, 2025)

## Approach

SLIMNet combines a graph neural network (GNN) with a physics-informed decoder built on polymer scaling laws:

```
QM9 pretraining → GNN → V_monomer → SLIMNet (scaling law + MLP) → polymer properties
```

- **Base Model**: GATConv-based GNN pretrained on QM9 to predict monomer properties (HOMO, LUMO, dipole, polarizability)
- **Disordered phase**: Scaling law prediction `ϕ_disordered = α · β^γ`
- **Ordered phase**: Neural network prediction from order parameters
- **Final prediction**: `φ = ϕ_disordered + ϕ_ordered`

## Project Structure

```
base_model_molecule_encoder/
├── download_qm9.py              QM9 dataset download & exploration
├── gnn_model.py                 GNN model definition (for import)
└── monomer_predict_GNN.py       QM9 pre-training script

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

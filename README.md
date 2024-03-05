# Why Attention Graphs Are All We Need: Pioneering Hierarchical Classification of Hematologic Cell Populations with LeukoGraph

This is the official PyTorch/PyG implementation for the paper "Why Attention Graphs Are All We Need: Pioneering Hierarchical Classification of Hematologic Cell Populations with LeukoGraph".

## Introduction

This paper introduces LeukoGraph, a novel graph attention network (GAT) based approach for hierarchical classification (HC) of hematologic cell populations from flow cytometry data. The key contributions are:

- Proposes a novel graph neural network framework, LeukoGraph, tailored for hierarchical multi-class classification problems. It uses a GAT as the base classification model.

- Introduces a max constraint loss function to exploit the hierarchical relationships during training. This prevents the model from getting stuck in spurious local optima.

- Achieves state-of-the-art performance in classifying 7 hematologic cell types arranged in a hierarchy. Outperforms DNNs and other graph learning methods like GCNs.

- Demonstrates high recall in identifying small but clinically significant cell populations like mast cells.

- Provides interpretability by identifying important cellular markers for classification.

## Installation

The code was developed with Python 3.8 and PyTorch 1.7. Install the dependencies using:

```
pip install -r requirements.txt
```

## Data

The flow cytometry dataset contains measurements for 12 cellular markers on bone marrow samples from 30 patients. On average, each patient's sample contains ~70,000 cells annotated into 7 cell types.

## Usage

The main model training, validation, and testing scripts are the following:

- `utils/data.py` - create the 30 CSV data from the original FCS, stored as *Case_{i}* in the `Data_hierarchical/` directory.
- `utils/graph_generation.py` - kNN patient-graphs construction from the saved data of `data.py`
- `utils/weight_generation.py` - weights generation for taking into account the strong class imbalance.
- `model/leukograph_main.py` - LeukoGraph model training/validation/testing with inductive learning and weighted BCE-MC loss function.
- `model/gnn_main.py` - GNN model training/validation/testing with inductive learning and weighted BCE-MC loss function.

## Models

- `LeukoGraph`: Proposed model. 
- `GCN`: Graph Convolutional Network.
- `GNN`: Graph Neural Network.  
- `DNN`: Deep Neural Network.

## Supplementary Material
- `suppl/LeukoGraph_suppl.pdf`: Supplementary explanations for coding the hierarchical structure and hierarchical loss in our proposed model, and discussion of time complexity.

## Citation

If you find this repository useful, please cite our paper:

```
@article{mojarrad2024attention,
  title={Why Attention Graphs Are All We Need: Pioneering Hierarchical Classification of Hematologic Cell Populations with LeukoGraph},
  author={Mojarrad, Fatemeh Nassajian and Bini, Lorenzo and Matthes, Thomas and Marchand-Maillet, St{\'e}phane},
  journal={arXiv preprint arXiv:2402.18610},
  year={2024}
}
```

## Contact

For any questions, please contact the authors or open an issue on GitHub. Data can be available upon request.

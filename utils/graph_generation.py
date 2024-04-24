'''
This script generates k-nearest neighbors graphs with k = 5 neighbors.
'''

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import itertools
import torch
from torch_geometric.data import Data
import os
import random
import networkx as nx

# Set seeds for reproducibility
seed_value = 77  
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

PRINT_MEMORY = False
device_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"

# Check if a GPU is available
if torch.cuda.is_available():
    # Get the current GPU device
    device = torch.cuda.current_device()
    
    # Get the GPU's memory usage in bytes
    memory_allocated = torch.cuda.memory_allocated(device)
    memory_cached = torch.cuda.memory_cached(device)
    
    # Convert bytes to a more human-readable format (e.g., megabytes or gigabytes)
    memory_allocated_mb = memory_allocated / 1024**2  # Megabytes
    memory_cached_mb = memory_cached / 1024**2  # Megabytes
    
    print(f"GPU Memory Allocated: {memory_allocated_mb:.2f} MB")
    print(f"GPU Memory Cached: {memory_cached_mb:.2f} MB")
else:
    print("No GPU available.")

to_skip = ['root']
ATTRIBUTE_class = "1,2,3,4,5,5_1,5_2,5_3"
g = nx.DiGraph()
for branch in ATTRIBUTE_class.split(','):
    term = branch.split('_')
    if len(term)==1:
        g.add_edge(term[0], 'root')
    else:
        for i in range(2, len(term) + 1):
            g.add_edge('.'.join(term[:i]), '.'.join(term[:i-1]))
nodes = sorted(g.nodes(), key=lambda x: (len(x.split('.')),x))
nodes_idx = dict(zip(nodes, range(len(nodes))))
g_t = g.reverse()
evall = [t not in to_skip for t in nodes]

AA = np.array(nx.to_numpy_array(g, nodelist=nodes))
R = np.zeros(AA.shape)
np.fill_diagonal(R, 1)
gg = nx.DiGraph(AA) # train.A is the matrix where the direct connections are stored 
for i in range(len(AA)):
    ancestors = list(nx.descendants(gg, i)) # here we need to use the function nx.descendants() because in the directed graph the edges have source from the descendant and point towards the ancestor 
    if ancestors:
        R[i, ancestors] = 1
R = torch.tensor(R)
#Transpose to get the descendants for each node 
R = R.transpose(1, 0)
R = R.unsqueeze(0).to(device)

K = 5
knn = NearestNeighbors(n_neighbors=K)
data_FC=[]
for j in range(30):
    # Data before min max normalization
    df = pd.read_csv(f"Data_hierarchical/Case_{j+1}.csv",dtype={"label": str})
    actual_labels = df[["label"]]
    labels = df.label.tolist()
    # Extract outputs
    Y = []
    for item in labels:
        y_ = np.zeros(len(nodes))
        for t in item.split('@'): 
            y_[[nodes_idx.get(a) for a in nx.ancestors(g_t, t.replace('_', '.'))]] =1
            y_[nodes_idx[t.replace('_', '.')]] = 1
        Y.append(y_)
    Y = np.stack(Y)
    #check the type
    df = df.drop('label', axis=1)
    # Select node features
    x = df[[ 'FS INT',  'SS INT', 'FL1 INT_CD14-FITC', 'FL2 INT_CD19-PE', 'FL3 INT_CD13-ECD', 'FL4 INT_CD33-PC5.5', 'FL5 INT_CD34-PC7', 'FL6 INT_CD117-APC', 'FL7 INT_CD7-APC700', 'FL8 INT_CD16-APC750', 'FL9 INT_HLA-PB', 'FL10 INT_CD45-KO']].to_numpy() # [num_nodes x num_features]
    knn.fit(x)
    # Extract edges
    all_edges = np.array([], dtype=np.int32).reshape((0, 2))
    for i in range(len(df)):
        p=knn.kneighbors([x[i]], return_distance=False)[0]
        # Build all combinations
        permutations = list(itertools.combinations(p[1:K], 2))
        edges_source = [e[0] for e in permutations]
        edges_target = [e[1] for e in permutations]
        edges = np.column_stack([edges_source, edges_target])
        all_edges = np.vstack([all_edges, edges])
    # Convert to Pytorch Geometric format
    edge_index = all_edges.transpose()
    edge_index # [2, num_edges]
    edge_index1 = e.t().clone().detach()
    for i in range(x.shape[1]):
        x[:,i] = (x[:,i] - x[:,i].min()) / (x[:,i].max() - x[:,i].min())
    data_FC.append(Data(x=torch.tensor(x, dtype=torch.float), edge_index=e.to().contiguous(), y=torch.tensor(Y),A=np.array(nx.to_numpy_array(g, nodelist=nodes)),terms=nodes,g=g,to_eval=evall,yy=list(actual_labels.to_numpy().flatten())))
torch.save(data_FC, 'graph_hierarchical_with_labels.pt')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Dataset
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import os
from sklearn.metrics import confusion_matrix
import random
import networkx as nx
import os
from utils.weight_generation import class_weights

# Set seeds for reproducibility
seed_value = 77  # You can use any integer value

# Set seed for random module
random.seed(seed_value)

# Set seed for NumPy
np.random.seed(seed_value)

# Set seed for PyTorch
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
print(nodes)

AA = np.array(nx.to_numpy_array(g, nodelist=nodes))
R = np.zeros(AA.shape)
np.fill_diagonal(R, 1)
gg = nx.DiGraph(AA) # train.A is the matrix where the direct connections are stored 
for i in range(len(AA)):
    ancestors = list(nx.descendants(gg, i)) #here we need to use the function nx.descendants() because in the directed graph the edges have source from the descendant and point towards the ancestor 
    if ancestors:
        R[i, ancestors] = 1
R = torch.tensor(R)
#Transpose to get the descendants for each node 
R = R.transpose(1, 0)
R = R.unsqueeze(0).to(device)


### graph with k=5 from kNN, and nodes have been min-max normalized ###
data_FC = torch.load('graph5_hierarchical_with_labels.pt') 

total_count=[]
for j in range(30):  
    df = pd.read_csv(f"Data_hierarchical/Case_{j+1}.csv", low_memory=False)
    
    total_count.append(len(df))


def get_constr_out(x, R):
    """ Given the output of the graph neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R """
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x),R.shape[1], R.shape[1])
    R_batch = R.expand(len(x),R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch*c_out.double(), dim = 2)
    return final_out

class MyGraphDataset(Dataset):
    def __init__(self,  num_samples,transform=None, pre_transform=None):
        super(MyGraphDataset, self).__init__(transform, pre_transform)
        self.num_samples = num_samples
        self.data_list = torch.load('graph5_hierarchical_with_labels.pt')   

    def len(self):
        return self.num_samples

    def get(self, idx):
        return self.data_list[idx]
    
class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads,concat_param,dropout_param):
        super(GATLayer, self).__init__()
        self.conv = GATConv(input_dim, output_dim , heads=num_heads,concat=concat_param, dropout=dropout_param)  # Adjusted output_dim
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x

class LeukoGraph(nn.Module):
    def __init__(self,R):
        super(LeukoGraph, self).__init__()
        self.R = R
        self.nb_layers = 2
        self.num_heads = 8
        self.out_head = 8
        self.hidden_dim = 64
        self.input_dim = 12
        self.output_dim= len(set(ATTRIBUTE_class.split(',')))+1  # We do not evaluate the performance of the model on the 'roots' node

        gat_layers = []
        for i in range(self.nb_layers):
            if i == 0:
                
                gat_layers.append(GATLayer(self.input_dim, self.hidden_dim, self.num_heads,True,0.4))
                
            elif i == self.nb_layers - 1:
              
                gat_layers.append(GATLayer(self.hidden_dim*self.num_heads, self.output_dim, self.out_head, False,0.4 ))
               
            else:
                
                gat_layers.append(GATLayer(self.hidden_dim*self.num_heads, self.hidden_dim, self.num_heads,True,0.4))
                
        self.gat_layers = nn.ModuleList(gat_layers)

        self.drop = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        self.f = nn.ReLU()
        self.reset_parameters()  # Initialize the weights

    def reset_parameters(self):
        for gat_layer in self.gat_layers:
            gat_layer.reset_parameters()

    def forward(self, data):
        x, edge_index= data.x, data.edge_index

        for i in range(self.nb_layers):
            x = self.gat_layers[i](x, edge_index)
            if i != self.nb_layers - 1:
                x = self.f(x)
                x = self.drop(x)
            else:
                x= self.sigmoid(x)
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R )  # Assuming get_constr_out is already defined
        return constrained_out

    
# One training epoch for the LeukoGraph model.
def train(train_loader, model, optimizer, device,criterion):
    model.train()
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)

        #MCLoss
        constr_output = get_constr_out(output, R)
        train_output = data.y *output.double()
        train_output = get_constr_out(train_output, R)
        train_output = (1-data.y)*constr_output.double() + data.y*train_output
        
        loss = criterion(train_output[:,data.to_eval[0]], data.y[:,data.to_eval[0]]) 

        predicted = constr_output.data > 0.4
 
        # Total number of labels
        total_train = data.y.size(0) * data.y.size(1)
        # Total correct predictions
        correct_train = (predicted == data.y.byte()).sum()

        loss.backward()
        optimizer.step()


# Get acc. of LeukoGraph model.
def test(loader, model, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        constrained_output = model(data)
        predss = constrained_output.data > 0.4

        correct += (predss == data.y.byte()).sum() / (predss.shape[0]*predss.shape[1])
    
    return correct / len(loader.dataset)



def gnn_evaluation(gnn, max_num_epochs, batch_size, start_lr, num_repetitions, min_lr=0.000001, factor=0.5, patience=5,all_std=True):
     '''
    Parameters:
    - max_num_epochs: Maximum number of training epochs
    - batch_size: Batch size for training and testing
    - start_lr: Initial learning rate for the optimizer
    - num_repetitions: Number of repetitions 
    - all_std: Boolean flag indicating whether to compute standard deviation of F1 scores across repetitions.

    Returns:
    - patient_dict: A dictionary containing hierarchical precision (hp), hierarchical recall (hr), hierarchical F-score (hf), and predicted labels for each patient
    '''
    dataset = MyGraphDataset(num_samples=len(torch.load('graph5_hierarchical_with_labels.pt'))).shuffle()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_model_state_dict = None
    # Add these lines before the cross-validation loop
    best_test_indices = None
    best_f1_score = 0.0
    patient_dict=dict()
    for i in range(num_repetitions):
        kf = KFold(n_splits=7, shuffle=True)
        dataset.shuffle()

        f1_scores = []
        acc_scores = []
        all_preds = []
        all_labels = []

        for train_index, test_index in kf.split(list(range(len(dataset)))):
            train_index, val_index = train_test_split(train_index, test_size=0.1)

            train_dataset = dataset[train_index.tolist()]
            val_dataset = dataset[val_index.tolist()]
            test_dataset = dataset[test_index.tolist()]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            num_patients = len(test_loader)
            print(f"Number of patients in the test loader: {num_patients}")

            model = gnn(R).to(device)
            model.reset_parameters()

            optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor,
                                                                   patience=patience, min_lr=0.0000001)
            criterion = nn.BCELoss(weight=class_weights_tensor)

            best_val_acc = 0.0
            best_test_acc = 0.0
            best_fold_f1_score = 0.0

            for epoch in range(1, max_num_epochs + 1):
                lr = scheduler.optimizer.param_groups[0]['lr']
                train(train_loader, model, optimizer, device,criterion)
                val_acc = test(val_loader, model, device)
                scheduler.step(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test(test_loader, model, device) * 100.0

            # Evaluation on the entire test set
            model.eval()
            preds = []
            labels = []
            for data in test_loader:
                model.eval()
                data = data.to(device)
                constrained_output = model(data)
                #predss = constrained_output.data > 0.5
                predss = constrained_output.data
       
                cell_preds=[]
                for row in predss:
                    #row_n=torch.cat((row[:5], row[6:]), axis = 0)
                    ind_max=row[:5].argmax().item()+1
                    if ind_max==5:
                        ind_n=row[6:].argmax().item()+1
                        cell_preds.append(str('5_')+str(ind_n))
                    else:
                        cell_preds.append(str(ind_max))
                    

                predsss = cell_preds
                labelss=data.yy[0]
                idx=total_count.index(len(labelss))+1
                # Compute confusion matrix, precision, recall, and F1 score
                alpha= [0] * len(predsss)
                beta= [0] * len(predsss)

                intersect = 0
                tot_alpha = 0
                tot_beta = 0

                total_K_label = 0
                true_K_label = 0
                for ii in range(len(predsss)):
                    
                    if labelss[ii] != '5_3':
                        if predsss[ii]=='5_1' or predsss[ii]=='5_2' or predsss[ii] == '5_3':
                            alpha[ii]=['root','5',predsss[ii]]
                        else:
                            alpha[ii]=['root',predsss[ii]]

                        if labelss[ii]=='5_1' or labelss[ii]=='5_2':
                            beta[ii]=['root','5',labelss[ii]]
                        else:
                            beta[ii]=['root',labelss[ii]]
                    elif (labelss[ii] == '5_1' or labelss[ii] == '5_2' or labelss[ii] == '5_3'):
                        total_K_label +=1
                        ind_max=predss[ii][:5].argmax().item()+1
                        if ind_max==5:
                            true_K_label +=1 
                            alpha[ii]=['root','5']
                            beta[ii]=['root','5']
                        else:
                            continue
                         
                        
                        

                    intersect += len(list(set(alpha[ii]) & set(beta[ii])))
                    tot_alpha  += len(alpha[ii])
                    tot_beta  += len(beta[ii])
                
                hP =  intersect /  tot_alpha
                hR =  intersect /  tot_beta
                hF = 2* hP * hR / (hP + hR)
                #precision, recall, f1, _ = precision_recall_fscore_support(labelss, predsss, average='weighted', zero_division=1) 
                
                if idx not in patient_dict.keys():
                    patient_dict[idx]=dict()
                    patient_dict[idx]['precision']=[hP]
                    patient_dict[idx]['recall']=[hR]
                    patient_dict[idx]['F-score']=[hF]
                    patient_dict[idx]['pred']=[predsss]
                    patient_dict[idx]['label']=[labelss]
                    patient_dict[idx]['K_label']=[true_K_label / total_K_label]
                else:
                    patient_dict[idx]['precision'].append(hP)
                    patient_dict[idx]['recall'].append(hR)
                    patient_dict[idx]['F-score'].append(hF)
                    patient_dict[idx]['pred'].append(predsss)
                    patient_dict[idx]['label'].append(labelss)
                    patient_dict[idx]['K_label'].append(true_K_label / total_K_label)

        
            
    
    return patient_dict

max_num_epochs=40
batch_size=1
start_lr=0.1
num_repetitions=5
patient_dict=gnn_evaluation(LeukoGraph, max_num_epochs, batch_size, start_lr, num_repetitions, all_std=True)

# Initialize a list to store the ratios for each label across all patients
average_ratio_per_label = []

F_scores = np.zeros((30,num_repetitions))
no = 0
for key in patient_dict.keys():
    F_scores[no,:] = patient_dict[key]['F-score']
    no += 1

F_scores_mean = F_scores.mean(axis = 0)
idx = F_scores_mean.argmax(axis=0)

for key in patient_dict.keys():
    #idx=patient_dict[key]['F-score'].index(max(patient_dict[key]['F-score']))
    df=pd.read_csv(f"Data_hierarchical/Case_{key}.csv", low_memory=False)  # Set low_memory=False to fix the warning
    df['predicted label']=patient_dict[key]['pred'][idx]
    if not os.path.exists("Data_hierarchical_predicted"):
        os.makedirs("Data_hierarchical_predicted")

    df.to_csv(f"Data_hierarchical_predicted/Case_{key}.csv", index=False)
    
     # Compute metrics for each patient
    conf_matrix = confusion_matrix(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx])[:-1,:-1]
    precision = patient_dict[key]['precision'][idx]
    recall = patient_dict[key]['recall'][idx]
    f1 = patient_dict[key]['F-score'][idx]
    
    print(f"Metrics for Patient {key}:")   
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
     # Calculate ratio  of correct predictions for each label
    total_right_cells = np.sum(np.diag(conf_matrix))
    ratio_per_label = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)

    for i, label in enumerate(range(conf_matrix.shape[0])):
        print(f"Label {label}:")
        print(f"Ratio of Correct Predictions: {ratio_per_label[i]:.4f}")

        # Add the ratio to the list for computing the average later
        
        if len(average_ratio_per_label) <= i:
            average_ratio_per_label.append([ratio_per_label[i]])
        else:
            average_ratio_per_label[i].append(ratio_per_label[i])
    print(f"Label 6:")
    print(f"Ratio of Correct Predictions: {patient_dict[key]['K_label'][idx]:.4f}")
   
    if len(average_ratio_per_label) <= 6:
        average_ratio_per_label.append([patient_dict[key]['K_label'][idx]])
    else:
        average_ratio_per_label[6].append(patient_dict[key]['K_label'][idx])
    print("-" * 50)

# Calculate the average ratio for each label across all patients
average_ratio_per_label = np.mean(average_ratio_per_label, axis=1)

# Print the average ratios 
print("\nAverage Ratios Across All Patients:")
label_dict = {0: 'O', 1: 'N', 2: 'G', 3: 'P', 4: 'M', 5: 'L', 6: 'K'}
for i, average_ratio in enumerate(average_ratio_per_label):
    print(f"Label {label_dict[i]}: {average_ratio:.4f}")

import torch
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import os


def adj2coo(adj_matrix, weight_threshold=None):
    # create some basic code for getting COO format from the adjacency matrix
    if weight_threshold is not None:
        rows, cols = np.where(np.absolute(adj_matrix)>weight_threshold)
    else:
        rows, cols = np.where(adj_matrix!=0)
    coo_matrix = np.array([rows, cols]).T
    # the matrix should be ordered but sort to ensure that is the case
    sorted_indices = np.lexsort((coo_matrix[:,1], coo_matrix[:,0]))
    coo_matrix = coo_matrix[sorted_indices]
    # grab weights based on the indices
    ind_rows = coo_matrix[:,0]
    ind_cols = coo_matrix[:,1]
    edge_features = adj_matrix[ind_rows, ind_cols]
    # get into correct shape and convert into tensor
    coo_tensor = torch.tensor(coo_matrix.T, dtype=torch.long)
    edge_tensor = torch.tensor(edge_features.reshape(-1,1), dtype=torch.float)
    
    return coo_tensor, edge_tensor

def get_connection_profile(adj_matrix, weight_threshold=0.025):
    adj_matrix[adj_matrix<weight_threshold] = 0
    adj_matrix = torch.tensor(adj_matrix).float()

    return adj_matrix

def get_dataset(all_filenames, base_dir, label_dir, weight_threshold=0.025, connection_profile=True):
    # record the files that we are actually going to use for training and inference
    graph_list = []
    for input_filename in all_filenames:
        initial_matrix = np.load(os.path.join(base_dir, input_filename))
        initial_matrix = np.absolute(initial_matrix)
        if 'gender' in label_dir:
            label = np.load(os.path.join(label_dir, input_filename)).item()
            label = torch.tensor(label)
        else: 
            label = np.load(os.path.join('graph_labels/bs', input_filename))
            label = torch.tensor(label, dtype=torch.float)
        coo_indices, edge_features = adj2coo(initial_matrix, weight_threshold)
        # using connection profile insetad of identity
        if connection_profile:
            node_features = get_connection_profile(initial_matrix, weight_threshold)
        else:
            node_features = torch.ones((82, 1), dtype=torch.float)
        data = Data(x=node_features, edge_index=coo_indices, edge_weight=edge_features, y=label)
        if data.edge_weight.shape[0]!=data.edge_index.shape[1]:
            raise Exception('Shapes do not match between edge indices and edge features!') 
        graph_list.append(data)
    
    return graph_list

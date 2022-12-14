import torch
import torch.nn as nn
from torch.utils.data import Subset
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from captum.attr import Saliency, IntegratedGradients
import os
import argparse

from graph_models import GCN_model, SagPoolGCN_model
from dataset_split import get_train_test_split
from dataset import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, choices={'regression', 'classification'}, help='type of machine learning task')
parser.add_argument('--data_dir', type=str, default='/ix/yufeihuang/Hugh', help='Parent directory that your data is stored in')
parser.add_argument('--home_dir', type=str, default='/ihome/yufeihuang/hug18/brain_graph_project', help='Parent directory that your data is stored in')
parser.add_argument('--modality', type=str, choices={'modal1', 'modal2'}, help='graph modality being used')
parser.add_argument('--model_subtype', type=str, default='GCN', choices={'SAGPool', 'GCN'}, help='graph modality being used')
parser.add_argument('--node_type', type=str, default='connection_profile', choices={'identity', 'connection_profile'}, help='type of node feature you want to use')
# just using zero and one to avoid possible errors when running classification
parser.add_argument('--class_instance', type=int, default=0, choices={0, 1}, help='The class you want to get attributions for ')
args = parser.parse_args()
task_type = args.task
data_dir = args.data_dir
home_dir = args.home_dir
modality = args.modality
model_subtype = args.model_subtype
node_type = args.node_type
class_instance = args.class_instance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# these are some simple functions to make getting the attributions easier - we are only performing attribution on test set predictions
def coo2adj(batch, edge_mask):
    adj_base = np.zeros((82,82))
    edge_numpy = data.edge_index.detach().cpu().numpy()
    adj_base[edge_numpy[1][:], edge_numpy[0][:]] = edge_mask
    return adj_base

def get_class_instances(model, loader, run_device, class_instance=0):
    instance_list = []
    model.eval()
    for data in loader:
        data = data.to(run_device)
        out = model(data.x, data.edge_index, data.edge_weight, data.batch)
        pred = out.argmax(dim=1)
        if pred.item() == class_instance:
            instance_list.append(data)
            
    return instance_list

def model_forward(edge_mask, data):
    data = data.to(device)
    batch = torch.zeros(data.x.shape[0], dtype=int).to(device)
    out = model(data.x, data.edge_index, edge_mask, batch)
    return out

def explain(method, data, target=1):
    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)
    if method == 'ig':
        ig = IntegratedGradients(model_forward)
        mask = ig.attribute(input_mask, target=target,
                            additional_forward_args=(data,),
                            internal_batch_size=data.edge_index.shape[1])
    elif method == 'saliency':
        saliency = Saliency(model_forward)
        mask = saliency.attribute(input_mask, target=target,
                                  additional_forward_args=(data,))
    else:
        raise Exception('Unknown explanation method')

    edge_mask = np.abs(mask.cpu().detach().numpy())
    if edge_mask.max() > 0:  # avoid division by zero
        edge_mask = edge_mask / edge_mask.max()
    return edge_mask

os.chdir(data_dir)

graphs_dir = os.path.join(data_dir, f'{modality}_data/raw')

if node_type == 'identity':
    c_profile = False
else:
    c_profile = True
    
_, test_files = get_train_test_split(graphs_dir, task_type, test=True)

if task_type == 'classification':
    label_dir = 'graph_labels/gender'
elif task_type == 'regression':
    label_dir = 'graph_labels/bs'

test_list = get_dataset(test_files, graphs_dir, label_dir, connection_profile=c_profile)
test_dataloader = DataLoader(test_list, batch_size=1, shuffle=False)

# load the model
if task_type == 'classification' and c_profile:
    model = GCN_model(82,64,2)
elif task_type == 'classification' and not c_profile:
    model = GCN_model(1,64,2)
elif task_type == 'regression' and c_profile:
    model = GCN_model(82,64,10)
elif task_type == 'regression' and not c_profile:
    model = GCN_model(1,64,10)

if c_profile:
    reg_path = os.path.join(home_dir, f'models/saved_best_models/GCN_{task_type}_{modality}_new.pt')
else:
    reg_path = os.path.join(home_dir, f'models/saved_best_models/GCN_{task_type}_{modality}.pt')

if device.type != 'cuda':
    model.load_state_dict(torch.load(reg_path, map_location=torch.device('cpu')))
else: 
    model.load_state_dict(torch.load(reg_path))

model.to(device)

# if the model is a classification model we need to grab samples that are classes of interest
if task_type == 'classification':
    data_list = get_class_instances(model, test_dataloader, device, class_instance)
    test_dataloader = test_dataloader = DataLoader(test_list, batch_size=1, shuffle=False)
# if the model is a regression model we can just do the attribution on every test sample
adj_list = []
for data in test_dataloader:
    edge_mask = explain('ig', data, target=class_instance)
    adj_attr = coo2adj(data, edge_mask)
    adj_list.append(adj_attr)

mean_attr = np.mean(adj_list, axis=0)

# grab scores
top_indices = np.argwhere(np.isin(mean_attr, np.sort(mean_attr, axis=None)))
top_scores = mean_attr[top_indices.T[0,:], top_indices.T[1,:]]
score_frame = pd.DataFrame([top_indices.T[0,:], top_indices.T[1,:], top_scores], index=['Adjacency Matrix Index Y', 'Adjacency Matrix Index X', 'Mean IG Score']).T
score_frame = score_frame.sort_values(by=['Mean IG Score'], ascending=False)

os.chdir(home_dir)
# aggregate and present the final matrix
fig, ax = plt.subplots(figsize=(5,5), dpi=300)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax.imshow(mean_attr, cmap='hot', aspect='auto')
fig.colorbar(im, cax=cax,orientation='vertical')
fig.savefig(f'figures/attribution_{model_subtype}_{node_type}_{task_type}_{class_instance}.png', dpi=450)
score_frame.to_csv(f'figures/attribution_{model_subtype}_{node_type}_{task_type}_{class_instance}_scores.tsv', sep='\t')

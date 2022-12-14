import torch
from torch.utils.data import Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
from torch_geometric.loader import DataLoader
from graph_models import GCN_model, SagPoolGCN_model, Meta_model, Identity
from dataset import *
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

from dataset_split import get_train_test_split
from dataset import get_dataset
from train import get_preds

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, choices={'regression', 'classification'}, help='type of machine learning task')
parser.add_argument('--data_dir', type=str, default='/ix/yufeihuang/Hugh', help='Parent directory that your data is stored in')
parser.add_argument('--home_dir', type=str, default='/ihome/yufeihuang/hug18/brain_graph_project', help='Parent directory that your data is stored in')
parser.add_argument('--modality', type=str, choices={'modal1', 'modal2', 'multimodality'}, help='graph modality being used')
parser.add_argument('--model_subtype', type=str, default='SAGPool', choices={'SAGPool', 'meta_model', 'GCN'}, help='graph modality being used')
parser.add_argument('--node_type', type=str, default='connection_profile', choices={'identity', 'connection_profile'}, help='type of node feature you want to use')
parser.add_argument('--model_name', type=str, help='What you want the model name to be in the figures')
args = parser.parse_args()
task_type = args.task
data_dir = args.data_dir
home_dir = args.home_dir
modality = args.modality
model_subtype = args.model_subtype
node_type = args.node_type
model_name = args.model_name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# load the testing dataset
os.chdir(data_dir)
if modality != 'multimodality':
    graphs_dir = os.path.join(data_dir, f'{modality}_data/raw')
else:
    graphs_dir = os.path.join(data_dir, f'modal1_data/raw')

if node_type == 'identity':
    c_profile = False
else:
    c_profile = True
    
_, test_files = get_train_test_split(graphs_dir, task_type, test=True)

if task_type == 'classification':
    label_dir = 'graph_labels/gender'
elif task_type == 'regression':
    label_dir = 'graph_labels/bs'

if modality == 'multimodality':
    modal1_dir = os.path.join(data_dir, 'modal1_data/raw')
    modal2_dir = os.path.join(data_dir, 'modal2_data/raw')
    modal1_test = get_dataset(test_files, modal1_dir, label_dir, connection_profile=c_profile)
    modal2_test = get_dataset(test_files, modal2_dir, label_dir, connection_profile=c_profile)

    modal1_test_loader = DataLoader(modal1_test, batch_size=32, shuffle=False)
    modal2_test_loader = DataLoader(modal2_test, batch_size=32, shuffle=False)

    test_dataloader = (modal1_test_loader, modal2_test_loader)
    
else:
    test_list = get_dataset(test_files, graphs_dir, label_dir, connection_profile=c_profile)
    test_dataloader = DataLoader(test_list, batch_size=32, shuffle=False)

# load the models
if model_subtype == 'GCN':
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

elif model_subtype == 'SAGPool':
    if task_type == 'classification' and c_profile:
        model = SagPoolGCN_model(82,64,2)
    elif task_type == 'classification' and not c_profile:
        model = SagPoolGCN_model(1,64,2)
    elif task_type == 'regression' and c_profile:
        model = SagPoolGCN_model(82,64,10)
    elif task_type == 'regression' and not c_profile:
        model = SagPoolGCN_model(1,64,10)
    
    if c_profile:
        reg_path = os.path.join(home_dir, f'models/saved_best_models/SAGPool_{task_type}_{modality}_new.pt')
    else:
        reg_path = os.path.join(home_dir, f'models/saved_best_models/SAGPool_{task_type}_{modality}.pt')
        
    if device.type != 'cuda':
        model.load_state_dict(torch.load(reg_path, map_location=torch.device('cpu')))
    else: 
        model.load_state_dict(torch.load(reg_path))
        
    model.to(device)


elif model_subtype == 'meta_model':
    if modality == 'multimodality':
        if task_type == 'classification':
            reg_path = os.path.join(home_dir, f'models/saved_best_models/meta_model_classification_multimodality_base.pt')
            model1 = GCN_model(1,64,2)
            model2 = GCN_model(1,64,2)
            model1.lin = Identity()
            model2.lin = Identity()
            model = Meta_model(64, model1, model2, 2, 2, 1)
            if device.type != 'cuda':
                model.load_state_dict(torch.load(reg_path, map_location=torch.device('cpu')))
            else: 
                model.load_state_dict(torch.load(reg_path))
            model = model.to(device)

        elif task_type == 'regression':
            reg_path = os.path.join(home_dir, f'models/saved_best_models/meta_model_regression_multimodality_base.pt')
            model1 = SagPoolGCN_model(1,64,10)
            model2 = SagPoolGCN_model(1,64,10)
            model1.lin1 = Identity()
            model1.out = Identity()
            model2.lin = Identity()
            model2.out = Identity()
            model = Meta_model(64, model1, model2, 10, 4, 3)
            if device.type != 'cuda':
                model.load_state_dict(torch.load(reg_path, map_location=torch.device('cpu')))
            else: 
                model.load_state_dict(torch.load(reg_path))
            model = model.to(device)

    else:
        if task_type == 'classification':
            reg_path = os.path.join(home_dir, f'models/saved_best_models/meta_model_classification_modal1_new.pt')
            model1 = SagPoolGCN_model(82,64,2)
            model2 = GCN_model(82,64,2)
            # remove the linear layers from the model
            model1.lin1 = Identity()
            model1.out = Identity()
            model2.lin = Identity()
            model2.out = Identity()
            model = Meta_model(64, model1, model2, 2, 4, 3)
            if device.type != 'cuda':
                model.load_state_dict(torch.load(reg_path, map_location=torch.device('cpu')))
            else: 
                model.load_state_dict(torch.load(reg_path))
            model = model.to(device)

# grab the predictions
if task_type == 'classification':
    if model_subtype == 'meta_model' and modality == 'multimodality':
        predicted_values, ground_truth = get_preds(model, test_dataloader, device, classification=True, multimodality=True)
    elif model_subtype =='meta_model':
        predicted_values, ground_truth = get_preds(model, test_dataloader, device, classification=True, meta=True)
    else:
        predicted_values, ground_truth = get_preds(model, test_dataloader, device, classification=True)
else:
    if modality == 'multimodality':
        predicted_values, ground_truth = get_preds(model, test_dataloader, device, multimodality=True)
    else:
        predicted_values, ground_truth = get_preds(model, test_dataloader, device)

    
# get relevant metrics and visualizations
if task_type == 'regression':
    # get regression metrics
    mae_val = mean_absolute_error(ground_truth, predicted_values)
    rmse_val = mean_squared_error(ground_truth, predicted_values, squared=False)
    mse_val = mean_squared_error(ground_truth, predicted_values)
    print(mae_val, rmse_val, mse_val)
    # save the metrics
    os.chdir(home_dir)
    metric_dict = {'Test Set MAE': mae_val, 'Test Set RMSE': rmse_val, 'Test Set MSE': mse_val}
    # pd.Series(metric_dict).to_csv(f'figures/{model_subtype}_{task_type}_{modality}_inference_metrics.tsv', sep='\t')

elif task_type == 'classification':
    # get classification metrics
    precision = precision_score(ground_truth, predicted_values)
    recall = recall_score(ground_truth, predicted_values)
    f1_result = f1_score(ground_truth, predicted_values)
    acc_result = accuracy_score(ground_truth, predicted_values)
    cm = confusion_matrix(ground_truth, predicted_values)
    cm_frame = pd.DataFrame(cm, index=[0,1], columns=[0,1])
    # save the confusion matrix
    os.chdir(home_dir)
    metric_dict = {'Test Precision': precision, 'Test Recall': recall, 'Test F1 Score': f1_result, 'Test Accuracy': acc_result}
    print(metric_dict)
    print(cm_frame)
    # pd.Series(metric_dict).to_csv(f'figures/{model_subtype}_{task_type}_{modality}_inference_metrics_new.tsv', sep='\t')
    fig, axes = plt.subplots(1, 1, tight_layout=True, facecolor=(1, 1, 1), figsize = (5,5), dpi=200)
    plt.title(f'Model: {model_name}')
    sns.heatmap(cm_frame, fmt='.0f', annot=True, annot_kws={"size": 10}, ax=axes)
    # fig.savefig(f'figures/{model_subtype}_{modality}_{task_type}_confusion_matrix_new.svg', bbox_inches='tight', dpi=350, transparent=False)
    plt.show()
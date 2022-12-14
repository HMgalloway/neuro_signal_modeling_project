import numpy as np
import pandas as pd
import argparse
import os
from torch_geometric.loader import DataLoader
from dataset_split import get_train_test_split
from dataset import get_dataset
from train import start_training

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, required=True, help='The number of epochs to use in training')
parser.add_argument('--data_dir', type=str, default='/ix/yufeihuang/Hugh', help='Parent directory that your data is stored in')
parser.add_argument('--home_dir', type=str, default='/ihome/yufeihuang/hug18/brain_graph_project', help='Parent directory that your data is stored in')
parser.add_argument('--task', type=str, choices={'regression', 'classification'}, help='type of machine learning task')
parser.add_argument('--modality', type=str, choices={'modal1', 'modal2', 'multimodality'}, help='graph modality being used')
parser.add_argument('--model_subtype', type=str, default='SAGPool', choices={'SAGPool', 'meta_model', 'GCN'}, help='graph modality being used')
parser.add_argument('--node_type', type=str, default='connection_profile', choices={'identity', 'connection_profile'}, help='type of node feature you want to use')
args = parser.parse_args()
total_epochs = args.num_epochs
data_dir = args.data_dir
home_dir = args.home_dir
task_type = args.task
modality = args.modality
model_subtype = args.model_subtype
node_type = args.node_type

if __name__=='__main__':

    os.chdir(data_dir)
    if modality != 'multimodality':
        graphs_dir = os.path.join(data_dir, f'{modality}_data/raw')
    else:
        graphs_dir = os.path.join(data_dir, f'modal1_data/raw')

    if node_type == 'identity':
        c_profile = False
    else:
        c_profile = True
    print('Splitting the dataset')

    train_files, test_files = get_train_test_split(graphs_dir, task_type)

    print('Processing the split dataset')
    if task_type == 'classification':
        label_dir = 'graph_labels/gender'
    elif task_type == 'regression':
        label_dir = 'graph_labels/bs'
    #  the list of files is the same for modal1 and modal2 so we can use one list to get both datasets
    if modality == 'multimodality':
        modal1_dir = os.path.join(data_dir, 'modal1_data/raw')
        modal2_dir = os.path.join(data_dir, 'modal2_data/raw')
        modal1_train = get_dataset(train_files, modal1_dir, label_dir, connection_profile=c_profile)
        modal1_test = get_dataset(test_files, modal1_dir, label_dir, connection_profile=c_profile)
        modal2_train = get_dataset(train_files, modal2_dir, label_dir, connection_profile=c_profile)
        modal2_test = get_dataset(test_files, modal2_dir, label_dir, connection_profile=c_profile)

        modal1_train_loader = DataLoader(modal1_train, batch_size=32, shuffle=False)
        modal1_test_loader = DataLoader(modal1_test, batch_size=32, shuffle=False)
        modal2_train_loader = DataLoader(modal2_train, batch_size=32, shuffle=False)
        modal2_test_loader = DataLoader(modal2_test, batch_size=32, shuffle=False)

        train_dataloader = (modal1_train_loader, modal2_train_loader)
        test_dataloader = (modal1_test_loader, modal2_test_loader)

    else:
        train_list = get_dataset(train_files, graphs_dir, label_dir, connection_profile=c_profile)
        test_list = get_dataset(test_files, graphs_dir, label_dir, connection_profile=c_profile)
        train_dataloader = DataLoader(train_list, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_list, batch_size=32, shuffle=False)

    print('Starting the training')
    start_training(total_epochs, train_dataloader, test_dataloader, task_type, model_subtype, modality, home_dir, c_profile)
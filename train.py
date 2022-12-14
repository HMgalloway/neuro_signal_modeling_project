import torch
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from graph_models import GCN_model, SagPoolGCN_model, Meta_model, Identity


def get_preds(model, loader, run_device, classification=False, meta=False, multimodality=False):
    pred_list, gt_list = [], []
    model.eval()
    correct = 0
    if multimodality:
        modal1_loader, modal2_loader = loader
        for data1, data2 in zip(modal1_loader, modal2_loader):
            data1 = data1.to(device)
            data2 = data2.to(device)
            out = model(data1, data2)
            if classification:
                pred = out.argmax(dim=1)
            else:
                pred = out
            data1.y = data1.y.reshape((out.shape[0],-1))
            pred_list.append(pred.detach().cpu().numpy())
            gt_list.append(data1.y.detach().cpu().numpy())
    else:
        for data in loader:
            data = data.to(run_device)
            if meta:
                out = model(data, data)
            else:
                out = model(data.x, data.edge_index, data.edge_weight, data.batch)
            if classification:
                pred = out.argmax(dim=1)
            else:
                pred = out
            data.y = data.y.reshape((out.shape[0],-1))
            pred_list.append(pred.detach().cpu().numpy())
            gt_list.append(data.y.detach().cpu().numpy())
            
    pred_arr = np.concatenate(pred_list)
    gt_arr = np.concatenate(gt_list)
    return pred_arr, gt_arr


def train_step_classification(loader, model, device, optimizer, criterion, meta=False, multimodal=False):
    model.train()
    if multimodal:
        modal1_loader, modal2_loader = loader
        for (data1, data2) in zip(modal1_loader, modal2_loader):
            data1 = data1.to(device)
            data2 = data2.to(device)
            out = model(data1, data2)
            loss = criterion(out, data1.y)
            loss.backward()
            optimizer.step()
    else:
        for data in loader:
            data = data.to(device)
            # print(data.x.shape)
            optimizer.zero_grad()
            if meta:
                out = model(data, data)
            else:
                out = model(data.x, data.edge_index, data.edge_weight, data.batch)
            # print(out.shape)
            #print(out)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

def train_step_regression(loader, model, device, optimizer, criterion, multimodal=False):
    model.train()
    if multimodal:
        modal1_loader, modal2_loader = loader
        for (data1, data2) in zip(modal1_loader, modal2_loader):
            data1 = data1.to(device)
            data2 = data2.to(device)
            out = model(data1, data2)
            data1.y = data1.y.reshape((out.shape[0],-1))
            loss = criterion(out, data1.y)
            loss.backward()
            optimizer.step()
    else: 
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_weight, data.batch)
            data.y = data.y.reshape((out.shape[0],-1))
            loss = criterion(torch.squeeze(out), data.y)
            loss.backward()
            optimizer.step()
        
def test_step_classification(loader, model, device, optimizer, criterion, meta=False, multimodal=False):
    model.eval()
    correct = 0
    test_loss = 0
    if multimodal:
        modal1_loader, modal2_loader = loader
        for data1, data2 in zip(modal1_loader, modal2_loader):
            data1 = data1.to(device)
            data2 = data2.to(device)
            out = model(data1, data2)
            test_loss += criterion(out, data1.y)
            pred = out.argmax(dim=1)
            correct += int((pred==data1.y).sum())
        return test_loss/len(modal1_loader.dataset), correct/len(modal1_loader.dataset)
    else: 
        for data in loader:
            data = data.to(device)
            if meta:
                out = model(data, data)
            else:
                out = model(data.x, data.edge_index, data.edge_weight, data.batch)
            test_loss += criterion(out, data.y)
            pred = out.argmax(dim=1)
            correct += int((pred==data.y).sum())
        return test_loss/len(loader.dataset), correct/len(loader.dataset)

def test_step_regression(loader, model, device, optimizer, criterion, multimodal=False):
    model.eval()
    test_mse = 0
    if multimodal:
        modal1_loader, modal2_loader = loader
        for data1, data2 in zip(modal1_loader, modal2_loader):
            data1 = data1.to(device)
            data2 = data2.to(device)
            out = model(data1, data2)
            data1.y = data1.y.reshape((out.shape[0],-1))
            test_mse += criterion(out, data1.y)
        return test_mse/len(modal1_loader.dataset)
    else:
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_weight, data.batch)
            pred = out.argmax(dim=1)
            data.y = data.y.reshape((out.shape[0],-1))
            test_mse += criterion(out, data.y)
        return test_mse/len(loader.dataset)

def start_training(max_epochs, train_loader, test_loader, task_type, model_subtype, modality, home_dir, connection_profile=True):
    # either using cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    # select a model
    if model_subtype == 'GCN':
        if task_type == 'classification' and connection_profile:
            model = GCN_model(82,64,2)
        elif task_type == 'classification' and not connection_profile:
            model = GCN_model(1,64,2)
        elif task_type == 'regression' and connection_profile:
            model = GCN_model(82,64,10)
        elif task_type == 'regression' and not connection_profile:
            model = GCN_model(1,64,10)
        model.to(device)

    elif model_subtype == 'SAGPool':
        if task_type == 'classification' and connection_profile:
            model = SagPoolGCN_model(82,64,2)
        elif task_type == 'classification' and not connection_profile:
            model = SagPoolGCN_model(1,64,2)
        elif task_type == 'regression' and connection_profile:
            model = SagPoolGCN_model(82,64,10)
        elif task_type == 'regression' and not connection_profile:
            model = SagPoolGCN_model(1,64,10)
        model.to(device)

    # training for the multimodal models are in a different file
    elif model_subtype == 'meta_model':
        if modality == 'multimodality':
            if task_type == 'classification':
                # TODO currently working on cpu remove the mapping lines if working on gpu
                reg1_path = os.path.join(home_dir, f'models/saved_best_models/GCN_{task_type}_modal1.pt')
                reg2_path = os.path.join(home_dir, f'models/saved_best_models/GCN_{task_type}_modal2.pt')
                model1 = GCN_model(1,64,2)
                model2 = GCN_model(1,64,2)
                if device.type != 'cuda':
                    model1.load_state_dict(torch.load(reg1_path, map_location=torch.device('cpu')))
                    model2.load_state_dict(torch.load(reg2_path, map_location=torch.device('cpu')))
                else:
                    model1.load_state_dict(torch.load(reg1_path))
                    model2.load_state_dict(torch.load(reg2_path))
                model1.train()
                model2.train()
                model1.lin = Identity()
                model2.lin = Identity()
                model = Meta_model(64, model1, model2, 2, 2, 1)
                model = model.to(device)

            elif task_type == 'regression':
                reg1_path = os.path.join(home_dir, f'models/saved_best_models/SAGPool_{task_type}_modal1.pt')
                reg2_path = os.path.join(home_dir, f'models/saved_best_models/SAGPool_{task_type}_modal2.pt')
                model1 = SagPoolGCN_model(1,64,10)
                model2 = SagPoolGCN_model(1,64,10)
                if device.type != 'cuda':
                    model1.load_state_dict(torch.load(reg1_path, map_location=torch.device('cpu')))
                    model2.load_state_dict(torch.load(reg2_path, map_location=torch.device('cpu')))
                else:
                    model1.load_state_dict(torch.load(reg1_path))
                    model2.load_state_dict(torch.load(reg2_path))
                model1.train()
                model2.train()
                model1.lin1 = Identity()
                model1.out = Identity()
                model2.lin = Identity()
                model2.out = Identity()
                model = Meta_model(64, model1, model2, 10, 4, 3)
                model = model.to(device)

        else:
            if task_type == 'classification':
                reg1_path = os.path.join(home_dir, f'models/saved_best_models/SAGPool_classification_modal1_new.pt')
                reg2_path = os.path.join(home_dir, f'models/saved_best_models/GCN_classification_modal1_new.pt')
                model1 = SagPoolGCN_model(82,64,2)
                model2 = GCN_model(82,64,2)
                if device.type != 'cuda':
                    model1.load_state_dict(torch.load(reg1_path, map_location=torch.device('cpu')))
                    model2.load_state_dict(torch.load(reg2_path, map_location=torch.device('cpu')))
                else:
                    model1.load_state_dict(torch.load(reg1_path))
                    model2.load_state_dict(torch.load(reg2_path))
                model1.train()
                model2.train()
                # remove the linear layers from the model
                model1.lin1 = Identity()
                model1.out = Identity()
                model2.lin = Identity()
                model2.out = Identity()
                model = Meta_model(64, model1, model2, 2, 4, 3)
                model = model.to(device)

    # set parameters in this block up here
    print('Start Training')
    sub_dict = {'train_loss': None, 'test_loss': None}
    loss_dict = {epoch: copy.deepcopy(sub_dict) for epoch in range(0, max_epochs)}
    if task_type == 'regression':
        min_loss=10000
        criterion = torch.nn.MSELoss()
    elif task_type == 'classification':
        min_loss = 0.0
        sub_acc_dict = {'train_accuracy': None, 'test_accuracy': None}
        accuracy_dict = {epoch: copy.deepcopy(sub_acc_dict) for epoch in range(0, max_epochs)}
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(0, max_epochs):
        if task_type == 'classification':
            if model_subtype != 'meta_model':
                train_step_classification(train_loader, model, device, optimizer, criterion)
                train_loss, train_acc = test_step_classification(train_loader, model, device, optimizer, criterion)
                test_loss, test_acc = test_step_classification(test_loader, model, device, optimizer, criterion)
            elif modality == 'multimodality':
                modal1_train_loader, modal2_train_loader = train_loader
                modal1_test_loader, modal2_test_loader = test_loader
                train_step_classification((modal1_train_loader, modal2_train_loader), model, device, optimizer, criterion, multimodal=True)
                train_loss, train_acc = test_step_classification((modal1_train_loader, modal2_train_loader), model, device, optimizer, criterion, multimodal=True)
                test_loss, test_acc = test_step_classification((modal1_test_loader, modal2_test_loader), model, device, optimizer, criterion, multimodal=True)
            elif model_subtype == 'meta_model':
                train_step_classification(train_loader, model, device, optimizer, criterion, meta=True)
                train_loss, train_acc = test_step_classification(train_loader, model, device, optimizer, criterion, meta=True)
                test_loss, test_acc = test_step_classification(test_loader, model, device, optimizer, criterion, meta=True)
            loss_dict[epoch]['train_loss'] = train_loss.detach().cpu().item()
            loss_dict[epoch]['test_loss'] = test_loss.detach().cpu().item()
            accuracy_dict[epoch]['train_accuracy'] = train_acc
            accuracy_dict[epoch]['test_accuracy'] = test_acc
            torch.save(model.state_dict(), os.path.join(home_dir, f'models/model_checkpoints/{model_subtype}_{task_type}_{modality}_epoch_{epoch}_base.pt'))
            if test_acc > min_loss:
                min_loss = test_acc
                torch.save(model.state_dict(), os.path.join(home_dir, f'models/saved_best_models/{model_subtype}_{task_type}_{modality}_base.pt'))

        elif task_type == 'regression':
            if modality == 'multimodality':
                modal1_train_loader, modal2_train_loader = train_loader
                modal1_test_loader, modal2_test_loader = test_loader
                train_step_regression((modal1_train_loader, modal2_train_loader), model, device, optimizer, criterion, multimodal=True)
                train_acc = test_step_regression((modal1_train_loader, modal2_train_loader), model, device, optimizer, criterion, multimodal=True)
                test_acc = test_step_regression((modal1_test_loader, modal2_test_loader), model, device, optimizer, criterion, multimodal=True)
            else:
                train_step_regression(train_loader, model, device, optimizer, criterion)
                train_acc = test_step_regression(train_loader, model, device, optimizer, criterion)
                test_acc = test_step_regression(test_loader, model, device, optimizer, criterion)
            loss_dict[epoch]['train_loss'] = train_acc.detach().cpu().item()
            loss_dict[epoch]['test_loss'] = test_acc.detach().cpu().item()
            torch.save(model.state_dict(), os.path.join(home_dir, f'models/model_checkpoints/{model_subtype}_{task_type}_{modality}_epoch_{epoch}_base.pt'))
            if test_acc < min_loss:
                min_loss = test_acc
                torch.save(model.state_dict(), os.path.join(home_dir, f'models/saved_best_models/{model_subtype}_{task_type}_{modality}_base.pt'))
        
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.8f}, Test Acc: {test_acc:.4f}')

    # everything here should not be in the loop
    # save all the losses for visualization later
    loss_frame = pd.DataFrame(loss_dict).T
    loss_frame.to_csv(os.path.join(home_dir, f'models/saved_best_models/{model_subtype}_{task_type}_{modality}_losses_base.csv'))

    # if doing a classification task save the accuracy scores as well - please plot too
    if task_type == 'classification':
        acc_frame = pd.DataFrame(accuracy_dict).T
        fig, ax = plt.subplots(figsize=(10,10))
        gx = sns.lineplot(acc_frame, ax=ax).get_figure()
        gx.savefig(os.path.join(home_dir, f'figures/{model_subtype}_{task_type}_{modality}_accuracies_base.png'), dpi=450, transparent=False, format='png')
        acc_frame.to_csv(os.path.join(home_dir, f'models/saved_best_models/{model_subtype}_{task_type}_{modality}_class_accuracy_base.csv'))

    fig, ax = plt.subplots(figsize=(10,10))
    g = sns.lineplot(loss_frame, ax=ax).get_figure()
    g.savefig(os.path.join(home_dir, f'figures/{model_subtype}_{task_type}_{modality}_losses_base.png'), dpi=450, transparent=False, format='png')

    print(f'Finished Training - for task: {task_type} and modality: {modality}')
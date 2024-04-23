import torch
import numpy as np
import streamlit as st
import pandas as pd
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, average_precision_score, precision_recall_fscore_support, precision_recall_curve, auc
from model import GTransCYPs
from dataset_featurizer import MoleculeDataset
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
from model import GTransCYPs

def smiles_to_mol(smiles_string):
    """
    Loads a rdkit molecule object from a given smiles string.
    If the smiles string is invalid, it returns None.
    """
    return Chem.MolFromSmiles(smiles_string)

def mol_file_to_mol(mol_file):
    """
    Checks if the given mol file is valid.
    """
    return Chem.MolFromMolFile(mol_file)

def draw_molecule(mol):
    """
    Draws a molecule in SVG format.
    """
    return MolToImage(mol)

def mol_to_tensor_graph(mol):
    """
    Convert molecule to a graph representation that
    can be fed to the model
    """
    featurizer = PagtnMolGraphFeaturizer(max_length=5)
    f = featurizer.featurize(Chem.MolToSmiles(mol))
    graph_data = f[0]
    x=torch.tensor(graph_data.node_features, dtype=torch.float),
    edge_attr = torch.tensor(graph_data.edge_features, dtype=torch.float)
    edge_index = torch.tensor(graph_data.edge_index, dtype=torch.long).t().contiguous()
    batch_index = torch.ones_like(x[0])    
    data = Data(x=x[0], edge_attr=edge_attr, edge_index=edge_index)
    data.batch_index = batch_index

    return data


def get_model_predictions(input_file, data_df):
    """
    Get model predictions  
    """
    try:
        test_dataset = MoleculeDataset(root="data/", filename=input_file.name, dataframe=data_df)
    except Exception as e:
        st.warning(e)
    params = {
            "batch_size": 100,
            "learning_rate": 0.01,
            "weight_decay": 0.0001,
            "sgd_momentum": 0.8,
            "scheduler_gamma": 0.8,
            "pos_weight": 1.3,
            "model_embedding_size": 64,
            "model_attention_heads": 1,
            "model_layers": 4,
            "model_dropout_rate": 0.2,
            "model_top_k_ratio": 0.5,
            "model_top_k_every_n": 1,
            "model_dense_neurons": 256
            }

    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    params["model_edge_dim"] = test_dataset[0].edge_attr.shape[1]
    model_params = {k: v for k, v in params.items() if k.startswith("model_")}
    model = GTransCYPs(feature_size=test_dataset[0].x.shape[1], model_params=model_params)  
    model.load_state_dict(torch.load('models/GTransCYPs_1A2.pth')) 

    prediction = test_one_epoch(model, test_loader)
    emoticon = []
    pred = []
    for mol_pred in prediction:
        pred_temp = (mol_pred *100)
        pred.append(pred_temp)
        if mol_pred >= 0.5:
            emoticon.append("✅")
        else:
            emoticon.append("❌")
    pred = np.array(pred)
    pred = pred.reshape(-1)
    prediction_output = pd.Series(pred, name='PConfInh (%)')
    prediction_output = prediction_output.map("{:.2f}".format)
    emoticon_output = pd.Series(emoticon, name='Inhibitor')
    molecule_name = pd.Series(data_df.Drug, name='Molecule SMILES Sequence')
    df = pd.concat([molecule_name, prediction_output, emoticon_output], axis=1)

    return df

def get_model_predictions2(input_file, data_df):
    try:
        test_dataset = MoleculeDataset(root="data/", filename=input_file.name, dataframe=data_df)
    except Exception as e:
        st.warning(e)
    params = {
        "batch_size": 128,
        "learning_rate": 0.1,
        "weight_decay": 0.0001,
        "sgd_momentum": 0.8,
        "scheduler_gamma": 0.8,
        "pos_weight": 1.3,
        "model_embedding_size": 64,
        "model_attention_heads": 1,
        "model_layers": 1,
        "model_dropout_rate": 0.2,
        "model_top_k_ratio": 0.5,
        "model_top_k_every_n": 1,
        "model_dense_neurons": 256
    }
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    params["model_edge_dim"] = test_dataset[0].edge_attr.shape[1]
    model_params = {k: v for k, v in params.items() if k.startswith("model_")}
    model = GTransCYPs(feature_size=test_dataset[0].x.shape[1], model_params=model_params)  
    model.load_state_dict(torch.load('models/GTransCYPs_2C9.pth'))  
    prediction = test_one_epoch(model, test_loader)
    emoticon = []
    pred = []
    for mol_pred in prediction:
        pred_temp = (mol_pred *100)
        pred.append(pred_temp)
        if mol_pred >= 0.5:
            emoticon.append("✅")
        else:
            emoticon.append("❌")
    pred = np.array(pred)
    pred = pred.reshape(-1)
    prediction_output = pd.Series(pred, name='PConfInh (%)')
    prediction_output = prediction_output.map("{:.2f}".format)
    emoticon_output = pd.Series(emoticon, name='Inhibitor')
    molecule_name = pd.Series(data_df.Drug, name='Molecule SMILES Sequence')
    df = pd.concat([molecule_name, prediction_output, emoticon_output], axis=1)
    return df

def get_model_predictions3(input_file, data_df):
    try:
        test_dataset = MoleculeDataset(root="data/", filename=input_file.name, dataframe=data_df)
    except Exception as e:
        st.warning(e)
    params = {
        "batch_size": 128,
        "learning_rate": 0.1,
        "weight_decay": 0.0001,
        "sgd_momentum": 0.8,
        "scheduler_gamma": 0.8,
        "pos_weight": 1.3,
        "model_embedding_size": 64,
        "model_attention_heads": 1,
        "model_layers": 1,
        "model_dropout_rate": 0.2,
        "model_top_k_ratio": 0.5,
        "model_top_k_every_n": 1,
        "model_dense_neurons": 256
    }
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    params["model_edge_dim"] = test_dataset[0].edge_attr.shape[1]
    model_params = {k: v for k, v in params.items() if k.startswith("model_")}
    model = GTransCYPs(feature_size=test_dataset[0].x.shape[1], model_params=model_params)  
    model.load_state_dict(torch.load('models/GTransCYPs_2C19.pth'))  
    prediction = test_one_epoch(model, test_loader)
    emoticon = []
    pred = []
    for mol_pred in prediction:
        pred_temp = (mol_pred *100)
        pred.append(pred_temp)
        if mol_pred >= 0.5:
            emoticon.append("✅")
        else:
            emoticon.append("❌")
    pred = np.array(pred)
    pred = pred.reshape(-1)
    prediction_output = pd.Series(pred, name='PConfInh (%)')
    prediction_output = prediction_output.map("{:.2f}".format)
    emoticon_output = pd.Series(emoticon, name='Inhibitor')
    molecule_name = pd.Series(data_df.Drug, name='Molecule SMILES Sequence')
    df = pd.concat([molecule_name, prediction_output, emoticon_output], axis=1)
    return df

def get_model_predictions4(input_file, data_df):
    try:
        test_dataset = MoleculeDataset(root="data/", filename=input_file.name, dataframe=data_df)
    except Exception as e:
        st.warning(e)
    params = {
        "batch_size": 128,
        "learning_rate": 0.1,
        "weight_decay": 0.0001,
        "sgd_momentum": 0.8,
        "scheduler_gamma": 0.8,
        "pos_weight": 1.3,
        "model_embedding_size": 64,
        "model_attention_heads": 1,
        "model_layers": 1,
        "model_dropout_rate": 0.2,
        "model_top_k_ratio": 0.5,
        "model_top_k_every_n": 1,
        "model_dense_neurons": 256
    }
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    params["model_edge_dim"] = test_dataset[0].edge_attr.shape[1]
    model_params = {k: v for k, v in params.items() if k.startswith("model_")}
    model = GTransCYPs(feature_size=test_dataset[0].x.shape[1], model_params=model_params)  
    model.load_state_dict(torch.load('models/GTransCYPs_2D6.pth'))  
    prediction = test_one_epoch(model, test_loader)
    emoticon = []
    pred = []
    for mol_pred in prediction:
        pred_temp = (mol_pred *100)
        pred.append(pred_temp)
        if mol_pred >= 0.5:
            emoticon.append("✅")
        else:
            emoticon.append("❌")
    pred = np.array(pred)
    pred = pred.reshape(-1)
    prediction_output = pd.Series(pred, name='PConfInh (%)')
    prediction_output = prediction_output.map("{:.2f}".format)
    emoticon_output = pd.Series(emoticon, name='Inhibitor')
    molecule_name = pd.Series(data_df.Drug, name='Molecule SMILES Sequence')
    df = pd.concat([molecule_name, prediction_output, emoticon_output], axis=1)
    return df

def get_model_predictions5(input_file, data_df):
    try:
        test_dataset = MoleculeDataset(root="data/", filename=input_file.name, dataframe=data_df)
    except Exception as e:
        st.warning(e)
    params = {
        "batch_size": 128,
        "learning_rate": 0.1,
        "weight_decay": 0.0001,
        "sgd_momentum": 0.8,
        "scheduler_gamma": 0.8,
        "pos_weight": 1.3,
        "model_embedding_size": 64,
        "model_attention_heads": 1,
        "model_layers": 1,
        "model_dropout_rate": 0.2,
        "model_top_k_ratio": 0.5,
        "model_top_k_every_n": 1,
        "model_dense_neurons": 256
    }
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    params["model_edge_dim"] = test_dataset[0].edge_attr.shape[1]
    model_params = {k: v for k, v in params.items() if k.startswith("model_")}
    model = GTransCYPs(feature_size=test_dataset[0].x.shape[1], model_params=model_params)  
    model.load_state_dict(torch.load('models/GTransCYPs_3A4.pth'))  
    prediction = test_one_epoch(model, test_loader)
    emoticon = []
    pred = []
    for mol_pred in prediction:
        pred_temp = (mol_pred *100)
        pred.append(pred_temp)
        if mol_pred >= 0.5:
            emoticon.append("✅")
        else:
            emoticon.append("❌")
    pred = np.array(pred)
    pred = pred.reshape(-1)
    prediction_output = pd.Series(pred, name='PConfInh (%)')
    prediction_output = prediction_output.map("{:.2f}".format)
    emoticon_output = pd.Series(emoticon, name='Inhibitor')
    molecule_name = pd.Series(data_df.Drug, name='Molecule SMILES Sequence')
    df = pd.concat([molecule_name, prediction_output, emoticon_output], axis=1)
    return df

def test_one_epoch(model, test_loader):
    all_preds = []
    all_score = []
    step = 0
    weight = torch.tensor([1.3], dtype=torch.float32)
    model.eval()    
    with torch.no_grad():
        for batch in test_loader:      
            pred = model(batch.x.float(),
                         batch.edge_attr.float(),
                         batch.edge_index,
                         batch.batch)
            step += 1
            all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
            all_score.append(torch.sigmoid(pred).cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_score = np.concatenate(all_score).ravel()

    return all_score

    







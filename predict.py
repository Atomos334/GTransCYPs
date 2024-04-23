import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, average_precision_score, precision_recall_fscore_support, precision_recall_curve, auc
import numpy as np
from tqdm import tqdm
from model import GCN, GAT, GIN, GraphSAGE, GAT_GCN, GTransCYPs
from featurizer import MoleculeDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_one_epoch(model, test_loader, loss_fn):
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch.to(device)
            pred = model(batch.x.float(),
                         batch.edge_attr.float(),
                         batch.edge_index,
                         batch.batch)
            loss = loss_fn(torch.squeeze(pred), batch.y.float())
            running_loss += loss.item()
            step += 1
            all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
            all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, 0, "test")  
    return running_loss / step

def calculate_metrics(y_pred, y_true, epoch, type):
    f1score = f1_score(y_true, y_pred)    
    acc_score = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aucpr = auc(recall, precision)
    roc = roc_auc_score(y_true, y_pred)
    print(f"Balanced Accuracy: {bal_acc}")
    print(f"Matthews Correlation Coefficient: {mcc}")
    print(f"Accuracy: {acc_score}")
    print(f"AUCPR: {aucpr}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1score}")
    print(f"ROC AUC: {roc}")
    
def main():
    print("Loading dataset...")
    test_dataset = MoleculeDataset(root="data/", filename="")
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
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    params["model_edge_dim"] = test_dataset[0].edge_attr.shape[1]
    model_params = {k: v for k, v in params.items() if k.startswith("model_")}
    model = GTransCYPs(feature_size=test_dataset[0].x.shape[1], model_params=model_params)  
    model.load_state_dict(torch.load(''))  
    model = model.to(device)
    print(f"Number of parameters: {count_parameters(model)}")
    weight = torch.tensor([1.3], dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    test_loss = test_one_epoch(model, test_loader, loss_fn)
    print(f"Test loss: {test_loss}")

if __name__ == "__main__":
    print("Starting testing...")
    main()

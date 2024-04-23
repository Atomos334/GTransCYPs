import torch 
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, average_precision_score, precision_recall_fscore_support, precision_recall_curve, auc
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
from mango import scheduler, Tuner
from config import HYPERPARAMETERS, SIGNATURE
from model import GTransCYPs
from featurizer import MoleculeDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(train_loader)):
        batch.to(device)  
        optimizer.zero_grad() 
        pred = model(batch.x.float(), 
                     batch.edge_attr.float(),
                     batch.edge_index, 
                     batch.batch) 
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()  
        optimizer.step()  
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss/step

def test(epoch, model, test_loader, loss_fn):
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
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
        all_preds_raw.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "test")
    log_conf_matrix(all_preds, all_labels, epoch)
    return running_loss/step

def log_conf_matrix(y_pred, y_true, epoch):
    cm = confusion_matrix(y_pred, y_true)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index = classes, columns = classes)

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

def run_one_training(params):
    params = params[0]
    print("Loading dataset...")
    train_dataset = MoleculeDataset(root="data/", filename="")
    test_dataset = MoleculeDataset(root="data/", filename="", test=True)
    params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=True)
    model_params = {k: v for k, v in params.items() if k.startswith("model_")}
    model = GTransCYPs(feature_size=train_dataset[0].x.shape[1], model_params=model_params) 
    model = model.to(device)
    print(f"Number of parameters: {count_parameters(model)}")
    weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=params["learning_rate"],
                                momentum=params["sgd_momentum"],
                                weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])
    
    best_loss = 1000
    early_stopping_counter = 0
    for epoch in range(20): 
        if early_stopping_counter <= 5: 
            model.train()
            loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
            print(f"Epoch {epoch} | Train Loss {loss}")

            model.eval()
            if epoch % 5 == 0:
                loss = test(epoch, model, test_loader, loss_fn)
                
                if float(loss) < best_loss:
                    best_loss = loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

            scheduler.step()
        else:
            print("Early stopping due to no improvement.")
            return [best_loss]
    print(f"Finishing training with best test loss: {best_loss}")
    return [best_loss]

print("Hyperparameter search...")
config = dict()
config["optimizer"] = "Bayesian"
config["num_iteration"] = 100

tuner = Tuner(HYPERPARAMETERS, 
              objective=run_one_training,
              conf_dict=config) 
results = tuner.minimize()

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, average_precision_score

# Function that creates a dataset from a dataframe
def create_dataset_from_timeseries(dataframe, labels):
    """
        Create a dataset from a dataframe.
        The dataset is a tuple with the features and the target
    """
    features_cols = [col for col in dataframe.columns if col not in ["RecordID", "Time"]]

    sequences = []
    for record_id, group in dataframe.groupby("RecordID"):
        seq = group[features_cols].to_numpy(dtype=np.float32)
        sequences.append(torch.tensor(seq)) 
    
    # Pad the sequences (timeseries may have different lengths)
    padded_sequences_a = pad_sequence(sequences, batch_first=True)

    X = padded_sequences_a
    y = torch.tensor(labels)
    return TensorDataset(X, y)

# Training loop for generic PyTorch model
def train_model_with_validation(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=10, scheduler=None):
    """
    Training loop for binary classification models with a validation phase.
    Computes AUCROC and AUPRC metrics for both training and validation sets.

    Parameters:
        model (torch.nn.Module): The binary classification model (e.g., LSTM, Transformer).
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device (CPU or GPU).
        num_epochs (int): Number of training epochs.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
    
    Returns:
        model (torch.nn.Module): The trained model.
    """
    model.to(device)
    
    for epoch in range(num_epochs):
        # ---------------------- Training Phase ----------------------
        model.train()
        running_loss_train = 0.0
        all_preds_train = []
        all_targets_train = []
        train_samples = 0
        
        train_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, targets in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Convert targets to float if needed
            if targets.dtype != outputs.dtype:
                targets = targets.to(outputs.dtype)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            batch_size = inputs.size(0)
            train_samples += batch_size
            running_loss_train += loss.item() * batch_size
            
            # Convert outputs to probabilities
            if outputs.dim() == 1 or outputs.shape[-1] == 1:
                probs = torch.sigmoid(outputs.view(-1))
            elif outputs.shape[-1] == 2:
                probs = torch.softmax(outputs, dim=-1)[:, 1]
            else:
                raise ValueError("Unexpected output shape for binary classification")
            
            all_preds_train.append(probs.detach().cpu())
            all_targets_train.append(targets.detach().cpu())
            
            avg_loss_train = running_loss_train / train_samples
            train_bar.set_postfix(loss=f"{avg_loss_train:.4f}")
        
        train_loss = running_loss_train / len(train_loader.dataset)
        all_preds_train = torch.cat(all_preds_train).numpy().ravel()
        all_targets_train = torch.cat(all_targets_train).numpy().ravel()
        train_auc = roc_auc_score(all_targets_train, all_preds_train)
        train_auprc = average_precision_score(all_targets_train, all_preds_train)
        
        # ---------------------- Validation Phase ----------------------
        model.eval()
        running_loss_val = 0.0
        all_preds_val = []
        all_targets_val = []
        val_samples = 0
        
        val_bar = tqdm(valid_loader, desc=f"Val Epoch {epoch+1}/{num_epochs}", leave=False)
        with torch.no_grad():
            for inputs, targets in val_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # Convert targets to float if needed
                if targets.dtype != outputs.dtype:
                    targets = targets.to(outputs.dtype)
                
                loss = criterion(outputs, targets)
                
                batch_size = inputs.size(0)
                val_samples += batch_size
                running_loss_val += loss.item() * batch_size
                
                if outputs.dim() == 1 or outputs.shape[-1] == 1:
                    probs = torch.sigmoid(outputs.view(-1))
                elif outputs.shape[-1] == 2:
                    probs = torch.softmax(outputs, dim=-1)[:, 1]
                else:
                    raise ValueError("Unexpected output shape for binary classification")
                
                all_preds_val.append(probs.detach().cpu())
                all_targets_val.append(targets.detach().cpu())
                
                val_bar.set_postfix(loss=f"{running_loss_val/val_samples:.4f}")
        
        val_loss = running_loss_val / len(valid_loader.dataset)
        all_preds_val = torch.cat(all_preds_val).numpy().ravel()
        all_targets_val = torch.cat(all_targets_val).numpy().ravel()
        val_auc = roc_auc_score(all_targets_val, all_preds_val)
        val_auprc = average_precision_score(all_targets_val, all_preds_val)
        
        if scheduler is not None:
            scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | AUCROC: {train_auc:.4f} | AUPRC: {train_auprc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | AUCROC: {val_auc:.4f} | AUPRC: {val_auprc:.4f}\n")
    
    return model

# Training loop for generic PyTorch model
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10, scheduler=None):
    """
    Generic training loop for PyTorch models without a validation phase.
    
    Parameters:
        model (torch.nn.Module): The model to train (e.g., LSTM, Transformer).
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run the model on (CPU or GPU).
        num_epochs (int): Number of training epochs.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        
    Returns:
        model (torch.nn.Module): The trained model.
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        samples_processed = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Convert targets to float if needed
            if targets.dtype != outputs.dtype:
                targets = targets.to(outputs.dtype)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            batch_size = inputs.size(0)
            samples_processed += batch_size
            running_loss += loss.item() * batch_size
            
            # Transform outputs to probabilities for binary classification.
            # If outputs are a single logit, use sigmoid; if there are two logits, use softmax.
            if outputs.dim() == 1 or outputs.shape[-1] == 1:
                probs = torch.sigmoid(outputs.view(-1))
            elif outputs.shape[-1] == 2:
                probs = torch.softmax(outputs, dim=-1)[:, 1]
            else:
                raise ValueError("Unexpected output shape for binary classification")
            
            all_preds.append(probs.detach().cpu())
            all_targets.append(targets.detach().cpu())
            
            avg_loss = running_loss / samples_processed
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        all_preds = torch.cat(all_preds).numpy().ravel()
        all_targets = torch.cat(all_targets).numpy().ravel()
        
        epoch_auc = roc_auc_score(all_targets, all_preds)
        epoch_auprc = average_precision_score(all_targets, all_preds)
        
        if scheduler is not None:
            scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - AUCROC: {epoch_auc:.4f} - AUPRC: {epoch_auprc:.4f}")
    
    return model

def evaluate_model(model, data_loader, criterion, device):
    """
    Generic evaluation function for PyTorch models.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        avg_loss (float): The average loss computed on the evaluation dataset.
    """
    model.to(device)
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    samples_processed = 0
    
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Convert targets to float if needed
            if targets.dtype != outputs.dtype:
                targets = targets.to(outputs.dtype)

            loss = criterion(outputs, targets)
            
            batch_size = inputs.size(0)
            samples_processed += batch_size
            running_loss += loss.item() * batch_size
            
            if outputs.dim() == 1 or outputs.shape[-1] == 1:
                probs = torch.sigmoid(outputs.view(-1))
            elif outputs.shape[-1] == 2:
                probs = torch.softmax(outputs, dim=-1)[:, 1]
            else:
                raise ValueError("Unexpected output shape for binary classification")
            
            all_preds.append(probs.detach().cpu())
            all_targets.append(targets.detach().cpu())
            
            progress_bar.set_postfix(loss=f"{running_loss/samples_processed:.4f}")
    
    avg_loss = running_loss / len(data_loader.dataset)
    all_preds = torch.cat(all_preds).numpy().ravel()
    all_targets = torch.cat(all_targets).numpy().ravel()
    
    aucroc = roc_auc_score(all_targets, all_preds)
    auprc = average_precision_score(all_targets, all_preds)
    
    print(f"Evaluation - Loss: {avg_loss:.4f} - AUCROC: {aucroc:.4f} - AUPRC: {auprc:.4f}")
    return avg_loss, aucroc, auprc



__all__ = ["create_dataset_from_timeseries", "train_model_with_validation", "train_model", "evaluate_model"]

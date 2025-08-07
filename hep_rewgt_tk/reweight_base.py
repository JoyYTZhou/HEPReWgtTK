from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch, os, logging
from torchviz import make_dot
import torch.nn as nn
from torch.nn.utils import spectral_norm
import seaborn as sns
import hiddenlayer as hl
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc, confusion_matrix

pjoin = os.path.join

def feature_extract_layer(input_dim, latent_dim) -> list:
    return [
        nn.Linear(input_dim, latent_dim),
        nn.ReLU(),
        nn.BatchNorm1d(latent_dim),
        nn.Dropout(0.2)
    ]

def output_layer(last_dim) -> list:
    return [
        nn.Linear(last_dim, 1),
        nn.Sigmoid()
    ]

def high_dim_discriminator(latent_dim) -> list:
    """High-dimensional discriminator.
    
    Return a list of hidden layers for a high-dimensional discriminator."""
    return [nn.Linear(latent_dim, 256),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(64),
    ]

def high_dim_with_att(latent_dim) -> list:
    return [nn.Linear(latent_dim, 512),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.MultiheadAttention(256, num_heads=8, dropout=0.3),
        nn.BatchNorm1d(256),
        nn.Linear(256, 128),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(64),
    ]

def stable_discriminator(latent_dim) -> list:
    """Stable discriminator with spectral normalization.
    
    Return a list of hidden layers for a stable discriminator with spectral normalization.
    """
    return [spectral_norm(nn.Linear(latent_dim, 256)),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(256),
        spectral_norm(nn.Linear(256, 128)),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(128),
        spectral_norm(nn.Linear(128, 64)),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(64),
    ]

def standard_discriminator(latent_dim) -> list:
    return [nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            ]

def add_hidden_layer(layers, in_dim, hidden_dims, activation):
    """Add hidden layers to the model."""
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(activation)
        in_dim = h_dim

def check_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")
    return device

class TrainingUtils:
    @staticmethod
    def train_epoch(model, train_loader, optimizer, device, criterion=nn.BCELoss()):
        """Train model for one epoch.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to train
        train_loader : DataLoader
            Training data loader
        optimizer : torch.optim.Optimizer
            Optimizer
        device : torch.device
            Device to use for training
            
        Returns
        -------
        float
            Average loss for this epoch
        """
        model.train()
        running_loss = 0.0
        
        for batch_data, batch_labels, batch_weights in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            batch_weights = batch_weights.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            weighted_loss = (loss * batch_weights).mean()
            weighted_loss.backward()
            optimizer.step()
            
            running_loss += weighted_loss.item()
            
        return running_loss / len(train_loader)

    @staticmethod
    def validate(model, val_loader, device, criterion=nn.BCELoss()):
        """Validate the model.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to validate
        val_loader : DataLoader
            Validation data loader
        device : torch.device
            Device to use for validation
            
        Returns
        -------
        float
            Validation loss
        """
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for val_data, val_labels, val_weights in val_loader:
                val_data = val_data.to(device)
                val_labels = val_labels.to(device)
                val_weights = val_weights.to(device)
                
                val_outputs = model(val_data)
                loss = criterion(val_outputs, val_labels)
                weighted_loss = (loss * val_weights).mean()
                val_loss += weighted_loss.item()
                
        return val_loss / len(val_loader)

    @staticmethod
    def save_checkpoint(model, epoch, save_path, optimizer=None, scheduler=None):
        """Save a model checkpoint.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to save
        epoch : int
            Current epoch number
        save_path : str
            Path to save the checkpoint
        optimizer : torch.optim.Optimizer, optional
            Optimizer state to save
        scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler state to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None
        }
        torch.save(checkpoint, save_path)

class ReweighterBase():
    """Base class for reweighting.

    Attributes:
    - `src`: pandas DataFrame for the source/original distribution (label 0).
    - `tgt`: pandas DataFrame for the target distribution (label 1).
    - `w_col`: Column name for the weights.
    - `src_w`: Weights for the source data.
    - `tgt_w`: Weights for the target data.
    - `out_dir`: Base directory to save the results, e.g. plots."""
    def __init__(self, src:'pd.DataFrame', tgt:'pd.DataFrame', w_col, out_dir):
        """
        Parameters:
        `src`: pandas DataFrame for the source/original distribution (label 0).
        `tgt`: pandas DataFrame for the target distribution (label 1)."""
        self.src = src
        self.tgt = tgt
        self.w_col = w_col
        self.src_w = None
        self.tgt_w = None
        self.out_dir = out_dir
        self.device = check_device()
        if out_dir is not None:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                logging.info(f"Created directory: {out_dir}")

    @staticmethod
    def drop_likes(df: 'pd.DataFrame', drop_kwd: 'list[str]' = []):
        """Drop columns containing the keywords in `drop_kwd`."""
        dropped = pd.DataFrame()
        for kwd in drop_kwd:
            cols_to_drop = df.filter(like=kwd).columns
            dropped = pd.concat([dropped, df[cols_to_drop]], axis=1)
            df = df.drop(columns=df.filter(like=kwd).columns, inplace=False)
        return df, dropped
    
    @staticmethod
    def clean_data(df_original, drop_kwd, wgt_col, drop_wgts=True, drop_neg_wgts=True) -> tuple['pd.DataFrame', 'pd.Series', 'pd.DataFrame', 'pd.DataFrame']:
        """Clean the data by dropping columns containing the keywords in `drop_kwd`.
        
        Return
        - `X`: Features, pandas DataFrame
        - `weights`: Weights, pandas Series
        - `neg_df`: DataFrame containing the events with negative weights.
        - `dropped_X`: DataFrame containing the dropped columns.
        """
        df = df_original.copy()
        neg_df = df[df[wgt_col] < 0].reset_index(drop=True)
        num_neg = len(neg_df)
        total = len(df_original)
        if drop_neg_wgts:
            df = df[df[wgt_col] > 0]
        num_dropped = total - len(df)
        logging.info(f"Dropped {num_dropped} events with negative weights out of {total} events.")

        drop_kwd_local = drop_kwd.copy()
        if drop_wgts and wgt_col not in drop_kwd_local:
            drop_kwd_local.append(wgt_col)
        X, dropped_X = ReweighterBase.drop_likes(df, drop_kwd_local)

        # Reset index to avoid reindexing errors later
        X = X.reset_index(drop=True)
        dropped_X = dropped_X.reset_index(drop=True)
        weights = df[wgt_col].reset_index(drop=True)
        
        return X, weights, neg_df, dropped_X
    
    @staticmethod
    def int_label(label, length):
        """Return the label as a pandas Series"""
        return pd.Series([label]*length)
    
    @staticmethod
    def prep_distributions(dist_a, dist_b, drop_kwd, wgt_col, drop_neg_wgts=True, drop_wgts=True):
        """Preprocess two distributions by dropping columns with keywords in `drop_kwd`.
        Assign label 0 to dist_a, 1 to dist_b. Returns features, labels, and weights."""
        X_a, w_a, _, _ = ReweighterBase.clean_data(dist_a, drop_kwd, wgt_col, drop_neg_wgts=drop_neg_wgts, drop_wgts=drop_wgts)
        y_a = ReweighterBase.int_label(0, len(X_a))
        X_b, w_b, _, _ = ReweighterBase.clean_data(dist_b, drop_kwd, wgt_col, drop_neg_wgts=drop_neg_wgts, drop_wgts=drop_wgts)
        y_b = ReweighterBase.int_label(1, len(X_b))
        
        return (
            pd.concat([X_a, X_b], ignore_index=True, axis=0),
            pd.concat([y_a, y_b], ignore_index=True, axis=0),
            pd.concat([w_a, w_b], ignore_index=True, axis=0)
        )
        
    @staticmethod
    def draw_distributions(distributions, labels, weights, column, bins=10, range=None, save_path=False):
        """Draw normalized histograms for multiple datasets."""
        hist_settings = {'bins': bins, 'density': True, 'alpha': 0.5}
        plt.figure(figsize=[12, 7])
        if range is None:
            all_vals = np.hstack([d[column] for d in distributions])
            range = np.percentile(all_vals, [0.01, 99.99])
        for dist, label, wgt in zip(distributions, labels, weights):
            plt.hist(dist[column], weights=wgt, range=range, label=label, **hist_settings)
        plt.legend(loc='best')
        plt.title(column)
        if save_path:
            plt.savefig(save_path)
    
    @staticmethod
    def compute_nn_auc(model, data_loader, device, save, save_path, title='ROC Curve'):
        """Compute the AUC score for a nn model."""
        all_labels = []
        all_preds = []
        all_weights = []
        
        model.eval()
        print("Using device: ", device)
        model.to(device)
        with torch.no_grad():
            for data, label, weight in data_loader:
                data, label, weight = data.to(device), label.to(device), weight.to(device)
                #! Bug here
                pred = model(data).squeeze()
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())
                all_weights.extend(weight.cpu().numpy())
        
        ReweighterBase.plot_roc(all_preds, all_labels, all_weights, save, save_path, title)
        
    @staticmethod
    def plot_roc(pred, label, weight, save, save_path, title='ROC Curve'):
        """Plot the ROC curve for both binary and multi-class classification.
        
        Parameters:
        - pred: Model predictions. For binary classification, a 1D array of probabilities.
                For multi-class, a 2D array where each column represents class probabilities.
        - label: True labels. For binary classification, a 1D array.
                For multi-class, a 1D array with class indices.
        - weight: Sample weights
        - save: Boolean indicating whether to save the plot
        - save_path: Path where to save the plot
        - title: Title of the plot
        """
        plt.figure(figsize=(10, 8))
        
        pred = np.array(pred)
        label = np.array(label)
        weight = np.array(weight)
        
        if len(pred.shape) == 2:
            n_classes = pred.shape[1]
            for i in range(n_classes):
                binary_label = (label == i).astype(int)
                fpr, tpr, _ = roc_curve(binary_label, pred[:, i], sample_weight=weight)
                auc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.2f})')
                print(f'ROC AUC Score for Class {i}: {auc_score:.3f}')
        
        else:
            fpr, tpr, _ = roc_curve(label, pred, sample_weight=weight)
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
            print(f'ROC AUC Score: {auc_score:.3f}')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend()
        
        if save:
            plt.savefig(save_path)
            plt.close()

    @staticmethod
    def plot_confusion(pred, label, weight, classes, save, save_path):
        """Plot the confusion matrix for both binary and multi-class classification.
        
        Parameters:
        - pred: Model predictions. For binary classification, a 1D array of probabilities.
                For multi-class, a 2D array
        - classes: List of class names"""
        cm = confusion_matrix(label, pred, sample_weight=weight)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f', 
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        if save:
            plt.savefig(save_path)
        plt.close()

class PredictionUtils:
    @staticmethod
    def predict_in_batches(model, data, device, batch_size=512) -> np.ndarray:
        """Make predictions in batches.
        
        Parameters
        ----------
        model : torch.nn.Module
        data : torch.Tensor
        batch_size : int, optional
            
        Returns
        -------
        numpy.ndarray
            Predictions
        """
        model.eval()
        model.to(device)
        
        predictions = []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size].to(device)
                pred = model(batch).cpu()
                predictions.append(pred)
                
        return torch.cat(predictions).numpy().squeeze()
    
    @staticmethod
    def compute_rewgt_ratio(infer_score, criterion=nn.BCELoss(), epsilon=1e-6):
        """Compute reweighting ratio based on infer_score.
        
        Parameters
        ----------
        infer_score : numpy.ndarray
            Model infer_score
        epsilon : float, optional
            Small value to prevent division by zero
            
        Returns
        -------
        numpy.ndarray
            Reweighting ratio
        """
        # Clip infer_score to avoid division by zero
        infer_score = np.clip(infer_score, epsilon, 1 - epsilon)
        
        if isinstance(criterion, nn.BCELoss) or isinstance(criterion, nn.MSELoss):
            reweight_ratio = infer_score / (1 - infer_score)
        
        return reweight_ratio
    
    @staticmethod
    def compute_weights(infer_score, original_weights, normalize_factor, epsilon=1e-6, criterion=nn.BCELoss()) -> tuple[np.ndarray, np.ndarray]:
        """Compute reweighting factors.
        
        Parameters
        ----------
        infer_score : numpy.ndarray
            Model infer_score
        original_weights : numpy.ndarray
            Original sample weights
        normalize_factor : float
            Normalization factor
        epsilon : float, optional
            Small value to prevent division by zero
            
        Returns
        -------
        numpy.ndarray
            New weights
        """
        reweight_ratio = PredictionUtils.compute_rewgt_ratio(infer_score, criterion, epsilon)
        
        # Calculate weights using the reweighting formula
        new_weights = original_weights * reweight_ratio
        
        # Normalize weights
        new_weights *= normalize_factor / new_weights.sum()
        
        return new_weights, reweight_ratio

class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for binary classification.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    hidden_dims : list
        List of integers for the number of nodes in each hidden layer
    dropout_rate : float, optional
        Dropout rate for regularization (default: 0.2)
    """
    def __init__(self, input_dim: int, first_latent_dim: int, hidden_choice: 'str'='high_dim', dropout_rate: float = 0.2):
        super().__init__()

        layers = []
        layers.extend(feature_extract_layer(input_dim, first_latent_dim))

        if hidden_choice == 'high_dim':
            layers.extend(high_dim_discriminator(first_latent_dim))
        elif hidden_choice == 'stable':
            layers.extend(stable_discriminator(first_latent_dim))
        elif hidden_choice == 'high_dim_with_att':
            layers.extend(high_dim_with_att(first_latent_dim))
        else:
            layers.extend(standard_discriminator(first_latent_dim))
        
        layers.extend(output_layer(64))

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
    def visualize_architecture(self, save_path):
        """Visualize the model architecture using torchviz."""
        batch_size = 32
        x = torch.randn(batch_size, self.model[0].in_features)
        y = self.model(x)
        
        dot = make_dot(y, params=dict(self.model.named_parameters()))
        
        # Save the visualization
        dot.render(save_path, format='png')


        
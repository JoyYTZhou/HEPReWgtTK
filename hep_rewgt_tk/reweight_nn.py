import torch
from torch import nn
from torch.nn.utils import spectral_norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from os.path import join as pjoin
import pandas as pd
from torchviz import make_dot
import os
from .reweight_base import ReweighterBase, WeightedDataset, MLPClassifier, check_device, TrainingUtils, PredictionUtils

def weighted_bce_loss(predictions, targets, weights):
    return torch.mean(weights * nn.BCELoss(reduction='none')(predictions, targets))

class SingleMLPRwgter(ReweighterBase):
    def __init__(self, ori_data, tar_data, weight_column, results_dir):
        super().__init__(ori_data, tar_data, weight_column, results_dir)
        self.scaler = MinMaxScaler()

    def prep_data(self, drop_kwd, drop_neg_wgts=True):
        """Preprocess the data into TensorDataset objects.
        Drop columns containing the keywords in `drop_kwd` and negatively weighted events if `drop_neg_wgts` is True."""
        X, y, weights = self.prep_ori_tar(self.ori_data, self.tar_data, drop_kwd, self.weight_column, drop_wgts=True, drop_neg_wgts=drop_neg_wgts)
        features = list(X.columns)
        print(f"Features: {features}")

        X[features] = self.scaler.fit_transform(X[features])

        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(X.values, y.values, weights.values, test_size=0.3, random_state=42)

        self.dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).view(-1, 1), torch.tensor(w_train, dtype=torch.float32))
        self.val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).view(-1, 1), torch.tensor(w_val, dtype=torch.float32))
        self.features = features

    def train(self, num_epochs, hidden_arch: 'str', batch_size, lr, dropout=0.2, save=True, 
             savename='SingleNNmodel.pth', save_interval=30, latent_dim=64):
        """Train the discriminator model.

        Parameters
        ----------
        num_epochs : int
            Number of epochs
        hidden_arch : str
            Hidden architecture of the model
        batch_size : int
            Batch size
        lr : float
            Learning rate
        dropout : float, optional
            Dropout rate
        save : bool, optional
            Whether to save checkpoints
        savename : str, optional
            Base name for saved model files
        save_interval : int, optional
            Interval between saving checkpoints
        """
        device = check_device()
        input_dim = len(self.features)

        # Model setup
        self.model = MLPClassifier(input_dim, latent_dim, hidden_choice=hidden_arch, dropout_rate=dropout
        ).to(device)
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Data loaders
        train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            # Train
            train_loss = TrainingUtils.train_epoch(
                self.model, train_loader, self.criterion, self.optimizer, device
            )
            train_losses.append(train_loss)

            # Validate
            val_loss = TrainingUtils.validate(
                self.model, val_loader, self.criterion, device
            )
            val_losses.append(val_loss)

            # Log progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")

            # Save checkpoint
            if save and (epoch + 1) % save_interval == 0:
                save_path = pjoin(self.results_dir, 
                                f"{savename.split('.')[0]}_{epoch+1}.pth")
                TrainingUtils.save_checkpoint(
                    self.model, epoch, save_path, self.optimizer
                )
                print(f"Model saved at {save_path}")

        self.history = {'train': train_losses, 'val': val_losses}

        if save:
            final_path = pjoin(self.results_dir, savename)
            TrainingUtils.save_checkpoint(
                self.model, num_epochs, final_path, self.optimizer
            )
            print(f"Final Model saved to {savename}")

        self._name = savename.split('.')[0]

    def load_checkpoint(self, checkpoint_path, input_dim, hidden_choice:'str', latent_dim=64):
        """Load model from a checkpoint.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file
        input_dim : int, optional
            Input dimension (required if model not initialized)
        hidden_dims : list, optional
            Hidden dimensions (required if model not initialized)
        """
        device = check_device()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Initialize model if not exists
        self.model = MLPClassifier(input_dim, latent_dim, hidden_choice=hidden_choice
        ).to(device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize and load optimizer if exists in checkpoint
        if checkpoint['optimizer_state_dict'] is not None:
            self.optimizer = optim.Adam(self.model.parameters())
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self._name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def visualize(self, vis_arch=False, save=False):
        """Visualize the training and validation losses + model architecture."""
        plt.figure(figsize=[12, 7])
        plt.plot(self.history['train'], label='Train')
        plt.plot(self.history['val'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        if save:
            plt.savefig(pjoin(self.results_dir, 'train_val_losses.png'))
        else:
            plt.show()
        
        if vis_arch:
            self.model = self.model.cpu()
            self.model.visualize_architecture(pjoin(os.path.abspath(self.results_dir), 'model_arch.png'))
    
    def evaluate(self):
        """Compute the AUC score for the model."""
        device = check_device()

        print("Validation AUC:", self.compute_nn_auc(self.model, DataLoader(self.val_dataset, batch_size=64, shuffle=False), device=device, save=False, save_path='', title='Validation ROC Curve'))
        print("Training AUC:", self.compute_nn_auc(self.model, DataLoader(self.dataset, batch_size=64, shuffle=False), device=device, save=False, save_path='', title='Training ROC Curve'))

    def reweight(self, original, normalize, drop_kwd, save_results=False, save_name='') -> 'pd.DataFrame':
        """Reweight the original data.

        Parameters
        ----------
        original : pd.DataFrame
            Original data to be reweighted
        normalize : float
            Normalization factor
        drop_kwd : list
            Keywords for columns to drop
        save_results : bool, optional
            Whether to save results to file
        save_name : str, optional
            Name of file to save results
            
        Returns
        -------
        pd.DataFrame
            Reweighted data
        """
        X_df, weights, neg_df, _ = self.clean_data(
            original, drop_kwd, self.weight_column, drop_neg_wgts=True
        )
        
        X = self.scaler.transform(X_df)
        data = torch.tensor(X, dtype=torch.float32)
        
        device = check_device()
        predictions = PredictionUtils.predict_in_batches(self.model, data, device)
        
        # Compute new weights
        normalize_factor = normalize - neg_df[self.weight_column].sum()
        new_weights = PredictionUtils.compute_weights(
            predictions, weights.values, normalize_factor
        )
        
        # Create output DataFrame
        X_df[self.weight_column] = new_weights
        reweighted = pd.concat([X_df, neg_df], ignore_index=True)
        
        # Save if requested
        if save_results:
            reweighted.to_csv(pjoin(self.results_dir, f'{save_name}.csv'))
        
        return reweighted 
 
import torch
import logging
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

class SubtractionRwgtMixin:
    def reweight(self):
        pass

class SingleMLPRwgter(ReweighterBase):
    def __init__(self, src, tgt, w_col, out_dir, drop_kwd: 'list', criterion=nn.BCELoss()):
        super().__init__(src, tgt, w_col, out_dir)
        self.scaler = MinMaxScaler()
        self.drop_kwd = drop_kwd
        self.criterion = criterion

    def prep_data(self, drop_neg_wgts=True):
        """Preprocess the data into TensorDataset objects.
        Drop columns containing the keywords in `drop_kwd` and negatively weighted events if `drop_neg_wgts` is True."""
        X, y, weights = self.prep_distributions(self.src, self.tgt, self.drop_kwd, self.w_col, drop_wgts=True, drop_neg_wgts=drop_neg_wgts)
        features = list(X.columns)
        logging.info(f"Features: {features}")

        X[features] = self.scaler.fit_transform(X[features])

        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(X.values, y.values, weights.values, test_size=0.3, random_state=42)

        self.dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).view(-1, 1), torch.tensor(w_train, dtype=torch.float32))
        self.val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).view(-1, 1), torch.tensor(w_val, dtype=torch.float32))
        self.features = features

    def train(self, num_epochs, hidden_arch: 'str', batch_size, lr, dropout=0.2, save=True, 
             savename='SingleNNmodel.pth', save_interval=30, latent_dim=64):
        """Train the model with given parameters: hidden_arch (str), batch_size (int), 
        lr (float), dropout (float, optional), save (bool, optional), savename (str, optional), 
        save_interval (int, optional).
        
        Available hidden architectures: 'high_dim', 'stable', 'high_dim_with_attention', 'stable_with_attention'."""
        device = self.device
        input_dim = len(self.features)

        # Model setup
        self.model = MLPClassifier(input_dim, latent_dim, hidden_choice=hidden_arch, dropout_rate=dropout
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Data loaders
        train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            # Train
            train_loss = TrainingUtils.train_epoch(
                self.model, train_loader, self.optimizer, device, criterion=self.criterion,
            )
            train_losses.append(train_loss)

            # Validate
            val_loss = TrainingUtils.validate(self.model, val_loader, device, self.criterion)
            val_losses.append(val_loss)

            # Log progress
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{num_epochs}, "
                             f"Train Loss: {train_loss:.4f}, "
                             f"Val Loss: {val_loss:.4f}")

            # Save checkpoint
            if save and (epoch + 1) % save_interval == 0:
                save_path = pjoin(self.out_dir, 
                                  f"{savename.split('.')[0]}_{epoch+1}.pth")
                TrainingUtils.save_checkpoint(
                    self.model, epoch, save_path, self.optimizer
                )
                logging.info(f"Model saved at {save_path}")

        self.history = {'train': train_losses, 'val': val_losses}

        if save:
            final_path = pjoin(self.out_dir, savename)
            TrainingUtils.save_checkpoint(
                self.model, num_epochs, final_path, self.optimizer
            )
            logging.info(f"Final model saved at {final_path}")

        self._name = savename.split('.')[0]
        logging.info(f"Training complete. Model name: {self._name}")

    def load_model(self, ckpt_path, in_dim, arch_choice: 'str', latent_dim=64):
        """Load model from a checkpoint.
        
        Parameters
        ----------
        ckpt_path : str
            Path to the checkpoint file
        in_dim : int
            Input dimension
        arch_choice : str
            Hidden architecture choice
        latent_dim : int, optional
            Latent dimension size
        """
        device = self.device
        
        ckpt = torch.load(ckpt_path, map_location=device)
        
        self.model = MLPClassifier(in_dim, latent_dim, hidden_choice=arch_choice).to(device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        
        if ckpt['optimizer_state_dict'] is not None:
            self.optimizer = optim.Adam(self.model.parameters())
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        self._name = os.path.splitext(os.path.basename(ckpt_path))[0]

        logging.info(f"Loading model from {ckpt_path}, name: {self._name}")
        logging.info(f"Loaded model from epoch {ckpt['epoch']}")
        return ckpt['epoch']
    
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
            plt.savefig(pjoin(self.out_dir, 'train_val_losses.png'))
        else:
            plt.show()
        
        if vis_arch:
            self.model = self.model.cpu()
            self.model.visualize_architecture(pjoin(os.path.abspath(self.out_dir), 'model_arch.png'))
    
    def evaluate(self):
        """Compute the AUC score for the model."""
        device = self.device

        print("Validation AUC:", self.compute_nn_auc(self.model, DataLoader(self.val_dataset, batch_size=64, shuffle=False), device=device, save=False, save_path='', title='Validation ROC Curve'))
        print("Training AUC:", self.compute_nn_auc(self.model, DataLoader(self.dataset, batch_size=64, shuffle=False), device=device, save=False, save_path='', title='Training ROC Curve'))

    def reweight(self, data, norm_factor, save=False, filename='') -> pd.DataFrame:
        """Reweight the input data using the trained model.

        Parameters
        ----------
        data : pd.DataFrame
            Input data to reweight.
        norm_factor : float
            Normalization factor for weights.
        save : bool, optional
            Save results to a file if True.
        filename : str, optional
            Name of the output file.

        Returns
        -------
        pd.DataFrame
            Data with updated weights and reweighting ratios.
        """
        logging.info("Starting reweighting process.")
        X, orig_weights, neg_data, dropped = self.clean_data(data, self.drop_kwd, self.w_col, drop_neg_wgts=True)
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        logging.info("Predicting scores using the trained model.")
        scores = PredictionUtils.predict_in_batches(self.model, X_tensor, self.device)

        norm_adj = norm_factor - neg_data[self.w_col].sum()
        new_weights, rwgt_ratio = PredictionUtils.compute_weights(
            scores, orig_weights.values, norm_adj, criterion=self.criterion)

        if self.w_col in X.columns:
            raise ValueError(f"Column '{self.w_col}' already exists in the DataFrame.")

        X[self.w_col] = new_weights
        X['rwgt_ratio'] = rwgt_ratio

        if not dropped.empty:
            dropped = dropped.rename(columns={self.w_col: 'original_weights'})
            result = pd.concat([X, dropped], axis=1)
        else:
            result = X

        if save:
            output_path = pjoin(self.out_dir, f'{filename}.csv')
            result.to_csv(output_path, index=False)
            logging.info(f"Reweighted data saved to {output_path}")

        logging.info("Reweighting process completed.")
        return result

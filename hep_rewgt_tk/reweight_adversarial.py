from .reweight_base import ReweighterBase, check_device
from os.path import join as pjoin
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pandas as pd
import logging

class AdversarialReweighter(ReweighterBase):
    """Adversarial Reweighting Network for distribution matching.
    
    This network uses adversarial training to learn weights that make the source
    distribution match the target distribution. It consists of:
    1. Feature extractor: learns meaningful representations
    2. Discriminator: tries to distinguish source from target
    3. Weight estimator: produces weights to fool the discriminator
    """
    def __init__(self, ori_data, tar_data, weight_column, results_dir):
        super().__init__(ori_data, tar_data, weight_column, results_dir)
        self.scaler = None
        self.features = None
        self.latent_dim = 64  # Dimension of learned representation

    def build_model(self, input_dim):
        """Build the three components of the ARN."""
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, self.latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.latent_dim)
        )

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Weight estimator
        self.weight_estimator = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensures positive weights
        )

    def prep_data(self, drop_kwd, drop_neg_wgts=True):
        """Prepare data for training."""
        X, y, weights = self.prep_ori_tar(
            self.ori_data, self.tar_data, drop_kwd, 
            self.weight_column, drop_wgts=True, 
            drop_neg_wgts=drop_neg_wgts
        )
        self.features = list(X.columns)
        
        # Initialize and fit scaler
        self.scaler = MinMaxScaler()
        X[self.features] = self.scaler.fit_transform(X[self.features])

        # Split source and target data
        source_mask = (y == 0)
        source_data = X[source_mask].values
        source_weights = weights[source_mask].values
        target_data = X[~source_mask].values
        target_weights = weights[~source_mask].values

        # Create datasets
        self.source_dataset = TensorDataset(
            torch.FloatTensor(source_data),
            torch.FloatTensor(source_weights)
        )
        self.target_dataset = TensorDataset(
            torch.FloatTensor(target_data),
            torch.FloatTensor(target_weights)
        )

    def train(self, num_epochs=100, batch_size=256, lr_d=0.0002, lr_w=0.0001,
              lambda_reg=0.1, save=True, savename='ARN_model.pth'):
        """Train the adversarial reweighting network.
        
        Parameters
        ----------
        num_epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        lr_d : float
            Learning rate for discriminator
        lr_w : float
            Learning rate for weight estimator
        lambda_reg : float
            Regularization parameter for weight estimation
        """
        device = check_device()
        
        # Build model
        self.build_model(len(self.features))
        
        # Move models to device
        self.feature_extractor = self.feature_extractor.to(device)
        self.discriminator = self.discriminator.to(device)
        self.weight_estimator = self.weight_estimator.to(device)

        # Create dataloaders
        source_loader = DataLoader(
            self.source_dataset, batch_size=batch_size, shuffle=True
        )
        target_loader = DataLoader(
            self.target_dataset, batch_size=batch_size, shuffle=True
        )

        # Optimizers
        d_optimizer = optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.discriminator.parameters()),
            lr=lr_d
        )
        w_optimizer = optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.weight_estimator.parameters()),
            lr=lr_w
        )

        criterion = nn.BCELoss()
        
        d_losses = []
        w_losses = []

        for epoch in range(num_epochs):
            d_epoch_loss = 0
            w_epoch_loss = 0
            
            for (source_data, source_weights), (target_data, target_weights) in zip(
                source_loader, target_loader):
                
                batch_size = len(source_data)
                
                # Move data to device
                source_data = source_data.to(device)
                target_data = target_data.to(device)
                source_weights = source_weights.to(device)
                target_weights = target_weights.to(device)

                # Train discriminator
                d_optimizer.zero_grad()
                
                # Extract features
                source_features = self.feature_extractor(source_data)
                target_features = self.feature_extractor(target_data)
                
                # Get reweighting factors
                reweight_factors = self.weight_estimator(source_features)
                
                # Discriminator predictions
                source_pred = self.discriminator(source_features)
                target_pred = self.discriminator(target_features)
                
                # Discriminator loss
                d_loss_source = criterion(
                    source_pred,
                    torch.zeros_like(source_pred)
                )
                d_loss_target = criterion(
                    target_pred,
                    torch.ones_like(target_pred)
                )
                d_loss = (d_loss_source + d_loss_target) / 2
                
                d_loss.backward()
                d_optimizer.step()
                
                # Train weight estimator
                w_optimizer.zero_grad()
                
                # Extract features again
                source_features = self.feature_extractor(source_data)
                source_weights = self.weight_estimator(source_features)
                source_pred = self.discriminator(source_features)
                
                # Weight estimator loss
                w_loss = criterion(
                    source_pred,
                    torch.ones_like(source_pred)
                )
                
                # Add regularization to prevent extreme weights
                w_reg = lambda_reg * torch.mean((source_weights - 1)**2)
                w_total_loss = w_loss + w_reg
                
                w_total_loss.backward()
                w_optimizer.step()
                
                d_epoch_loss += d_loss.item()
                w_epoch_loss += w_total_loss.item()
            
            d_losses.append(d_epoch_loss)
            w_losses.append(w_epoch_loss)
            
            if epoch % 10 == 0:
                logging.info(
                    f"Epoch {epoch}/{num_epochs}: "
                    f"D_loss = {d_epoch_loss:.4f}, "
                    f"W_loss = {w_epoch_loss:.4f}"
                )

        self.history = {
            'discriminator_loss': d_losses,
            'weight_loss': w_losses
        }

        if save:
            state = {
                'feature_extractor': self.feature_extractor.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'weight_estimator': self.weight_estimator.state_dict()
            }
            torch.save(state, pjoin(self.results_dir, savename))

    def reweight(self, original, normalize, drop_kwd, 
                save_results=False, save_name=''):
        """Reweight the original data using the trained ARN."""
        X_df, weights, neg_df, _ = self.clean_data(
            original, drop_kwd, self.weight_column, drop_neg_wgts=True
        )
        
        # Transform features
        X = self.scaler.transform(X_df[self.features])
        data = torch.FloatTensor(X)
        
        # Get predictions
        device = check_device()
        self.feature_extractor.eval()
        self.weight_estimator.eval()
        
        with torch.no_grad():
            features = self.feature_extractor(data.to(device))
            new_weights = self.weight_estimator(features).cpu().numpy()
        
        # Normalize weights
        normalize_factor = normalize - neg_df[self.weight_column].sum()
        new_weights = new_weights.squeeze() * weights.values
        new_weights *= normalize_factor / new_weights.sum()
        
        # Create output DataFrame
        X_df[self.weight_column] = new_weights
        reweighted = pd.concat([X_df, neg_df], ignore_index=True)
        
        if save_results:
            reweighted.to_csv(pjoin(self.results_dir, save_name))
        
        return reweighted
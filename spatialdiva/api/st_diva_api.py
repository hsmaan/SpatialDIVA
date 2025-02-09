import sys 
sys.path.append('..')

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import wandb    
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from sklearn.decomposition import PCA
import umap

from utils.diva_data import find_knn, return_dataset, dataset_to_dataloader, adata_process
from models import SpatialDIVA, LitSpatialDIVA

def process_labels(labels, label_type):
    if label_type == "numeric":
        if ptypes.is_categorical_dtype(labels):
            return labels.values.to_numpy(dtype="float64")
        else:
            return labels.values
    else:
        if ptypes.is_categorical_dtype(labels):
            values = labels.values
            le = LabelEncoder()
            return le.fit_transform(values)
        else:
            le = LabelEncoder()
            return le.fit_transform(labels.values)


class StDIVA:
    def __init__(
        self, 
        counts_dim,
        hist_dim, 
        y1_dim,
        y2_dim,
        y3_dim,
        d_dim,
        y1_latent_dim = 20,
        y2_latent_dim = 20,
        y3_latent_dim = 20,
        d_latent_dim = 5,
        residual_latent_dim = 20,
        hidden_layers_x = [128, 64],
        hidden_layers_y = [128, 64],
        hidden_layers_d = [128, 64],
        num_y_covars = 3,
        y_covar_space = ["discrete", "continuous", "discrete"],
        y_dists = ["categorical", "NA", "categorical"],
        d_dist = "categorical",
        linear_decoder = True,
        spatial_loss_testing = False,
        spatial_gnn_encoder = False,
        spatial_covar_number = 2,
        restrict_recon_pos = False,
        restrict_recon_pos_cutoff = None,
        lr = 1e-3,
        betas = [1, 1, 1, 1, 1, 1],
        zys_betas_kl = [1.0, 1.0, 1.0],
        zys_betas_aux = [1.0, 1.0, 1.0],
        train_data = None,
        val_data = None,
        test_data = None
    ):
        
        # Set a distribution mask based on the counts and histology data - 
        # use neg_binomial for counts data and gaussian for histology data
        # by default
        distribution_mask = ["neg_binomial"]*counts_dim + ["gaussian"]*hist_dim

        self.model = SpatialDIVA(       
            x_dim = counts_dim + hist_dim,
            y_dims = [y1_dim, y2_dim, y3_dim],
            d_dim = d_dim,
            zx_dim = residual_latent_dim,
            zy_dims = [y1_latent_dim, y2_latent_dim, y3_latent_dim],
            zd_dim = d_latent_dim,
            hidden_layers_x = hidden_layers_x,
            hidden_layers_y = hidden_layers_y,
            hidden_layers_d = hidden_layers_d,
            num_y_covars = num_y_covars,
            y_covar_space = y_covar_space,
            y_dists = y_dists,
            d_dist = d_dist,
            linear_decoder = linear_decoder,
            spatial_loss_testing = spatial_loss_testing,
            spatial_gnn_encoder = spatial_gnn_encoder,
            spatial_covar_number = spatial_covar_number,
            restrict_recon_pos = restrict_recon_pos,
            restrict_recon_pos_cutoff = restrict_recon_pos_cutoff,
            distribution_mask = distribution_mask,
            lr = lr,
            betas = betas,
            zys_betas_kl = zys_betas_kl,
            zys_betas_aux = zys_betas_aux,
            train_data = train_data,
            val_data = val_data,
            test_data = test_data
        )
        
    def add_data(self, adata, train_index, val_index = None, label_key_y1 = None, 
                 label_key_y2 = None, label_key_y3 = None, label_key_d = None, 
                 pos_key = None, hist_col_key = "UNI"):
        
        # Process the anndata object 
        adata = adata_process(
            normalize=False,
            standardize_sct=False,
            standardize_uni=True,
            n_top_genes=2500,
            n_neighbors_pca=15,
            knn_type="spatial",
        )
        
        self.train_data = adata[train_index]
        if val_index is not None:   
            self.val_data = adata[val_index]
            
        # Create the train dataloader 
        count_data = np.asarray(self.train_data.X)
        hist_cols = [col for col in self.train_data.obs.columns if hist_col_key in col]
        hist_data = self.train_data.obsm[hist_cols].values
        
        count_hist_data = np.concatenate([count_data, hist_data], axis=1)
        
        st_labels = process_labels(self.train_data.obs[label_key_y1], "categorical")
        morpho_labels = process_labels(self.train_data.obs[pos_key], "categorical")
        
        neighbor_data = self.train_data.obsm["X_pca_neighbors_avg"]
        
        spatial_positions = self.train_data.obsm["spatial"]
        
        num_classes_morpho = len(np.unique(morpho_labels))
        morpho_labels_onehot = np.eye(num_classes_morpho)[morpho_labels]
        
        num_classes_st = len(np.unique(st_labels))
        st_labels_onehot = np.eye(num_classes_st)[st_labels]
        
        batch_labels = process_labels(self.train_data.obs["batch"], "categorical")
        num_classes_batch = len(np.unique(batch_labels))
        batch_labels_onehot = np.eye(num_classes_batch)[batch_labels]
        
        count_hist_data_tensor = torch.from_numpy(count_hist_data)
        st_labels_tensor = torch.from_numpy(st_labels_onehot)
        morpho_labels_tensor = torch.from_numpy(morpho_labels_onehot)
        batch_labels_tensor = torch.from_numpy(batch_labels_onehot)
        neighbor_data_tensor = torch.from_numpy(neighbor_data)
        spatial_positions_tensor = torch.from_numpy(spatial_positions)
        
        self.train_dataset = torch.utils.data.TensorDataset(
            count_hist_data_tensor,
            st_labels_tensor,
            morpho_labels_tensor,
            batch_labels_tensor,
            neighbor_data_tensor,
            spatial_positions_tensor
        )
        
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=128, shuffle=True)
        
        # Do the same for the validation data
        if self.val_data is not None:
            count_data_val = np.asarray(self.val_data.X)
            hist_cols_val = [col for col in self.val_data.obs.columns if hist_col_key in col]
            hist_data_val = self.val_data.obsm[hist_cols_val].values
            
            count_hist_data_val = np.concatenate([count_data_val, hist_data_val], axis=1)
            
            st_labels_val = process_labels(self.val_data.obs[label_key_y1], "categorical")
            morpho_labels_val = process_labels(self.val_data.obs[pos_key], "categorical")
            
            num_classes_morpho = len(np.unique(morpho_labels_val))
            morpho_labels_onehot_val = np.eye(num_classes_morpho)[morpho_labels_val]
            
            num_classes_st = len(np.unique(st_labels_val))
            st_labels_onehot_val = np.eye(num_classes_st)[st_labels_val]
            
            batch_labels_val = process_labels(self.val_data.obs["batch"], "categorical")
            num_classes_batch = len(np.unique(batch_labels_val))
            batch_labels_onehot_val = np.eye(num_classes_batch)[batch_labels_val]   
            
            count_hist_data_val_tensor = torch.from_numpy(count_hist_data_val)
            st_labels_val_tensor = torch.from_numpy(st_labels_onehot_val)
            morpho_labels_val_tensor = torch.from_numpy(morpho_labels_onehot_val)
            batch_labels_val_tensor = torch.from_numpy(batch_labels_onehot_val)
            
            neighbor_data_val = self.val_data.obsm["X_pca_neighbors_avg"]
            neighbor_data_val_tensor = torch.from_numpy(neighbor_data_val)
            
            spatial_positions_val = self.val_data.obsm["spatial"]
            spatial_positions_val_tensor = torch.from_numpy(spatial_positions_val)  
            
            self.val_dataset = torch.utils.data.TensorDataset(
                count_hist_data_val_tensor,
                st_labels_val_tensor,
                morpho_labels_val_tensor,
                batch_labels_val_tensor,
                neighbor_data_val_tensor,
                spatial_positions_val_tensor
            )
            
            self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=128, shuffle=False)
     
    def train(self, max_epochs = 100, early_stopping = True, patience = 10):
        # Train the model using torch lightning 
        trainer = Trainer(
            max_epochs = max_epochs,
            callbacks = [EarlyStopping(monitor="val_loss", patience=patience)] if early_stopping else None
        )
        
        trainer.fit(self.model, self.train_loader, self.val_loader)
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        
        
    def get_embeddings(self, type = "train"):
        if type == "train":
            loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=128, shuffle=False)
        elif type == "val":
            loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=128, shuffle=False)
        else:
            raise ValueError("Invalid type")
        
        # Get the posterior embeddings for the data 
        self.model.eval()
        y1_labels = []
        y2_labels = []
        y3_labels = []
        d_labels = []
        zy1_samples = []
        zy2_samples = []
        zy3_samples = []
        zd_samples = []
        zx_samples = []
            
        with torch.no_grad():
            for batch in loader:
                x, y1, y2, y3, d, spo_var = batch
                x = x.double()
                y1 = y1.double()
                y2 = y2.double()
                y3 = y3.double()
                d = d.double()
                d = d.unsqueeze(dim=-1)

                y = [y1, y2, y3]
                zx, zys, zd = self.model.get_posterior(x)

                # Detach, move to CPU and convert to numpy
                y1 = y1.detach().cpu().numpy()
                y2 = spo_var.detach().cpu().numpy()
                y3 = y3.detach().cpu().numpy()
                d = d.detach().cpu().numpy()
                zd = zd.detach().cpu().numpy()
                zx = zx.detach().cpu().numpy()
                zys = [zy.detach().cpu().numpy() for zy in zys]

                # Append the labels
                y1_labels.append(y1)
                y2_labels.append(y2)
                y3_labels.append(y3)
                d_labels.append(d)
                zd_samples.append(zd)
                zy1_samples.append(zys[0])
                zy2_samples.append(zys[1])
                zy3_samples.append(zys[2])
                zx_samples.append(zx)
                    
        y1_labels = np.concatenate(y1_labels, axis=0)
        y2_labels = np.concatenate(y2_labels, axis=0)
        y3_labels = np.concatenate(y3_labels, axis=0)
        d_labels = np.concatenate(d_labels, axis=0)
        zd_samples = np.concatenate(zd_samples, axis=0)
        zy1_samples = np.concatenate(zy1_samples, axis=0)
        zy2_samples = np.concatenate(zy2_samples, axis=0)
        zy3_samples = np.concatenate(zy3_samples, axis=0)
        zx_samples = np.concatenate(zx_samples, axis=0)
        
        # Return the embeddings 
        return y1_labels, y2_labels, y3_labels, d_labels, zd_samples, zy1_samples, zy2_samples, \
            zy3_samples, zx_samples
        
    def reduce_embedding(self, embedding, method = "pca"):
        if method == "pca":
            pca = PCA(n_components=2)
            return pca.fit_transform(embedding  )
        elif method == "umap":
            reducer = umap.UMAP(n_components=2)
            return reducer.fit_transform(embedding)
        else:
            raise ValueError("Invalid method")
        
        
    
        
        
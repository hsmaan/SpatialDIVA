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
import anndata as ann 
from scvi.distributions import NegativeBinomial

from utils.diva_data import adata_process
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

        self.model = LitSpatialDIVA(
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
        
        self.train_loader = None
        self.val_loader = None
        
    def add_data(self, adata, train_index = None, val_index = None, label_key_y1 = None, 
                 label_key_y2 = None, label_key_y3 = None, hist_col_key = "UNI", hvg = True):
        
        print("Processing data..")
        self.label_key_y1 = label_key_y1
        self.label_key_y2 = label_key_y2
        self.label_key_y3 = label_key_y3
        
        # Process the anndata object 
        adatas = adata_process(
            adata_files = adata,
            normalize=False,
            standardize_sct=False,
            standardize_uni=True,
            n_top_genes=2500,
            n_neighbors_pca=15,
            knn_type="spatial",
        )
        
        # Get the union of the HVGs 
        self.adata = ann.concat(adatas)
        
        # Retain information about highly variable genes from
        # the individual datasets
        hvg = adatas[0].var.highly_variable.copy()
        self.adata.var["highly_variable"] = hvg
        
        if train_index is None and val_index is None:
            val_pct = 0.1
            self.val_index = np.random.choice(self.adata.shape[0], int(val_pct*self.adata.shape[0]), replace=False)
            self.train_index = np.setdiff1d(np.arange(self.adata.shape[0]), val_index)
        print("Creating dataloaders..")
            
        # Create the train and val dataloaders
        if hvg is True:
            count_data = np.asarray(self.adata[:, self.adata.var["highly_variable"]].X)
        else:
            count_data = np.asarray(self.adata.X)
        hist_cols = [col for col in self.adata.obs.columns if hist_col_key in col]
        hist_data = self.adata.obs[hist_cols].values
        
        count_hist_data = np.concatenate([count_data, hist_data], axis=1)
        
        st_labels = process_labels(self.adata.obs[label_key_y1], "categorical")
        morpho_labels = process_labels(self.adata.obs[label_key_y3], "categorical")
        
        if label_key_y2 is None:
            label_key_y2 = "X_pca_neighbors_avg"
        neighbor_data = self.adata.obsm[label_key_y2]
        
        spatial_positions = self.adata.obsm["spatial"]
        
        num_classes_morpho = len(np.unique(morpho_labels))
        morpho_labels_onehot = np.eye(num_classes_morpho)[morpho_labels]
        
        num_classes_st = len(np.unique(st_labels))
        st_labels_onehot = np.eye(num_classes_st)[st_labels]
        
        batch_labels = process_labels(self.adata.obs["batch"], "categorical")
        num_classes_batch = len(np.unique(batch_labels))
        batch_labels_onehot = np.eye(num_classes_batch)[batch_labels]
        
        count_hist_data_tensor = torch.from_numpy(count_hist_data)
        st_labels_tensor = torch.from_numpy(st_labels_onehot)
        morpho_labels_tensor = torch.from_numpy(morpho_labels_onehot)
        batch_labels_tensor = torch.from_numpy(batch_labels_onehot)
        neighbor_data_tensor = torch.from_numpy(neighbor_data)
        spatial_positions_tensor = torch.from_numpy(spatial_positions)
        
        self.full_datasets = torch.utils.data.TensorDataset(
            count_hist_data_tensor,
            st_labels_tensor,
            neighbor_data_tensor,
            morpho_labels_tensor,
            batch_labels_tensor,
            spatial_positions_tensor
        )
        
        # Split the data into train and validation based on the TensorDataset
        self.train_dataset = torch.utils.data.Subset(self.full_datasets, self.train_index)
        self.val_dataset = torch.utils.data.Subset(self.full_datasets, self.val_index)
        
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=128, 
                                                        shuffle=True, num_workers=0)
        
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=128, 
                                                      shuffle=False, num_workers=0)
        
        print("Data processing complete!")
     
    def train(self, max_epochs = 100, early_stopping = True, patience = 10):
        
        print("Starting training..")
        
        # Train the model using torch lightning 
        trainer = Trainer(
            max_epochs = max_epochs,
            callbacks = [EarlyStopping(monitor="val_loss", patience=patience)] if early_stopping else None
        )
        trainer.fit(self.model, self.train_loader, self.val_loader)
        
        print("Training complete!")
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
              
    def get_embeddings(self, type = "full", sample_number = 1000):
        if type == "train":
            loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=128, shuffle=False)
        elif type == "val":
            loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=128, shuffle=False)
        elif type == "full":
            loader = torch.utils.data.DataLoader(self.full_datasets, batch_size=128, shuffle=False)
        elif type == "sample":
            sample_dataset = torch.utils.data.Subset(
                self.full_datasets, 
                np.random.choice(np.arange(self.full_datasets.tensors[0].shape[0]),
                sample_number, 
                replace=False)
            )
            batch_size = 128 if sample_number > 128 else sample_number
            loader = torch.utils.data.DataLoader(sample_dataset, batch_size=batch_size, shuffle=False)
        else:
            raise ValueError("Invalid type")
        
        # Get the posterior embeddings for the data 
        self.model.eval()
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
                
                y = [y1, y2, y3]
                zx, zys, zd = self.model.model.get_posterior(x)

                # Detach, move to CPU and convert to numpy
                y1 = y1.detach().cpu().numpy()
                y2 = spo_var.detach().cpu().numpy()
                y3 = y3.detach().cpu().numpy()
                d = d.detach().cpu().numpy()
                zd = zd.detach().cpu().numpy()
                zx = zx.detach().cpu().numpy()
                zys = [zy.detach().cpu().numpy() for zy in zys]

                # Append the embeddings to the list
                zd_samples.append(zd)
                zy1_samples.append(zys[0])
                zy2_samples.append(zys[1])
                zy3_samples.append(zys[2])
                zx_samples.append(zx)
                    
        zd_samples = np.concatenate(zd_samples, axis=0)
        zy1_samples = np.concatenate(zy1_samples, axis=0)
        zy2_samples = np.concatenate(zy2_samples, axis=0)
        zy3_samples = np.concatenate(zy3_samples, axis=0)
        zx_samples = np.concatenate(zx_samples, axis=0)
        
        # Return the embeddings 
        return zd_samples, zy1_samples, zy2_samples, \
            zy3_samples, zx_samples
        
    def get_labels(self, type = "full"):
        if type == "train":
            adata_sub = self.adata[self.train_index]
        elif type == "val":
            adata_sub = self.adata[self.val_index]
        elif type == "full":
            adata_sub = self.adata
            
        y1_labels = adata_sub.obs[self.label_key_y1].values
        y2_labels = adata_sub.obsm["spatial"]
        y3_labels = adata_sub.obs[self.label_key_y3].values
        d_labels = adata_sub.obs["batch"].values
        
        return y1_labels, y2_labels, y3_labels, d_labels
        
    def reduce_embedding(self, embedding, method = "pca"):
        if method == "pca":
            pca = PCA(n_components=2)
            return pca.fit_transform(embedding)
        elif method == "umap":
            reducer = umap.UMAP(n_components=2)
            return reducer.fit_transform(embedding)
        else:
            raise ValueError("Invalid method")
        
    def conditional_generation(self, covariate_num, num_samples = 1000):
        # Extract the samples for all covariates 
        zd_samples, zy1_samples, zy2_samples, zy3_samples, \
            zx_samples = self.get_embeddings(type = "sample", sample_number = num_samples)

        # Zero out samples for all but the covariate of interest
        if covariate_num == 1:
            zy2_samples = np.zeros_like(zy2_samples)
            zy3_samples = np.zeros_like(zy3_samples)
            zd_samples = np.zeros_like(zd_samples)
            zx_samples = np.zeros_like(zx_samples)
        elif covariate_num == 2:
            zy1_samples = np.zeros_like(zy1_samples)
            zy3_samples = np.zeros_like(zy3_samples)
            zd_samples = np.zeros_like(zd_samples)
            zx_samples = np.zeros_like(zx_samples)
        elif covariate_num == 3:
            zy1_samples = np.zeros_like(zy1_samples)
            zy2_samples = np.zeros_like(zy2_samples)
            zd_samples = np.zeros_like(zd_samples)
            zx_samples = np.zeros_like(zx_samples)
        else:
            raise ValueError("Invalid covariate number - must be 1, 2, or 3")
    
        # Create a dataloader with the modified embeddings
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(zx_samples),
            torch.from_numpy(zy1_samples),
            torch.from_numpy(zy2_samples),
            torch.from_numpy(zy3_samples),
            torch.from_numpy(zd_samples)
        )
        batch_size = 128 if num_samples > 128 else num_samples
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Generate the gaussian and negative binomial parameters
        gauss_mu_params = []
        gauss_logvar_params = []
        
        negbin_mu_params = []
        negbin_theta_params = []
        
        with torch.no_grad():
            for batch in loader:
                zx, zy1, zy2, zy3, zd = batch
                zx = zx.double()
                zy1 = zy1.double()
                zy2 = zy2.double()
                zy3 = zy3.double()
                zd = zd.double()
                
                zy_cond_full = torch.cat([zy1, zy2, zy3], dim=1)
                
                mu_gauss, logvar_gauss, mu_negbin, theta_negbin = self.model.model.px(
                    zx, zd, zy_cond_full
                )
                
                # Detach, move to CPU and convert to numpy
                mu_gauss = mu_gauss.detach().cpu().numpy()
                logvar_gauss = logvar_gauss.detach().cpu().numpy()
                
                mu_negbin = mu_negbin.detach().cpu().numpy()
                theta_negbin = theta_negbin.detach().cpu().numpy()
                
                # Append the parameters to the list
                gauss_mu_params.append(mu_gauss)
                gauss_logvar_params.append(logvar_gauss)
                
                negbin_mu_params.append(mu_negbin)
                negbin_theta_params.append(theta_negbin)
                
        # Concatenate the parameters
        gauss_mu_params = np.concatenate(gauss_mu_params, axis=0)
        gauss_logvar_params = np.concatenate(gauss_logvar_params, axis=0)
        
        negbin_mu_params = np.concatenate(negbin_mu_params, axis=0)
        negbin_theta_params = np.concatenate(negbin_theta_params, axis=0)
        
        # Reconvert negbin paramters to torch
        negbin_mu_params = torch.from_numpy(negbin_mu_params)
        negbin_theta_params = torch.from_numpy(negbin_theta_params)
        
        # Create a negative binomial distribution object
        negbin_zy_cond = NegativeBinomial(mu=negbin_mu_params, theta=negbin_theta_params)
        
        # sample from the negative binomial distributions
        negbin_sample = negbin_zy_cond.sample()
        
        # Convert sample to numpy
        negbin_sample = negbin_sample.detach().cpu().numpy()
        
        # Return the parameters and samples
        return gauss_mu_params, gauss_logvar_params, negbin_mu_params, negbin_theta_params, \
            negbin_sample
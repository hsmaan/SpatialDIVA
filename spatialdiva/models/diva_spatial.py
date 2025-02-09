import math

import torch 
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
from torch.nn.functional import gaussian_nll_loss
from torch.distributions.kl import kl_divergence 
from torch.utils.data import TensorDataset, DataLoader
import wandb
import numpy as np 
import lightning as L
import umap
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from torch_geometric.nn import GATConv
from torch_geometric.nn import Sequential as GCNSequential
from torch.special import gammaln

# Set figure parameters
FIGSIZE = (6, 6)
rcParams["figure.figsize"] = FIGSIZE

# Function for applying a selective softplus to a tensor
class PartialSoftplus(nn.Module):
    def __init__(self, feature_indices):
        super(PartialSoftplus, self).__init__()
        self.feature_indices = feature_indices

    def forward(self, x):
        # Compute Softplus over the entire tensor
        x_softplus = F.softplus(x)

        # Create a mask for the features to apply Softplus
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, self.feature_indices] = True

        # Use torch.where to select between the original x and x_softplus
        x = torch.where(mask, x_softplus, x)
        return x

# Define decoders for DIVA model for spatial transcriptomic data
class px(nn.Module):
    def __init__(
        self,
        hidden_layers,
        x_dim,
        zx_dim,
        zd_dim,
        zy_dim,
        decoder_type="likelihood",
        linear_decoder=False,
        restrict_recon_pos=False,
        restrict_recon_pos_cutoff=None,
        distribution_mask=None,  
    ):
        super(px, self).__init__()

        self.decoder_type = decoder_type
        self.restrict_recon_pos = restrict_recon_pos
        self.restrict_recon_pos_cutoff = restrict_recon_pos_cutoff

        # If none provided, assume all features are Gaussian
        if distribution_mask is None:
            distribution_mask = ["gaussian"] * x_dim
        self.distribution_mask = distribution_mask
        self.x_dim = x_dim

        if linear_decoder:
            modules = []
            hidden_dim_rev = hidden_layers[::-1]
            latent_dim_concat = zd_dim + zx_dim + zy_dim
            modules.append(
                nn.Sequential(
                    nn.Linear(latent_dim_concat, hidden_dim_rev[0]),
                )
            )
            latent_dim_concat = hidden_dim_rev[0]
        else:
            modules = []
            hidden_dim_rev = hidden_layers[::-1]
            latent_dim_concat = zd_dim + zx_dim + zy_dim
            for hidden_dim in hidden_dim_rev:
                modules.append(
                    nn.Sequential(
                        nn.Linear(latent_dim_concat, hidden_dim),
                        nn.ReLU()
                    )
                )
                latent_dim_concat = hidden_dim

        # For "likelihood"-type decoder, produce distribution parameters
        if decoder_type == "likelihood":
            # For Gaussian features: we produce mu & logvar
            self.fc_mu_x = nn.Linear(latent_dim_concat, x_dim)
            self.fc_logvar_x = nn.Linear(latent_dim_concat, x_dim)
            torch.nn.init.xavier_uniform_(self.fc_mu_x.weight)
            torch.nn.init.xavier_uniform_(self.fc_logvar_x.weight)

            # For NB features: we produce mu_nb & theta_nb
            # We'll do a linear layer for each: 
            # mu_nb > 0 so we can apply softplus, or exponent
            self.fc_mu_nb = nn.Linear(latent_dim_concat, x_dim)
            # theta_nb > 0 so we can apply softplus, or exponent
            self.fc_theta_nb = nn.Linear(latent_dim_concat, x_dim)
            torch.nn.init.xavier_uniform_(self.fc_mu_nb.weight)
            torch.nn.init.xavier_uniform_(self.fc_theta_nb.weight)
        elif decoder_type == "recon":
            # If "recon", do direct output
            if self.restrict_recon_pos:
                modules.append(
                    nn.Sequential(
                        nn.Linear(latent_dim_concat, x_dim),
                        PartialSoftplus(feature_indices=range(self.restrict_recon_pos_cutoff))
                    )
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Linear(latent_dim_concat, x_dim)
                    )
                )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
        # Xavier init for any direct Linear layers in modules
        for block in modules:
            for m in block:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)

        self.decoder_px = nn.Sequential(*modules)

    def forward(self, zx, zd, zy):
        # Concat latents
        if zx is None:
            latent = torch.cat((zd, zy), dim=1)
        else:
            latent = torch.cat((zd, zx, zy), dim=1)

        # Pass through MLP stack => final "h"
        h = self.decoder_px(latent)

        # If not "likelihood", return direct reconstruction
        if self.decoder_type != "likelihood":
            return h

        # Otherwise, produce distribution parameters for each feature
        mu_gauss = self.fc_mu_x(h)        # (batch, x_dim)
        logvar_gauss = self.fc_logvar_x(h)
        mu_nb = self.fc_mu_nb(h)          # (batch, x_dim)
        theta_nb = self.fc_theta_nb(h)

        # If restrict_recon_pos for Gaussian, partial softplus on the relevant subset
        if self.restrict_recon_pos:
            psp = PartialSoftplus(feature_indices=range(self.restrict_recon_pos_cutoff))
            mu_gauss = psp(mu_gauss)
        mu_nb = F.softplus(mu_nb)     # ensure mu > 0
        theta_nb = F.softplus(theta_nb)   # ensure theta > 0

        return mu_gauss, logvar_gauss, mu_nb, theta_nb

class pzd(nn.Module):
    def __init__(self, hidden_layers, d_dim, zd_dim):
        super(pzd, self).__init__()
        
        # Reconstruction of d using hidden layers
        modules = []
        hidden_dim_rev = hidden_layers[::-1]
        latent_dim_concat = d_dim
        for hidden_dim in hidden_dim_rev:
            modules.append(
                nn.Sequential(
                    nn.Linear(latent_dim_concat, hidden_dim),
                    nn.ReLU()
                )
            )
            latent_dim_concat = hidden_dim
        self.fc_mu_d = nn.Linear(latent_dim_concat, zd_dim)
        self.fc_logvar_d = nn.Linear(latent_dim_concat, zd_dim)
        
        # Xavier initialization for all modules
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                
        # Define the decoder
        self.decoder_pzd = nn.Sequential(*modules)
        
    def forward(self, d):
        h = self.decoder_pzd(d)
        mu_zd = self.fc_mu_d(h)
        logvar_zd = self.fc_logvar_d(h)
        return mu_zd, logvar_zd

class pzy(nn.Module):
    def __init__(self, hidden_layers, y_dim, zy_dim):
        super(pzy, self).__init__()
        
        # Reconstruction of y using hidden layers
        modules = []
        hidden_dim_rev = hidden_layers[::-1]
        latent_dim_concat = y_dim
        for hidden_dim in hidden_dim_rev:
            modules.append(
                nn.Sequential(
                    nn.Linear(latent_dim_concat, hidden_dim),
                    nn.ReLU()
                )
            )
            latent_dim_concat = hidden_dim
        self.fc_mu_y = nn.Linear(latent_dim_concat, zy_dim)
        self.fc_logvar_y = nn.Linear(latent_dim_concat, zy_dim)
        
        # Xavier initialization for all modules
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                
        # Define the decoder
        self.decoder_pzy = nn.Sequential(*modules)
        
    def forward(self, zy):
        h = self.decoder_pzy(zy)
        mu_zy = self.fc_mu_y(h)
        logvar_zy = self.fc_logvar_y(h)
        return mu_zy, logvar_zy
    
# Define the encoders for the DIVA model
class qzx(nn.Module):
    def __init__(self, hidden_layers, x_dim, zx_dim):
        super(qzx, self).__init__()
        
        # Inference of zx using hidden layers
        modules = []
        latent_dim_concat = x_dim
        for hidden_dim in hidden_layers:
            modules.append(
                nn.Sequential(
                    nn.Linear(latent_dim_concat, hidden_dim),
                    nn.ReLU()
                )
            )
            latent_dim_concat = hidden_dim
        self.fc_mu_zx = nn.Linear(latent_dim_concat, zx_dim)
        self.fc_logvar_zx = nn.Linear(latent_dim_concat, zx_dim)
        
        # Xavier initialization for all modules
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                
        # Define the encoder
        self.encoder_qzx = nn.Sequential(*modules)
        
    def forward(self, x):
        h = self.encoder_qzx(x)
        mu_zx = self.fc_mu_zx(h)
        logvar_zx = self.fc_logvar_zx(h)
        return mu_zx, logvar_zx
    
class qzd(nn.Module):
    def __init__(self, hidden_layers, x_dim, zd_dim):
        super(qzd, self).__init__()
        
        # Inference of zd using hidden layers
        modules = []
        latent_dim_concat = x_dim
        for hidden_dim in hidden_layers:
            modules.append(
                nn.Sequential(
                    nn.Linear(latent_dim_concat, hidden_dim),
                    nn.ReLU()
                )
            )
            latent_dim_concat = hidden_dim
        self.fc_mu_zd = nn.Linear(latent_dim_concat, zd_dim)
        self.fc_logvar_zd = nn.Linear(latent_dim_concat, zd_dim)
        
        # Xavier initialization for all modules
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                
        # Define the encoder
        self.encoder_qzd = nn.Sequential(*modules)
        
    def forward(self, d):
        h = self.encoder_qzd(d)
        mu_zd = self.fc_mu_zd(h)
        logvar_zd = self.fc_logvar_zd(h)
        return mu_zd, logvar_zd

class qzy(nn.Module):
    def __init__(self, hidden_layers, x_dim, zy_dim):
        super(qzy, self).__init__()
        
        # Inference of zy using hidden layers
        modules = []
        latent_dim_concat = x_dim
        for hidden_dim in hidden_layers:
            modules.append(
                nn.Sequential(
                    nn.Linear(latent_dim_concat, hidden_dim),
                    nn.ReLU()
                )
            )
            latent_dim_concat = hidden_dim
        self.fc_mu_zy = nn.Linear(latent_dim_concat, zy_dim)
        self.fc_logvar_zy = nn.Linear(latent_dim_concat, zy_dim)
        
        # Xavier initialization for all modules
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                
        # Define the encoder
        self.encoder_qzy = nn.Sequential(*modules)
        
    def forward(self, y):
        h = self.encoder_qzy(y)
        mu_zy = self.fc_mu_zy(h)
        logvar_zy = self.fc_logvar_zy(h)
        return mu_zy, logvar_zy
    
class qzy_spatial(nn.Module):
    def __init__(self, hidden_layers, x_dim, zy_dim, heads=4, dropout=0.5):
        # @TODO - expand hidden layers to beyond 2 for later cases
        super(qzy_spatial, self).__init__()
        self.hidden_layers = hidden_layers            
        self.dropout = dropout
    
        self.conv1 = GATConv(x_dim, hidden_layers[0], heads = heads, dropout = dropout)
        if len(hidden_layers) > 1:
            self.conv2 = GATConv(hidden_layers[0] * heads, hidden_layers[1], heads = heads, 
                                 concat = False, dropout = dropout)
            self.fc_mu_zy = nn.Linear(hidden_layers[1], zy_dim)
            self.fc_logvar_zy = nn.Linear(hidden_layers[1], zy_dim)
        else:
            self.fc_mu_zy = nn.Linear(hidden_layers[0], zy_dim)
            self.fc_logvar_zy = nn.Linear(hidden_layers[0], zy_dim)
        
    def forward(self, x, edge_index):
        if len(self.hidden_layers) > 1: 
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.conv2(x, edge_index))
            mu_zy = self.fc_mu_zy(x)
            logvar_zy = self.fc_logvar_zy(x)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
            mu_zy = self.fc_mu_zy(x)
            logvar_zy = self.fc_logvar_zy(x)
            
        return mu_zy, logvar_zy
    
# Define auxillary task heads 
class qd(nn.Module):
    def __init__(self, d_dim, zd_dim):
        super(qd, self).__init__()
        
        # Prediction of d using single linear layer
        self.fc1 = nn.Linear(zd_dim, d_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        
    def forward(self, zd):
        h = F.relu(zd)
        loc_d = self.fc1(h)
        return loc_d
    
class qy(nn.Module):
    def __init__(self, y_dim, zy_dim):
        super(qy, self).__init__()
        
        # Prediction of y using single linear layer
        self.fc1 = nn.Linear(zy_dim, y_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        
    def forward(self, zy):
        h = F.relu(zy)
        loc_y = self.fc1(h)
        return loc_y
    
# Reparametrization trick 
def reparametrize(mu, logvar, current_device):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std).to(current_device).double()
    return mu + eps*std

# Negative binomial loss formulation borrowed from scvitools - 
# https://github.com/scverse/scvi-tools/blob/main/src/scvi/distributions/_negative_binomial.py#L86
def log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
) -> torch.Tensor:
    """Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
    lgamma_fn
        log gamma function
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    return res

# Mixed likelihood loss for Gaussian and Negative Binomial
def mixed_recon_loss_gauss_nb(
    x: torch.Tensor,              # (batch, x_dim)
    mu_gauss: torch.Tensor,       # (batch, x_dim)
    logvar_gauss: torch.Tensor,   # (batch, x_dim)
    mu_nb: torch.Tensor,          # (batch, x_dim)
    theta_nb: torch.Tensor,       # (batch, x_dim)
    distribution_mask: list       # length = x_dim, each "gaussian" or "neg_binomial"
):
    gauss_idx = [i for i, dist_name in enumerate(distribution_mask) if dist_name == "gaussian"]
    nb_idx    = [i for i, dist_name in enumerate(distribution_mask) if dist_name == "neg_binomial"]

    x_gauss = x[:, gauss_idx] if gauss_idx else None
    mu_gauss_ = mu_gauss[:, gauss_idx] if gauss_idx else None
    logvar_gauss_ = logvar_gauss[:, gauss_idx] if gauss_idx else None

    x_nb = x[:, nb_idx] if nb_idx else None
    mu_nb_ = mu_nb[:, nb_idx] if nb_idx else None
    theta_nb_ = theta_nb[:, nb_idx] if nb_idx else None

    nll_gauss = 0.0
    if x_gauss is not None and x_gauss.shape[1] > 0:
        # sum across dimension -> shape (batch,)
        nll_gauss = gaussian_nll_loss(x_gauss, mu_gauss_, torch.ones_like(logvar_gauss_), reduction="none")
        nll_gauss = nll_gauss.sum(dim=-1)  

    nll_nb = 0.0
    if x_nb is not None and x_nb.shape[1] > 0:
        log_ll_nb = log_nb_positive(
            x=x_nb,
            mu=mu_nb_,
            theta=theta_nb_,
            eps=1e-8,
            log_fn=torch.log,
            lgamma_fn=torch.lgamma
        )
        nll_nb = -log_ll_nb.sum(dim=-1)  # shape (batch,)

    # Take the mean across both the Gaussian and Negative Binomial losses
    nll_gauss = nll_gauss.mean(dim=0)
    nll_nb = nll_nb.mean(dim=0)
    nll_total = nll_gauss + nll_nb

    return nll_total
    
# Define the DIVA model
class SpatialDIVA(nn.Module):
    def __init__(self, x_dim, y_dims, d_dim, zx_dim, zy_dims, zd_dim, 
                 hidden_layers_x, hidden_layers_y, hidden_layers_d, 
                 num_y_covars = 1, y_covar_space = ["discrete"], 
                 y_dists = ["bernoulli"], d_dist = "bernoulli",
                 linear_decoder = False, spatial_loss_testing = False,
                 spatial_gnn_encoder = False, spatial_covar_number = 2,
                 restrict_recon_pos = False, restrict_recon_pos_cutoff = None,
                 distribution_mask = None):
        super(SpatialDIVA, self).__init__()
        self.x_dim = x_dim
        self.y_dims = y_dims
        self.d_dim = d_dim
        self.zx_dim = zx_dim
        self.zy_dims = zy_dims
        self.zd_dim = zd_dim
        self.hidden_layers_x = hidden_layers_x
        self.hidden_layers_y = hidden_layers_y
        self.hidden_layers_d = hidden_layers_d
        self.num_covars_y = num_y_covars
        self.y_covar_space = y_covar_space
        self.y_dists = y_dists
        self.d_dist = d_dist
        self.linear_decoder = linear_decoder
        self.spatial_loss_testing = spatial_loss_testing
        self.spatial_gnn_encoder = spatial_gnn_encoder
        self.spatial_covar_number = spatial_covar_number
        self.restrict_recon_pos = restrict_recon_pos
        self.restrict_recon_pos_cutoff = restrict_recon_pos_cutoff
        self.distribution_mask = distribution_mask
        
        # Define the encoder and decoder modules
        zy_dim_sum = np.sum(zy_dims)
        self.px = px(
            hidden_layers_x, x_dim, zx_dim, zd_dim, zy_dim_sum, decoder_type = "likelihood",
            linear_decoder = self.linear_decoder, restrict_recon_pos = self.restrict_recon_pos,
            restrict_recon_pos_cutoff = self.restrict_recon_pos_cutoff,
            distribution_mask = self.distribution_mask
        )
        self.pzd = pzd(hidden_layers_d, d_dim, zd_dim)
        self.pzy = nn.ModuleList()
        for i in range(self.num_covars_y):
            self.pzy.append(pzy(hidden_layers_y, y_dims[i], zy_dims[i]))
        
        self.qzx = qzx(hidden_layers_x, x_dim, zx_dim)
        self.qzd = qzd(hidden_layers_d, x_dim, zd_dim)
        self.qzy = nn.ModuleList()
        for i in range(self.num_covars_y):
            if i == self.spatial_covar_number:
                if self.spatial_gnn_encoder:
                    self.qzy.append(qzy_spatial(hidden_layers_y, x_dim, zy_dims[i]))
                else:
                    self.qzy.append(qzy(hidden_layers_y, x_dim, zy_dims[i]))
            else:
                self.qzy.append(qzy(hidden_layers_y, x_dim, zy_dims[i]))
        self.qd = qd(d_dim, zd_dim)
        self.qy = nn.ModuleList()
        for i in range(self.num_covars_y):
            self.qy.append(qy(y_dims[i], zy_dims[i]))
        
    def forward(self, x, d, y, edge_index = None):
        # Encode the data
        qzx_loc, qzx_logvar = self.qzx(x)
        qzd_loc, qzd_logvar = self.qzd(x)
        qzy_locs = []
        qzy_logvars = []
        for i in range(self.num_covars_y):
            if i == self.spatial_covar_number:
                if self.spatial_gnn_encoder:
                    qzy_loc, qzy_logvar = self.qzy[i](x, edge_index)
                    qzy_locs.append(qzy_loc)
                    qzy_logvars.append(qzy_logvar)
                else:
                    qzy_loc, qzy_logvar = self.qzy[i](x)
                    qzy_locs.append(qzy_loc)
                    qzy_logvars.append(qzy_logvar)
            else:
                qzy_loc, qzy_logvar = self.qzy[i](x)
                qzy_locs.append(qzy_loc)
                qzy_logvars.append(qzy_logvar)
                
        # Reparameterization trick
        zx = reparametrize(qzx_loc, qzx_logvar, x.device)
        zd = reparametrize(qzd_loc, qzd_logvar, x.device)
        zys = []
        for i in range(self.num_covars_y):
            zy = reparametrize(qzy_locs[i], qzy_logvars[i], x.device)
            zys.append(zy)
        
        # Decode the latent variables
        zy = torch.cat(zys, dim = 1)
        px_mu_gauss, px_logvar_gauss, px_mu_nb, px_theta_nb = self.px(zx, zd, zy)
        pzd_loc, pzd_logvar = self.pzd(d)
        pzy_locs = []
        pzy_logvars = []
        for i in range(self.num_covars_y):
            y_covar = y[i]
            if y_covar.shape.__len__() == 1:
                y_covar = y_covar.unsqueeze(-1)
                pzy_loc, pzy_logvar = self.pzy[i](y_covar)
                pzy_locs.append(pzy_loc)
                pzy_logvars.append(pzy_logvar)
            elif y_covar.shape.__len__() == 3:
                y_var_mean = torch.mean(y_covar, dim = 1)
                pzy_loc, pzy_logvar = self.pzy[i](y_var_mean)
                pzy_locs.append(pzy_loc)
                pzy_logvars.append(pzy_logvar)
            else:
                pzy_loc, pzy_logvar = self.pzy[i](y_covar)
                pzy_locs.append(pzy_loc)
                pzy_logvars.append(pzy_logvar)

        # Compute the auxiliary task predictions
        qd_loc = self.qd(zd)
        qy_locs = []
        for i in range(self.num_covars_y):
            qy_loc = self.qy[i](zys[i])
            qy_locs.append(qy_loc)
        
        return px_mu_gauss, px_logvar_gauss, px_mu_nb, px_theta_nb, pzd_loc, pzd_logvar, pzy_locs, \
                pzy_logvars, qzx_loc, qzx_logvar, qzd_loc, qzd_logvar, qzy_locs, qzy_logvars, qd_loc, \
                qy_locs
            
    def get_qzx(self, x):
        # Encode 
        qzx_loc, qzx_logvar = self.qzx(x)
        
        # Sample
        zx = reparametrize(qzx_loc, qzx_logvar, x.device)
        
        # Return 
        return zx
    
    def get_posterior(self, x, edge_index = None, means = False):
        # Encode the data pzy_logvar
        qzx_loc, qzx_logvar = self.qzx(x)
        qzd_loc, qzd_logvar = self.qzd(x)
        qzy_locs = []
        qzy_logvars = []
        for i in range(self.num_covars_y):
            if i == self.spatial_covar_number:
                if self.spatial_gnn_encoder:
                    qzy_loc, qzy_logvar = self.qzy[i](x, edge_index)
                else:
                    qzy_loc, qzy_logvar = self.qzy[i](x)
            else:
                qzy_loc, qzy_logvar = self.qzy[i](x)
            qzy_locs.append(qzy_loc)
            qzy_logvars.append(qzy_logvar)
            
        # If means is True, return the means of the latent variables,
        # otherwise, sample from the distributions
        if means:
            return qzx_loc, qzy_locs, qzd_loc
        else:   
            # Reparameterization trick
            zx = reparametrize(qzx_loc, qzx_logvar, x.device)
            zd = reparametrize(qzd_loc, qzd_logvar, x.device)
            zys = []
            for i in range(self.num_covars_y):
                zy = reparametrize(qzy_locs[i], qzy_logvars[i], x.device)
                zys.append(zy)
            
            # Return embeddings
            return zx, zys, zd
    
    def get_prior(self, d, y):
        # Sample from the prior distributions
        zd_loc, zd_logvar = self.pzd(d)
        zy_locs = []
        zy_logvars = []
        for i in range(self.num_covars_y):
            y_covar = y[i]
            if y_covar.shape.__len__() == 1:
                y_covar = y_covar.unsqueeze(-1)
            zy_loc, zy_logvar = self.pzy[i](y_covar)
            zy_locs.append(zy_loc)
            zy_logvars.append(zy_logvar)
        
        # Reparameterization trick
        zd = reparametrize(zd_loc, zd_logvar, d.device)
        zys = []
        for i in range(self.num_covars_y):
            zy = reparametrize(zy_locs[i], zy_logvars[i], d.device)
            zys.append(zy)
        
        # Return embeddings
        return zys, zd
    
    def get_likelihood(self, zx, zd, zy):
        # Decode the latent variables
        px_loc, px_logvar = self.px(zx, zd, zy)
        
        # Return the likelihood
        return px_loc, px_logvar
            
    def loss(self, x, d, y,mu_gauss, logvar_gauss, mu_nb, theta_nb,
             pzd_loc, pzd_logvar, pzy_locs, pzy_logvars,
             qzx_loc, qzx_logvar, qzd_loc, qzd_logvar, qzy_locs, qzy_logvars,
             qd_loc, qy_locs, zys_beta_kl, zys_beta_aux, current_device):
        
        # Reconstruction loss - set variance to one for now 
        recon_loss = mixed_recon_loss_gauss_nb(
            x,
            mu_gauss,
            logvar_gauss,
            mu_nb,
            theta_nb,
            distribution_mask=self.px.distribution_mask 
        )
        
        # KL divergence losses 
        kl_zx = kl_divergence(
            dist.Normal(qzx_loc, torch.exp(0.5*qzx_logvar)),
            dist.Normal(
                torch.zeros(self.zx_dim).to(current_device),
                torch.ones(self.zx_dim).to(current_device)
            )
        ).sum(dim = 1).mean(dim = 0)
        
        kl_zys = []
        for i in range(self.num_covars_y):
            kl_zy = kl_divergence(
                dist.Normal(qzy_locs[i], torch.exp(0.5*qzy_logvars[i])),
                dist.Normal(pzy_locs[i], torch.exp(0.5*pzy_logvars[i]))
            ).sum(dim = 1).mean(dim = 0)
            kl_zys.append(kl_zy)
        kl_zys_raw = kl_zys.copy()
        for i in range(self.num_covars_y):
            kl_zys[i] = kl_zys[i] * zys_beta_kl[i]
        kl_zys_mean = torch.stack(kl_zys).mean(dim = 0)
        
        kl_zd = kl_divergence(
            dist.Normal(qzd_loc, torch.exp(0.5*qzd_logvar)),
            dist.Normal(pzd_loc, torch.exp(0.5*pzd_logvar))
        ).sum(dim = 1).mean(dim = 0)
        
        
        
        # Auxilliary task losses - classification and regression losss
        if self.d_dist == "bernoulli":
            bce_loss = nn.BCEWithLogitsLoss()
            aux_loss_d = bce_loss(qd_loc, d.float())
        elif self.d_dist == "categorical":
            cce_loss = nn.CrossEntropyLoss()
            aux_loss_d = cce_loss(qd_loc, d)
        
        aux_losses_y = []
        for i in range(self.num_covars_y):
            if self.spatial_loss_testing:
                if i == self.spatial_covar_number:
                    spatial_loss = spatial_loss_iter_1(x, y[i], qy_locs[i], current_device)
                    aux_losses_y.append(spatial_loss)
                    continue 
            if self.y_covar_space[i] == "discrete":
                if self.y_dists[i] == "bernoulli":
                    bce_loss = nn.BCEWithLogitsLoss()
                    aux_loss_y = bce_loss(qy_locs[i], y[i].float())
                    aux_losses_y.append(aux_loss_y)
                elif self.y_dists[i] == "categorical":
                    cce_loss = nn.CrossEntropyLoss()
                    aux_loss_y = cce_loss(qy_locs[i], y[i])
                    aux_losses_y.append(aux_loss_y)
            elif self.y_covar_space[i] == "continuous":
                aux_loss_y = gaussian_nll_loss(qy_locs[i], y[i], torch.ones_like(qy_locs[i]))
                aux_losses_y.append(aux_loss_y)
            elif self.y_covar_space[i] == "proportional":
                    # Ensure predictions are positive
                    qy_loc_positive = F.softplus(qy_locs[i])
                    
                    # Create Dirichlet distribution
                    dirichlet = dist.Dirichlet(qy_loc_positive)
                    
                    # Compute negative log-likelihood
                    aux_loss_y = -dirichlet.log_prob(y[i]).mean()
                    aux_losses_y.append(aux_loss_y)
            
        aux_losses_y_raw = aux_losses_y.copy()
        for i in range(self.num_covars_y):
            aux_losses_y[i] = aux_losses_y[i] * zys_beta_aux[i]
        aux_losses_y_mean = torch.stack(aux_losses_y).mean(dim = 0)
        
        # Return losses
        return recon_loss, kl_zx, kl_zys_mean, kl_zd, aux_loss_d, aux_losses_y_mean, kl_zys_raw, \
            aux_losses_y_raw
        
def spatial_loss_iter_1(x_input, neighbor_input, qy_loc, current_device):
    # This loss uses qzy to aim to predict the difference in expression between
    # the x input and the neighbors of the x input. x_input is of shape 
    # (batch_size, x_dim) and neighbor_input is of shape (batch_size, num_neighbors, x_dim).
    
    # qy_loc is of shape (batch_size, x_dim)
    
    # For each dimension (feature) and for each neighbor, get the difference between x_input and 
    # neighbor_input, and then average this difference across all neighbors.
    num_neighbors = neighbor_input.shape[1]
    batch_size = x_input.shape[0]
    x_dim = x_input.shape[1]
    neighbor_diffs = torch.zeros(batch_size, num_neighbors, x_dim).to(current_device).double()
    for i in range(batch_size):
        for j in range(num_neighbors):
            neighbor_diffs[i, j, :] = x_input[i, :] - neighbor_input[i, j, :]
    neighbor_diffs = neighbor_diffs.mean(dim = 1)

    # Use the qy_loc to predict the neighbor_diffs
    loss = F.mse_loss(qy_loc, neighbor_diffs)
    return loss

class LitSpatialDIVA(L.LightningModule):
    """PyTorch Lightning implementation of the Spatial DIVA model.
    
    Parameters
    ----------
    x_dim : int
        Dimension of the input data (number of features), considering the combined
        counts and histology data
    y_dims : list of int
        List of dimensions for each y covariate
    d_dim : int 
        Dimension of the batch variable
    zx_dim : int, optional
        Dimension of the residual latent space, by default 20
    zy_dims : list of int, optional
        List of dimensions for each y covariate latent space, by default [20, 20, 20]
    zd_dim : int, optional
        Dimension of the batch latent space, by default 5
    hidden_layers_x : list of int, optional
        Hidden layer sizes for residual encoder/decoder, by default [128, 64]
    hidden_layers_y : list of int, optional
        Hidden layer sizes for label encoder/decoder, by default [128, 64]
    hidden_layers_d : list of int, optional
        Hidden layer sizes for batch encoder/decoder, by default [128, 64]
    num_y_covars : int, optional
        Number of y covariates, by default 3
    y_covar_space : list of str, optional
        List specifying the space of each y covariate ("discrete", "continuous", or "proportional"), 
        by default ["discrete", "discrete", "discrete"]
    y_dists : list of str, optional
        List specifying the distribution of each y covariate ("bernoulli" or "categorical"),
        by default ["bernoulli", "bernoulli", "bernoulli"]
    d_dist : str, optional
        Distribution of batch variable ("bernoulli" or "categorical"), by default "bernoulli"
    linear_decoder : bool, optional
        Whether to use linear decoder without activation functions, by default False
    spatial_loss_testing : bool, optional
        Whether to use spatial loss testing, by default False
    spatial_gnn_encoder : bool, optional
        Whether to use GNN encoder for spatial data, by default False
    spatial_covar_number : int, optional
        Index of the spatial covariate, by default 2
    restrict_recon_pos : bool, optional
        Whether to restrict reconstruction to positive values, by default False
    restrict_recon_pos_cutoff : int, optional
        Cutoff index for positive value restriction, by default None
    distribution_mask : list, optional
        List specifying distribution type for each feature ("gaussian" or "neg_binomial"), by default None
        For example, if the first 10 features are counts data and the last 10 features are histology data,
        the distribution_mask should be [["neg_binomial"]*10 + ["gaussian"]*10]
    lr : float, optional
        Learning rate for the ADAM optimizer, by default 1e-3
    betas : list of float, optional
        List of beta values for loss weighting, by default [1, 1, 1, 1, 1, 1]
        The first beta value is for the reconstruction loss, the second is for the KL divergence of the residual latent space,
        the third is for the KL divergence of the y covariate latent spaces, the fourth is for the KL divergence of the batch latent space,
        the fifth is for the auxiliary loss of the batch variable, and the sixth is for the auxiliary loss of the y covariates.
    zys_betas_kl : list of float, optional
        List of beta values for KL divergence terms of y covariates, by default [1.0, 1.0, 1.0]
    zys_betas_aux : list of float, optional
        List of beta values for auxiliary loss terms of y covariates, by default [1.0, 1.0, 1.0]
    train_data : torch.utils.data.Dataset, optional
        Training dataset, by default None
    val_data : torch.utils.data.Dataset, optional
        Validation dataset, by default None
    test_data : torch.utils.data.Dataset, optional
        Test dataset, by default None
    batch_size : int, optional
        Batch size for training, by default 64
    plot_prior : bool, optional
        Whether to plot samples from prior distributions, by default False
    plot_posterior : bool, optional
        Whether to plot samples from posterior distributions, by default False
    plot_reduction : str, optional
        Dimensionality reduction method for plotting ("umap" or "pca"), by default "umap"
    """
    
    def __init__(
        self,
        x_dim,
        y_dims,
        d_dim,
        zx_dim=20,
        zy_dims=[20, 20, 20],
        zd_dim=5,
        hidden_layers_x=[128, 64],
        hidden_layers_y=[128, 64],
        hidden_layers_d=[128, 64],
        num_y_covars=3,
        y_covar_space=["discrete", "discrete", "discrete"],
        y_dists=["bernoulli", "bernoulli", "bernoulli"],
        d_dist="bernoulli",
        linear_decoder=False,
        spatial_loss_testing=False,
        spatial_gnn_encoder=False,
        spatial_covar_number=2,
        restrict_recon_pos=False,
        restrict_recon_pos_cutoff=None,
        distribution_mask=None,
        lr=1e-3,
        betas=[1, 1, 1, 1, 1, 1],
        zys_betas_kl=[1.0, 1.0, 1.0],
        zys_betas_aux=[1.0, 1.0, 1.0],
        train_data=None,
        val_data=None,
        test_data=None,
        batch_size=64,
        plot_prior=False,
        plot_posterior=False,
        plot_reduction="umap"
    ):
        super(LitSpatialDIVA, self).__init__()
        self.save_hyperparameters()

        # Verify all parameters match what SpatialDIVA expects
        if len(y_dims) != num_y_covars:
            raise ValueError(f"Length of y_dims ({len(y_dims)}) must match num_y_covars ({num_y_covars})")
        if len(zy_dims) != num_y_covars:
            raise ValueError(f"Length of zy_dims ({len(zy_dims)}) must match num_y_covars ({num_y_covars})")
        if len(y_covar_space) != num_y_covars:
            raise ValueError(f"Length of y_covar_space ({len(y_covar_space)}) must match num_y_covars ({num_y_covars})")
        if len(y_dists) != num_y_covars:
            raise ValueError(f"Length of y_dists ({len(y_dists)}) must match num_y_covars ({num_y_covars})")
        if len(zys_betas_kl) != num_y_covars:
            raise ValueError(f"Length of zys_betas_kl ({len(zys_betas_kl)}) must match num_y_covars ({num_y_covars})")
        if len(zys_betas_aux) != num_y_covars:
            raise ValueError(f"Length of zys_betas_aux ({len(zys_betas_aux)}) must match num_y_covars ({num_y_covars})")

        self.lr = lr
        self.betas = betas
        self.zys_betas_kl = zys_betas_kl
        self.zys_betas_aux = zys_betas_aux
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.plot_prior = plot_prior
        self.plot_posterior = plot_posterior
        self.plot_reduction = plot_reduction

        self.model = SpatialDIVA(
            x_dim = x_dim,
            y_dims = y_dims,
            d_dim = d_dim,
            zx_dim = zx_dim,
            zy_dims = zy_dims,
            zd_dim = zd_dim,
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
            distribution_mask = distribution_mask
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            batch_size=self.val_data.dataset.tensors[0].shape[0]
        )

    def test_dataloader(self):
        if self.test_data is not None:
            return DataLoader(
                self.test_data, 
                batch_size=self.test_data.dataset.tensors[0].shape[0]
            )

    def training_step(self, batch, batch_idx):
        # Example structure: batch = (x, y1, y2, y3, d, spo_var, edge_index) if using a GNN
        if self.model.spatial_gnn_encoder:
            x, y1, y2, y3, d, spo_var, edge_index = batch
            edge_index = edge_index.to(self.device)
        else:
            x, y1, y2, y3, d, spo_var = batch
            edge_index = None
        
        # Move data to device
        x = x.to(self.device)
        y1 = y1.to(self.device)
        y2 = y2.to(self.device)
        y3 = y3.to(self.device)
        d = d.to(self.device).unsqueeze(dim=-1)
        spo_var = spo_var.to(self.device)

        # If your spatial covariate is y2 or spo_var (adjust indexing as appropriate)
        if self.model.spatial_covar_number == 1:
            y = [y1, spo_var, y3]
        else:
            y = [y1, y2, y3]

        # Forward pass (updated return signature with distribution mask usage inside the model)
        (
            px_mu_gauss, px_logvar_gauss, px_mu_nb, px_theta_nb,
            pzd_loc, pzd_logvar, pzy_locs, pzy_logvars,
            qzx_loc, qzx_logvar, qzd_loc, qzd_logvar,
            qzy_locs, qzy_logvars, qd_loc, qy_locs
        ) = self.model.forward(x, d, y, edge_index=edge_index)

        # Compute loss (loss function also uses distribution_mask internally)
        (
            recon_loss, 
            kl_zx, 
            kl_zys_mean, 
            kl_zd, 
            aux_loss_d, 
            aux_losses_y_mean, 
            kl_zys_raw, 
            aux_losses_y_raw
        ) = self.model.loss(
            x=x,
            d=d,
            y=y,
            mu_gauss=px_mu_gauss,
            logvar_gauss=px_logvar_gauss,
            mu_nb=px_mu_nb,
            theta_nb=px_theta_nb,
            pzd_loc=pzd_loc,
            pzd_logvar=pzd_logvar,
            pzy_locs=pzy_locs,
            pzy_logvars=pzy_logvars,
            qzx_loc=qzx_loc,
            qzx_logvar=qzx_logvar,
            qzd_loc=qzd_loc,
            qzd_logvar=qzd_logvar,
            qzy_locs=qzy_locs,
            qzy_logvars=qzy_logvars,
            qd_loc=qd_loc,
            qy_locs=qy_locs,
            zys_beta_kl=self.zys_betas_kl,
            zys_beta_aux=self.zys_betas_aux,
            current_device=self.device
        )

        # --- Logging individual loss components ---
        self.log("train_recon_loss", recon_loss, prog_bar=False)
        self.log("train_kl_zx", kl_zx, prog_bar=False)
        self.log("train_kl_zys_mean", kl_zys_mean, prog_bar=False)
        self.log("train_kl_zd", kl_zd, prog_bar=False)
        self.log("train_aux_loss_d", aux_loss_d, prog_bar=False)
        self.log("train_aux_losses_y_mean", aux_losses_y_mean, prog_bar=False)
        
        # Log each component for y-covariates
        for i in range(self.model.num_covars_y):
            self.log(f"train_kl_zys_raw_{i}", kl_zys_raw[i], prog_bar=False)
            self.log(f"train_aux_loss_y_raw_{i}", aux_losses_y_raw[i], prog_bar=False)
        
        # --- Summation of unscaled loss components ---
        loss_unscaled = recon_loss + kl_zx + kl_zys_mean + kl_zd + aux_loss_d + aux_losses_y_mean
        self.log("train_loss_unscaled", loss_unscaled, prog_bar=True)

        # --- Weighted final loss using self.betas ---
        loss = (
            self.betas[0] * recon_loss +
            self.betas[1] * kl_zx +
            self.betas[2] * kl_zys_mean +
            self.betas[3] * kl_zd +
            self.betas[4] * aux_loss_d +
            self.betas[5] * aux_losses_y_mean
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.model.spatial_gnn_encoder:
            x, y1, y2, y3, d, spo_var, edge_index = batch
            edge_index = edge_index.to(self.device)
        else:
            x, y1, y2, y3, d, spo_var = batch
            edge_index = None
        
        x = x.to(self.device)
        y1 = y1.to(self.device)
        y2 = y2.to(self.device)
        y3 = y3.to(self.device)
        d = d.to(self.device).unsqueeze(dim=-1)
        spo_var = spo_var.to(self.device)

        if self.model.spatial_covar_number == 1:
            y = [y1, spo_var, y3]
        else:
            y = [y1, y2, y3]

        (
            px_mu_gauss, px_logvar_gauss, px_mu_nb, px_theta_nb,
            pzd_loc, pzd_logvar, pzy_locs, pzy_logvars,
            qzx_loc, qzx_logvar, qzd_loc, qzd_logvar,
            qzy_locs, qzy_logvars, qd_loc, qy_locs
        ) = self.model.forward(x, d, y, edge_index=edge_index)

        (
            recon_loss, 
            kl_zx, 
            kl_zys_mean, 
            kl_zd, 
            aux_loss_d, 
            aux_losses_y_mean, 
            kl_zys_raw, 
            aux_losses_y_raw
        ) = self.model.loss(
            x=x,
            d=d,
            y=y,
            mu_gauss=px_mu_gauss,
            logvar_gauss=px_logvar_gauss,
            mu_nb=px_mu_nb,
            theta_nb=px_theta_nb,
            pzd_loc=pzd_loc,
            pzd_logvar=pzd_logvar,
            pzy_locs=pzy_locs,
            pzy_logvars=pzy_logvars,
            qzx_loc=qzx_loc,
            qzx_logvar=qzx_logvar,
            qzd_loc=qzd_loc,
            qzd_logvar=qzd_logvar,
            qzy_locs=qzy_locs,
            qzy_logvars=qzy_logvars,
            qd_loc=qd_loc,
            qy_locs=qy_locs,
            zys_beta_kl=self.zys_betas_kl,
            zys_beta_aux=self.zys_betas_aux,
            current_device=self.device
        )

        self.log("val_recon_loss", recon_loss, prog_bar=False)
        self.log("val_kl_zx", kl_zx, prog_bar=False)
        self.log("val_kl_zys_mean", kl_zys_mean, prog_bar=False)
        self.log("val_kl_zd", kl_zd, prog_bar=False)
        self.log("val_aux_loss_d", aux_loss_d, prog_bar=False)
        self.log("val_aux_losses_y_mean", aux_losses_y_mean, prog_bar=False)
        
        for i in range(self.model.num_covars_y):
            self.log(f"val_kl_zys_raw_{i}", kl_zys_raw[i], prog_bar=False)
            self.log(f"val_aux_loss_y_raw_{i}", aux_losses_y_raw[i], prog_bar=False)

        loss_unscaled = recon_loss + kl_zx + kl_zys_mean + kl_zd + aux_loss_d + aux_losses_y_mean
        self.log("val_loss_unscaled", loss_unscaled, prog_bar=True)

        loss = (
            self.betas[0] * recon_loss +
            self.betas[1] * kl_zx +
            self.betas[2] * kl_zys_mean +
            self.betas[3] * kl_zd +
            self.betas[4] * aux_loss_d +
            self.betas[5] * aux_losses_y_mean
        )
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        if self.model.spatial_gnn_encoder:
            x, y1, y2, y3, d, spo_var, edge_index = batch
            edge_index = edge_index.to(self.device)
        else:
            x, y1, y2, y3, d, spo_var = batch
            edge_index = None
        
        x = x.to(self.device)
        y1 = y1.to(self.device)
        y2 = y2.to(self.device)
        y3 = y3.to(self.device)
        d = d.to(self.device).unsqueeze(dim=-1)
        spo_var = spo_var.to(self.device)

        if self.model.spatial_covar_number == 1:
            y = [y1, spo_var, y3]
        else:
            y = [y1, y2, y3]

        (
            px_mu_gauss, px_logvar_gauss, px_mu_nb, px_theta_nb,
            pzd_loc, pzd_logvar, pzy_locs, pzy_logvars,
            qzx_loc, qzx_logvar, qzd_loc, qzd_logvar,
            qzy_locs, qzy_logvars, qd_loc, qy_locs
        ) = self.model.forward(x, d, y, edge_index=edge_index)

        (
            recon_loss, 
            kl_zx, 
            kl_zys_mean, 
            kl_zd, 
            aux_loss_d, 
            aux_losses_y_mean, 
            kl_zys_raw, 
            aux_losses_y_raw
        ) = self.model.loss(
            x=x,
            d=d,
            y=y,
            mu_gauss=px_mu_gauss,
            logvar_gauss=px_logvar_gauss,
            mu_nb=px_mu_nb,
            theta_nb=px_theta_nb,
            pzd_loc=pzd_loc,
            pzd_logvar=pzd_logvar,
            pzy_locs=pzy_locs,
            pzy_logvars=pzy_logvars,
            qzx_loc=qzx_loc,
            qzx_logvar=qzx_logvar,
            qzd_loc=qzd_loc,
            qzd_logvar=qzd_logvar,
            qzy_locs=qzy_locs,
            qzy_logvars=qzy_logvars,
            qd_loc=qd_loc,
            qy_locs=qy_locs,
            zys_beta_kl=self.zys_betas_kl,
            zys_beta_aux=self.zys_betas_aux,
            current_device=self.device
        )

        loss_unscaled = recon_loss + kl_zx + kl_zys_mean + kl_zd + aux_loss_d + aux_losses_y_mean
        self.log("test_loss_unscaled", loss_unscaled, prog_bar=True)

        loss = (
            self.betas[0] * recon_loss +
            self.betas[1] * kl_zx +
            self.betas[2] * kl_zys_mean +
            self.betas[3] * kl_zd +
            self.betas[4] * aux_loss_d +
            self.betas[5] * aux_losses_y_mean
        )
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_train_end(self):
        # Optional: sampling & plotting
        if self.plot_prior:
            self.prior_sample_and_plot(reduction=self.plot_reduction)
        if self.plot_posterior:
            self.posterior_sample_and_plot(reduction=self.plot_reduction)
            



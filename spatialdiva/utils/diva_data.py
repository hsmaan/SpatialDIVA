import numpy as np
import faiss
import torch
import scanpy as sc
import anndata as ann
from sklearn.decomposition import PCA
#from torch_geometric.data import Data, Dataset
#from torch_geometric.loader import NeighborLoader


def find_knn(data_list, k=5):
    """Gets k nearest-neighbors for all datasets.

    Using each data subset in data_list, gets k-nearest-neighbors for each subset and
    returns concatenated k-nearest-neighbors for each index across all datasets.

    Args:
        data_list (list): List of numpy arrays corresponding to data subsets. Datasets
            must have an equivalent number of features.
        k (integer): Positive integer value indicating how many neighbors to consider
            in the k-nearest-neighbors algorithm. Default value is 15.

    Returns:
        knn_concat (array): Array of values corresponding to query KNN indices for all
            datasets in data_list, indexed based on their concatenation. Will return
            k nearest-neighbors at query position for each index from input data.
    """
    # Get lengths of all datasets for reindexing
    data_lens = [len(dataset) for dataset in data_list]

    # Get all indices
    indices = [i for i in range(len(data_list))]

    # Get knn pairs for each dataset and reindex as necessary
    knn_list = []
    for idx in indices:
        dataset = data_list[idx]
        dataset = np.ascontiguousarray(dataset, dtype=np.float32)
        index = faiss.IndexFlatL2(dataset.shape[1])
        index.add(dataset)
        knn_vals, knn = index.search(dataset, k)
        if idx == 0:
            knn_corrected = knn
            knn_list.append(knn_corrected)
        else:
            knn_corrected = []
            len_addition = sum(data_lens[0:idx])
            for i in range(len(knn)):
                knn_corrected.append(knn[i] + len_addition)
            knn_corrected_arr = np.asarray(knn_corrected)
            knn_list.append(knn_corrected_arr)

    # Concatenate all KNNs corresponding to all dataset indices and return
    knn_concat = np.concatenate(knn_list)
    return knn_concat


def uni_st_knn(st_counts, uni_counts, k=15, standardize=False):
    # Standardize data if needed
    if standardize:
        st_counts = (st_counts - np.mean(st_counts, axis=0)) / np.std(st_counts, axis=0)
        uni_counts = (uni_counts - np.mean(uni_counts, axis=0)) / np.std(
            uni_counts, axis=0
        )
    else:
        pass

    # Find knn for concatenated data
    data = np.concatenate([st_counts, uni_counts], axis=1)
    knn = find_knn([data], k=k)

    # Return knn
    return knn


# def return_dataset(st_uni_data, knn, y1, y2, y3, d):
#     # Convert data to torch tensors
#     num_samples, num_neighbors = knn.shape

#     source_nodes = torch.arange(num_samples).repeat_interleave(num_neighbors)
#     target_nodes = torch.tensor(knn).flatten()

#     edge_index = torch.stack([source_nodes, target_nodes], dim=0)

#     # Create a torch geometric dataset object
#     data = Data(x=st_uni_data, edge_index=edge_index, y1=y1, y2=y2, y3=y3, d=d)
#     data.validate()

#    return data


# def dataset_to_dataloader(dataset, num_neighbors, batch_size, shuffle):
#     # Create a torch geometric loader object
#     loader = NeighborLoader(
#         dataset, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=shuffle
#     )

#     return loader


def adata_process(
    adata_files,
    normalize=True,
    standardize_sct=False,
    standardize_uni=True,
    n_top_genes=2500,
    n_neighbors_pca=15,
    knn_type="spatial",
):
    # Load the anndata objects from adata files (list)
    adata_list = []
    for adata_file in adata_files:
        adata = sc.read_h5ad(adata_file)
        adata_list.append(adata)

    # Perform preprocessing steps of lognorm if indicated
    if normalize:
        for adata in adata_list:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

    # Perform hvg selection (concatenate anndatas to do this)
    adata_concat = ann.concat(adata_list)
    if normalize:
        sc.pp.highly_variable_genes(
            adata_concat, 
            n_top_genes=n_top_genes,
            flavor="seurat"
        )
    else:
        sc.pp.highly_variable_genes(
            adata_concat, 
            n_top_genes=n_top_genes,
            flavor="seurat_v3"
        )
    hvg = adata_concat.var.highly_variable
    for adata in adata_list:
        adata.var["highly_variable"] = hvg

    # Standardize the counts and UNI features if indicated
    if standardize_sct:
        for adata in adata_list:
            # Standardize the counts
            sc.pp.scale(adata)
    if standardize_uni:
        for adata in adata_list:
            # Standardize the UNI features
            obs_columns = [col for col in adata.obs.columns if "UNI" in col]
            adata.obs[obs_columns] = (
                adata.obs[obs_columns] - adata.obs[obs_columns].mean()
            ) / adata.obs[obs_columns].std()

    # If counts not standardized, convert them to dense matrices
    for adata in adata_list:
        if not standardize_sct:
            adata.X = adata.X.todense()

    # Determine which way to do the knn
    # Perform neighborhood PCA averaging (for SpatialDIVA) - using both
    # the count and UNI features
    for i, adata in enumerate(adata_list):
        # Get the PCA reduction of the count data - log normalize if not done
        if not normalize:
            adata_copy = adata.copy()
            sc.pp.normalize_total(adata_copy, target_sum=1e4)
            sc.pp.log1p(adata_copy)
            sc.pp.pca(adata_copy)
            count_pc = adata_copy.obsm["X_pca"]
        else:
            sc.pp.pca(adata)
            count_pc = adata.obsm["X_pca"]
            
        # Get the PCA reduction of the UNI data
        obs_columns = [col for col in adata.obs.columns if "UNI" in col]
        uni_data = adata.obs[obs_columns].values
        uni_pca = PCA(n_components=50)
        uni_pca.fit(uni_data)
        uni_pc = uni_pca.transform(uni_data)

        # Perform k-nearest neighbors on the combined PCA, and determine the
        # type of knn to perform - either spatial coordinates or PCA
        # reduction of the count and UNI data
        if knn_type == "spatial":
            spatial_coords = adata.obsm["spatial"]
            knn = find_knn([spatial_coords], k=n_neighbors_pca)
            # Append KNN to the adata object - change dataset to the
            # PCA because we're only using the spatial coordinates for the
            # KNN, but the representation of the data is the PCA
            dataset = np.concatenate([count_pc, uni_pc], axis=1)
            adata.uns["neighbors"] = knn
        elif knn == "pca":
            dataset = np.concatenate([count_pc, uni_pc], axis=1)
            knn = find_knn([dataset], k=n_neighbors_pca)
            # Append KNN to the adata object
            adata.uns["neighbors"] = knn
        else:
            raise ValueError("Invalid knn type specified.")

        # Get the mean value of the neighbors for each cell based
        # on PC reduction of both count and UNI data
        mean_pca_neighbors = []
        for i in range(dataset.shape[0]):
            indices = knn[i]
            mean_pca = dataset[indices].mean(axis=0)
            mean_pca_neighbors.append(mean_pca)

        # Append the mean PCA neighbors to the adata object
        adata.obsm["X_pca_neighbors_avg"] = np.array(mean_pca_neighbors)

        # Append the UNI PCA to the adata object
        adata.obsm["X_pca_uni"] = uni_pc

        # Append the count PCA to the adata object
        adata.obsm["X_pca_count"] = count_pc

    # Add batch information to the adatas
    for i, adata in enumerate(adata_list):
        adata.obs = adata.obs.copy()
        adata.obs["batch"] = i

    # Return the processed adatas
    return adata_list

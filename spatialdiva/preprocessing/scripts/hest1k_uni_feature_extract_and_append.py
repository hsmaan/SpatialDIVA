import os
import sys
import argparse
import json
sys.path.append("../src")

import numpy as np
import pandas as pd 
import scanpy as sc 
import h5py

from utils import load_uni_model, convert_arr_pil
    
def main(cfg, sample_index):
    # Parse the config for the indicated data 
    adata_paths = cfg["adata_paths"]
    image_paths = cfg["image_paths"]
    adata_names = cfg["adata_names"]
    model_path = cfg["model_path"]
    adata_save_path = cfg["adata_save_path"]
    sample_index = int(sample_index)
    
    # Load the anndata and image objects, and process them to extract UNI
    # features for the spot-associated histology patches
    adata_path = adata_paths[sample_index]
    image_path = image_paths[sample_index]
    adata_name = adata_names[sample_index]
    adata = sc.read_h5ad(adata_path)
    image_data = h5py.File(image_path)
    
    # Check to ensure image_data barcodes and anndata barcodes
    # are equivalent and are in the same order. If not, subset
    # the image_data to match the anndata object
    image_barcodes = image_data["barcode"][:,:].squeeze()
    images = image_data["img"][:,:]
    image_barcodes = np.array([image_barcode.decode('UTF-8') for image_barcode in image_barcodes])
    anndata_barcodes = adata.obs.index.values
    if not np.array_equiv(image_barcodes, anndata_barcodes):
        anndata_barcodes_valid = anndata_barcodes[[barcode in image_barcodes for barcode in anndata_barcodes]]
        image_barcodes_valid = np.array([barcode for barcode in image_barcodes if barcode in anndata_barcodes_valid])
        images_valid = np.array([image for image, barcode in zip(images, image_barcodes) if barcode in anndata_barcodes_valid])
        adata_sub = adata[image_barcodes_valid]
    else:
        adata_sub = adata.copy()
        image_barcodes_valid = image_barcodes
        images_valid = images
        
    # Check if barcodes match between adata sub and image data
    if not np.array_equiv(adata_sub.obs.index.values, image_barcodes_valid):
        raise ValueError("Barcodes do not match between adata and image data")
    
    # Load the model 
    os.chdir(model_path)
    model, transform = load_uni_model()
    model.eval()
        
    # Zip the barcode and image together and extract the features
    features = []
    for image in images_valid:
        image_pil = convert_arr_pil(arr = image, transform = transform)
        feature_values = model(image_pil).detach().numpy().squeeze()
        features.append(feature_values)      

    # Append the features to the anndata object
    features_arr = np.array(features)
    print(features_arr.shape)
    column_names = [f"UNI-{i+1}" for i in range(features_arr.shape[1])]
    features_df = pd.DataFrame(features_arr, index=adata_sub.obs.index, columns=column_names)
    adata_sub.obs = pd.concat([adata_sub.obs, features_df], axis=1)
    
    # Save the appended anndata object
    adata_sub.write_h5ad(adata_save_path + adata_name + ".h5ad")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract UNI features from histology images and append to anndata object"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON configuration file",
    )
    parser.add_argument(
        "--sample_index",
        required=True,
        help="Specific sample index to process in this run"
    )
    args=parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    main(cfg = config, sample_index = args.sample_index)
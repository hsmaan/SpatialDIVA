import os
import sys
import argparse
import json
sys.path.append("../src")


import numpy as np
import pandas as pd 

from hest import VisiumReader

def main(config, sample_idx):
    # Get the paths to the objects based on the config 
    image_paths = config['image_paths']
    bc_matrix_paths = config['bc_matrix_paths']
    spatial_coord_paths = config['spatial_coord_paths']
    save_dirs = config['save_dirs']
    
    # Subset for the specific sample
    sample_idx = int(sample_idx)
    image = image_paths[sample_idx]
    bc_matrix = bc_matrix_paths[sample_idx]
    spatial_coord = spatial_coord_paths[sample_idx]
    save_dir = save_dirs[sample_idx]
    
    # Create save dir if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the hest st object
    st = VisiumReader().read(
        image, # path to a full res image
        bc_matrix, # path to filtered_feature_bc_matrix.h5
        spatial_coord_path=spatial_coord# path to a .json alignment file
    )
    
    # Visualize spots over downscaled wsi
    st.save_spatial_plot(save_dir)
    
    # Save pyramidal tiff and h5
    st.save(save_dir, pyramidal=True)
    
    # Segment the tissue using fine-tuned DeepLabV3 and ResNet50
    st.segment_tissue(method='otsu') 
    st.save_tissue_seg_pkl(save_dir, 'otsu_segmentation')
    
    # Patch the images using pre-set target_pixel_size (this may 
    # have to be adjusted later depending on data size)
    patch_save_dir = save_dir + '/patches'
    os.makedirs(patch_save_dir, exist_ok=True)
    
    st.dump_patches(
        patch_save_dir,
        target_patch_size=224,
        target_pixel_size=0.5,
        verbose=True
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process the Visium data for HEST"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON configuration file",
    )
    parser.add_argument(
        "--sample_idx",
        required=True,
        help="Index of the sample to process",
    )
    args=parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    main(config = config, sample_idx = args.sample_idx)    
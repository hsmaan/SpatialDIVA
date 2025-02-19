## SpatialDIVA - preprocessing code for ST and histopathology data

This README contains information on how to preprocess in-house Visium and VisiumHD datasets to extract spot-aligned patches and UNI features.

### To install environment

```
pip install -r requirements.txt
```

### To install UNI

The UNI repository has already been cloned in the `UNI` folder. 

1. Create a virtual env named UNI with python=3.10

```
virtualenv --no-download UNI
source UNI/bin/activate
```

2. Change to UNI directory and do pip install 

```
cd UNI
pip install --no-index --upgrade pip
pip install -e .
```

This is a modified approach to the instructions in the UNI installation directory (https://github.com/mahmoodlab/UNI). After doing this, the environment can be activated anytime with `source UNI/bin/activate`

3. Add additional dependencies

```
pip install scanpy anndata seaborn matplotlib jupyter ipykernel pillow
```

### To run the preprocessing code 

### Converting Visium data to spot-aligned image patches via the HEST pipeline

To take Visium data and return spot-aligned patches of images for UNI feature extraction, the `hest_spot_align_process.py` script can be used. 

This script take arguments:

- `--config` - A JSON config with the following necessary parameters:
    - "image_paths" - A list of paths to the Visium images
    - "bc_matrix_paths" - A list of paths to the Visium barcode matrix files
    - "spatial_coord_paths" - A list of paths to the Visium spatial coordinate files
    - "save_dirs" - A list of paths to save the spot-aligned patches and other HEST outputs

- `--sample_idx` - The index of the sample to process

### Extracting UNI features from spot-aligned Visium patches 

To extract UNI features from the spot-aligned patches, the `hest1k_uni_feature_extract_and_append.py` script can be used.

This script takes arguments:

- `--config` - A JSON config with the following necessary parameters:
    - "image_paths" - A list of paths to the spot-aligned patches for each slide
    - "adata_paths" - A list of paths for the slide level AnnData objects
    - "adata_names" - A list of names for the slide level AnnData objects
    - "model_path" - The path to the UNI model
    - "adata_save_path" - The path to save the slide level AnnData object with saved UNI features

- `--sample_idx` - The index of the sample to process
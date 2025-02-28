![Build Status](https://github.com/hsmaan/SpatialDIVA/actions/workflows/ci.yml/badge.svg?branch=main)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.02.19.638201-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2025.02.19.638201v1)
[![PyPI - Version](https://img.shields.io/pypi/v/SpatialDIVA)](https://pypi.org/project/SpatialDIVA/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<img src="docs/spatialdiva_logo.png" width="75" height="75" style="float;" /> 

# SpatialDIVA - disentangling spatial transcriptomics and histopathology imaging data 

This repository contains code for the SpatialDIVA method, associated preprocessing, and evaluations performed in the manuscript - "Multi-modal disentanglement of spatial transcriptomics and histopathology imaging".

If you use our work, please consider [citing the preprint](#citation).

## Table of Contents

- [Installation](#installation)
- [Datasets](#processed-datasets)
- [Usage](#usage)
- [Tutorials](#tutorials)
- [Preprocessing](#preprocessing)
- [Paper evaluation code](#paper-evaluation-code)
- [Citation](#citation)

## Installation

The requirements can be installed via pip:

```
pip install SpatialDIVA
```

Alternatively, poetry can be used to install the package after cloning:

```
git clone https://github.com/hsmaan/SpatialDIVA.git
poetry install
```

We also recommend cloning the git repository to access the tutorials and preprocessing code.

## Processed datasets

#### Valdeolivas et al. - colorectal cancer (https://www.nature.com/articles/s41698-023-00488-4)

This dataset can be downloaded from Figshare at https://figshare.com/s/e12b576b1b05cb1ab77d. After downloading, please move the data to the following directory:

```
mkdir spatialdiva/data
unzip valdeolivas_processed.zip
mv valdeolivas_processed spatialdiva/data
```

#### Zhou et al. - pancreatic ductal adenocarcinoma (https://www.nature.com/articles/s41588-022-01157-1)

Similar to the Valdeolivas et al. data, the data can be downloaded from Figshare https://figshare.com/s/1cd18b65fc0cd7079e41. After downloaded, the data can be moved to the appropriate directory:

```
unzip zhou_processed.zip
mv zhou_processed spatialdiva/data
```

## Usage

The easiest way to get set up and use SpatialDIVA is the through the high-level API defined in `spatialdiva/api`. 

For more full control, users can directly import the lightning module (`LitSpatialDIVA`) from `spatialdiva/models/diva_spatial.py` - a full tutorial on this is coming soon! 

You'll need an anndata object where each observation is a spot from Visium or similar technologies, and `.X` quantifies the expression of different genes per spot. To use the model outlined in the manuscript, clusters or annotations for the histology data and spatial transcriptomics (ST) data will be needed and stored in the `obs` attribute of the anndata object. Sample information (i.e. batch) will also be stored in `.obs` and the spatial coordinates are stored in `.obsm['spatial']`. 

We'll need to featurize the histopathology data, and the best way to do this is to use a histpathology foundation model and spot-aligned image patches, such that we obtain spot-aligned features for the histology modality. For using the UNI foundation model, code to do this is available in `spatialdiva/preprocessing`.

Once the data is preprocessed, the SpatialDIVA model can be used as follows:

```python
import scanpy as sc 
import anndata as ann 
from api import StDIVA

# Load the anndata object
adata = sc.read_h5ad("path/to/adata.h5ad") 

# Initialize the SpatialDIVA model 
stdiva = StDIVA(
    counts_dim = 30000, # We're assuming 30'000 genes are present in the data
    hist_dim = 1024, # The number of features extracted from the histology data - e.g. via the UNI foundation model
    y1_dim = 10, # The number of clusters/classes for the ST-labels 
    y2_dim = 100, # The dimensionality of the spatial covariate - the API will automatically infer this from the spatial coordinates
    y3_dim = 2, # The number of clusters/classes for the pathology labels 
    d_dim = 5 # The number of batches/samples in the data
)

# Add the anndata object to the model for preprocessing
stdiva.add_data(
    adata = adata,
    label_key_y1 = "Cell-type", # We're assuming the label in .obs for y1 is "Cell-type"
    label_key_y3 = "Pathologist Annotation", # We're assuming the label in .obs for y3 is "Pathologist Annotation"
    hist_col_key = "UNI" # We're assuming the histology features are stored in columns starting with "UNI" in .obs
)

# Train the model - this will train using the default parameters and pytorch lightning 
stdiva.train(max_epochs=100)

# Extract embeddings
zd_samples, zy1_samples, zy2_samples, zy3_samples, zx_samples = stdiva.get_embeddings(type = "full") 

```

To see a full workup of this process, please see the tutorials below.

## Tutorials 

The following notebooks offer more in-depth tutorials on how to use the SpatialDIVA model for relevant analyses of histopathology and spatial transcriptomics data:

1. #### Factor covariance analysis with SpatialDIVA - `spatialdiva/tutorials/01_colorectal_cancer_spdiva_analysis.ipynb`

2. #### Conditional generation analysis with SpatialDIVA - Coming soon!

3. #### Tumor annotation and subtyping with SpatialDIVA - Coming soon!

## Preprocessing

The preprocessed data for Valdeolivas et al. (colorectal cancer) and Zhou et al. (pancreatic cancer) contains spot-aligned features for histopathology imaging extracted using the UNI foundation model (https://github.com/mahmoodlab/UNI). 

Code for preprocessing in-house datasets in a similar manner, as well as environment and installation information is available in the `spatialdiva/preprocessing` directory.

## Paper evaluation code

Coming soon!

## Citation

*Multi-Modal Disentanglement of Spatial Transcriptomics and Histopathology Imaging*

Hassaan Maan, Zongliang Ji, Elliot Sicheri, Tiak Ju Tan, Alina Selega, Ricardo Gonzalez, Rahul G. Krishnan, Bo Wang, Kieran R. Campbell

bioRxiv 2025.02.19.638201; doi: https://doi.org/10.1101/2025.02.19.638201  

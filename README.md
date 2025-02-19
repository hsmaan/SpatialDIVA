![Build Status](https://github.com/hsmaan/SpatialDIVA/actions/workflows/ci.yml/badge.svg?branch=main)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2022.10.06.511156-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2022.10.06.511156v1)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<img style="float: right;" src="docs/spatialdiva_logo.png" width="125" height="125" />

# SpatialDIVA - disentangling spatial transcriptomics and histopathology data 

This repository contains code for the SpatialDIVA method, associated preprocessing, and evaluations performed in the manuscript - "Multi-modal disentanglement of spatial transcriptomics and histopathology imaging".

## Table of Contents

- [Installation](#installation)
- [Datasets](#processed-datasets)
- [Usage](#usage)
- [Tutorials](#tutorials)
- [Preprocessing](#preprocessing)
- [Paper evaluation code](#paper-evaluation-code)
- [Citation](#citation)

## Installation

Coming soon!

## Processed datasets

#### Valdeolivas et al. - colorectal cancer (https://www.nature.com/articles/s41698-023-00488-4)

This dataset can be downloaded from Figshare at https://figshare.com/s/e12b576b1b05cb1ab77d. After downloading, please move the data to the following directory:

```
mkdir spatialdiva/data
mv valdeolivas_processed spatialdiva/data
```

#### Zhou et al. - pancreatic ductal adenocarcinoma (https://www.nature.com/articles/s41588-022-01157-1)

Coming soon!

## Usage

Coming soon!

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

TBD 
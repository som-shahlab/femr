# FEMR
### Framework for Electronic Medical Records

**FEMR** is a python package for building models using EHR data.

**FEMR** offers the following four main types of functionality. In order, they are the ability to:
1. Convert EHR and claims data into a common schema, where each patient is associated with a timeline of events extracted from the EHR
2. Apply labeling functions on that schema in order to derive labels for each patient
3. Apply featurization schemes to obtain feature matrices for each patient
4. Perform other common tasks necessary for research with EHR data

# Installation

Special note for NERO users:

You will need to install cuda manually until cuda version is updated on the nero. Follow the following steps for nero.

1. Download the right version of cuda on your local machine and transfer it over a folder in nero [link](https://developer.nvidia.com/cuda-11.1.1-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)
2. Run the installer as bash command by providing path to install such as `bash <CUDA_PATH> --installpath=<INSTALL_PATH>`, where `<CUDA_PATH>` is the path to the file you downloaded/transferred in step 1 and `<INSTALL_PATH>` is where you'd like to store the CUDA installation files. During installation, uncheck all the boxes except cuda toolkit.
3. After the installation completes, it will spit out two paths on terminal that should be put into your .bashrc file:
```
export PATH="<INSTALL_PATH>/bin:$PATH"
export LD_LIBRARY_PATH="<INSTALL_PATH>/lib64:$LD_LIBRARY_PATH"
```
4. Delete the `/tmp/cuda-installer.log` file or it will create problems (e.g., segmentation fault) for other users

Note: you may need to restart your terminal for the changes to reflect

As Nero does not have internet access, you must run the following before running the code below.

```
export DISTDIR=/local-scratch/nigam/distdir
```



Run the following:

```
conda create -n FEMR_ENV python=3.10 bazel=5.3 clangxx=14 -c conda-forge
conda activate FEMR_ENV

export BAZEL_USE_CPP_ONLY_TOOLCHAIN=1

git clone https://github.com/som-shahlab/femr.git
cd femr
pip install -e .
```

If you want to use PyTorch for deep learning, you can install it as follows (first install numpy dependency):
```python
conda install numpy
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

# Precommit checks

## Installation
```
conda install pre-commit -y
conda install pytest -y
pre-commit install
```

## Pytest test

```
pytest tests
```

## Pre-commit test

This will run automatically on every commit.

You can also run it manually with:

```
pre-commit run --all-files
```


## Running one of our tools
```
export OMOP_SOURCE=/share/pi/nigam...
export MODIFIED_OMOP_DESTINATION=/share/pi/nigam...
python tools/blah.py $OMOP_SOURCE $MODIFIED_OMOP_DESTINATION
```


## GZIP decompression commands
```
export OMOP_SOURCE=/share/pi/nigam...
gunzip $OMOP_SOURCE/**/*.csv.gz
```

## Zstandard compression commands
```
export OMOP_SOURCE=/share/pi/nigam...
zstd -1 --rm $OMOP_SOURCE/**/*.csv
```

## Generating extract

```
# Set up environment variables
#   Path to a folder containing your raw STARR-OMOP download, generated via `tools.stanford.download_bigquery.py`
export OMOP_SOURCE=/path/to/omop/folder...
#   Path to any arbitrary folder where you want to store your Piton extract
export EXTRACT_DESTINATION=/path/to/piton/extract/folder...
#   Path to any arbitrary folder where you want to store your Piton extract logs
export EXTRACT_LOGS=/path/to/piton/extract/logs...

# Do some data preprocessing with Stanford-specific helper scripts
#   Extract data from flowsheets
python tools/stanford/flowsheet_cleaner.py --num_threads 5 $OMOP_SOURCE "${EXTRACT_DESTINATION}_flowsheets"
#   Normalize visits
python tools/omop/normalize_visit_detail.py --num_threads 5 "${EXTRACT_DESTINATION}_flowsheets" "${EXTRACT_DESTINATION}_flowsheets_detail"

# Run actual Piton extraction
etl_stanford_omop "${EXTRACT_DESTINATION}_flowsheets_detail" $EXTRACT_DESTINATION $EXTRACT_LOGS --num_threads 10
```

Example usage (Note: This should take ~10 minutes on a 1% extract of STARR-OMOP)

```
export OMOP_SOURCE=/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_1pcent_2022_11_09
export EXTRACT_DESTINATION=/local-scratch/nigam/projects/mwornow/piton_starr_omop_cdm5_deid_1pcent_2022_11_09
export EXTRACT_LOGS=/local-scratch/nigam/projects/mwornow/piton_starr_omop_cdm5_deid_1pcent_2022_11_09_logs

python tools/stanford/flowsheet_cleaner.py --num_threads 5 $OMOP_SOURCE "${EXTRACT_DESTINATION}_flowsheets"
python tools/omop/normalize_visit_detail.py --num_threads 5 "${EXTRACT_DESTINATION}_flowsheets" "${EXTRACT_DESTINATION}_flowsheets_detail"

etl_stanford_omop "${EXTRACT_DESTINATION}_flowsheets_detail" $EXTRACT_DESTINATION $EXTRACT_LOGS --num_threads 10
```

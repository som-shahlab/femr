# FEMR
### Framework for Electronic Medical Records

**FEMR** is a python package for building models using EHR data.

**FEMR** offers the following four main types of functionality. In order, they are the ability to:
1. Convert EHR and claims data into a common schema, where each patient is associated with a timeline of events extracted from the EHR
2. Apply labeling functions on that schema in order to derive labels for each patient
3. Apply featurization schemes to obtain feature matrices for each patient
4. Perform other common tasks necessary for research with EHR data

# Installation

There are two variants of the **FEMR** package, a CPU only version and a CUDA enabled version.

## How to install **FEMR** without CUDA

```bash
pip install femr
```

If you have a particularly old CPU, we offer a variant of femr without CPU optimations.

```bash
pip install femr_oldcpu
```

## How to install **FEMR** with CUDA support

Note that CUDA-enabled **FEMR** requires jax in order to function.

```bash
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "femr_cuda[models]"
```

# Getting Started

The first step of using **FEMR** is to convert your patient data into a femr.datasets.PatientDatabase, a file format that allows you to easily query patient timelines.

There are three options for doing this (in order from most to least recommended):

a) Convert your data to OMOP form and run the etl_generic_omop program to convert OMOP datasets to PatientDatabases. See our MIMIC OMOP ETL tutorial.

b) Convert your data to FEMR's custom simple csv format and run the etl_simple_femr program to convert that format into a PatientDatabase. See our simple format ETL tutorial.

c) Write a custom ETL script to handle special cases. See both the Stanford and Sickkid's ETL scripts.

# Development

The following guides are for developers who want to contribute to **FEMR**.

## Building from source

In some scenarios (such as contributing to **FEMR**), you might want to compile the package from source.

In order to do so, follow the following instructions.

```bash
conda create -n FEMR_ENV python=3.10 bazel=6 -c conda-forge -y
conda activate FEMR_ENV

export BAZEL_USE_CPP_ONLY_TOOLCHAIN=1

git clone https://github.com/som-shahlab/femr.git
cd femr
pip install -e .
```

## Precommit checks

Before committing, please run the following commands to ensure that your code is formatted correctly and passes all tests.

### Installation
```bash
conda install pre-commit pytest -y
pre-commit install
```

### Running

#### Test Functions

```bash
pytest tests
```

### Formatting Checks

```bash
pre-commit run --all-files
```

# Miscellaneous

## GZIP decompression commands
```bash
export OMOP_SOURCE=/share/pi/nigam...
gunzip $OMOP_SOURCE/**/*.csv.gz
```

## Zstandard compression commands
```bash
export OMOP_SOURCE=/share/pi/nigam...
zstd -1 --rm $OMOP_SOURCE/**/*.csv
```

## Generating extract

```bash
# Set up environment variables
#   Path to a folder containing your raw STARR-OMOP download, generated via `tools.stanford.download_bigquery.py`
export OMOP_SOURCE=/path/to/omop/folder...
#   Path to any arbitrary folder where you want to store your FEMR extract
export EXTRACT_DESTINATION=/path/to/femr/extract/folder...
#   Path to any arbitrary folder where you want to store your FEMR extract logs
export EXTRACT_LOGS=/path/to/femr/extract/logs...

# Do some data preprocessing with Stanford-specific helper scripts
#   Extract data from flowsheets
python tools/stanford/flowsheet_cleaner.py --num_threads 5 $OMOP_SOURCE "${EXTRACT_DESTINATION}_flowsheets"
#   Normalize visits
python tools/omop/normalize_visit_detail.py --num_threads 5 "${EXTRACT_DESTINATION}_flowsheets" "${EXTRACT_DESTINATION}_flowsheets_detail"

# Run actual FEMR extraction
etl_stanford_omop "${EXTRACT_DESTINATION}_flowsheets_detail" $EXTRACT_DESTINATION $EXTRACT_LOGS --num_threads 10
```

Example usage (Note: This should take ~10 minutes on a 1% extract of STARR-OMOP)

```bash
export OMOP_SOURCE=/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_1pcent_2022_11_09
export EXTRACT_DESTINATION=/local-scratch/nigam/projects/mwornow/femr_starr_omop_cdm5_deid_1pcent_2022_11_09
export EXTRACT_LOGS=/local-scratch/nigam/projects/mwornow/femr_starr_omop_cdm5_deid_1pcent_2022_11_09_logs

python tools/stanford/flowsheet_cleaner.py --num_threads 5 $OMOP_SOURCE "${EXTRACT_DESTINATION}_flowsheets"
python tools/omop/normalize_visit_detail.py --num_threads 5 "${EXTRACT_DESTINATION}_flowsheets" "${EXTRACT_DESTINATION}_flowsheets_detail"

etl_stanford_omop "${EXTRACT_DESTINATION}_flowsheets_detail" $EXTRACT_DESTINATION $EXTRACT_LOGS --num_threads 10
```

### (Optional) Installing PyTorch

If you are on Nero, you need to install PyTorch using:

```bash
conda install numpy -y
pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu111
```

If you are on Carina, you need to install PyTorch using:

```bash
conda install numpy pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
```

# FEMR
### Framework for Electronic Medical Records

**FEMR** is a Python package for manipulating longitudinal EHR data for machine learning, with a focus on supporting the creation of foundation models and verifying their [presumed benefits](https://hai.stanford.edu/news/how-foundation-models-can-advance-ai-healthcare) in healthcare. Such a framework is needed given the [current state of large language models in healthcare](https://hai.stanford.edu/news/shaky-foundations-foundation-models-healthcare) and the need for better evaluation frameworks.

The currently supported foundation models are [CLMBR](https://arxiv.org/pdf/2001.05295.pdf) and [MOTOR](https://arxiv.org/abs/2301.03150).

**FEMR** by default supports the [OMOP Common Data Model](https://www.ohdsi.org/data-standardization/) developed by the OHDSI community, but can also be used with other forms of EHR / claims data with minimal processing. Data that has been used with **FEMR** includes MIMIC-IV, Optum, Truven, STARR-OMOP, and SickKids-OMOP. 

**FEMR** helps users:
1. [Manipulate events in the EHR data comprising a patient's timeline](https://github.com/som-shahlab/femr/blob/main/tutorials/1_Overview.ipynb)
2. [Algorithmically label patient records based on structured data](https://github.com/som-shahlab/femr/blob/main/tutorials/3_Labeling.ipynb)
3. [Generate tabular features from patient timelines for use with traditional gradient boosted tree models](https://github.com/som-shahlab/femr/blob/main/tutorials/4_Count%20Featurization%20And%20Modeling.ipynb)
4. [Train](https://github.com/som-shahlab/femr/blob/main/tutorials/5_Train%20CLMBR.ipynb) and [finetune](https://github.com/som-shahlab/femr/blob/main/tutorials/6_CLMBR%20Featurization%20And%20Modeling.ipynb) CLMBR-derived models for binary classification and prediction tasks.
5. Train and finetune MOTOR-derived models for making time-to-event predictions.

We recommend users start with our [tutorial folder](https://github.com/som-shahlab/femr/tree/main/tutorials)

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
pip install --upgrade "jax[cuda11_pip]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
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

### Special note for NERO users

As Nero does not have internet access, you must run the following before running the code above.

```bash
export DISTDIR=/local-scratch/nigam/distdir
```


### (Optional) Installing CUDA on Nero / Carina

As a side note for Nero/Carina users, do not use your home directory to save the femr repo and installation files due to limited storage. We recommend using the shared project folder, e.g., on nero, use '/local-scratch/nigam/project/...'

If you are using Nero, you will need to install CUDA manually until the CUDA version on Nero is updated. To do so, follow these steps:

1. Download version 11.8 of CUDA onto your local machine [from here](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=runfile_local)
2. Copy your CUDA download from your local machine onto Nero, into whatever folder you'd like. We'll refer to the path to this folder as `<PATH_TO_CUDA_INSTALLER>` from now on.
    - *Note:* Nero doesn't work with `scp`. You can use an alternative like `pscp`, which functions basically identically to `scp`. You can install `pscp` on a Mac by using `brew install putty`.
3. `ssh` into Nero using `ssh <username>@nero-nigam.compute.stanford.edu`
4. On Nero, run the CUDA installer as a bash command as follows: `bash <PATH_TO_CUDA_INSTALLER> --installpath=<INSTALL_PATH>`, where `<PATH_TO_CUDA_INSTALLER>` is the path to the file you downloaded/transferred in Step #2, and `<INSTALL_PATH>` is where you'd like to save your CUDA installation files. We recommend using `~` or something similar.
5. The CUDA installer will pop-up a window during installation. Uncheck all of the boxes it presents except for the box labeled "cuda toolkit".
6. After the installation completes, the installer will print out two paths to your console. Take note of these paths, and copy them into your `.bashrc` file by running the following commands.

7. Install cuDNN v8.7.0 (November 28th, 2022) for CUDA. Go to this [link](https://developer.nvidia.com/rdp/cudnn-archive) and download the file
`Download cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x` -> `Local Installer for Linux x86_64 (Tar)` on your local computer and transfer it
over to your local folder in nero. Then follow the instruction [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
section 1.3. Note that you need to copy over cudnn files to your local cuda. For example,

- `cp cudnn-*-archive/include/cudnn*.h <path_to_your_cuda>/include`
- `cp -P cudnn-*-archive/lib/libcudnn* <path_to_your_cuda>/lib64`
- `chmod a+r <path_to_your_cuda>/include/cudnn*.h <path_to_your_cuda>/lib64/libcudnn*`

8. Add the following to your .bashrc file. You may need to restart your terminal for the changes to be reflected.

```bash
export PATH="<INSTALL_PATH>/bin:$PATH"
export LD_LIBRARY_PATH="<INSTALL_PATH>/lib64:$LD_LIBRARY_PATH"
```

To write in a .bashrc file, use
```bash
nano ~/.bashrc
```

9. Run `rm /tmp/cuda-installer.log` to remove the installer log (if you don't do this, it will cause a segmentation fault for other users when they try to install CUDA).


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

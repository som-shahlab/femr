# FEMR
### Framework for Electronic Medical Records

**FEMR** is a python package for building models using EHR data.

**FEMR** offers the following four main types of functionality. In order, they are the ability to:
1. Convert EHR and claims data into a common schema, where each patient is associated with a timeline of events extracted from the EHR
2. Apply labeling functions on that schema in order to derive labels for each patient
3. Apply featurization schemes to obtain feature matrices for each patient
4. Perform other common tasks necessary for research with EHR data

# Installation

To install **FEMR** run the following, replacing `<FEMR_ENV>` with the desired name of your conda environment:

```bash
conda create -n FEMR_ENV python=3.10 bazel=5.3 clangxx=14 -c conda-forge
conda activate FEMR_ENV

export BAZEL_USE_CPP_ONLY_TOOLCHAIN=1

git clone https://github.com/som-shahlab/femr.git
cd femr
pip install -e .
```

### Note: FEMR currently is only tested on Linux. Installation might not work on macOS or Windows. Support for those platforms is currently in progress.

### Special note for NERO users

As Nero does not have internet access, you must run the following before running the code below.

```bash
export DISTDIR=/local-scratch/nigam/distdir
```

1. Download the right version of cuda on your local machine and transfer it over a folder in nero [link](https://developer.nvidia.com/cuda-11.1.1-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)
2. Run the installer as bash command by providing path to install such as `bash <CUDA_PATH> --installpath=<INSTALL_PATH>`, where `<CUDA_PATH>` is the path to the file you downloaded/transferred in step 1 and `<INSTALL_PATH>` is where you'd like to store the CUDA installation files. During installation, uncheck all the boxes except cuda toolkit.
3. After the installation completes, it will spit out two paths on terminal that should be put into your .bashrc file:
```bash
export PATH="<INSTALL_PATH>/bin:$PATH"
export LD_LIBRARY_PATH="<INSTALL_PATH>/lib64:$LD_LIBRARY_PATH"
```
4. Delete the `/tmp/cuda-installer.log` file or it will create problems (e.g., segmentation fault) for other users

Note: you may need to restart your terminal for the changes to reflect

### (Optional) Installing PyTorch

If you want to use PyTorch for deep learning, you can install it as follows:

```bash
conda install numpy
pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu111
```

### (Optional) Installing CUDA on Nero

If you are using Nero, you will need to install CUDA manually until the CUDA version on Nero is updated. To do so, follow these steps:

1. Download version 11.1.1 of CUDA onto your local machine [from here](https://developer.nvidia.com/cuda-11.1.1-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)
2. Copy your CUDA download from your local machine onto Nero, into whatever folder you'd like. We'll refer to the path to this folder as `<PATH_TO_CUDA_INSTALLER>` from now on.
    - *Note:* Nero doesn't work with `scp`. You can use an alternative like `pscp`, which functions basically identically to `scp`. You can install `pscp` on a Mac by using `brew install putty`.
3. `ssh` into Nero using `ssh <username>@nero-nigam.compute.stanford.edu`
4. On Nero, run the CUDA installer as a bash command as follows: `bash <PATH_TO_CUDA_INSTALLER> --installpath=<INSTALL_PATH>`, where `<PATH_TO_CUDA_INSTALLER>` is the path to the file you downloaded/transferred in Step #2, and `<INSTALL_PATH>` is where you'd like to save your CUDA installation files. We recommend using `~` or something similar.
5. The CUDA installer will pop-up a window during installation. Uncheck all of the boxes it presents except for the box labeled "cuda toolkit".
6. After the installation completes, the installer will print out two paths to your console. Take note of these paths, and copy them into your `.bashrc` file by running the following commands. You may need to restart your terminal for the changes to be reflected.
```bash
export PATH="<INSTALL_PATH>/bin:$PATH"
export LD_LIBRARY_PATH="<INSTALL_PATH>/lib64:$LD_LIBRARY_PATH"
```
4. Run `rm /tmp/cuda-installer.log` to remove the installer log (if you don't do this, it will cause a segmentation fault for other users when they try to install CUDA).

# Development

The following guides are for developers who want to contribute to **FEMR**.

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
export EXTRACT_DESTINATION=/local-scratch/nigam/projects/mwornow/piton_starr_omop_cdm5_deid_1pcent_2022_11_09
export EXTRACT_LOGS=/local-scratch/nigam/projects/mwornow/piton_starr_omop_cdm5_deid_1pcent_2022_11_09_logs

python tools/stanford/flowsheet_cleaner.py --num_threads 5 $OMOP_SOURCE "${EXTRACT_DESTINATION}_flowsheets"
python tools/omop/normalize_visit_detail.py --num_threads 5 "${EXTRACT_DESTINATION}_flowsheets" "${EXTRACT_DESTINATION}_flowsheets_detail"

etl_stanford_omop "${EXTRACT_DESTINATION}_flowsheets_detail" $EXTRACT_DESTINATION $EXTRACT_LOGS --num_threads 10
```


## Training CLMBR

The tutorial to run CLMBR model is in `tutorials/4_train_clmbr_model.py`. However, before running this script, some additional dependencies need to be installed.
Please follow the steps below:

1. Jax is one of the important package to train clmbr, but before installing that, you will need to have cuda and cudnn installed. For Nero users, please follow the
instructions above to install cuda. Additionally, install the latest version of cudnn from nvidia. You will need to create developer account and download the file for specific version
of your system. For Nero users, please go to this [link](https://developer.nvidia.com/rdp/cudnn-archive) and download the file
`Download cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x` -> `Local Installer for Linux x86_64 (Tar)` on your local computer and transfer it
over to your local folder in nero. Then follow the instruction [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
section 1.3. Note that you need to copy over cudnn files to your local cuda. For example,

- `cp cudnn-*-archive/include/cudnn*.h <path_to_your_cuda>/include`
- `cp -P cudnn-*-archive/lib/libcudnn* <path_to_your_cuda>/lib64`
- `chmod a+r <path_to_your_cuda>/include/cudnn*.h <path_to_your_cuda>/lib64/libcudnn*`

2. Install Jax by running `pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`, as shown [here](https://github.com/google/jax#installation).
3. Run `pip install dm-haiku msgpack optax`

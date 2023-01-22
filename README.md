# piton

**piton** is a Python package for ingesting EHR data for building ML models.

**piton** offers the following four main types of functionality. In order, they are the ability to:
1. Convert EHR and claims data into a common schema, where each patient is associated with a timeline of events
2. Apply labeling functions on that schema in order to derive labels for each patient
3. Apply featurization schemes to obtain feature matrices for each patient
4. Perform other common tasks necessary for research with EHR data

# Installation

> **NOTE FOR NERO USERS:** As Nero does not have internet access, you must run the following command before running the code below.
> ```
> export DISTDIR=/local-scratch/nigam/distdir
> ```

To install **piton** run the following, replacing `<PITON_ENV>` with the desired name of your conda environment:

```bash
conda create -n <PITON_ENV> python=3.10 bazel=5.3 clangxx=14 -c conda-forge
conda activate <PITON_ENV>
git clone https://github.com/som-shahlab/piton.git
cd piton
pip install -e .
```
### (Optional) Installing PyTorch

If you want to use PyTorch for deep learning, you can install it as follows:

```bash
conda install numpy
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
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
```
export PATH="<INSTALL_PATH>/bin:$PATH"
export LD_LIBRARY_PATH="<INSTALL_PATH>/lib64:$LD_LIBRARY_PATH"
```
4. Run `rm /tmp/cuda-installer.log` to remove the installer log (if you don't do this, it will cause a segmentation fault for other users when they try to install CUDA).

# Development

The following guides are for developers who want to contribute to **piton**.

## Precommit checks

Before committing, please run the following commands to ensure that your code is formatted correctly and passes all tests.

### Installation
```
conda install pre-commit pytest -y
pre-commit install
```

### Running

#### Test Functions

```
pytest tests
```

### Formatting Checks

```
pre-commit run --all-files
```

# Miscellaneous

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
export OMOP_SOURCE=/share/pi/nigam...
export EXTRACT_DESTINATION=/share/pi/nigam...
export EXTRACT_LOGS=/share/pi/nigam...

etl_stanford_omop $OMOP_SOURCE $EXTRACT_DESTINATION $EXTRACT_LOGS --num_threads 10
```

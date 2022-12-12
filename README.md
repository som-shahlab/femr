# piton

**piton** is a python package for building models using EHR data.

**piton** offers the following four main types of functionality. In order, they are the ability to:
1. Convert EHR and claims data into a common schema, where each patient is associated with a timeline of events extracted from the EHR
2. Apply labeling functions on that schema in order to derive labels for each patient
3. Apply featurization schemes to obtain feature matrices for each patient
4. Perform other common tasks necessary for research with EHR data

As part of the model building process, it offers the ability to learn clinical language model based representations (CLMBR) as described in Steinberg et al at https://pubmed.ncbi.nlm.nih.gov/33290879/.


# Documentation

https://ehr-ml.readthedocs.io/en/latest/ has (outdated) documentation, including setup instructions and a tutorial using SynPuf data.

# Installation

Special note for NERO users:

You will need to install cuda manually until cuda version is updated on the nero. Follow the following steps for nero. 

1. Download the right version of cuda on your local machine and transfer it over a folder in nero.
2. Run the installer and make sure to provide correct install path in your home directory. 
3. Delete /tmp/cuda-insatll.log file cause it will create problems for other users. 

As Nero does not have internet access, you must run the following before running the code below.

```
export DISTDIR=/local-scratch/nigam/distdir
```

Run the following:

```
conda create -n piton_env python=3.10 bazel=5 clangxx=14 -c conda-forge
conda activate piton_env
git clone https://github.com/som-shahlab/piton.git
cd piton
pip install -e .
```


# Precommit checks

## Installation
```
conda install pre-commit
conda install pytest
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

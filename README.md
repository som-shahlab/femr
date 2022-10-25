# piton

piton is a python package for building models using EHR data. As part of the model building process, it offers the ability to learn clinical language model based representations (CLMBR) as described in Steinberg et al at https://pubmed.ncbi.nlm.nih.gov/33290879/.

There are four main groups of functionality in piton. The ability to:
1. Convert EHR and claims data into a common schema
2. Apply labeling functions on that schema in order to derive labels
3. Apply featurization schemes on those patients to obtain feature matrices
4. Perform other common tasks necessary for research with EHR data

https://ehr-ml.readthedocs.io/en/latest/ has the full documentation, including setup instructions and a tutorial using SynPuf data.

Installation instructions:

```
conda create -n env_name python=3.10 bazel=5 clangxx=14 -c conda-forge

conda activate env_name

git clone https://github.com/som-shahlab/piton.git

cd piton

pip install -e .
```

Special note for NERO users:

As Nero does not have internet access, you must run the following before pip install -e .

```
export DISTDIR=/local-scratch/nigam/distdir
```

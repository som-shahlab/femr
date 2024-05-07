# FEMR (EHRSHOT compatible version)

### Framework for Electronic Medical Records

Please [see here](https://github.com/som-shahlab/femr/tree/main) for a discussion of what FEMR is.

This is an EHRSHOT compatible version of FEMR, based on FEMR version 0.0.20.

# Installation

```bash
pip install ehrshot_femr
pip install --upgrade "jax[cuda11_pip]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

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
#   GZIP decompression
gunzip $OMOP_SOURCE/**/*.csv.gz
#   Apply zstd compression
zstd -1 --rm $OMOP_SOURCE/**/*.csv
#   Extract data from flowsheets
python tools/stanford/flowsheet_cleaner.py --num_threads 5 $OMOP_SOURCE "${EXTRACT_DESTINATION}_flowsheets"
#   Normalize visits
python tools/omop/normalize_visit_detail.py --num_threads 5 "${EXTRACT_DESTINATION}_flowsheets" "${EXTRACT_DESTINATION}_flowsheets_detail"

# Run actual FEMR extraction
etl_stanford_omop "${EXTRACT_DESTINATION}_flowsheets_detail" $EXTRACT_DESTINATION $EXTRACT_LOGS --num_threads 10
```

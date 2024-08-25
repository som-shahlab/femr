# FEMR
### Framework for Electronic Medical Records

**FEMR** is a Python package for manipulating longitudinal EHR data for machine learning, with a focus on supporting the creation of foundation models and verifying their [presumed benefits](https://hai.stanford.edu/news/how-foundation-models-can-advance-ai-healthcare) in healthcare. Such a framework is needed given the [current state of large language models in healthcare](https://hai.stanford.edu/news/shaky-foundations-foundation-models-healthcare) and the need for better evaluation frameworks.

The currently supported foundation models are [CLMBR](https://arxiv.org/pdf/2001.05295.pdf) and [MOTOR](https://arxiv.org/abs/2301.03150).

**FEMR** works with data that has been converted to the [MEDS](https://github.com/Medical-Event-Data-Standard/) schema, a simple schema that supports a wide variety of EHR / claims datasets. Please see the MEDS documentation, and in particular its [provided ETLs](https://github.com/Medical-Event-Data-Standard/meds_etl) for help converting your data to MEDS.

**FEMR** helps users:
1. [Use ontologies to better understand / featurize medical codes](http://github.com/som-shahlab/femr/blob/main/tutorials/1_Ontology.ipynb)
2. [Algorithmically label patient records based on structured data](https://github.com/som-shahlab/femr/blob/main/tutorials/2_Labeling.ipynb)
3. [Generate tabular features from patient timelines for use with traditional gradient boosted tree models](https://github.com/som-shahlab/femr/blob/main/tutorials/3_Count%20Featurization%20And%20Modeling.ipynb)
4. [Train](https://github.com/som-shahlab/femr/blob/main/tutorials/4_Train%20CLMBR.ipynb) and [finetune](https://github.com/som-shahlab/femr/blob/main/tutorials/5_CLMBR%20Featurization%20And%20Modeling.ipynb) CLMBR-derived models for binary classification and prediction tasks.
5. [Train](https://github.com/som-shahlab/femr/blob/main/tutorials/6_Train%20MOTOR.ipynb) and [finetune](https://github.com/som-shahlab/femr/blob/main/tutorials/7_MOTOR%20Featurization%20And%20Modeling.ipynb) MOTOR-derived models for binary classification and prediction tasks.

We recommend users start with our [tutorial folder](https://github.com/som-shahlab/femr/tree/main/tutorials)

# Installation

```bash
pip install femr

# If you are using deep learning, you also need to install xformers
#
# Note that xformers has some known issues with MacOS.
# If you are using MacOS you might also need to install llvm. See https://stackoverflow.com/questions/60005176/how-to-deal-with-clang-error-unsupported-option-fopenmp-on-travis
pip install xformers

```
# Getting Started

The first step of using **FEMR** is to convert your patient data into [MEDS](https://github.com/Medical-Event-Data-Standard), the standard input format expected by **FEMR** codebase.

**Note: FEMR currently only supports MEDS v1, so you will need to install MEDS v1 versions of packages. Aka pip install meds-etl==0.1.3**

The best way to do this is with the [ETLs provided by MEDS](https://github.com/Medical-Event-Data-Standard/meds_etl).


## OMOP Data

If you have OMOP CDM formated data, follow these instructions:

1. Download your OMOP dataset to `[PATH_TO_SOURCE_OMOP]`.
2. Convert OMOP => MEDS using the following:
```bash
# Convert OMOP => MEDS data format
meds_etl_omop [PATH_TO_SOURCE_OMOP] [PATH_TO_OUTPUT_MEDS]
```

3. Use HuggingFace's Datasets library to load our dataset in Python
```bash
import datasets
dataset = datasets.Dataset.from_parquet(PATH_TO_OUTPUT_MEDS + 'data/*')

# Print dataset stats
print(dataset)
>>> Dataset({
>>>   features: ['patient_id', 'events'],
>>>   num_rows: 6732
>>> })

# Print number of events in first patient in dataset
print(len(dataset[0]['events']))
>>> 2287
```

## Stanford STARR-OMOP Data

If you are using the STARR-OMOP dataset from Stanford (which uses the OMOP CDM), we add an initial Stanford-specific preprocessing step. Otherwise this should be identical to the **OMOP Data** section. Follow these instructions:

1. Download your STARR-OMOP dataset to `[PATH_TO_SOURCE_OMOP]`.
2. Convert STARR-OMOP => MEDS using the following:
```bash
# Convert OMOP => MEDS data format
meds_etl_omop [PATH_TO_SOURCE_OMOP] [PATH_TO_OUTPUT_MEDS]_raw

# Apply Stanford fixes
femr_stanford_omop_fixer [PATH_TO_OUTPUT_MEDS]_raw [PATH_TO_OUTPUT_MEDS]
```

3. Use HuggingFace's Datasets library to load our dataset in Python
```bash
import datasets
dataset = datasets.Dataset.from_parquet(PATH_TO_OUTPUT_MEDS + 'data/*')

# Print dataset stats
print(dataset)
>>> Dataset({
>>>   features: ['patient_id', 'events'],
>>>   num_rows: 6732
>>> })

# Print number of events in first patient in dataset
print(len(dataset[0]['events']))
>>> 2287
```

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

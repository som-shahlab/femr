# FEMR
### Framework for Electronic Medical Records

**FEMR** is a Python package for manipulating longitudinal EHR data for machine learning, with a focus on supporting the creation of foundation models and verifying their [presumed benefits](https://hai.stanford.edu/news/how-foundation-models-can-advance-ai-healthcare) in healthcare. Such a framework is needed given the [current state of large language models in healthcare](https://hai.stanford.edu/news/shaky-foundations-foundation-models-healthcare) and the need for better evaluation frameworks.

The currently supported foundation models is [MOTOR](https://arxiv.org/abs/2301.03150).

(Users who want to train auto-regressive CLMBR-style models should use [FEMR 0.1.16](https://github.com/som-shahlab/femr/releases/tag/0.1.16) or https://github.com/som-shahlab/hf_ehr)

**FEMR** works with data that has been converted to the [MEDS](https://github.com/Medical-Event-Data-Standard/) schema, a simple schema that supports a wide variety of EHR / claims datasets. Please see the MEDS documentation, and in particular its [provided ETLs](https://github.com/Medical-Event-Data-Standard/meds_etl) for help converting your data to MEDS.

**FEMR** helps users:
1. [Use ontologies to better understand / featurize medical codes](http://github.com/som-shahlab/femr/blob/main/tutorials/1_Ontology.ipynb)
2. [Algorithmically label subject records based on structured data](https://github.com/som-shahlab/femr/blob/main/tutorials/2_Labeling.ipynb)
3. [Generate tabular features from subject timelines for use with traditional gradient boosted tree models](https://github.com/som-shahlab/femr/blob/main/tutorials/3_Count%20Featurization%20And%20Modeling.ipynb)
4. [Train](https://github.com/som-shahlab/femr/blob/main/tutorials/4_Train%20MOTOR.ipynb) and [finetune](https://github.com/som-shahlab/femr/blob/main/tutorials/5_MOTOR%20Featurization%20And%20Modeling.ipynb) MOTOR-derived models for binary classification and prediction tasks.

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

The first step of using **FEMR** is to convert your subject data into [MEDS](https://github.com/Medical-Event-Data-Standard), the standard input format expected by **FEMR** codebase.

**Note: FEMR currently only supports MEDS v3, so you will need to install MEDS v3 versions of packages. Aka pip install meds-etl==0.3.11**

The best way to do this is with the [ETLs provided by MEDS](https://github.com/Medical-Event-Data-Standard/meds_etl).


## OMOP Data

If you have OMOP CDM formated data, follow these instructions:

1. Download your OMOP dataset to `[PATH_TO_SOURCE_OMOP]`.
2. Convert OMOP => MEDS using the following:
```bash
# Convert OMOP => MEDS data format
meds_etl_omop [PATH_TO_SOURCE_OMOP] [PATH_TO_OUTPUT_MEDS]
```

## Site-specific post-ETL fixes

OMOP extracts from different sites have different data-quality quirks (midnight timestamps, billing codes stamped at
visit start, unreliable flowsheet measurements, ...). FEMR provides a configurable post-ETL fix system via
`femr_omop_fixer`, which applies a sequence of small, named "fixes" (subject-level transforms) to a MEDS dataset:

```bash
# Apply the generic profile: site-agnostic timing/code fixes for any OMOP => MEDS output
femr_omop_fixer [PATH_TO_OUTPUT_MEDS]_raw [PATH_TO_OUTPUT_MEDS]

# Apply the Stanford profile (exactly reproduces femr_stanford_omop_fixer)
femr_omop_fixer [PATH_TO_OUTPUT_MEDS]_raw [PATH_TO_OUTPUT_MEDS] --profile stanford

# Or describe the fixes yourself in a JSON config file
femr_omop_fixer [PATH_TO_OUTPUT_MEDS]_raw [PATH_TO_OUTPUT_MEDS] --config my_site.json
```

A config file selects a base profile and/or lists the fixes to apply (a `"fixes"` list replaces the profile's list):

```json
{
    "profile": "generic",
    "fixes": [
        {"name": "move_pre_birth"},
        {"name": "move_to_day_end"},
        {"name": "remove_codes", "params": {"codes": ["MY_OBS/Flowsheet"]}},
        {
            "name": "move_billing_codes",
            "params": {
                "billing_code_tables": ["mysite_pat_enc_dx"],
                "encounter_tables": ["mysite_pat_enc"]
            }
        }
    ]
}
```

Built-in fixes: `move_pre_birth`, `move_visit_start_to_first_event_start`, `move_to_day_end`, `switch_to_icd10cm`,
`move_billing_codes`, `remove_nones`, `delta_encode`, `remove_codes`, `fix_events`.

Sites can add their own fixes without forking FEMR internals by registering a fix factory:

```python
from femr.post_etl_pipelines import site_fixes

@site_fixes.register_fix("drop_test_patients")
def _make_drop_test_patients() -> site_fixes.SubjectTransform:
    def drop_test_patients(subject):
        ...
        return subject
    return drop_test_patients
```

and then referencing `"drop_test_patients"` by name in the config.

## Stanford STARR-OMOP Data

If you are using the STARR-OMOP dataset from Stanford (which uses the OMOP CDM), we add an initial Stanford-specific
preprocessing step. Otherwise this should be identical to the **OMOP Data** section. Follow these instructions:

1. Download your STARR-OMOP dataset to `[PATH_TO_SOURCE_OMOP]`.
2. Convert STARR-OMOP => MEDS using the following:
```bash
# Convert OMOP => MEDS data format
meds_etl_omop [PATH_TO_SOURCE_OMOP] [PATH_TO_OUTPUT_MEDS]_raw

# Apply Stanford fixes
femr_stanford_omop_fixer [PATH_TO_OUTPUT_MEDS]_raw [PATH_TO_OUTPUT_MEDS]
```

`femr_stanford_omop_fixer` is retained for backwards compatibility and is exactly equivalent to
`femr_omop_fixer --profile stanford`; see **Site-specific post-ETL fixes** above for the general mechanism.

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

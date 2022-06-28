ehr_ml.clmbr
==================================

:py:mod:`ehr_ml.clmbr` is an implementation of the CLMBR representation learning algorithm (see https://arxiv.org/abs/2001.05295).
It consists of components for both training a language model and extraction representations from that language model.


*******************************
Additional Dependencies
*******************************

CLMBR requires additional dependencies that are not needed for the rest of ehr_ml. In particular, it requires both pytorch and https://github.com/Lalaland/embedding_dot.

https://github.com/Lalaland/embedding_dot can be installed through the following commands:


.. code-block:: console

   git clone https://github.com/Lalaland/embedding_dot
   cd embedding_dot
   pip install -e .


*******************************
Language Model Training
*******************************

There are two programs contained within ehr_ml that are necessary for training a CLMBR language model.

clmbr_create_info

clmbr_create_info is a command line program for preprocessing the dataset before training.
clmbr_create_info is responsible for two very important hyperparameters, the time splits and the minimum patient feature count.
CLMBR operates under a time split assumption such that training data, validation data and test data come from different times.
Everything between train_start_date and train_end_date is training. Everything between val_start_date and val_end_date is for validation.
Everything past the val_end_date is assumed to be the test set and is not touched by CLMBR, either for internal validation or training.

The second major hyperparameter is the minimum patient count. This controls how many patients are required to have a feature before it is used in the model.

clmbr_create_info takes the following parameters:
    - extract_dir: The extract directory
    - save_dir: The name of the output directory
    - train_end_date: The end date of the train split
    - val_end_date: The end date of the validation split
    - min_patient_count: The minimum patient count for each feature
    - banned_patient_file: A file containing a list of patient ids to exclude from the model
    - train_start_date: The start date for representation learning training. Defaults to 1900-01-01
    - val_start_date: The start date for representation learning validation. Defaults to train_end_date.

clmbr_train_model

clmbr_train_model is a command line program responsible for model training. It controls most of the hyperameters for the model architecture.
Please see our paper for the fine details on these hyperameters. We recommend leaving most parameters unspecified as the defaults.

clmbr_train_model takes the following parameters:
    - model_dir: The location to store the model
    - info_dir: The info directory
    - lr: The learning rate
    - size: The primary embedding size
    - use_gru: Whether to use a GRU or a transformer
    - no_tied_weights: A flag to disable tied weights
    - gru_layers: How many GRU layers to use
    - gru_hidden_size: The size of the GRU hidden layers
    - dropout: Internal dropout within the model
    - l2: The L2 regularization
    - batch_size: The batch size
    - no_cuda: Force disable CUDA
    - code_dropout: Dropout codes at the input layer

***********************************
Language Model Feature Extraction
***********************************

Once a language model is trained, it is then possible to extract features for use in other tasks. 

The function featurize_patients provides code that performs this tasks

.. py:function:: ehr_ml.clmbr.featurize_patients(model_dir: str, extract_dir: str, l: labeler.SavedLabeler) -> np.array
    
    Featurize patients using the given model and labeler.
    The result is a numpy array aligned with l.get_labeler_data().
    This function will use the GPU if it is available.

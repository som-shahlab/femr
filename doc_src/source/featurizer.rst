ehr_ml.featurizer
==================================

:py:mod:`ehr_ml.featurizer` is a module for defining feature extractors.

*********************************
Definition Of A Feature Extractor
*********************************

A feature extractor is a class which takes a patient and a list of labeled timepoints and then returns a row for each timepoint. Feature extractors must be trained before they are used to compute normalization statistics. Most of the candidate features within STRIDE are sparse, so a sparse representation named :py:class:`ColumnValue` is used to represent the values returned by a feature extractor. :py:class:`Featurizer` is an interface for feature extractors that descripes the required methods.

.. autoclass:: ehr_ml.featurizer.ColumnValue

.. autoclass:: ehr_ml.featurizer.Featurizer
    :members:

.. autoclass:: ehr_ml.featurizer.FeaturizerList
    :members:


****************************
Provided Feature Extractors
****************************

.. autoclass:: ehr_ml.featurizer.AgeFeaturizer

.. autoclass:: ehr_ml.featurizer.CountFeaturizer
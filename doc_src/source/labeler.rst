ehr_ml.labeler
==================================

:py:mod:`stride_ml.labeler` is a module for defining labeling functions.

****************************
Definition Of A Labeler
****************************

:py:mod:`stride_ml.labeler` works on the assumption that there may be multiple or zero labels per patient. 

Each label is defined by :py:class:`Label` to consist of a day offset and a label as defined by :py:class:`Label`.
Labels can be either time to event labels, numeric labels, categorical or binary labels.

:py:class:`Label` provides the primary interface for specifying labeling functions as code that takes in timelines and returns lists of labels.

.. autoclass:: ehr_ml.labeler.Label
    :members:

.. autoclass:: ehr_ml.labeler.Labeler
    :members:



****************************
Provided Labeler Utilities
****************************

`ehr_ml` provides some utility classes to assist with building labelers.


.. autoclass:: ehr_ml.labeler.SavedLabeler

.. autoclass:: ehr_ml.labeler.FixedTimeHorizonEventLabeler

.. autoclass:: ehr_ml.labeler.InfiniteTimeHorizonEventLabeler

.. autoclass:: ehr_ml.labeler.RandomSelectionLabeler

.. autoclass:: ehr_ml.labeler.YearHistoryRequiredLabeler

.. autoclass:: ehr_ml.labeler.CodeLabeler

.. autoclass:: ehr_ml.labeler.PredictionAfterDateLabeler

.. autoclass:: ehr_ml.labeler.PatientSubsetLabeler



****************************
Provided Labelers
****************************


`ehr_ml` also contains some prebuilt labelers for common use.


.. autoclass:: ehr_ml.labeler.MortalityLabeler

.. autoclass:: ehr_ml.labeler.OpioidOverdoseLabeler

.. autoclass:: ehr_ml.labeler.LupusDiseaseLabeler

.. autoclass:: ehr_ml.labeler.InpatientMortalityLabeler

ehr_ml.index
==================================

:py:mod:`ehr_ml.index` is a module for quickly obtaining the set of patients who have a particular code in their records.
It is especially useful for rapid phenotyping queries and related tasks.

.. py:class:: Index

    A class for finding patients that have certain codes.

    .. py:method:: __init__(filename: str)

        Construct an index reader given the filename.

    .. py:method:: get_patient_ids(code: int) -> Sequence[int]

        Get the patient ids with a particular code.

    .. py:method:: get_all_patient_ids(codes: Iterable[int]) -> Sequence[int]

        Get the patient ids with one of a set of particular codes.
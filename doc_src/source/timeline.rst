ehr_ml.timeline
==================================

:py:mod:`ehr_ml.timeline` is a module for handling patient timelines.

It consists of a compact and efficient patient timeline format as well as some tools for working with patient timelines.

************************
:py:mod:`ehr_ml` schema
************************

:py:mod:`ehr_ml` has a simple schema for EHR data. One special note is that all observations are encoded with integers. See TermDictionary for how those integers get mapped to and from strings.

.. py:class:: Patient

    The records for a single patient.

    .. py:attribute:: patient_id
        :type: int

        The id for this particular patient.

    .. py:attribute:: days
        :type: Sequence[Day]

        The days with events for this patient.

.. py:class:: PatientDay

    A day of observations for a patient.

    .. py:attribute:: date
        :type: datetime.date

        The calendar date.

    .. py:attribute:: age
        :type: int

        The age in days.

    .. py:attribute:: observations
        :type: Sequence[int]

        The events that occurred on that day.

    .. py:attribute:: observations_with_values
        :type: Sequence[ObservationWithValue]

        The events with values that occurred on that day.


.. py:class:: ObservationWithValue

    An observation code that is attached to either a text or numeric value.

    .. py:attribute:: code
        :type: int

        The observation code.

    .. py:attribute:: is_text
        :type: bool

        Whether the observation is a text observation

    .. py:attribute:: numeric_value
        :type: float

        The numeric value if the observation is not a text observation.

        This field is only present if is_text == False.

    .. py:attribute:: text_value
        :type: int

        The text value if the observation is a text observation.

        This fields is only present if is_text == True.


.. py::class:: TermDictionary:
    A utility for mapping dictionary codes to and from string terms.

    .. py:attribute:: map(term: str) -> Optional[int]:

        Convert a dictionary term into the corresponding integer code.
        Returns None if that term is not in the dictionary.

    .. py:attribute:: get_word(self, code: int) -> Optional[str]:

        Convert a code back to the string term.
        Returns None if the code is not in the dictionary.


*******************************
Storing and retreiving patients
*******************************

Storing and retreiving patients is done through the :py:mod:`ehr_ml.timeline.TimelineReader` class.

.. py:class:: TimelineReader

    A utility class for retreiving patient timelines.

    .. py:method:: __init__(filename: str, readall: bool = False)

        Construct a timeline reader with the given filename.
        The readall parameter indicates whether the reader should be optimized for reading the entire dataset.
        Set readall to false if it's expected that only a small number of records will be read.

    .. py:method:: get_patient(patient_id: int, start_date: Optional[datetime.date] = None, end_date: Optional[dataset.date] = None) -> Patient

        Get a particular patient with the given patient id. start_date and end_date allow control over loading only a certain fraction of the patient.

    .. py:method:: get_patients(patient_ids: Optional[Sequence[int]], start_date: Optional[datetime.date] = None, end_date: Optional[dataset.date] = None) -> Iterator[Patient]

        Retrieve a sequence of patients (or all the patients if no sequence is provided).

    .. py:method:: get_patient_ids() -> Iterator[Int]

        Return the list of patient ids stored in this dataset.


    .. py:method:: get_original_patient_ids() -> Iterator[Int]

        Return the list of original patient ids stored in this dataset.
        This method is necessary when there is an additional patient id mapping going on.
        Note that the order of the result is the same as get_patient_ids().

    .. py:method:: get_dictionary(self) -> TermDictionary:

        Obtain the dictionary used for mapping observation codes.

    .. py:method:: get_value_dictionary(self) -> TermDictionary:

        Obtain the dictionary used for mapping text observation values.


*******************************
Manually Inspecting Timelines
*******************************

The commind line tool inspect_timelines enables manual inspection of patient timelines.

.. code-block:: console

   pip install ehr_ml

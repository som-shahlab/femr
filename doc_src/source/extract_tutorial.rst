Extract Tutorial
==================================

This tutorial assumes that you already have installed ehr_ml. See the Setup page for more details.

***************************
Overview
***************************

The goal of the extraction process is to create a database of patients where each patient has information in the following schema:

.. code-block:: python

    @dataclass
    class Patient:
        person_id: int
        events: List[Event]

    @dataclass
    class Event:
        start: datetime.datetime
        end: datetime.datetime | None

        code: str
        value: str | float | None

        event_type: str

        id: int | None
        parent_id: int | None


Extract creations consists of four main components

1. Creating events

2. Creating patients

3. Normalizing patients

4. Generating extracts

This tutorial will walk through each of those steps

***************************
Creating Events
***************************

The first step of extraction is to generate raw event files for each patient.

You simply create a directory with event files generated in any manner of your choosing.

This can be done using the EventWriter class 

.. autoclass:: ehr_ml.extract.EventWriter
    :members:

For this tutorial, we will simply create one event file with just two hard coded events

.. code-block:: python

    os.mkdir('events')

    dummy_patient = ehr_ml.extract.Patient(
        person_id = 10,
        events = [
            Event(start = datetime.date(2010, 1, 3), code='E11.4', event_type = 'diagnosis_code')
            Event(start = datetime.date(2010, 1, 5), code='C80.1', event_type = 'diagnosis_code')
        ]
    )

    with ehr_ml.extract.EventWriter(events) as w:
        w.add_patient(dummy_patient)

One particularly common usecase is generating events from csv files. The CSVConverter API can be used to more easily implement that approach in a nice multithreaded manner.

.. autoclass:: ehr_ml.extract.csv_converter.CSVConverter
    :members:

.. automethod:: ehr_ml.extract.csv_converter.run_csv_converters

***************************
Creating Patients
***************************

Given events, it is then necessary to convert them to patients.

This can done using the convert_events_to_patients method. Note that you should select a number of shards that corresponds to how many processors you have in your machine.

.. automethod:: ehr_ml.extract.convert_events_to_patients

Example

.. code-block:: python

    os.mkdir('raw_patients')

    ehr_ml.extract.convert_events_to_patients('events', 'raw_patients', 10)

***************************
Transforming Patients
***************************

Given the patient directory, you can then optionally apply a transformation step on those patients.

This allows you to perform ontology level cleanup, invalid data removal, and other sorts of postprocessing.

It also lets you generate or add additional events as necessary.

The core api for this is transform_patients.


.. automethod:: ehr_ml.extract.transform_patients

Example

.. code-block:: python

    os.mkdir('processed_patients')

    def transform(input: Patient) -> Patient:
        input.events = [a for a in input.events if a.code != 'E11.4'] # Remove E11.4 for some reason
        return input

    ehr_ml.transform_patients('raw_patients', 'proccessed_patients', transform)


***************************
Generating Extract
***************************

Finally, you can generate an extract given patients.

.. automethod:: ehr_ml.extract.convert_patients_to_extract

Example:

.. code-block:: python

    ehr_ml.extract.convert_patients_to_extract('processed_patient', 'extract')

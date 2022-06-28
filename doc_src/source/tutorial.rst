Tutorial
==================================

This tutorial assumes that you already have installed ehr_ml and have created an ehr_ml extract. See the Setup page for more details.

***************************
Exploring Timeline Objects
***************************

First, import ehr_ml.timeline and create an instance of the TimelineReader class.
Note that you shoudl replace EHR_ML_EXTRACT_DIR with the corresponding extract folder.

.. code-block:: python

   import ehr_ml.timeline

   timelines = ehr_ml.timeline.TimelineReader(EHR_ML_EXTRACT_DIR + "/extract.db")


You are now able to find what patients are in your dataset.
Note that all patients have both a primary id and an original id. The original id allows you to lookup the patients in the source tables.

.. code-block:: python

   patient_ids = timelines.get_patient_ids()

   print(len(patient_ids))

   original_patient_ids = timelines.get_original_patient_ids()

   print(patient_ids[0], original_patient_ids[0])
   

It's now possible to extract a given patient using the get_patient function of a TimelineReader.
Each patient has a patient id and a list of days. The first day is the birth date.

.. code-block:: python

   patient = timelines.get_patient(patient_ids[0])
   print(patient.patient_id, "has", len(patient.days), "days of information")
   birth_day = patient.days[0]
   print("Born on", birth_day.date)


Each day for each patient has an age, a date, and a list of observations.
We can inspect these for this patient's birth date.
Note that we can use the dictionary attached to the timeline object to map the observation codes into textual descriptions.
Patients are assigned their demographic codes at birth.

.. code-block:: python

   dictionary = timelines.get_dictionary()

   print("Age at birth (should be 0)", birth_day.age)

   for obs in birth_day.observations:
      print("Got obs", dictionary.get_word(obs))

We can now inspect the day after birth to see more traditional medical observations

.. code-block:: python

   dictionary = timelines.get_dictionary()

   next_day = patient.days[1]

   print("Next age (should be 0)", next_day.age)

   for obs in next_day.observations:
      print("Got obs", dictionary.get_word(obs))

******************************
Ontology And Index Utilities
******************************

ehr_ml's ontology and index utilities make it easy to find relationships between codes and patients with particular codes.
These are particular useful for phenotyping queries and feature engineering.

First, import both the ontology and index and construct the corresponding classes.


.. code-block:: python

   import ehr_ml.ontology
   import ehr_ml.index

   ontologies = ehr_ml.ontology.OntologyReader(EHR_ML_EXTRACT_DIR + "/ontology.db")
   index = ehr_ml.index.Index(EHR_ML_EXTRACT_DIR + "/index.db")

We can now use the ontology class to find all Type 2 diabetes codes.

.. code-block:: python

   diabetes_codes = ontologies.get_words_for_subword_term("ICD10/E11")

   for code in diabetes_codes:
      # Print the codes that are diabetes codes
      print(dictionary.get_word(code))

We can also perform the reverse operation of getting all parent codes for a particular code. Note the use of the ontology dictionary.

.. code-block:: python

   ontology_dictionary = ontologies.get_dictionary()

   print(dictionary.get_word(diabetes_codes[0]))
   for ontology_code in ontologies.get_subwords(diabetes_codes[0]):
      
      print("Got parent", ontology_dictionary.get_word(ontology_code))

Finally, we can use the index as a tool to quickly find patients with particular codes.
For example, we can find all patients with diabetes codes.

.. code-block:: python

   diabetes_patient_ids = index.get_all_patient_ids(diabetes_codes)

   print(len(diabetes_patient_ids))


******************************
Labeling
******************************

ehr_ml contains an API and utilities for easily defining labeling functions.
The core part of a labeling function is that it takes in a patient and returns labels for that patient. 

First, we need to import the labeling utilities.

.. code-block:: python

   import ehr_ml.labeler

Now we can start defining labelers in terms of our labeling utilites. One of the simplest utilities is the CodeLabeler class.
The CodeLabeler class enables the definition of simple time based code labelers that predict whether something will happen in a particular amount of time.
We can define a simple diabetes labeler given the CodeLabeler class as follows.


.. code-block:: python

   class DiabetesLabeler(ehr_ml.labeler.CodeLabeler):
      """
      The mortality task is defined as predicting whether or not an
      patient will get a diabetes code in the next 3 months.
      """

      def __init__(self, timelines: ehr_ml.timeline.TimelineReader):
         diabetes_code = timelines.get_dictionary().map("ICD10CM/E11.9")
         if diabetes_code is None:
            raise ValueError("Could not find the diabetes code")
         else:
            super().__init__(code=diabetes_code)

      def get_time_horizon(self) -> int:
         return 90

We can now use this labeler to label patients in our dataset.

.. code-block:: python

   labeler = DiabetesLabeler(timelines)
   labels = labeler.label(timelines.get_patient(patient_ids[4]))

   print(labels)


Whenever using a labeler, it's often worth considering whether SavedLabeler can improve the effectiveness of your pipelines. 
SavedLabeler enables labelers to be saved and then loaded later, which makes debugging easier and code run faster.

See the rest of the labeler package for additional utilities and the full labeler API.

******************************
Featurization
******************************

The final part of the ehr_ml toolkit is featurizers. 
Featurizers take in a patient record and a set of labels and generate features for each label.

First, we must import the featurization library:

.. code-block:: python

   import ehr_ml.featurizer

We can now define and train featurization scheme in terms of a list of featurizers.
We use a labeling function in order to define the set of patients and timepoints that get featurized.

.. code-block:: python

   featurizers = ehr_ml.featurizer.FeaturizerList(
        [ehr_ml.featurizer.AgeFeaturizer(normalize=False), ehr_ml.featurizer.CountFeaturizer(timelines.get_dictionary())])

   # This training step is necessary for featurizers that need to do normalization of various sorts
   featurizers.train_featurizers(timelines, labeler)

Finally, we can apply our featurizers to our data to obtain matrices. Note that a single patient might have multiple rows if the labeler triggers multiple times on a single patient.

.. code-block:: python

   features, labels, patient_ids, day_offsets = featurizers.featurize(timelines, labeler)

   print(features.shape, labels.shape, patient_ids.shape, day_offsets.shape)


******************************
CLMBR Featurization
******************************


ehr_ml contains an implementation of CLMBR (https://arxiv.org/abs/2001.05295), a representation learning technique for electronic health records.

CLMBR also comes with some additional dependencies. In particular, it requires both pytorch and https://github.com/Lalaland/embedding_dot.

https://github.com/Lalaland/embedding_dot can be installed through the following commands:

.. code-block:: console

   git clone https://github.com/Lalaland/embedding_dot
   cd embedding_dot
   pip install -e .


First, it's necessary to create an info directory that provides information on the patients used to train a CLMBR model.
Note that the date parameters might need to be changed depending on your dataset.

.. code-block:: console

   clmbr_create_info  --save_dir INFO_DIR --extract_dir EHR_ML_EXTRACT_DIR --train_end_date '2010-01-01' --val_end_date '2011-01-01' 

Second, given an info directory it's necessary to train a CLMBR model.

.. code-block:: console

   clmbr_train_model --model_dir MODEL_DIR --info_dir INFO_DIR --lr 0.0001 --use_gru --size 800 --code_dropout 0 

Finally, it is now possible to use that model to generate features.

For our example, we will use the labeler defined in the previous tutorial sections.

.. code-block:: python

   import ehr_ml.clmbr

   # We need to create a saved labeler in order to use the CLMBR API
   ehr_ml.labeler.SavedLabeler.save(labeler, timelines, "tmp_labeler_saved")

   with open("tmp_labeler_saved") as f:
      saved_labeler = ehr_ml.labeler.SavedLabeler(f)

   features, labels, patient_ids, day_offsets = ehr_ml.clmbr.featurize_patients("MODEL_DIR", saved_labeler)

   print(features.shape, labels.shape, patient_ids.shape, day_offsets.shape)


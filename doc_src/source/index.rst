.. ehr_ml documentation master file, created by
   sphinx-quickstart on Wed Oct 21 11:27:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ehr_ml: Machine learning + EHRs
=======================================================================

:py:mod:`ehr_ml` is a Python package for performing machine learning on EHR data.

:py:mod:`ehr_ml` makes it easy to:

* Convert EHR and claims data into a common schema
* Apply labeling functions on that schema in order to derive labels
* Apply featurization schemes on those patients to obtain feature matrices
* Perform many other common tasks necessary in order to perform research with EHR data

*********************************************
How is :py:mod:`ehr_ml` organized?
*********************************************

ehr_ml is designed as a combination of multiple relatively independent components.

The main modules are:

:py:mod:`ehr_ml.timeline`
    Contains both a common data format and utilities interacting with it.
:py:mod:`ehr_ml.labeler`
    Contains an abstract definition of a labeling function as well as some useful shared labeling functions.
:py:mod:`ehr_ml.featurizer`
    Contains an abstract definition of a featurizer and some useful default choices.
:py:mod:`ehr_ml.utils`
    Contains various utility classes which are not directly connected to ehr_ml.
:py:mod:`ehr_ml.ontology`
    Contains utilities for working with ontologies and mapping codes to subcodes.
:py:mod:`ehr_ml.index`
    Enables a user to quickly find patients who have certain codes.
:py:mod:`ehr_ml.clmbr`
    An implementation of the CLMBR EHR representation learning algorithm.

****************
Documentation
****************

.. toctree::
   :maxdepth: 1
   :caption: Getting Started:

   setup
   tutorial
   extract_tutorial


.. toctree::
   :maxdepth: 1
   :caption: Components:

   timeline
   labeler
   featurizer
   utils
   ontology
   m-index
   clmbr

ehr_ml.ontology
==================================

:py:mod:`ehr_ml.ontology` is a module for handling relationships between observation codes.

The fundamental ontological structure is in the form of a rooted network.
There is a root node at the top that contains multiple children and each child in turn contains children.
Nodes can have multiple parents.

Note that the ontologies operate in a different code space from the raw timeline codes and thus have a seperate TermDictionary.

Accessing this ontology is done through the OntologyReader class.

.. py:class:: OntologyReader

    A class for accessing onotlogy information

    
    .. py:method:: __init__(filename: str)

        Construct an ontology reader given the filename.

    .. py:method:: get_subwords(word: int) -> Sequence[int]

        Get the child codes for a particular timeline code.

    .. py:method:: get_parents(word: int) -> Sequence[int]

        Get the set of direct parent codes for an ontology code.

    .. py:method:: get_all_parents(word: int) -> Sequence[int]

        Get the set of all parents (direct and indirect) for an ontology code.

    .. py:method:: get_children(word: int) -> Sequence[int]

        Get the set of direct children for an ontology code.

    .. py:method:: get_words_for_subword(word: int) -> Sequence[int]

        Given an ontology code, retrieve all corresponding timeline codes.

    
    .. py:method:: get_words_for_subword_term(term: str) -> Sequence[int]

        Given an ontology string term, retrieve all corresponding timeline codes.

        
    .. py:method:: get_recorded_date_codes() -> Sequence[int]
        
        Some types of observations are more or less reliable than others.
        This returns the set of reliable timeline codes that should be used for prediction models.


    .. py:method:: get_dictionary() -> timeline.TermDictionary

        Return the dictionary used to define the ontology code space.

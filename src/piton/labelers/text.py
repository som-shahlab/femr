"""Labeling functions for OMOP data based on lab values."""
from __future__ import annotations

import datetime
from typing import Callable, Dict, List, Optional
import torch

from .. import Event, Patient
from ..extension import datasets as extension_datasets
from .core import (
    Label,
    Labeler,
    LabelType,
    TimeHorizon,
    TimeHorizonEventLabeler,
)
from .omop import (
    WithinVisitLabeler,
    get_death_concepts,
    get_inpatient_admission_discharge_times,
    group_inpatient_events_by_visit_id,
    map_omop_concept_codes_to_piton_codes,
    move_datetime_to_end_of_day,
)


def is_radiology_note(text: str) -> bool:
    phrases = [ x.lower() for x in [
        'I have personally reviewed the images for this examination',
        'Physician to Physician Radiology Consult Line',
        'Interpreted by Attending Radiologist',
    ]]
    return any([p in text for p in phrases])


class PulmonaryEmbolismLabeler(TimeHorizonEventLabeler):
    """
    This predicts whether a patient will have a pulmonary embolism within the `time_horizon` 
        after discharge from an INPATIENT admission.

    Prediction Time: At discharge after each INPATIENT visit (adjusted 
        by `self.prediction_time_adjustment_func()` if provided)
    Label: TRUE if the patient has a pulmonary embolism written in a note within 
        the `time_horizon` after discharge from an INPATIENT admission.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        time_horizon: TimeHorizon,
        text_tokenizer: Callable,
        text_model: Callable,
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.time_horizon: TimeHorizon = time_horizon
        self.text_tokenizer: Callable = text_tokenizer
        self.text_model: Callable = text_model
    
    def predict_if_is_outcome(self, text: str) -> bool:
        max_length: int = self.text_model.config.max_position_embeddings
        tokens = self.text_tokenizer(
                    [ text ],
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
        with torch.no_grad():
            results = self.text_model(tokens['input_ids'])
            probs = torch.sigmoid(results.logits.squeeze()).tolist()
            pe_acute, pe_subsegmentalonly, pe_positive = probs
        return pe_positive > 0.5

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        times: List[datetime.datetime] = []
        for event in patient.events:
            if isinstance(event.value, str) and is_radiology_note(event.value):
                # If this is a radiology note, check if it's labeled as a pulmonary embolism
                if self.predict_if_is_outcome(event.value):
                    times.append(event.start)
        return times
    
    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        __, discharge_times = get_inpatient_admission_discharge_times(patient, self.ontology)
        return discharge_times

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon
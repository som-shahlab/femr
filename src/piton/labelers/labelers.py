from __future__ import annotations

import collections
import datetime
import json
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    DefaultDict,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    TextIO,
    Tuple,
    Union,
)

from .. import Event, Patient
from ..extension import datasets as extension_datasets

import numpy as np

@dataclass
class SurvivalValue:
    event_time: int # TODO - rename
    is_censored: bool # TRUE if this patient was censored

LabelType = Union[
    Literal["boolean"],
    Literal["numeric"],
    Literal["survival"],
    Literal["categorical"],
]

VALID_LABEL_TYPES = ["boolean", "numeric", "survival", "categorical"]


class Label:
    """
        An individual label on a particular patient at a particular time.
    """

    __slots__ = [
        "time", # Arbitrary timestamp (datetime.datetime)
        "label_type",
        "value",
    ]

    label_type: LabelType

    def __init__(
        self,
        time: datetime.datetime,
        label_type: str,
        value: Optional[Union[bool, int, float, SurvivalValue]] = None
    ):
        """Construct a label for datetime `time` and value `value`. 

        Args:
            time (datetime.datetime): Time in this patient's timeline that corresponds to this label
            label_type (str): Type of label. Must be an element in `VALID_LABEL_TYPES`
            value (Optional[Union[bool, int, float, SurvivalValue]], optional): Value of label. Defaults to None.
        """
        assert label_type in VALID_LABEL_TYPES
        self.time = time
        self.label_type = label_type
        self.value = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Label):
            return NotImplemented
        return (
            self.time,
            self.label_type,
            self.value,
        ) == (
            other.time,
            other.label_type,
            other.value,
        )

    def __repr__(self) -> str:
        return f"Label(time={self.time}, value={self.value}, type={self.label_type})"

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = { 
                                  "time": self.time,
                                  "label_type": self.label_type,
                                  "value" : None, 
                                }
        if self.label_type in ["boolean", "numeric", "categorical"]:
            result["value"] = self.value
        elif self.label_type == "survival":
            assert self.value is not None
            result["value"] = {
                "event_time": self.value.event_time,
                "is_censored": self.value.is_censored,
            }
        return result

    @classmethod
    def from_dict(cls, stored: Dict[str, Any]) -> Label:
        assert "label_type" in stored and "value" in stored and "time" in stored
        assert stored["label_type"] in VALID_LABEL_TYPES
        
        # Handle SurvivalValue
        if stored["label_type"] == "survival":
            assert "event_time" in stored["value"]
            assert "is_censored" in stored["value"]
            value = SurvivalValue(
                stored["value"]["event_time"],
                stored["value"]["is_censored"],
            )

        return Label(
            stored["time"],
            stored["label_type"],
            stored["value"],
        )


class Labeler(ABC):
    """
    An interface for labeling functions.
    
    A labeling function applies a label to a specific datetime in a given patient's timeline.
    It can be thought of as generating the following list given a specific patient:
        [(patient ID, datetime_1, label_1), (patient ID, datetime_2, label_2), ... ]
    
    """

    @abstractmethod
    def label(self, patient: Patient) -> List[Label]:
        """
        Applies every label that is applicable to the provided patient.
        This is only called once per patient.

        Args:
            patient (Patient): A patient object

        Returns:
            List[Label]: A list of :class:`Label` containing every label for the given patient
        """
        pass

    def get_required_codes(self) -> List[int]:
        """Set of codes that a patient must have at least one of in order to qualify for this labeler
        This allows us to only extract patients from the :class:`PatientDatabase` who have a code that matches one of these

        Returns:
            List[int]: List of applicable OMOP codes
        """
        pass

    @abstractmethod
    def get_labeler_type(self) -> LabelType:
        """Return what type of labels this labeler returns. See the Label class."""
        pass


##########################################################
# Labeler Utilities
##########################################################

class FixedTimeHorizonEventLabeler(Labeler):
    """A fixed time horizon labeler enables you to label events that occur within a particular time horizon (i.e. timedelta).
        It is a boolean event that is TRUE if the event of interest occurs within the next 6 months

    No labels are generated if the patient record is "censored" before the end of the horizon

    You are required to implement two methods:
        get_event_times() for defining the datetimes of the events
        get_time_horizon() for defining the length of time (i.e. timedelta) to use for the time horizon
    """

    @abstractmethod
    def get_event_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a sorted list containing the datetime that each event "occurs".
            You define what "occurs" means in your implementation, i.e. the start/end/middle/etc. of an event
        
        IMPORTANT: Must be sorted ascending (i.e. start -> end of timeline)

        Args:
            patient (Patient): A patient object

        Returns:
            List[datetime.datetime]: A list of datetimes, one corresponding to each event
        """
        pass

    @abstractmethod
    def get_time_horizon(self) -> datetime.timedelta:
        """
        Return the length of the time horizon used to make labels. 
        Can be an arbitrary length of time.
        """
        pass

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """
        Return a sorted list containing the datetimes for which it's valid to make a prediction,
        i.e. the start of each time horizon
        
        IMPORTANT: Must be sorted ascending (i.e. start -> end of timeline)
        """
        return [ e.start for e in patient.events ]

    def get_patient_start_end_times(self, patient: Patient) -> Tuple[datetime.datetime, datetime.datetime]:
        """Returns the (start, end) of the patient timeline that we want to consider for this labeler

        Returns:
            Tuple[datetime.datetime, datetime.datetime]: (start, end)
        """
        return (patient.events[0].start, patient.events[-1].end)

    def label(self, patient: Patient) -> List[Label]:
        """
        Assumes that events in `patient.events` are already sorted in chronologically 
        ascending order (i.e. start -> end).

        Args:
            patient (Patient): A patient object

        Returns:
            List[Label]: A list containing a label for each datetime returned by `get_prediction_times()`
        """
        if len(patient.events) == 0:
            return []

        __, end_time = self.get_patient_start_end_times(patient)
        event_times: List[datetime.datetime] = self.get_event_times(patient)
        time_horizon: datetime.timedelta = self.get_time_horizon()
        
        results: List[Label] = []
        next_event_index: int = 0
        for time in self.get_prediction_times(patient):
            while (
                next_event_index < len(event_times)
                and event_times[next_event_index] <= time
            ):
                next_event_index += 1

            # TODO - what exactly are we doing here? TRUE if there's an event within window after `time`, FALSE if not, NONE if patient timeline ends?
            event_soon: bool = (
                next_event_index < len(event_times)
                and event_times[next_event_index] <= time + time_horizon
            ) # TRUE if an event occurs within the time horizon starting from `time`
            evidence_of_alive = time + time_horizon <= end_time # TRUE if patient timeline ends AFTER this time horizon starting with `time`

            if event_soon:
                results.append(Label(time=time, value=True, label_type="boolean"))
            elif evidence_of_alive:
                results.append(Label(time=time, value=False, label_type="boolean"))

        return results

    def get_labeler_type(self) -> LabelType:
        return "boolean"

class SavedLabeler(Labeler):
    """A class for loading and saving labeled examples"""

    def __init__(self, fp: TextIO):
        """Initialize the saved labeler from a file object
            Expects the file to be a dictionary with two keys:
                - labeler_type: LabelType
                - labels: Dict[int, List[Label]] where [key] is patient ID, [value] is a list of Labels for that patient

        Args:
            fp (TextIO): String contents of a JSON file
        """
        data: Dict = json.load(fp)
        assert "labeler_type" in data and "labels" in data
        
        self.labeler_type: LabelType = data["labeler_type"]

        # Load labels for each patient
        self.labels: DefaultDict[int, List[Label]] = collections.defaultdict(list)
        for patient_id, labels in data["labels"]:
            for label in labels:
                actual_label: Dict = Label.from_dict(label)
                self.labels[patient_id].append(actual_label)
        self.labels = dict(self.labels)

    def label(
        self, 
        patient: Optional[Patient] = None, 
        patient_id: Optional[int] = None
    ) -> List[Label]:
        """Return list of Labels for the specified patient

        Args:
            patient (Optional[Patient], optional): Use `patient.patient_id` if `patient_id` is None. Defaults to None.
            patient_id (Optional[int], optional): If None, then use `patient.patient_id`. Defaults to None.

        Returns:
            List[Label]: List of Labels for this patient
        """        
        assert patient is not None or patient_id is not None
        patient_id: int = patient.patient_id if patient_id is None else patient_id
        return self.labels.get(patient_id, [])

    @classmethod
    def from_boolean_label_data(
        cls,
        result_labels: np.array,
        patient_ids: np.array,
        label_times: np.array,
    ) -> SavedLabeler:
        """Create a :class:`SavedLabeler` from boolean labels

        Args:
            result_labels (np.array): Array of boolean values, used to set the `boolean_value` part of :class:`Label`
            patient_ids (np.array): Patient for the corresponding label.
            label_times (np.array): Time that the corresponding label occurs.

        Returns:
            SavedLabeler: A :class:`SavedLabeler` instantiated from these boolean labels
        """
        # Dict mapping patient_id => list of Label objects
        label_dict: DefaultDict[List[Label]] = collections.defaultdict(list)
        for label, patient_id, patient_time in zip(
            result_labels, patient_ids, label_times
        ):
            label_dict[patient_id].append(
                Label(time=int(patient_time), value=bool(label), label_type="boolean")
            )

        # List of (patient_id, dict version of Labels for that patient)
        labels: List[Tuple[int, List[Dict[str, Any]]]] = []
        for patient_id in patient_ids:
            labels.append((
                int(patient_id),
                [ label.to_dict() for label in label_dict[patient_id] ],
            ))

        data_str: str = json.dumps({"labels": labels, "labeler_type": "boolean"})
        return SavedLabeler(io.StringIO(data_str))

    @classmethod
    def save(
        cls,
        labeler: Labeler,
        timelines: Union[extension_datasets.PatientDatabase, List[Patient]],
        filepath: str,
        end_date: Optional[datetime.date] = None,
    ) -> None:
        """Save the provided `labeler` to the given filepath.

        Args:
            labeler (Labeler): Labeler class
            timelines (Union[extension_datasets.PatientDatabase, List[Patient]]): Iterable of Patient objects.
            filepath (str): Path to file where JSON will be written.
            end_date (Optional[datetime.date], optional): _description_. Defaults to None.

        Raises:
            ValueError: If `time` of a label is before the acceptable start/end time for its corresponding patient
        """        
        labels: List[Tuple[int, List[Dict]]] = []
        all_patient_ids: List[int] = []
        for patient_id in timelines.get_patient_ids():
            patient: Patient = timelines.get_patient(patient_id, end_date=end_date)
            generated_labels: List[Label] = labeler.label(patient)

            start_time, end_time = labeler.get_patient_start_end_times(patient)
            for label in generated_labels:
                if (
                    label.time < start_time
                    or label.time > end_time
                ):
                    raise ValueError(
                        f"The labeler {labeler} produced a label with an out-of-range time ({label.time}) -- should be bounded by ({start_time},{end_time})"
                    )

            if len(generated_labels) > 0:
                labels.append((
                    patient_id,
                    [ label.to_dict() for label in generated_labels ],
                ))
                all_patient_ids.append(patient_id)

        with open(filepath, "w") as fp:
            json.dump(
                {
                    "labels": labels, 
                    "labeler_type": labeler.get_labeler_type()
                },
                fp,
            )

    def get_label_data(self) -> Tuple[np.array, np.array, np.array]:
        """Splits `self.labels` up into its three component parts -- (patient ID, time, label value)
        
        Raises:
            ValueError: _description_

        Returns:
            Tuple[np.array, np.array, np.array]: _description_
        """
        result_labels: List[Any] = []
        patient_ids: List[int] = []
        label_times: List[datetime.datetime] = []
        
        if self.labeler_type in ["boolean", "numerical", "categorical"]:
            for patient_id, labels in self.labels.items():
                for label in labels:
                    result_labels.append(label.value)
                    patient_ids.append(patient_id)
                    label_times.append(label.time)
        elif self.labeler_type == "survival":
            raise
        else:
            raise ValueError(
                "Other label types are not implemented yet for this method"
            )
        return (
            np.array(result_labels),
            np.array(patient_ids),
            np.array(label_times),
        )

    def get_labeler_type(self) -> LabelType:
        return self.labeler_type

class CodeLabeler(FixedTimeHorizonEventLabeler):
    """
        Applies a label based on a single code's occurrence over a fixed time horizon
    """

    def __init__(self, code: int):
        """Label the code whose index in your Ontology equal to `code`
        """        
        self.code = code

    def get_event_times(self, patient: Patient) -> List[datetime.datetime]:
        """Returns a list of datetimes corresponding to the start time of the Events
            in this patient's timeline which have the exact same `code` as `self.code`
        """        
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.code == self.code:
                times.append(event.start)
        return times

class MortalityLabeler(CodeLabeler):
    """
        The mortality task is defined as predicting whether or not a
        patient will die within the next 3 months.
    """

    def __init__(self, ontology: extension_datasets.Ontology):
        CODE_DEATH_PREFIX = "Death Type/"

        death_codes: Set[Tuple[str, int]] = set()
        for code_idx, code_str in enumerate(ontology.get_dictionary().values()):
            if code_str.startswith(CODE_DEATH_PREFIX):
                death_codes.add((code_str, code_idx))

        if len(death_codes) != 1:
            raise ValueError(
                f"Could not find exactly one death code -- instead found {len(death_codes)} codes: {str(death_codes)}"
            )
        else:
            death_code: int = list(death_codes)[0][1]
            super().__init__(code=death_code)

    def get_time_horizon(self) -> datetime.timedelta:
        """Three month time horizon
        """        
        return datetime.timedelta(months=3)

class IsMaleLabeler(Labeler):
    # TODO
    """
        This labeler tries to predict whether or not a patient is male or not.
        The prediction time is on admission.

        This is primarily intended as a "debugging" labeler that should be "trivial" and get 1.0 AUROC.
    """

    def __init__(self, ontology: extension_datasets.Ontology):
        self.male_code: int = ontology.get_dictionary().map("demographics/gender/Male")
        self.admission_helper = InpatientAdmissionHelper(timelines)

    def label(self, patient: Patient) -> List[Label]:
        if len(patient.events) == 0:
            return []

        current_admission_index = 0
        admissions = self.admission_helper.get_inpatient_admissions(patient)

        labels = []

        is_male = self.male_code in patient.days[0].observations

        for i, day in enumerate(patient.days):
            if current_admission_index >= len(admissions):
                continue
            current_admission = admissions[current_admission_index]
            assert day.age <= current_admission.start_age

            if day.age == current_admission.start_age:
                current_admission_index += 1
                labels.append(Label(event_index=i, boolean_value=is_male))
                
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"
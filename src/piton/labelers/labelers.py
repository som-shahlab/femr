from __future__ import annotations

import collections
import datetime
import hashlib
import json
import random
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    TextIO,
    Tuple,
    Union,
    cast,
)
from .. import Event, Patient, ValueType

import numpy as np

from . import index, ontology, timeline

@dataclass
class SurvivalValue:
    event_time: int
    is_outcome: bool

LabelType = Union[
    Literal["boolean"],
    Literal["numeric"],
    Literal["survival"],
    Literal["categorical"],
]


class Label:
    """
        An individual label on a particular patient at a particular time.
    """

    __slots__ = [
        "event_index",
        "boolean_value",
        "numeric_value",
        "survival_value",
        "categorical_value",
        "label_type",
    ]

    label_type: LabelType

    def __init__(
        self,
        event_index: int,
        boolean_value: Optional[bool] = None,
        numeric_value: Optional[float] = None,
        survival_value: Optional[SurvivalValue] = None,
        categorical_value: Optional[int] = None,
    ):
        """Construct a label with a event_index and a value. 
            You must only provide a single value option.

        Args:
            event_index (int): Index in the `events` property of a `Patient` object that corresponds to this label
            boolean_value (Optional[bool], optional): Value for label - if set, then other `_value` variables must be `None`. Defaults to None.
            numeric_value (Optional[float], optional): Value for label - if set, then other `_value` variables must be `None`. Defaults to None.
            survival_value (Optional[SurvivalValue], optional): Value for label - if set, then other `_value` variables must be `None`. Defaults to None.
            categorical_value (Optional[int], optional): Value for label - if set, then other `_value` variables must be `None`. Defaults to None.
        """        

        self.event_index = event_index
        self.boolean_value = boolean_value
        self.numeric_value = numeric_value
        self.survival_value = survival_value
        self.categorical_value = categorical_value

        if boolean_value is not None:
            assert numeric_value is None
            assert survival_value is None
            assert categorical_value is None
            self.label_type = "boolean"
        elif numeric_value is not None:
            assert boolean_value is None
            assert survival_value is None
            assert categorical_value is None
            self.label_type = "numeric"
        elif survival_value is not None:
            assert boolean_value is None
            assert numeric_value is None
            assert categorical_value is None
            self.label_type = "survival"
        elif categorical_value is not None:
            assert boolean_value is None
            assert numeric_value is None
            assert survival_value is None
            self.label_type = "categorical"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Label):
            return NotImplemented
        return (
            self.event_index,
            self.boolean_value,
            self.numeric_value,
            self.survival_value,
            self.categorical_value,
        ) == (
            other.event_index,
            other.boolean_value,
            other.numeric_value,
            other.survival_value,
            other.categorical_value,
        )

    def __repr__(self) -> str:
        if self.label_type == "boolean":
            return f"Label(event_index={self.event_index}, boolean_value={self.boolean_value})"
        elif self.label_type == "numeric":
            return f"Label(event_index={self.event_index}, numeric_value={self.numeric_value})"
        elif self.label_type == "survival":
            return f"Label(event_index={self.event_index}, survival_value={self.survival_value})"
        elif self.label_type == "categorical":
            return f"Label(event_index={self.event_index}, categorical_value={self.categorical_value})"
        else:
            raise ValueError(f"Invalid label type of {self.label_type}")

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"event_index": self.event_index}

        if self.label_type == "boolean":
            result["boolean_value"] = self.boolean_value
        elif self.label_type == "numeric":
            result["numeric_value"] = self.numeric_value
        elif self.label_type == "survival":
            assert self.survival_value is not None
            result["survival_value"] = {
                "event_time": self.survival_value.event_time,
                "is_outcome": self.survival_value.is_outcome,
            }
        elif self.label_type == "categorical":
            result["categorical_value"] = self.categorical_value

        return result

    @classmethod
    def from_dict(cls, stored: Dict[str, Any]) -> Label:
        survival_value: Optional[SurvivalValue] = None
        if stored.get("survival_value") is not None:
            survival_value = SurvivalValue(
                stored["survival_value"]["event_time"],
                stored["survival_value"]["is_outcome"],
            )

        return Label(
            stored["event_index"],
            boolean_value=stored.get("boolean_value"),
            numeric_value=stored.get("numeric_value"),
            survival_value=survival_value,
            categorical_value=stored.get("categorical_value"),
        )


class Labeler(ABC):
    """
    An interface for labeling functions.
    
    A labeling function applies a label to each event in a given patient's timeline.
    """

    @abstractmethod
    def label(self, patient: Patient) -> List[Label]:
        """
        Label each event in the provided patient.

        Args:
            patient (Patient): A patient object

        Returns:
            List[Label]: A list of :class:`Label` containing every label for the given patient
        """

    def get_possible_patient_ids(self) -> Mapping[bool, Optional[Set[int]]]:
        # TODO - what is this method? what are the keys of the returned dict?
        """
        Get the possible patient ids for each class for this labeler. This method should only be
        provided if the labeler knows in advance the patients which are labeled with what label.

        Returns:
            `None` if the patient ids are not known, otherwise a mapping where each label maps to the set of patient
            ids for that label.
        """
        return {True: None, False: None}

    def get_all_patient_ids(self) -> Optional[Set[int]]:
        """
        A helper method that takes the output of get_possible_patient_ids() and combines the True and False classes
        into a single set.

        Returns:
            None if the patients ids are not known, otherwise a set of patient ids which are labeled.
        """
        possible_patient_ids = self.get_possible_patient_ids()
        if (
            possible_patient_ids[True] is not None
            and possible_patient_ids[False] is not None
        ):
            true_examples = cast(Set[int], possible_patient_ids[True])
            false_examples = cast(Set[int], possible_patient_ids[False])
            return true_examples | false_examples
        else:
            return None

    @abstractmethod
    def get_labeler_type(self) -> LabelType:
        """Return what type of labels this labeler returns. See the Label class."""
        ...


##########################################################
# Labeler Utilities
##########################################################

class FixedTimeHorizonEventLabeler(Labeler):
    """A fixed time horizon labeler enables you to label events that occur within a particular time horizon.

    No labels are generated if the patient record is "censored" before the end of the horizon

    You are required to implement two methods:
        get_event_ages() for specifying the times of the events
        get_time_horizon() for specifying the length of time to use for the time horizon
    """

    @abstractmethod
    def get_event_ages(self, patient: Patient) -> List[int]:
        """
        Return a sorted list (oldest -> most recent) containing the age of the patient at which each event occurs
        """
        pass

    @abstractmethod
    def get_time_horizon(self) -> int:
        # TODO - make it more flexible than days (i.e. hours)?
        """
        Return an integer which represents the length of the time horizon in days
        """
        pass

    def get_prediction_days(self, patient: Patient) -> List[int]:
        """
        Return a sorted list containing the indices in which it's valid to make a prediction.
        """
        return list(range(len(patient.events)))

    def label(self, patient: Patient) -> List[Label]:
        """_summary_

        Args:
            patient (Patient): _description_

        Returns:
            List[Label]: _description_
        """
        if len(patient.events) == 0:
            return []

        # is_use_last_event_start_as_end_date: bool = True
        # start_date = patient.events[0].start
        # start_age = XXX
        # end_date = patient.events[-1].end if is_use_last_event_start_as_end_date else patient.events[-1].start
        # event_ages = self.get_event_ages(patient)

        last_age = patient.days[-1].age
        event_ages = self.get_event_ages(patient)
        next_event_index = 0

        results = []
        for i in self.get_prediction_days(patient):
            day = patient.days[i]

            while (
                next_event_index < len(event_ages)
                and event_ages[next_event_index] <= day.age
            ):
                next_event_index += 1

            event_soon = (
                next_event_index < len(event_ages)
                and (event_ages[next_event_index] - day.age)
                <= self.get_time_horizon()
            )
            evidence_of_alive = (last_age - day.age) >= self.get_time_horizon()

            if event_soon:
                results.append(Label(event_index=i, boolean_value=True))
            elif evidence_of_alive:
                results.append(Label(event_index=i, boolean_value=False))

        return results

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class InfiniteTimeHorizonEventLabeler(Labeler):
    """An infinite time horizon labeler enables you to label patients that have an event at any time in the future.

    You are required to implement get_first_event_age for specifying the time of the first event
    """

    @abstractmethod
    def get_first_event_age(self, patient: Patient) -> Optional[int]:
        """
        Return the time of the first event
        """
        pass

    def get_prediction_days(self, patient: Patient) -> List[int]:
        """
        Return a sorted list containing the indices in which it's valid to make a prediction.
        """
        return list(range(len(patient.days)))

    def label(self, patient: Patient) -> List[Label]:
        event_age = self.get_first_event_age(patient)

        results = []
        for i in self.get_prediction_days(patient):
            day = patient.days[i]
            if event_age is not None and day.age >= event_age:
                break

            if event_age is not None:
                results.append(Label(event_index=i, boolean_value=True))
            else:
                results.append(Label(event_index=i, boolean_value=False))

        return results

    def get_labeler_type(self) -> LabelType:
        return "boolean"


def hash_rand_rang(seed: int, number: int, max_num: int) -> int:
    m = hashlib.sha256()
    m.update(seed.to_bytes(8, byteorder="little"))
    m.update(number.to_bytes(8, byteorder="little"))
    return int.from_bytes(m.digest(), byteorder="little") % max_num


class RandomSelectionLabeler(Labeler):
    """A labeler which enables you to randomly select one label per patient by using a random seed."""

    def __init__(self, sublabeler: Labeler, random_seed: int):
        """
        Initialize the RandomSelectionLabeler

        Args:
            sublabeler: A Labeler to use for the initial labeling
            random_seed: A random 64-bit integer seed

        """
        self.sublabeler = sublabeler
        self.random_seed = random_seed

    def label(self, patient: Patient) -> List[Label]:
        labels = self.sublabeler.label(patient)
        if len(labels) == 0:
            return []

        i = hash_rand_rang(self.random_seed, patient.patient_id, len(labels))
        return [labels[i]]

    def get_possible_patient_ids(self) -> Mapping[bool, Optional[Set[int]]]:
        return self.sublabeler.get_possible_patient_ids()

    def get_labeler_type(self) -> LabelType:
        return self.sublabeler.get_labeler_type()


class YearHistoryRequiredLabeler(Labeler):
    """A composite labeler which enables you to require at least one year of history for every label."""

    def __init__(self, sublabeler: Labeler):
        """
        Initialize the YearHistoryRequiredLabeler

        Args:
            sublabeler: A Labeler to use for the initial labeling

        """
        self.sublabeler = sublabeler

    def label(self, patient: Patient) -> List[Label]:
        if len(patient.days) <= 1:
            return []

        first_age = patient.days[1].age

        labels = self.sublabeler.label(patient)

        return [
            label
            for label in labels
            if patient.days[label.event_index].age >= (first_age + 365)
        ]

    def get_possible_patient_ids(self) -> Mapping[bool, Optional[Set[int]]]:
        return self.sublabeler.get_possible_patient_ids()

    def get_all_patient_ids(self) -> Optional[Set[int]]:
        return self.sublabeler.get_all_patient_ids()

    def get_labeler_type(self) -> LabelType:
        return self.sublabeler.get_labeler_type()


class SavedLabeler(Labeler):
    """A class for loading and saving labeled examples"""

    def __init__(self, fp: TextIO):
        """Initialize the saved labeler from a file object"""

        self.labels: Dict[int, List[Label]] = {}

        data = json.load(fp)

        self.labeler_type: LabelType = data["labeler_type"]

        if self.labeler_type == "boolean":
            self.possible_patient_ids: Dict[bool, Set[int]] = {
                True: set(),
                False: set(),
            }

        for patient_id, labels in data["labels"]:
            self.labels[patient_id] = []

            for label in labels:
                actual_label = Label.from_dict(label)
                self.labels[patient_id].append(actual_label)

                if self.labeler_type == "boolean":
                    self.possible_patient_ids[
                        cast(bool, actual_label.boolean_value)
                    ].add(patient_id)

    def label(
        self, patient: Patient, patient_id: Optional[int] = None
    ) -> List[Label]:
        if patient_id is None:
            patient_id = patient.patient_id
        return self.labels.get(patient_id, [])

    def get_possible_patient_ids(self) -> Mapping[bool, Optional[Set[int]]]:
        if self.labeler_type == "boolean":
            return self.possible_patient_ids
        else:
            return {True: None, False: None}

    def get_all_patient_ids(self) -> Optional[Set[int]]:
        return set(self.labels.keys())

    @classmethod
    def from_boolean_label_data(
        cls,
        result_labels: np.array,
        patient_ids: np.array,
        patient_day_indices: np.array,
    ) -> SavedLabeler:
        """Create a saved labeler from boolean label data."""
        labels = []
        all_patient_ids = []

        label_dict = collections.defaultdict(list)

        for label, patient_id, patient_index in zip(
            result_labels, patient_ids, patient_day_indices
        ):
            label_dict[patient_id].append(
                Label(event_index=int(patient_index), boolean_value=bool(label))
            )

        for patient_id in patient_ids:
            labels.append(
                (
                    int(patient_id),
                    [label.to_dict() for label in label_dict[patient_id]],
                )
            )

        data_str = json.dumps({"labels": labels, "labeler_type": "boolean"},)

        return SavedLabeler(io.StringIO(data_str))

    @classmethod
    def save(
        cls,
        labeler: Labeler,
        timelines: timeline.TimelineReader,
        filename: str,
        end_date: Optional[datetime.date] = None,
        force_prevalence: Optional[float] = None,
        prevalence_seed: Optional[int] = None,
    ) -> None:
        """Save the provided labeler to the given filename."""
        if force_prevalence is not None:
            if prevalence_seed is None:
                raise ValueError(
                    "Need a seed if you are going to force a particular prevelence"
                )
            if labeler.get_labeler_type() != "boolean":
                raise ValueError(
                    f"Can only force a prevelence on a boolean labeler, not {labeler.get_labeler_type()}"
                )

            label_indices: Mapping[bool, List[Tuple[int, int]]] = {
                True: [],
                False: [],
            }

        possible_patient_ids = labeler.get_all_patient_ids()

        labels = []
        all_patient_ids = []

        for patient_id in timelines.get_patient_ids():
            if (
                possible_patient_ids is None
                or patient_id in possible_patient_ids
            ):
                patient = timelines.get_patient(patient_id, end_date=end_date)
                generated_labels = labeler.label(patient)

                for label in generated_labels:
                    if label.event_index < 0 or label.event_index >= len(
                        patient.days
                    ):
                        raise ValueError(
                            f"The labeler {labeler} produced a label with invalid day index {label.event_index}"
                        )

                if len(generated_labels) > 0:
                    labels.append(
                        (
                            patient_id,
                            [label.to_dict() for label in generated_labels],
                        )
                    )
                    all_patient_ids.append(patient_id)

                    if force_prevalence is not None:
                        for i, label in enumerate(generated_labels):
                            label_indices[cast(bool, label.boolean_value)].append(
                                (patient_id, i)
                            )

        if force_prevalence is not None:
            counts = {k: len(v) for k, v in label_indices.items()}
            print(counts)
            current_prevalence = counts[True] / (counts[True] + counts[False])
            if current_prevalence < force_prevalence:
                wanted_removed_negatives = int(
                    counts[False]
                    - counts[True] * (1 - force_prevalence) / force_prevalence
                )
                # I need to subsample the negatives
                rng = random.Random(prevalence_seed)
                negatives_to_remove = set(
                    rng.sample(label_indices[False], wanted_removed_negatives)
                )

                filtered_labels = []
                filtered_patient_ids = []

                for patient_id, patient_labels in labels:
                    sampled_labels = [
                        label
                        for i, label in enumerate(patient_labels)
                        if (patient_id, i) not in negatives_to_remove
                    ]
                    if len(sampled_labels) > 0:
                        filtered_labels.append((patient_id, sampled_labels))
                        filtered_patient_ids.append(patient_id)

                labels = filtered_labels
                all_patient_ids = filtered_patient_ids

        with open(filename, "w") as fp:
            json.dump(
                {"labels": labels, "labeler_type": labeler.get_labeler_type()},
                fp,
            )

    def get_label_data(self) -> Tuple[np.array, np.array, np.array]:
        if self.labeler_type == "boolean":

            result_labels = []
            patient_ids = []
            patient_day_indices = []

            for patient_id, labels in self.labels.items():
                for label in labels:
                    result_labels.append(label.boolean_value)
                    patient_ids.append(patient_id)
                    patient_day_indices.append(label.event_index)

            return (
                np.array(result_labels),
                np.array(patient_ids),
                np.array(patient_day_indices),
            )
        else:
            raise ValueError(
                "Other label types are not implemented yet for this method"
            )

    def get_labeler_type(self) -> LabelType:
        return self.labeler_type


class CodeLabeler(FixedTimeHorizonEventLabeler):
    """
        Labels for a given code in a fixed time horizon
    """

    def __init__(self, code: int):
        self.code = code

    def get_event_ages(self, patient: Patient) -> List[int]:
        ages = []
        for day in patient.days:
            observation_pos = self.code in day.observations
            observations_with_values_pos = self.code in set(
                [temp.code for temp in day.observations_with_values]
            )
            if observation_pos or observations_with_values_pos:
                ages.append(day.age)
        return ages


class PredictionAfterDateLabeler(Labeler):
    """Filter a sublabeler to require that each label occurs within at or after a given date"""

    def __init__(self, sublabeler: Labeler, start_date: datetime.date):
        self.sublabeler = sublabeler
        self.start_date = start_date

    def label(self, patient: Patient) -> List[Label]:
        result = []

        for label in self.sublabeler.label(patient):
            if patient.days[label.event_index].date >= self.start_date:
                result.append(label)

        return result

    def get_possible_patient_ids(self) -> Mapping[bool, Optional[Set[int]]]:
        return self.sublabeler.get_possible_patient_ids()

    def get_all_patient_ids(self) -> Optional[Set[int]]:
        return self.sublabeler.get_all_patient_ids()

    def get_labeler_type(self) -> LabelType:
        return self.sublabeler.get_labeler_type()


class PatientSubsetLabeler(Labeler):
    """Filter a sublabeler to only label certain patients"""

    def __init__(self, sublabeler: Labeler, patients_to_label: Iterable[int]):
        self.sublabeler = sublabeler
        self.patients_to_label = set(patients_to_label)

    def label(self, patient: Patient) -> List[Label]:
        if patient.patient_id not in self.patients_to_label:
            return []
        else:
            return self.sublabeler.label(patient)

    def get_possible_patient_ids(self) -> Mapping[bool, Optional[Set[int]]]:
        possible_patient_ids = self.sublabeler.get_possible_patient_ids()

        positive_ids = possible_patient_ids[True]

        result = {}

        if positive_ids is None:
            result[True] = self.patients_to_label
        else:
            result[True] = positive_ids & self.patients_to_label

        negative_ids = possible_patient_ids[False]

        if negative_ids is None:
            result[False] = self.patients_to_label
        else:
            result[False] = negative_ids & self.patients_to_label

        return result

    def get_all_patient_ids(self) -> Optional[Set[int]]:
        all_patient_ids = self.sublabeler.get_all_patient_ids()

        if all_patient_ids is not None:
            return self.patients_to_label & all_patient_ids
        else:
            return self.patients_to_label

    def get_labeler_type(self) -> LabelType:
        return self.sublabeler.get_labeler_type()


class OlderThanAgeLabeler(Labeler):
    """
        A composite labeler which enables you to require that all of your labels are for dates above some age threshold
    """

    def __init__(
        self, sublabeler: Labeler, age_cutoff_in_days: float = 18 * 365.25
    ):
        """
        Initialize the OlderThanAgeLabeler
        Args:
            sublabeler: A Labeler to use for the initial labeling
            age_cutoff_in_days: All labels prior to this age are filtered out. Default 18 years.
        """
        self.sublabeler = sublabeler
        self.age_cutoff_in_days = age_cutoff_in_days

    def label(self, patient: Patient) -> List[Label]:
        if len(patient.days) == 0:
            return []

        labels = self.sublabeler.label(patient)

        return [
            label
            for label in labels
            if patient.days[label.event_index].age >= self.age_cutoff_in_days
        ]

    def get_possible_patient_ids(self) -> Mapping[bool, Optional[Set[int]]]:
        return self.sublabeler.get_possible_patient_ids()

    def get_all_patient_ids(self) -> Optional[Set[int]]:
        return self.sublabeler.get_all_patient_ids()

    def get_labeler_type(self) -> LabelType:
        return self.sublabeler.get_labeler_type()


@dataclass
class InpatientAdmission:
    start_age: int
    end_age: int


class InpatientAdmissionHelper:
    """The inpatient admission helper enables users to gather all of the inpatient
    admissions for a particular patient.
    See the extractor for a percise query for what an inpatient admission is. Do note
    that if you want a more sophisticated definition, please use the ADT data directly.

    Do note that this does one further step of processing on the that query by merging overlapping
    "admissions".
    """

    def __init__(self, timelines: timeline.TimelineReader):
        dictionary = timelines.get_dictionary()

        inpatient_visit_code = "Visit/IP"
        admission_code = dictionary.map(inpatient_visit_code)
        if admission_code is None:
            raise ValueError(
                f"Could not find inpatient visit code? {inpatient_visit_code}"
            )
        else:
            self.admission_code = admission_code

    def get_inpatient_admissions(
        self, patient: Patient
    ) -> List[InpatientAdmission]:
        results = []
        current_admission: Optional[InpatientAdmission] = None
        for i, day in enumerate(patient.days):
            admission_values = []

            for obs_value in day.observations_with_values:
                if obs_value.code == self.admission_code:
                    if obs_value.is_text:
                        raise ValueError(
                            f"Got a text admission code? {patient.patient_id} {day.date} {obs_value.code}"
                        )
                    else:
                        admission_values.append(int(obs_value.numeric_value))

            if (
                current_admission is not None
                and day.age > current_admission.end_age
            ):
                # We can close out the current admission
                results.append(current_admission)
                current_admission = None

            if len(admission_values) > 0:
                max_discharge = max(admission_values)
                if current_admission is None:
                    current_admission = InpatientAdmission(
                        day.age, day.age + max_discharge
                    )
                else:
                    current_admission = InpatientAdmission(
                        current_admission.start_age,
                        max(day.age + max_discharge, current_admission.end_age),
                    )

        if current_admission is not None:
            results.append(current_admission)

        return results

    def get_all_patient_ids(self, ind: index.Index) -> Set[int]:
        return set(ind.get_patient_ids(self.admission_code))


##########################################################
# Commonly used shared labelers
##########################################################


class ObservationGreaterThanValue(Labeler):
    """
        Labeler than predicts whether some obersevationWithValue is greater than some amount
    """

    def __init__(self, code: int, greater_than: float):
        self.code = code
        self.greater_than = greater_than

    def label(self, patient: Patient) -> List[Label]:
        results = []
        for i, day in enumerate(patient.days):
            if i == 0:
                # We can't provide a label for the first day
                continue

            for observation_with_value in day.observations_with_values:
                if (
                    observation_with_value.code == self.code
                    and not observation_with_value.is_text
                ):
                    results.append(
                        Label(
                            event_index=i - 1,
                            boolean_value=observation_with_value.numeric_value
                            > self.greater_than,
                        )
                    )

        return results

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class MortalityLabeler(CodeLabeler):
    """
        The mortality task is defined as predicting whether or not an
        patient will die within the next 3 months.
    """

    def __init__(self, timelines: timeline.TimelineReader, ind: index.Index):
        death_codes = set()
        for code_str, _ in timelines.get_dictionary().get_items():
            if code_str.startswith("Death Type/"):
                code_id = timelines.get_dictionary().map(code_str)
                death_codes.add((code_str, code_id))

        if len(death_codes) != 1:
            raise ValueError(
                "Could not find a single death code " + str(death_codes)
            )
        else:
            death_code = list(death_codes)[0][1]
            super().__init__(code=death_code)
            self.possible_patients = ind.get_patient_ids(death_code)

    def get_possible_patient_ids(self) -> Mapping[bool, Optional[Set[int]]]:
        return {True: self.possible_patients, False: None}

    def get_time_horizon(self) -> int:
        return 180


class IsMaleLabeler(Labeler):
    """
        This labeler tries to predict whether or not a patient is male or not.
        The prediction time is on admission.

        This is primarily intended as a "debugging" labeler that should be "trivial" and get 1.0 AUROC.
    """

    def __init__(self, timelines: timeline.TimelineReader, ind: index.Index):
        dictionary = timelines.get_dictionary()

        self.admission_helper = InpatientAdmissionHelper(timelines)

        self.male_code = dictionary.map("demographics/gender/Male")

        self.all_patient_ids = self.admission_helper.get_all_patient_ids(ind)

    def label(self, patient: Patient) -> List[Label]:
        if len(patient.days) == 0:
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

    def get_all_patient_ids(self) -> Optional[Set[int]]:
        return self.all_patient_ids

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class OpioidOverdoseLabeler(FixedTimeHorizonEventLabeler):
    """
    The opioid overdose labeler predicts whether or not an opioid overdose will occur in the next 3 months
    It is conditioned on the patient being prescribed opioids.
    """

    def __init__(self, ontologies: ontology.OntologyReader, ind: index.Index):
        self.overdose_word_ids: Set[int] = set()

        icd9_codes = [
            "E850.0",
            "E850.1",
            "E850.2",
            "965.00",
            "965.01",
            "965.02",
            "965.09",
        ]

        icd10_codes = ["T40.0", "T40.1", "T40.2", "T40.3", "T40.4"]

        for code in icd9_codes:
            self.overdose_word_ids |= set(
                ontologies.get_words_for_subword_term("ICD9CM/" + code)
            )
        for code in icd10_codes:
            self.overdose_word_ids |= set(
                ontologies.get_words_for_subword_term("ICD10CM/" + code)
            )

        self.opioid_word_ids = set(
            ontologies.get_words_for_subword_term("ATC/N02A")
        )

        opioid_patient_ids = ind.get_all_patient_ids(self.opioid_word_ids)
        overdose_patient_ids = ind.get_all_patient_ids(self.overdose_word_ids)

        possible_negatives = set(opioid_patient_ids)
        possible_positives = set(opioid_patient_ids) & set(overdose_patient_ids)

        self.possible_patients = {
            True: possible_positives,
            False: possible_negatives,
        }

    def get_event_ages(self, patient: Patient) -> List[int]:
        overdose_ages = []
        for day in patient.days:
            for code in day.observations:
                if code in self.overdose_word_ids:
                    overdose_ages.append(day.age)

        return overdose_ages

    def get_time_horizon(self) -> int:
        return 90

    def get_prediction_days(self, patient: Patient) -> List[int]:
        result = []

        for i, day in enumerate(patient.days):
            if len(set(day.observations) & self.opioid_word_ids) > 0:
                result.append(i)

        return result

    def get_possible_patient_ids(self) -> Mapping[bool, Optional[Set[int]]]:
        return self.possible_patients

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class LupusDiseaseLabeler(InfiniteTimeHorizonEventLabeler):
    """
        TODO-document
    """

    def __init__(self, ontologies: ontology.OntologyReader, ind: index.Index):
        icd9_codes = ["710.0"]

        icd10_codes = ["M32"]

        self.lupus_word_ids = set()
        for code in icd9_codes:
            self.lupus_word_ids |= set(
                ontologies.get_words_for_subword_term("ICD9CM/" + code)
            )
        for code in icd10_codes:
            self.lupus_word_ids |= set(
                ontologies.get_words_for_subword_term("ICD10CM/" + code)
            )

        lupus_patient_ids = set(ind.get_all_patient_ids(self.lupus_word_ids))

        self.possible_patients = {True: lupus_patient_ids, False: None}

    def get_first_event_age(self, patient: Patient) -> Optional[int]:
        for day in patient.days:
            for code in day.observations:
                if code in self.lupus_word_ids:
                    return day.age

        return None

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class InpatientMortalityLabeler(Labeler):
    """
    The inpatient labeler predicts whether or not a patient will die within the current admission.
    The prediction time is before the day of admission.
    """

    def __init__(self, timelines: timeline.TimelineReader, ind: index.Index):
        dictionary = timelines.get_dictionary()

        self.admission_helper = InpatientAdmissionHelper(timelines)

        self.death_code = dictionary.map("Death Type/OMOP generated")

        self.all_patient_ids = self.admission_helper.get_all_patient_ids(ind)

    def label(self, patient: Patient) -> List[Label]:
        admissions = self.admission_helper.get_inpatient_admissions(patient)

        labels: List[Label] = []

        death_age = None

        for day in patient.days:
            if self.death_code in day.observations:
                death_age = day.age
                break

        current_admission_index = 0

        for i, day in enumerate(patient.days):
            if current_admission_index >= len(admissions):
                continue
            current_admission = admissions[current_admission_index]

            assert day.age <= current_admission.start_age

            if death_age is not None and death_age < day.age:
                # Hmm. This is odd. The patient dies before the end of the record
                # We will just discard past this point as it's probability junk (for this labeler at least)
                return labels

            if day.age == current_admission.start_age:
                current_admission_index += 1

                died_in_admission = (
                    death_age is not None
                    and death_age <= current_admission.end_age
                )

                if i != 0:
                    labels.append(
                        Label(event_index=i - 1, boolean_value=died_in_admission)
                    )

        return labels

    def get_all_patient_ids(self) -> Optional[Set[int]]:
        return self.all_patient_ids

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class InpatientReadmissionLabeler(FixedTimeHorizonEventLabeler):
    """
    This labeler is designed to predict whether a patient will be readmitted within 30 days.

    It explicitly does not try to deal with categorizing admissions as "unexpected" or not and is thus
    not comparable to other work.

    It predicts at the end of the admission.

    """

    def __init__(self, timelines: timeline.TimelineReader, ind: index.Index):
        self.admission_helper = InpatientAdmissionHelper(timelines)

        self.all_patient_ids = self.admission_helper.get_all_patient_ids(ind)

    def get_event_ages(self, patient: Patient) -> List[int]:
        """
        Return a sorted list containing the ages at which the event occurs
        """
        admissions = self.admission_helper.get_inpatient_admissions(patient)

        return [admission.start_age for admission in admissions]

    def get_time_horizon(self) -> int:
        """
        Return an integer which represents the length of the time horizon in days
        """
        return 30

    def get_prediction_days(self, patient: Patient) -> List[int]:
        """
        Return a sorted list containing the indices in which it's valid to make a prediction.
        """
        admissions = self.admission_helper.get_inpatient_admissions(patient)

        result = []

        current_admission_index = 0

        for i, day in enumerate(patient.days):
            if current_admission_index >= len(admissions):
                continue
            current_admission = admissions[current_admission_index]

            if day.age < current_admission.end_age:
                continue
            else:
                result.append(i)
                current_admission_index += 1

        return result

    def get_all_patient_ids(self) -> Optional[Set[int]]:
        return self.all_patient_ids

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class LongAdmissionLabeler(Labeler):
    """
    The inpatient labeler predicts whether or not a patient will be admitted for a long time (defined
    as greater than 7 days).
    The prediction time is before they get admitted
    """

    def __init__(self, timelines: timeline.TimelineReader, ind: index.Index):
        self.admission_helper = InpatientAdmissionHelper(timelines)

        self.all_patient_ids = self.admission_helper.get_all_patient_ids(ind)

    def label(self, patient: Patient) -> List[Label]:
        admissions = self.admission_helper.get_inpatient_admissions(patient)

        labels = []

        current_admission_index = 0

        for i, day in enumerate(patient.days):
            if current_admission_index >= len(admissions):
                continue
            current_admission = admissions[current_admission_index]

            assert day.age <= current_admission.start_age

            if day.age == current_admission.start_age:
                current_admission_index += 1

                long_admission = (
                    current_admission.end_age - current_admission.start_age >= 7
                )

                if i != 0:
                    labels.append(
                        Label(event_index=i - 1, boolean_value=long_admission)
                    )

        return labels

    def get_all_patient_ids(self) -> Optional[Set[int]]:
        return self.all_patient_ids

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class CeliacTestLabeler(Labeler):
    """
    The Celiac test labeler predicts whether or not a celiac test will be positive or negative.
    The prediction time is right before the lab results come in.

    Note: This labeler excludes patients who either already had a celiac test or were previously diagnosed.
    """

    def __init__(
        self,
        timelines: timeline.TimelineReader,
        ontologies: ontology.OntologyReader,
        ind: index.Index,
    ):
        self.timelines = timelines

        self.labs = set(ontologies.get_words_for_subword_term("LNC/31017-7"))

        self.celiac_codes = set(
            ontologies.get_words_for_subword_term("ICD9CM/579.0")
        ) | set(ontologies.get_words_for_subword_term("ICD10CM/K90.0"))

        self.positive_text_value = timelines.get_value_dictionary().map(
            "Positive"
        )
        self.negative_text_value = timelines.get_value_dictionary().map(
            "Negative"
        )

        self.all_patient_ids = set(ind.get_all_patient_ids(self.labs))

    def get_all_patient_ids(self) -> Optional[Set[int]]:
        return self.all_patient_ids

    def label(self, patient: Patient) -> List[Label]:
        for i, day in enumerate(patient.days):
            if i > 0:
                for obsValue in day.observations_with_values:
                    if (
                        obsValue.code in self.labs
                        and obsValue.is_text
                        and obsValue.text_value
                        in (self.positive_text_value, self.negative_text_value)
                    ):
                        return [
                            Label(
                                event_index=i - 1,
                                boolean_value=obsValue.text_value
                                == self.positive_text_value,
                            )
                        ]
            if len(self.celiac_codes & set(day.observations)) > 0:
                return []  # This patient has celiac

        return []

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class HighHbA1cLabeler(Labeler):
    """
    The high HbA1c labeler tries to predict whether a non-diabetic patient will test as diabetic.

    Note: This labeler will only trigger at most once every 6 months.
    """

    def __init__(
        self,
        timelines: timeline.TimelineReader,
        ontologies: ontology.OntologyReader,
        ind: index.Index,
    ):
        self.timelines = timelines

        self.labs = set(ontologies.get_words_for_subword_term("LNC/4548-4"))

        self.diabetes_codes = set(
            ontologies.get_words_for_subword_term("ICD9CM/250")
        ) | set(ontologies.get_words_for_subword_term("ICD10CM/E08-E13"))

        self.all_patient_ids = set(ind.get_all_patient_ids(self.labs))

        self.timelines = timelines

    def label(self, patient: Patient) -> List[Label]:
        results = []
        last_trigger: Optional[int] = None
        had_previous_negative_test = False
        for i, day in enumerate(patient.days):
            if i == 0:
                # We can't provide a label for the first day
                continue

            got_diabetes_code = False

            for code in day.observations:
                if code in self.diabetes_codes:
                    got_diabetes_code = True

            for observation_with_value in day.observations_with_values:
                if (
                    observation_with_value.code in self.labs
                    and not observation_with_value.is_text
                ):
                    is_diabetes = observation_with_value.numeric_value > 6.5

                    if last_trigger is None or (day.age - last_trigger) > 180:
                        if had_previous_negative_test:
                            results.append(
                                Label(event_index=i - 1, boolean_value=is_diabetes)
                            )
                            last_trigger = day.age

                    if is_diabetes:
                        got_diabetes_code = True
                    else:
                        had_previous_negative_test = True

            if got_diabetes_code:
                break

        return results

    def get_labeler_type(self) -> LabelType:
        return "boolean"

    def get_all_patient_ids(self) -> Optional[Set[int]]:
        return self.all_patient_ids


##########################################################
# Consult related labelers
##########################################################


class Blayney1ConsultInterventionLabeler(Labeler):
    def __init__(
        self,
        timelines: timeline.TimelineReader,
        ontologies: ontology.OntologyReader,
        ind: index.Index,
    ):
        self.za_drug_ids = ontologies.get_words_for_subword_term("RXNORM/77655")
        self.denosumab_drug_ids = ontologies.get_words_for_subword_term(
            "RXNORM/993449"
        )

        cancer_icd9_codes = [
            "140-149.99",
            "150-159.99",
            "160-165.99",
            "170-176.99",
            "179-189.99",
            "190-199.99",
            "200-208.99",
        ]

        cancer_icd10_codes = [
            "C00-C14",
            "C15-C26",
            "C30-C39",
            "C40-C41",
            "C43-C44",
            "C45-C49",
            "C50-C50",
            "C51-C58",
            "C60-C63",
            "C64-C68",
            "C69-C72",
            "C73-C75",
            "C76-C80",
            "C7A-C7A",
            "C7B-C7B",
            "C81-C96",
            "D00-D09",
        ]

        self.cancer_word_ids = set()

        for code in cancer_icd9_codes:
            self.cancer_word_ids |= set(
                ontologies.get_words_for_subword_term("ICD9CM/" + code)
            )
        for code in cancer_icd10_codes:
            self.cancer_word_ids |= set(
                ontologies.get_words_for_subword_term("ICD10CM/" + code)
            )

        self.all_patient_ids = set(
            ind.get_all_patient_ids(self.cancer_word_ids)
        ) & (
            set(ind.get_all_patient_ids(self.za_drug_ids))
            | set(ind.get_all_patient_ids(self.denosumab_drug_ids))
        )

    def label(self, patient: Patient) -> List[Label]:
        has_cancer = False

        for i, day in enumerate(patient.days):
            for code in day.observations:
                if code in self.cancer_word_ids:
                    has_cancer = True

                if code in self.za_drug_ids or code in self.denosumab_drug_ids:
                    if i == 0 or not has_cancer:
                        # Ignore people who don't have cancer
                        return []
                    else:
                        return [
                            Label(
                                event_index=i - 1,
                                boolean_value=code in self.za_drug_ids,
                            )
                        ]

        # No events
        return []

    def get_all_patient_ids(self) -> Optional[Set[int]]:
        return self.all_patient_ids

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class Knowles2ConsultInterventionLabeler(Labeler):
    def __init__(
        self,
        timelines: timeline.TimelineReader,
        ontologies: ontology.OntologyReader,
        ind: index.Index,
    ):
        self.penicillin = ontologies.get_words_for_subword_term("ATC/J01C")

        self.cephalosporin = {
            a
            for code in ["J01DB", "J01DC", "J01DE"]
            for a in ontologies.get_words_for_subword_term(f"ATC/{code}")
        }

        tract_infection_icd9 = [
            "599",
        ]

        tract_infection_icd10 = [
            "N39.0",
        ]

        self.tract_infection_codes = set()

        for code in tract_infection_icd9:
            self.tract_infection_codes |= set(
                ontologies.get_words_for_subword_term("ICD9CM/" + code)
            )
        for code in tract_infection_icd10:
            self.tract_infection_codes |= set(
                ontologies.get_words_for_subword_term("ICD10CM/" + code)
            )

        self.all_patient_ids = set(
            ind.get_all_patient_ids(self.tract_infection_codes)
        ) & (
            set(ind.get_all_patient_ids(self.penicillin))
            | set(ind.get_all_patient_ids(self.cephalosporin))
        )

    def label(self, patient: Patient) -> List[Label]:
        has_cancer = False

        for i, day in enumerate(patient.days):
            for code in day.observations:
                if code in self.tract_infection_codes:
                    has_cancer = True

                if code in self.penicillin or code in self.cephalosporin:
                    if i == 0 or not has_cancer:
                        # Ignore people who don't have cancer
                        return []
                    else:
                        return [
                            Label(
                                event_index=i - 1,
                                boolean_value=code in self.penicillin,
                            )
                        ]

        # No events
        return []

    def get_all_patient_ids(self) -> Optional[Set[int]]:
        return self.all_patient_ids

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class Tabata1ConsultInterventionLabeler(Labeler):
    def __init__(
        self,
        timelines: timeline.TimelineReader,
        ontologies: ontology.OntologyReader,
        ind: index.Index,
    ):
        self.pd1 = {
            a
            for code in ["1597876", "1547545", "1792776"]
            for a in ontologies.get_words_for_subword_term(f"RXNORM/{code}")
        }

        self.ana = ontologies.get_words_for_subword_term("ATC/L01")

        melanoma_icd9 = [
            "172",
        ]

        melanoma_icd10 = [
            "C43",
        ]

        self.melanoma_codes = set()

        for code in melanoma_icd9:
            self.melanoma_codes |= set(
                ontologies.get_words_for_subword_term("ICD9CM/" + code)
            )
        for code in melanoma_icd10:
            self.melanoma_codes |= set(
                ontologies.get_words_for_subword_term("ICD10CM/" + code)
            )

        self.all_patient_ids = set(
            ind.get_all_patient_ids(self.melanoma_codes)
        ) & (
            set(ind.get_all_patient_ids(self.pd1))
            | set(ind.get_all_patient_ids(self.ana))
        )

    def label(self, patient: Patient) -> List[Label]:
        has_cancer = False

        for i, day in enumerate(patient.days):
            for code in day.observations:
                if code in self.melanoma_codes:
                    has_cancer = True

                if code in self.pd1 or code in self.ana:
                    if i == 0 or not has_cancer:
                        # Ignore people who don't have cancer
                        return []
                    else:
                        return [
                            Label(event_index=i - 1, boolean_value=code in self.pd1)
                        ]

        # No events
        return []

    def get_all_patient_ids(self) -> Optional[Set[int]]:
        return self.all_patient_ids

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class NextVisitCodeCategoryLabeler(FixedTimeHorizonEventLabeler):
    def __init__(
        self,
        timelines: timeline.TimelineReader,
        ontologies: ontology.OntologyReader,
        ind: index.Index,
        category: str,
    ):
        self.codes = set(ontologies.get_words_for_subword_term(category))

    def get_event_ages(self, patient: Patient) -> List[int]:
        """
        Return a sorted list containing the ages at which the event occurs
        """
        result = []
        for day in patient.days:
            has_code = False
            for code in day.observations:
                if code in self.codes:
                    has_code = True

            if has_code:
                result.append(day.age)

        return result

    def get_time_horizon(self) -> int:
        """
        Return an integer which represents the length of the time horizon in days
        """
        return 365

    def get_prediction_days(self, patient: Patient) -> List[int]:
        """
        Return a sorted list containing the indices in which it's valid to make a prediction.
        """
        event_ages = self.get_event_ages(patient)

        first_age: Optional[int]
        if event_ages:
            first_age = event_ages[0]
        else:
            first_age = None

        last_prediction_age: Optional[int] = None
        indices = []
        for i, day in enumerate(patient.days):
            if i == 0:
                continue

            if first_age is not None and day.age >= first_age:
                break

            if (
                last_prediction_age is None
                or day.age >= last_prediction_age + 365
            ):
                indices.append(i)
                last_prediction_age = day.age

        return indices[1:]

    def get_all_patient_ids(self) -> Optional[Set[int]]:
        return None

    def get_labeler_type(self) -> LabelType:
        return "boolean"
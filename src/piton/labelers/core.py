"""Core labeling functionality/schemas, shared across all labeling functions."""
from __future__ import annotations

import collections
import datetime
import multiprocessing
import pprint
import random
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from nptyping import NDArray

from .. import Patient
from ..datasets import PatientDatabase


@dataclass(frozen=True)
class TimeHorizon:
    """An interval of time."""

    start: datetime.timedelta
    end: datetime.timedelta


@dataclass(frozen=True)
class SurvivalValue:
    """Used for survival tasks."""

    event_time: int  # TODO - rename
    is_censored: bool  # TRUE if this patient was censored


LabelType = Union[
    Literal["boolean"],
    Literal["numeric"],
    Literal["survival"],
    Literal["categorical"],
]

VALID_LABEL_TYPES = ["boolean", "numeric", "survival", "categorical"]


@dataclass
class Label:
    """An individual label for a particular patient at a particular time.
    The prediction for this label is made with all data <= time."""

    time: datetime.datetime
    value: Union[bool, int, float, SurvivalValue]


def _apply_labeling_function(
    args: Tuple[LabelingFunction, str, List[int]]
) -> Dict[int, List[Label]]:
    """Apply a labeling function to the set of patients included in `patient_ids`.
    Gets called as a parallelized subprocess of the .apply() method of `LabelingFunction`.
    """
    labeling_function: LabelingFunction = args[0]
    database_path: str = args[1]
    patient_ids: List[int] = args[2]

    database: PatientDatabase = PatientDatabase(database_path)
    patients_to_labels: Dict[int, List[Label]] = {}
    for patient_id in patient_ids:
        patient: Patient = database[patient_id]  # type: ignore
        labels: List[Label] = labeling_function.label(patient)

        # Only add a patient to the `patient_to_labels` dict if they have at least one label
        # This allows for faster iteration over the .keys() of this dict, since we know
        # that each patient in this dict must have a label (which is faster than iterating over each
        # patient and checking len(labels) > 0).
        if len(labels) > 0:
            patients_to_labels[patient_id] = labels

    return patients_to_labels


class LabelingFunction(ABC):
    """An interface for labeling functions.

    A labeling function applies a label to a specific datetime in a given patient's timeline.
    It can be thought of as generating the following list given a specific patient:
        [(patient ID, datetime_1, label_1), (patient ID, datetime_2, label_2), ... ]
    Usage:
    ```
        labeling_function: LabelingFunction = LF(...)
        patients: Sequence[Patient] = ...
        labels: LabeledPatient = labeling_function.apply(patients)
    ```
    """

    @abstractmethod
    def label(self, patient: Patient) -> List[Label]:
        """Apply every label that is applicable to the provided patient.

        This is only called once per patient.

        Args:
            patient (Patient): A patient object

        Returns:
            List[Label]: A list of :class:`Label` containing every label for the given patient
        """
        pass

    def get_required_codes(self) -> List[int]:  # type: ignore
        """Return the set of codes that a patient must have at least one to qualify for this labeler.

        This allows us to only extract patients from the :class:`PatientDatabase` who have a code
        that matches one of these "required codes."

        Returns:
            List[int]: List of applicable OMOP codes
        """
        return []

    def get_patient_start_end_times(
        self, patient: Patient
    ) -> Tuple[datetime.datetime, datetime.datetime]:
        """Return the (start, end) of the patient timeline.

        TODO: Evaluate whether this can be removed

        Returns:
            Tuple[datetime.datetime, datetime.datetime]: (start, end)
        """
        return (patient.events[0].start, patient.events[-1].start)

    @abstractmethod
    def get_labeler_type(self) -> LabelType:
        """Return what type of labels this labeler returns. See the Label class."""
        pass

    def apply(
        self,
        database_path: str,
        num_threads: int = 1,
        num_patients: Optional[int] = None,
    ) -> LabeledPatients:
        """Apply the `label()` function one-by-one to each Patient in a sequence of Patients.

        Args:
            database_path (str): Path to `PatientDatabase` on disk
            num_threads (int, optional): Number of CPU threads to parallelize across. Defaults to 1.
            num_patients (Optional[int], optional): Number of patients to process - useful for debugging.
                If None, use all patients.

        Returns:
            LabeledPatients: Maps patients to labels
        """
        # Split patient IDs across parallelized processes
        if num_patients is None:
            num_patients = len(PatientDatabase(database_path))
        pids = list(range(num_patients))
        pids_parts = np.array_split(pids, num_threads)

        # Multiprocessing
        tasks = [(self, database_path, pid_part) for pid_part in pids_parts]
        ctx = multiprocessing.get_context("forkserver")
        with ctx.Pool(num_threads) as pool:
            results: List[Dict[int, List[Label]]] = list(
                pool.imap(_apply_labeling_function, tasks)
            )

        # Join results and return
        patients_to_labels: Dict[int, List[Label]] = dict(
            collections.ChainMap(*results)
        )
        return LabeledPatients(patients_to_labels, self.get_labeler_type())


class LabeledPatients(MutableMapping[int, List[Label]]):
    """Maps patients to labels.

    Wrapper class around the output of an LF's `apply()` function
    """

    def __init__(
        self,
        patients_to_labels: Dict[int, List[Label]],
        labeler_type: LabelType,
    ):
        """Construct a `LabeledPatients` object from the output of an LF's `apply()` function.

        Args:
            patients_to_labels (Dict[int, List[Label]]): [key] = patient ID, [value] = labels for this patient
            labeler_type (LabelType): Type of labeler
        """
        self.patients_to_labels: Dict[int, List[Label]] = patients_to_labels
        self.labeler_type: LabelType = labeler_type

    def get_labels_from_patient_idx(self, idx: int) -> List[Label]:
        return self.patients_to_labels[idx]

    def get_all_patient_ids(self) -> List[int]:
        return sorted(list(self.patients_to_labels.keys()))

    def get_patients_to_labels(self) -> Dict[int, List[Label]]:
        return self.patients_to_labels

    def get_labeler_type(self) -> LabelType:
        return self.labeler_type

    def as_numpy_arrays(
        self,
    ) -> Tuple[
        NDArray[Literal["n_patients, 1"], np.int64],
        NDArray[Literal["n_patients, 1"], Any],
        NDArray[Shape["n_patients, 1"], np.datetime64],
    ]:
        """Convert `patients_to_labels` to a tuple of NDArray's.

        One NDArray for each of:
            Patient ID, Label value, Label time

        Returns:
            Tuple[NDArray, NDArray, NDArray]: (Patient IDs, Label values, Label time)
        """
        patient_ids: List[int] = []
        label_values: List[Any] = []
        label_times: List[datetime.datetime] = []
        if self.labeler_type in ["boolean", "numerical", "categorical"]:
            for patient_id, labels in self.patients_to_labels.items():
                for label in labels:
                    patient_ids.append(patient_id)
                    label_values.append(label.value)
                    label_times.append(label.time)
        else:
            raise ValueError(
                "Other label types are not implemented yet for this method"
            )
        return (
            np.array(patient_ids),
            np.array(label_values),
            np.array(label_times),
        )

    def get_num_patients(self) -> int:
        """Return the total number of patients."""
        return len(self)

    def get_num_labels(self) -> int:
        """Return the total number of labels across all patients."""
        total: int = 0
        for labels in self.patients_to_labels.values():
            total += len(labels)
        return total

    def as_list_of_label_tuples(self) -> List[Tuple[int, Label]]:
        """Convert `patients_to_labels` to a list of (patient_id, Label) tuples."""
        result: List[Tuple[int, Label]] = []
        for patient_id, labels in self.patients_to_labels.items():
            for label in labels:
                result.append((int(patient_id), label))
        return result

    @classmethod
    def load_from_numpy(
        cls,
        patient_ids: NDArray[Literal["n_patients, 1"], np.int64],
        label_values: NDArray[Literal["n_patients, 1"], Any],
        label_times: NDArray[Literal["n_patients, 1"], datetime.datetime],
        labeler_type: LabelType,
    ) -> LabeledPatients:
        """Create a :class:`LabeledPatients` from NDArray labels.

            Inverse of `as_numpy_arrays()`

        Args:
            patient_ids (NDArray): Patient IDs for the corresponding label.
            label_values (NDArray): Values for the corresponding label.
            label_times (NDArray): Times that the corresponding label occurs.
            labeler_type (LabelType): LabelType of the corresponding labels.
        """
        patients_to_labels: DefaultDict[
            int, List[Label]
        ] = collections.defaultdict(list)
        for patient_id, l_value, l_time in zip(
            patient_ids, label_values, label_times
        ):
            patients_to_labels[patient_id].append(
                Label(time=l_time, value=l_value)
            )
        return LabeledPatients(dict(patients_to_labels), labeler_type)

    def __str__(self):
        """Return string representation."""
        return "LabeledPatients:\n" + pprint.pformat(self.patients_to_labels)

    def __getitem__(self, key):
        """Necessary for implementing MutableMapping."""
        return self.patients_to_labels[key]

    def __setitem__(self, key, item):
        """Necessary for implementing MutableMapping."""
        self.patients_to_labels[key] = item

    def __delitem__(self, key):
        """Necessary for implementing MutableMapping."""
        del self.patients_to_labels[key]

    def __iter__(self):
        """Necessary for implementing MutableMapping."""
        return iter(self.patients_to_labels)

    def __len__(self):
        """Necessary for implementing MutableMapping."""
        return len(self.patients_to_labels)


##########################################################
# Specific Labeler Superclasses
##########################################################


class FixedTimeHorizonEventLF(LabelingFunction):
    """Label events that occur within a particular time horizon.

    A fixed time horizon labeler enables you to label events that occur within a particular
    time horizon (i.e. `TimeHorizon`). It is a boolean event that is TRUE if the event of interest
    occurs within the next 6 months.

    No labels are generated if the patient record is "censored" before the end of the horizon.

    You are required to implement three methods:
        get_outcome_times() for defining the datetimes of the event of interset
        get_prediction_times() for defining the datetimes at which we make our predictions
        get_time_horizon() for defining the length of time (i.e. `TimeHorizon`) to use for the time horizon
    """

    @abstractmethod
    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a sorted list containing the datetimes that the event of interest "occurs".

        IMPORTANT: Must be sorted ascending (i.e. start -> end of timeline)

        Args:
            patient (Patient): A patient object

        Returns:
            List[datetime.datetime]: A list of datetimes, one corresponding to an occurrence of the outcome
        """
        pass

    @abstractmethod
    def get_time_horizon(self) -> TimeHorizon:
        """Return time horizon for making predictions with this labeling function.

        Return the (start offset, end offset) of the time horizon (from the prediction time)
        used for labeling whether an outcome occurred or not. These can be arbitrary timedeltas.

        NOTE: Must have end offset >= start offset.

        Example:
            X is the time that you're making a prediction (given by `get_prediction_times()`)
            (A,B) is your time horizon (given by `get_time_horizon()`)
            O is an outcome (given by `get_outcome_times()`)

            Then given a patient timeline:
                X-----(X+A)------(X+B)------


            This has a label of TRUE:
                X-----(X+A)--O---(X+B)------

            This has a label of TRUE:
                X-----(X+A)--O---(X+B)----O-

            This has a label of FALSE:
                X---O-(X+A)------(X+B)------

            This has a label of FALSE:
                X-----(X+A)------(X+B)--O---
        """
        pass

    @abstractmethod
    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a sorted list containing the datetimes at which we'll make a prediction.

        IMPORTANT: Must be sorted ascending (i.e. start -> end of timeline)
        """
        pass

    def label(self, patient: Patient) -> List[Label]:
        """Return a list of Labels for an individual patient.

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
        outcome_times: List[datetime.datetime] = self.get_outcome_times(patient)
        time_horizon: TimeHorizon = self.get_time_horizon()

        results: List[Label] = []
        curr_outcome_idx: int = 0
        # For each prediction time, check if there is an outcome which occurs within the (start, end)
        # of the time horizon
        for time in self.get_prediction_times(patient):
            while (
                curr_outcome_idx < len(outcome_times)
                and outcome_times[curr_outcome_idx] < time + time_horizon.start
            ):
                # This is the idx in `outcome_times` that corresponds to the first outcome EQUAL or AFTER
                # the time horizon for this prediction time starts (if one exists)
                curr_outcome_idx += 1

            # TRUE if an event occurs within the time horizon
            is_outcome_occurs_in_time_horizon: bool = curr_outcome_idx < len(
                outcome_times
            ) and (
                time + time_horizon.start
                <= outcome_times[curr_outcome_idx]
                <= time + time_horizon.end
            )
            # TRUE if patient is censored (i.e. timeline ends BEFORE this time horizon ends,
            # so we don't know if the outcome happened after the patient timeline ends)
            is_censored: bool = end_time < time + time_horizon.end

            if is_outcome_occurs_in_time_horizon:
                results.append(Label(time=time, value=True))
            elif not is_censored:
                # Not censored + no outcome => FALSE
                results.append(Label(time=time, value=False))

        # # checks that we have a label for each prediction time (even if `None``)
        # assert len(results) == len(self.get_prediction_times(patient))
        return results

    def get_patient_start_end_times(
        self, patient: Patient
    ) -> Tuple[datetime.datetime, datetime.datetime]:
        """Return the datetimes that we consider the (start, end) of this patient."""
        return (patient.events[0].start, patient.events[-1].start)

    def get_labeler_type(self) -> LabelType:
        """Return boolean labels (TRUE if event occurs in TimeHorizon, FALSE otherwise)."""
        return "boolean"


class OneLabelPerPatient(LabelingFunction):
    # TODO - update
    def __init__(self, labeling_function, seed=10):
        self.labeling_function = labeling_function
        self.seed = seed

    def label(self, patient: Patient) -> List[Label]:
        labels = self.labeling_function.label(patient)
        if len(labels) == 0:
            return labels
        return [random.choice(labels)]

    def get_labeler_type(self) -> LabelType:
        """Return boolean labels (TRUE if event occurs in TimeHorizon, FALSE otherwise)."""
        return self.labeling_function.get_labeler_type()

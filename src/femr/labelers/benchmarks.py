"""Labeling functions for OMOP data."""
from __future__ import annotations

import datetime
import warnings
from typing import List, Optional, Tuple

from femr import Event, Patient
from femr.extension import datasets as extension_datasets
from femr.labelers.core import Label, Labeler, LabelType, TimeHorizon
from femr.labelers.omop import (
    CodeLabeler,
    WithinVisitLabeler,
    does_exist_event_within_time_range,
    get_death_concepts,
    get_icu_events,
    map_omop_concept_codes_to_femr_codes,
    move_datetime_to_end_of_day,
)
from femr.labelers.omop_inpatient_admissions import (
    InpatientLongAdmissionLabeler,
    InpatientReadmissionLabeler,
    WithinInpatientVisitLabeler,
    get_inpatient_admission_discharge_times,
)

##########################################################
##########################################################
# CLMBR Benchmark Tasks
# See: https://arxiv.org/pdf/2001.05295.pdf
# details on how this was reproduced.
#
# Citation: Guo et al.
# "EHR foundation models improve robustness in the presence of temporal distribution shift"
# Scientific Reports. 2023.
##########################################################
##########################################################


class Guo_LongLOSLabeler(Labeler):
    """Long LOS prediction task from Guo et al. 2023.

    Binary prediction task @ 11:59PM on the day of admission whether the patient stays in hospital for >=7 days.

    Excludes:
        - Admissions with LOS < 1 day
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.long_time: datetime.timedelta = datetime.timedelta(days=7)
        self.prediction_time_adjustment_func = move_datetime_to_end_of_day

    def label(self, patient: Patient) -> List[Label]:
        """Label all admissions with admission length >= `self.long_time`"""
        labels: List[Label] = []
        for admission_time, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
            # If admission and discharge are on the same day, then ignore
            if admission_time.date() == discharge_time.date():
                continue
            is_long_admission: bool = (discharge_time - admission_time) >= self.long_time
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(admission_time)
            labels.append(Label(prediction_time, is_long_admission))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class Guo_30DayReadmissionLabeler(InpatientReadmissionLabeler):
    """30-day readmissions prediction task from Guo et al. 2023.

    Binary prediction task @ 11:59PM on the day of disharge whether the patient will be readmitted within 30 days.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        time_horizon: TimeHorizon = TimeHorizon(
            start=datetime.timedelta(minutes=1), end=datetime.timedelta(days=30)
        )  # type: ignore
        super().__init__(
            ontology=ontology,
            time_horizon=time_horizon,
            prediction_time_adjustment_func=move_datetime_to_end_of_day,
        )


class Guo_ICUAdmissionLabeler(WithinInpatientVisitLabeler):
    """ICU admission prediction task from Guo et al. 2023.

    Binary prediction task @ 11:59PM on the day of admission whether the patient will be readmitted to the ICU.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        super().__init__(
            ontology=ontology,
            visit_start_adjust_func=None,
            visit_end_adjust_func=None,
        )

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        return [e.start for e in get_icu_events(patient, self.ontology)]  # type: ignore


##########################################################
##########################################################
# MIMIC-III Benchmark Tasks
# See: https://www.nature.com/articles/s41597-019-0103-9/figures/7 for
# details on how this was reproduced.
#
# Citation: Harutyunyan, H., Khachatrian, H., Kale, D.C. et al.
# Multitask learning and benchmarking with clinical time series data.
# Sci Data 6, 96 (2019). https://doi.org/10.1038/s41597-019-0103-9
##########################################################
##########################################################


class Harutyunyan_DecompensationLabeler(CodeLabeler):
    """Decompensation prediction task from Harutyunyan et al. 2019.

    Hourly binary prediction task on whether the patient dies in the next 24 hours.
    Make prediction every 60 minutes after ICU admission, starting at hour 4.

    Excludes:
        - ICU admissions with no length-of-stay (i.e. `event.end is None` )
        - ICU admissions < 4 hours
        - ICU admissions with no events
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        # Next 24 hours
        time_horizon = TimeHorizon(datetime.timedelta(hours=0), datetime.timedelta(hours=24))
        # Death events
        outcome_codes = list(
            map_omop_concept_codes_to_femr_codes(ontology, get_death_concepts(), is_ontology_expansion=True)
        )
        # Save ontology for `get_prediction_times()`
        self.ontology = ontology

        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
        )

    def is_apply_censoring(self) -> bool:
        """Consider censored patients to be alive."""
        return False

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a list of every hour after every ICU visit, up until death occurs or end of visit.
        Note that this requires creating an artificial event for each hour since there will only be one true
        event per ICU admission, but we'll need to create many subevents (at each hour) within this event.
        Also note that these events may not align with :00 minutes if the ICU visit does not start exactly "on the hour".

        Excludes:
            - ICU admissions with no length-of-stay (i.e. `event.end is None` )
            - ICU admissions < 4 hours
            - ICU admissions with no events
        """
        times: List[datetime.datetime] = []
        icu_events: List[Tuple[int, Event]] = get_icu_events(patient, self.ontology, is_return_idx=True)  # type: ignore
        icu_event_idxs = [idx for idx, __ in icu_events]
        death_times: List[datetime.datetime] = self.get_outcome_times(patient)
        earliest_death_time: datetime.datetime = min(death_times) if len(death_times) > 0 else datetime.datetime.max
        for __, e in icu_events:
            if (
                e.end is not None
                and e.end - e.start >= datetime.timedelta(hours=4)
                and does_exist_event_within_time_range(patient, e.start, e.end, exclude_event_idxs=icu_event_idxs)
            ):
                # Record every hour after admission (i.e. every hour between `e.start` and `e.end`),
                # but only after 4 hours have passed (i.e. start at `e.start + 4 hours`)
                # and only until the visit ends (`e.end`) or a death event occurs (`earliest_death_time`)
                end_of_stay: datetime.datetime = min(e.end, earliest_death_time)
                event_time = e.start + datetime.timedelta(hours=4)
                while event_time < end_of_stay:
                    times.append(event_time)
                    event_time += datetime.timedelta(hours=1)
        return times


class Harutyunyan_MortalityLabeler(WithinVisitLabeler):
    """In-hospital mortality prediction task from Harutyunyan et al. 2019.
    Single binary prediction task of whether patient dies within ICU admission 48 hours after admission.
    Make prediction 48 hours into ICU admission.

    Excludes:
        - ICU admissions with no length-of-stay (i.e. `event.end is None` )
        - ICU admissions < 48 hours
        - ICU admissions with no events before 48 hours
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        visit_start_adjust_func = lambda x: x + datetime.timedelta(
            hours=48
        )  # Make prediction 48 hours into ICU admission
        visit_end_adjust_func = lambda x: x
        super().__init__(ontology, visit_start_adjust_func, visit_end_adjust_func)

    def is_apply_censoring(self) -> bool:
        """Consider censored patients to be alive."""
        return False

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a list of all times when the patient experiences an outcome"""
        outcome_codes = list(
            map_omop_concept_codes_to_femr_codes(self.ontology, get_death_concepts(), is_ontology_expansion=True)
        )
        times: List[datetime.datetime] = []
        for e in patient.events:
            if e.code in outcome_codes:
                times.append(e.start)
        return times

    def get_visit_events(self, patient: Patient) -> List[Event]:
        """Return a list of all ICU visits > 48 hours.

        Excludes:
            - ICU admissions with no length-of-stay (i.e. `event.end is None` )
            - ICU admissions < 48 hours
            - ICU admissions with no events before 48 hours
        """
        icu_events: List[Tuple[int, Event]] = get_icu_events(patient, self.ontology, is_return_idx=True)  # type: ignore
        icu_event_idxs = [idx for idx, __ in icu_events]
        valid_events: List[Event] = []
        for __, e in icu_events:
            if (
                e.end is not None
                and e.end - e.start >= datetime.timedelta(hours=48)
                and does_exist_event_within_time_range(
                    patient, e.start, e.start + datetime.timedelta(hours=48), exclude_event_idxs=icu_event_idxs
                )
            ):
                valid_events.append(e)
        return valid_events


class Harutyunyan_LengthOfStayLabeler(Labeler):
    """LOS remaining regression task from Harutyunyan et al. 2019.

    Hourly regression task on the patient's remaining length-of-stay (in hours) in the ICU.
    Make prediction every 60 minutes after ICU admission, starting at hour 4.

    Excludes:
        - ICU admissions with no length-of-stay (i.e. `event.end is None` )
        - ICU admissions < 4 hours
        - ICU admissions with no events
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology = ontology

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a list of all times when the patient experiences an outcome"""
        outcome_codes = list(
            map_omop_concept_codes_to_femr_codes(self.ontology, get_death_concepts(), is_ontology_expansion=True)
        )
        times: List[datetime.datetime] = []
        for e in patient.events:
            if e.code in outcome_codes:
                times.append(e.start)
        return times

    def get_labeler_type(self) -> LabelType:
        return "numerical"

    def label(self, patient: Patient) -> List[Label]:
        """Return a list of Labels at every hour after every ICU visit, where each Label is the # of hours
        until the visit ends (or a death event occurs).
        Note that this requires creating an artificial event for each hour since there will only be one true
        event per ICU admission, but we'll need to create many subevents (at each hour) within this event.
        Also note that these events may not align with :00 minutes if the ICU visit does not start exactly "on the hour".

        Excludes:
            - ICU admissions with no length-of-stay (i.e. `event.end is None` )
            - ICU admissions < 4 hours
            - ICU admissions with no events
        """
        labels: List[Label] = []
        icu_events: List[Tuple[int, Event]] = get_icu_events(patient, self.ontology, is_return_idx=True)  # type: ignore
        icu_event_idxs = [idx for idx, __ in icu_events]
        death_times: List[datetime.datetime] = self.get_outcome_times(patient)
        earliest_death_time: datetime.datetime = min(death_times) if len(death_times) > 0 else datetime.datetime.max
        for __, e in icu_events:
            if (
                e.end is not None
                and e.end - e.start >= datetime.timedelta(hours=4)
                and does_exist_event_within_time_range(patient, e.start, e.end, exclude_event_idxs=icu_event_idxs)
            ):
                # Record every hour after admission (i.e. every hour between `e.start` and `e.end`),
                # but only after 4 hours have passed (i.e. start at `e.start + 4 hours`)
                # and only until the visit ends (`e.end`) or a death event occurs (`earliest_death_time`)
                end_of_stay: datetime.datetime = min(e.end, earliest_death_time)
                event_time = e.start + datetime.timedelta(hours=4)
                while event_time < end_of_stay:
                    los: float = (end_of_stay - event_time).total_seconds() / 3600
                    labels.append(Label(event_time, los))
                    event_time += datetime.timedelta(hours=1)
                    assert (
                        los >= 0
                    ), f"LOS should never be negative, but end_of_stay={end_of_stay} - event_time={event_time} = {end_of_stay - event_time} for patient {patient.patient_id}"
        return labels


##########################################################
##########################################################
# Few-Shot Benchmark Tasks
# See: https://github.com/som-shahlab/few_shot_ehr/tree/main
#
# Citation: van Uden, Cara et al.
# Few shot EHRs
##########################################################
##########################################################


class FirstDiagnosisTimeHorizonCodeLabeler(Labeler):
    """Predict if patient will have their *first* diagnosis of `self.root_concept_code` in the next (1, 365) days.
    Make prediction at end of day of discharge from inpatient admission.
    If patient has already had this outcome, then ignore the rest of their timeline.
    """

    root_concept_code = None  # OMOP concept code for outcome, e.g. "SNOMED/57054005"

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        assert (
            self.root_concept_code is not None
        ), "Must specify `root_concept_code` for `FirstDiagnosisTimeHorizonCodeLabeler`"
        self.ontology = ontology
        self.outcome_codes = list(
            map_omop_concept_codes_to_femr_codes(ontology, [self.root_concept_code], is_ontology_expansion=True)
        )
        self.time_horizon: TimeHorizon = TimeHorizon(datetime.timedelta(days=1), datetime.timedelta(days=365))

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return discharge as prediction timm."""
        times: List[datetime.datetime] = []
        for __, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
            prediction_time: datetime.datetime = move_datetime_to_end_of_day(discharge_time)
            times.append(prediction_time)
        times = sorted(list(set(times)))
        return times

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of this patient's events whose `code` is in `self.outcome_codes`."""
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.code in self.outcome_codes:
                times.append(event.start)
        return times

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon

    def is_apply_censoring(self) -> bool:
        return False

    def allow_same_time_labels(self) -> bool:
        return False

    def get_labeler_type(self) -> LabelType:
        return "boolean"

    def label(self, patient: Patient) -> List[Label]:
        if len(patient.events) == 0:
            return []

        __, end_time = self.get_patient_start_end_times(patient)
        prediction_times: List[datetime.datetime] = self.get_prediction_times(patient)
        outcome_times: List[datetime.datetime] = self.get_outcome_times(patient)
        time_horizon: TimeHorizon = self.get_time_horizon()

        # Get (start, end) of time horizon. If end is None, then it's infinite (set timedelta to max)
        time_horizon_start: datetime.timedelta = time_horizon.start
        time_horizon_end: Optional[datetime.timedelta] = time_horizon.end  # `None` if infinite time horizon

        # For each prediction time, check if there is an outcome which occurs within the (start, end)
        # of the time horizon
        results: List[Label] = []
        curr_outcome_idx: int = 0
        last_time = None
        for time_idx, time in enumerate(prediction_times):
            if last_time is not None:
                assert (
                    time > last_time
                ), f"Must be ascending prediction times, instead got last_prediction_time={last_time} <= prediction_time={time} for patient {patient.patient_id} at curr_outcome_idx={curr_outcome_idx} | prediction_time_idx={time_idx} | start_prediction_time={prediction_times[0]}"

            last_time = time

            while curr_outcome_idx < len(outcome_times) and outcome_times[curr_outcome_idx] < time + time_horizon_start:
                # `curr_outcome_idx` is the idx in `outcome_times` that corresponds to the first
                # outcome EQUAL or AFTER the time horizon for this prediction time starts (if one exists)
                curr_outcome_idx += 1

            if curr_outcome_idx < len(outcome_times) and outcome_times[curr_outcome_idx] == time:
                if not self.allow_same_time_labels():
                    continue
                warnings.warn(
                    "You are making predictions at the same time as the target outcome."
                    "This frequently leads to label leakage."
                )

            if curr_outcome_idx > 0:
                # We only want to track the first outcome that occurs to this patient, so if we've
                # already seen an outcome (i.e. `curr_outcome_idx > 0`), then we stop labeling
                # NOTE: This is the only difference from the `TimeHorizonLabeler`
                break

            # TRUE if an event occurs within the time horizon
            is_outcome_occurs_in_time_horizon: bool = (
                (
                    # ensure there is an outcome
                    # (needed in case there are 0 outcomes)
                    curr_outcome_idx
                    < len(outcome_times)
                )
                and (
                    # outcome occurs after time horizon starts
                    time + time_horizon_start
                    <= outcome_times[curr_outcome_idx]
                )
                and (
                    # outcome occurs before time horizon ends (if there is an end)
                    (time_horizon_end is None)
                    or outcome_times[curr_outcome_idx] <= time + time_horizon_end
                )
            )
            # TRUE if patient is censored (i.e. timeline ends BEFORE this time horizon ends,
            # so we don't know if the outcome happened after the patient timeline ends)
            # If infinite time horizon labeler, then assume no censoring
            is_censored: bool = end_time < time + time_horizon_end if (time_horizon_end is not None) else False

            if is_outcome_occurs_in_time_horizon:
                results.append(Label(time=time, value=True))
            elif not is_censored:
                # Not censored + no outcome => FALSE
                results.append(Label(time=time, value=False))
            else:
                if self.is_apply_censoring():
                    # Censored + no outcome => CENSORED
                    pass
                else:
                    # Censored + no outcome => FALSE
                    results.append(Label(time=time, value=False))

        return results


class PancreaticCancerCodeLabeler(FirstDiagnosisTimeHorizonCodeLabeler):
    # n = 200684
    root_concept_code = "SNOMED/372003004"


class CeliacDiseaseCodeLabeler(FirstDiagnosisTimeHorizonCodeLabeler):
    # n = 60270
    root_concept_code = "SNOMED/396331005"


class LupusCodeLabeler(FirstDiagnosisTimeHorizonCodeLabeler):
    # n = 176684
    root_concept_code = "SNOMED/55464009"


class AcuteMyocardialInfarctionCodeLabeler(FirstDiagnosisTimeHorizonCodeLabeler):
    # n = 21982
    root_concept_code = "SNOMED/57054005"


class CTEPHCodeLabeler(FirstDiagnosisTimeHorizonCodeLabeler):
    # n = 1433
    root_concept_code = "SNOMED/233947005"


class EssentialHypertensionCodeLabeler(FirstDiagnosisTimeHorizonCodeLabeler):
    # n = 4644483
    root_concept_code = "SNOMED/59621000"


class HyperlipidemiaCodeLabeler(FirstDiagnosisTimeHorizonCodeLabeler):
    # n = 3048320
    root_concept_code = "SNOMED/55822004"


if __name__ == "__main__":
    pass
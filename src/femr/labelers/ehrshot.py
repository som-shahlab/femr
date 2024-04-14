"""EHRSHOT tasks from Wornow et al. 2023."""

from __future__ import annotations

import datetime
from typing import Any, Callable, List, Optional, Set, Tuple, Union

import meds
import pandas as pd

import femr.ontology
from femr.labelers.core import Labeler, TimeHorizon, TimeHorizonEventLabeler, move_datetime_to_end_of_day
from femr.labelers.omop import WithinVisitLabeler
from femr.labelers.omop_labs import InstantLabValueLabeler


def get_icu_visit_detail_care_site_ids(ontology: femr.ontology.Ontology) -> Set[str]:
    return ontology.get_all_children(
        [
            # All care sites with "ICU" (case insensitive) in the name
            "528292",
            "528612",
            "528604",
            "528623",
            "528396",
            "528377",
            "528314",
            "528478",
            "528112",
            "528024",
            "527323",
            "527858",
        ]
    )


def get_icu_measurements(
    patient: meds.Patient, ontology: femr.ontology.Ontology
) -> List[Tuple[datetime.datetime, meds.Measurement]]:
    """Return all ICU events for this patient."""
    icu_visit_detail_care_site_ids: Set[str] = get_icu_visit_detail_care_site_ids(ontology)
    measurements: List[Tuple[datetime.datetime, meds.Measurement]] = []  # type: ignore
    for idx, e in enumerate(patient["events"]):
        # `visit_detail` is more accurate + comprehensive than `visit_occurrence` for
        #   ICU measurements for STARR OMOP for some reason
        for m in e["measurements"]:
            if (
                m["metadata"]["table"] == "visit_detail"
                and "care_site_id" in m["metadata"]
                and m["metadata"]["care_site_id"] in icu_visit_detail_care_site_ids  # no ontology expansion for ICU
            ):
                # Error checking
                if isinstance(m["metadata"]["end"], str):
                    m["metadata"]["end"] = datetime.datetime.fromisoformat(m["metadata"]["end"])
                if e["time"] is None or m["metadata"]["end"] is None:
                    raise RuntimeError(
                        f"Event {e} for patient {patient['patient_id']} cannot have `None` as its `start` or `end` attribute."
                    )
                elif e["time"] > m["metadata"]["end"]:
                    raise RuntimeError(
                        f"Event {e} for patient {patient['patient_id']} cannot have `start` after `end`."
                    )
                # Drop single point in time measurements
                if e["time"] == m["metadata"]["end"]:
                    continue
                measurements.append((e["time"], m))  # type: ignore
    return measurements


def get_visit_codes(ontology: femr.ontology.Ontology) -> Set[str]:
    return ontology.get_all_children(get_inpatient_admission_codes().union(get_outpatient_visit_codes()))


def get_inpatient_admission_codes(ontology: femr.ontology.Ontology) -> Set[str]:
    # Don't get children here b/c it adds noise (i.e. "Medicare Specialty/AO")
    return {
        "Visit/IP",
        "Visit/ERIP",
    }


def get_outpatient_visit_codes(ontology: femr.ontology.Ontology) -> Set[str]:
    # Don't get children here b/c it adds noise (i.e. "Medicare Specialty/AO")
    return {
        "Visit/OP",
        "Visit/OMOP4822036",
        "Visit/OMOP4822458",
    }


def get_outpatient_visit_measurements(
    patient: meds.Patient, ontology: femr.ontology.Ontology
) -> List[Tuple[datetime.datetime, meds.Measurement]]:
    admission_codes: Set[str] = get_outpatient_visit_codes(ontology)
    measurements: List[meds.Measurement] = []
    for e in patient["events"]:
        for m in e["measurements"]:
            if (
                m["metadata"]["table"] == "visit"
                and m["code"] in admission_codes
            ):
                if isinstance(m["metadata"]["end"], str):
                    m["metadata"]["end"] = datetime.datetime.fromisoformat(m["metadata"]["end"])
                # Error checking
                if e["time"] is None or m["metadata"]["end"] is None:
                    raise RuntimeError(f"Event {e} cannot have `None` as its `start` or `end` attribute.")
                elif e["time"] > m["metadata"]["end"]:
                    raise RuntimeError(f"Event {e} cannot have `start` after `end`.")
                # Drop single point in time events
                if e["time"] == m["metadata"]["end"]:
                    continue
                measurements.append((e["time"], m))
    return measurements


def get_inpatient_admission_measurements(
    patient: meds.Patient, ontology: femr.ontology.Ontology
) -> List[Tuple[datetime.datetime, meds.Measurement]]:
    admission_codes: Set[str] = get_inpatient_admission_codes(ontology)
    measurements: List[Tuple[datetime.datetime, meds.Measurement]] = []
    for e in patient["events"]:
        for m in e["measurements"]:
            if (
                m["metadata"]["table"] == "visit" 
                and m["code"] in admission_codes
            ):
                if isinstance(m["metadata"]["end"], str):
                    m["metadata"]["end"] = datetime.datetime.fromisoformat(m["metadata"]["end"])
                # Error checking
                if e["time"] is None or m["metadata"]["end"] is None:
                    raise RuntimeError(f"Event {e} cannot have `None` as its `start` or `end` attribute.")
                elif e["time"] > m["metadata"]["end"]:
                    raise RuntimeError(f"Event {e} cannot have `start` after `end`.")
                # Drop single point in time events
                if e["time"] == m["metadata"]["end"]:
                    continue
                measurements.append((e["time"], m))
    return measurements


def get_inpatient_admission_discharge_times(
    patient: meds.Patient, ontology: femr.ontology.Ontology
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """Return a list of all admission/discharge times for this patient."""
    measurements: List[Tuple[datetime.datetime, meds.Measurement]] = get_inpatient_admission_measurements(
        patient, ontology
    )
    times: List[Tuple[datetime.datetime, datetime.datetime]] = []
    for start, m in measurements:
        if isinstance(m["metadata"]["end"], str):
            m["metadata"]["end"] = datetime.datetime.fromisoformat(m["metadata"]["end"])
        if m["metadata"]["end"] is None:
            raise RuntimeError(f"Event {m} cannot have `None` as its `end` attribute.")
        if start > m["metadata"]["end"]:
            raise RuntimeError(f"Event {m} cannot have `start` after `end`.")
        times.append((start, m["metadata"]["end"]))
    return times


##########################################################
##########################################################
# "Operational Outcomes" Tasks
#
# See: https://www.medrxiv.org/content/10.1101/2022.04.15.22273900v1
# details on how this was reproduced.
# Citation: Guo et al.
# "EHR foundation models improve robustness in the presence of temporal distribution shift"
# Scientific Reports. 2023.
##########################################################
##########################################################


class Guo_LongLOSLabeler(Labeler):
    """Long LOS prediction task from Guo et al. 2023.

    Binary prediction task @ 11:59PM on the day of admission whether the patient stays in hospital for >=7 days.

    Excludes:
        - Visits where discharge occurs on the same day as admission
    """

    def __init__(
        self,
        ontology: femr.ontology.Ontology,
    ):
        self.ontology: femr.ontology.Ontology = ontology
        self.long_time: datetime.timedelta = datetime.timedelta(days=7)
        self.prediction_time_adjustment_func = move_datetime_to_end_of_day

    def label(self, patient: meds.Patient) -> List[meds.Label]:
        """Label all admissions with admission length >= `self.long_time`"""
        labels: List[meds.Label] = []
        for admission_time, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
            # If admission and discharge are on the same day, then ignore
            if admission_time.date() == discharge_time.date():
                continue
            is_long_admission: bool = (discharge_time - admission_time) >= self.long_time
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(admission_time)
            labels.append(
                meds.Label(
                    patient_id=patient["patient_id"], prediction_time=prediction_time, boolean_value=is_long_admission
                )
            )
        return labels


class Guo_30DayReadmissionLabeler(TimeHorizonEventLabeler):
    """30-day readmissions prediction task from Guo et al. 2023.

    Binary prediction task @ 11:59PM on the day of disharge whether the patient will be readmitted within 30 days.

    Excludes:
        - Patients readmitted on same day as discharge
    """

    def __init__(
        self,
        ontology: femr.ontology.Ontology,
    ):
        self.ontology: femr.ontology.Ontology = ontology
        self.time_horizon: TimeHorizon = TimeHorizon(
            start=datetime.timedelta(minutes=1), end=datetime.timedelta(days=30)
        )
        self.prediction_time_adjustment_func = move_datetime_to_end_of_day

    def get_outcome_times(self, patient: meds.Patient) -> List[datetime.datetime]:
        """Return the start times of inpatient admissions."""
        times: List[datetime.datetime] = []
        for admission_time, __ in get_inpatient_admission_discharge_times(patient, self.ontology):
            times.append(admission_time)
        return times

    def get_prediction_times(self, patient: meds.Patient) -> List[datetime.datetime]:
        """Return end of admission as prediction timm."""
        times: List[datetime.datetime] = []
        admission_times = set()
        for admission_time, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(discharge_time)
            # Ignore patients who are readmitted the same day they were discharged b/c of data leakage
            if prediction_time.replace(hour=0, minute=0, second=0, microsecond=0) in admission_times:
                continue
            times.append(prediction_time)
            admission_times.add(admission_time.replace(hour=0, minute=0, second=0, microsecond=0))
        times = sorted(list(set(times)))
        return times

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon


class Guo_ICUAdmissionLabeler(WithinVisitLabeler):
    """ICU admission prediction task from Guo et al. 2023.

    Binary prediction task @ 11:59PM on the day of admission
    whether the patient will be admitted to the ICU during their admission.

    Excludes:
        - Patients transfered on same day as admission
        - Visits where discharge occurs on the same day as admission
    """

    def __init__(
        self,
        ontology: femr.ontology.Ontology,
    ):
        super().__init__(
            ontology=ontology,
            visit_start_adjust_func=move_datetime_to_end_of_day,
            visit_end_adjust_func=None,
        )

    def get_outcome_times(self, patient: meds.Patient) -> List[datetime.datetime]:
        # Return the start times of all ICU admissions -- this is our outcome
        return [time for time, __ in get_icu_measurements(patient, self.ontology)]  # type: ignore

    def get_visit_measurements(self, patient: meds.Patient) -> List[Tuple[datetime.datetime, meds.Measurement]]:
        """Return all inpatient visits where ICU transfer does not occur on the same day as admission."""
        # Get all inpatient visits -- each visit comprises a prediction (start, end) time horizon
        measurements: List[Tuple[datetime.datetime, meds.Measurement]] = get_inpatient_admission_measurements(
            patient, self.ontology
        )
        # Exclude visits where ICU admission occurs on the same day as admission
        icu_transfer_dates: List[datetime.datetime] = [
            x.replace(hour=0, minute=0, second=0, microsecond=0) for x in self.get_outcome_times(patient)
        ]
        valid_visits: List[Tuple[datetime.datetime, meds.Measurement]] = []
        for time, m in measurements:
            # If admission and discharge are on the same day, then ignore
            if isinstance(m["metadata"]["end"], str):
                m["metadata"]["end"] = datetime.datetime.fromisoformat(m["metadata"]["end"])
            if time.date() == m["metadata"]["end"].date():
                continue
            # If ICU transfer occurs on the same day as admission, then ignore
            if time.replace(hour=0, minute=0, second=0, microsecond=0) in icu_transfer_dates:
                continue
            valid_visits.append((time, m))
        return valid_visits


##########################################################
##########################################################
# "Abnormal Lab Value" Tasks
#
# See: https://arxiv.org/abs/2307.02028
# Citation: Wornow et al.
# EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models
# NeurIPS (2023).
##########################################################
##########################################################


class ThrombocytopeniaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for thrombocytopenia based on platelet count (10^9/L).
    Thresholds: mild (<150), moderate(<100), severe(<50), and reference range."""

    original_omop_concept_codes = [
        "LOINC/LP393218-5",
        "LOINC/LG32892-8",
        "LOINC/777-3",
    ]

    def value_to_label(self, value: Union[float, str], unit: Optional[str]) -> str:
        if str(value).lower() in ["normal", "adequate"]:
            return "normal"
        value = float(value)
        if value < 50:
            return "severe"
        elif value < 100:
            return "moderate"
        elif value < 150:
            return "mild"
        return "normal"


class HyperkalemiaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for hyperkalemia using blood potassium concentration (mmol/L).
    Thresholds: mild(>5.5),moderate(>6),severe(>7), and abnormal range."""

    original_omop_concept_codes = [
        "LOINC/LG7931-1",
        "LOINC/LP386618-5",
        "LOINC/LG10990-6",
        "LOINC/6298-4",
        "LOINC/2823-3",
    ]

    def value_to_label(self, value: Union[float, str], unit: Optional[str]) -> str:
        if str(value).lower() in ["normal", "adequate"]:
            return "normal"
        value = float(value)
        if unit is not None:
            unit = unit.lower()
            if unit.startswith("mmol/l"):
                # mmol/L
                # Original OMOP concept ID: 8753
                value = value
            elif unit.startswith("meq/l"):
                # mEq/L (1-to-1 -> mmol/L)
                # Original OMOP concept ID: 9557
                value = value
            elif unit.startswith("mg/dl"):
                # mg / dL (divide by 18 to get mmol/L)
                # Original OMOP concept ID: 8840
                value = value / 18.0
            else:
                raise ValueError(f"Unknown unit: {unit}")
        else:
            raise ValueError(f"Unknown unit: {unit}")
        if value > 7:
            return "severe"
        elif value > 6.0:
            return "moderate"
        elif value > 5.5:
            return "mild"
        return "normal"


class HypoglycemiaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for hypoglycemia using blood glucose concentration (mmol/L).
    Thresholds: mild(<3), moderate(<3.5), severe(<=3.9), and abnormal range."""

    original_omop_concept_codes = [
        "SNOMED/33747003",
        "LOINC/LP416145-3",
        "LOINC/14749-6",
    ]

    def value_to_label(self, value: Union[float, str], unit: Optional[str]) -> str:
        if str(value).lower() in ["normal", "adequate"]:
            return "normal"
        value = float(value)
        if unit is not None:
            unit = unit.lower()
            if unit.startswith("mg/dl"):
                # mg / dL
                # Original OMOP concept ID: 8840, 9028
                value = value / 18
            elif unit.startswith("mmol/l"):
                # mmol / L (x 18 to get mg/dl)
                # Original OMOP concept ID: 8753
                value = value
            else:
                raise ValueError(f"Unknown unit: {unit}")
        else:
            raise ValueError(f"Unknown unit: {unit}")
        if value < 3:
            return "severe"
        elif value < 3.5:
            return "moderate"
        elif value <= 3.9:
            return "mild"
        return "normal"


class HyponatremiaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for hyponatremia based on blood sodium concentration (mmol/L).
    Thresholds: mild (<=135),moderate(<130),severe(<125), and abnormal range."""

    original_omop_concept_codes = ["LOINC/LG11363-5", "LOINC/2951-2", "LOINC/2947-0"]

    def value_to_label(self, value: Union[float, str], unit: Optional[str]) -> str:
        if str(value).lower() in ["normal", "adequate"]:
            return "normal"
        value = float(value)
        if value < 125:
            return "severe"
        elif value < 130:
            return "moderate"
        elif value <= 135:
            return "mild"
        return "normal"


class AnemiaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for anemia based on hemoglobin levels (g/L).
    Thresholds: mild(<120),moderate(<110),severe(<70), and reference range"""

    original_omop_concept_codes = [
        "LOINC/LP392452-1",
    ]

    def value_to_label(self, value: Union[float, str], unit: Optional[str]) -> str:
        if str(value).lower() in ["normal", "adequate"]:
            return "normal"
        value = float(value)
        if unit is not None:
            unit = unit.lower()
            if unit.startswith("g/dl"):
                # g / dL
                # Original OMOP concept ID: 8713
                # NOTE: This weird *10 / 100 is how Lawrence did it
                value = value * 10
            elif unit.startswith("mg/dl"):
                # mg / dL (divide by 1000 to get g/dL)
                # Original OMOP concept ID: 8840
                # NOTE: This weird *10 / 100 is how Lawrence did it
                value = value / 100
            elif unit.startswith("g/l"):
                value = value
            else:
                raise ValueError(f"Unknown unit: {unit}")
        else:
            raise ValueError(f"Unknown unit: {unit}")
        if value < 70:
            return "severe"
        elif value < 110:
            return "moderate"
        elif value < 120:
            return "mild"
        return "normal"


##########################################################
##########################################################
# "New Diagnosis" Tasks
#
# See: https://arxiv.org/abs/2307.02028
# Citation: Wornow et al.
# EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models
# NeurIPS (2023).
##########################################################
##########################################################


class FirstDiagnosisTimeHorizonCodeLabeler(TimeHorizonEventLabeler):
    """Predict if patient will have their *first* diagnosis of `self.root_concept_code` in the next (1, 365) days.

    Make prediction at 11:59pm on day of discharge from inpatient admission.

    Excludes:
        - Patients who have already had this diagnosis
    """

    root_concept_code: Optional[str] = None  # OMOP concept code for outcome, e.g. "SNOMED/57054005"

    def __init__(
        self,
        ontology: femr.ontology.Ontology,
    ):
        assert (
            self.root_concept_code is not None
        ), "Must specify `root_concept_code` for `FirstDiagnosisTimeHorizonCodeLabeler`"
        self.ontology = ontology
        self.outcome_codes = ontology.get_all_children(self.root_concept_code)
        self.time_horizon: TimeHorizon = TimeHorizon(datetime.timedelta(minutes=1), datetime.timedelta(days=365))

    def get_prediction_times(self, patient: meds.Patient) -> List[datetime.datetime]:
        """Return discharges that occur before first diagnosis of outcome as prediction times."""
        times: List[datetime.datetime] = []
        for __, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
            prediction_time: datetime.datetime = move_datetime_to_end_of_day(discharge_time)
            times.append(prediction_time)
        times = sorted(list(set(times)))

        # Drop all times that occur after first diagnosis
        valid_times: List[datetime.datetime] = []
        outcome_times: List[datetime.datetime] = self.get_outcome_times(patient)
        if len(outcome_times) == 0:
            return times
        else:
            first_diagnosis_time: datetime.datetime = min(outcome_times)
            for t in times:
                if t < first_diagnosis_time:
                    valid_times.append(t)
            return valid_times

    def get_outcome_times(self, patient: meds.Patient) -> List[datetime.datetime]:
        """Return the start times of this patient's events whose `code` is in `self.outcome_codes`."""
        times: List[datetime.datetime] = []
        for event in patient["events"]:
            for m in event["measurements"]:
                if m["code"] in self.outcome_codes:
                    times.append(event["time"])
        return times

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon

    def is_discard_censored_labels(self) -> bool:
        return True

    def allow_same_time_labels(self) -> bool:
        return False


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


class EssentialHypertensionCodeLabeler(FirstDiagnosisTimeHorizonCodeLabeler):
    # n = 4644483
    root_concept_code = "SNOMED/59621000"


class HyperlipidemiaCodeLabeler(FirstDiagnosisTimeHorizonCodeLabeler):
    # n = 3048320
    root_concept_code = "SNOMED/55822004"


##########################################################
##########################################################
# "Chest X-Ray" Tasks
#
# See: https://arxiv.org/abs/2307.02028
# Citation: Wornow et al.
# EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models
# NeurIPS (2023).
##########################################################
##########################################################


CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


class ChexpertLabeler(Labeler):
    """CheXpert labeler.

    Multi-label classification task of patient's radiology reports.
    Make prediction 24 hours before radiology note is recorded.

    Excludes:
        - Radiology reports that are written <=24 hours of a patient's first event (i.e. `patient.events[0].start`)
    """

    def __init__(
        self,
        path_to_chexpert_csv: str,
    ):
        self.path_to_chexpert_csv = path_to_chexpert_csv
        self.prediction_offset: datetime.timedelta = datetime.timedelta(hours=-24)
        self.df_chexpert = pd.read_csv(self.path_to_chexpert_csv, sep="\t").sort_values(by=["start"], ascending=True)

    def label(self, patient: meds.Patient) -> List[meds.Label]:  # type: ignore
        labels: List[meds.Label] = []
        patient_start_time, _ = self.get_patient_start_end_times(patient)
        df_patient = self.df_chexpert[self.df_chexpert["patient_id"] == patient.patient_id].sort_values(
            by=["start"], ascending=True
        )

        for idx, row in df_patient.iterrows():
            label_time: datetime.datetime = datetime.datetime.fromisoformat(row["start"])
            prediction_time: datetime.datetime = label_time + self.prediction_offset
            if prediction_time <= patient_start_time:
                # Exclude radiology reports where our prediction time would be before patient's first timeline event
                continue

            bool_labels = row[CHEXPERT_LABELS].astype(int).to_list()
            label_string = "".join([str(x) for x in bool_labels])
            label_num: int = int(label_string, 2)
            labels.append(
                meds.Label(patient_id=patient["patient_id"], prediction_time=prediction_time, integer_value=label_num)
            )

        return labels

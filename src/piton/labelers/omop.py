"""Labeling functions for OMOP data."""
from __future__ import annotations

import datetime
from collections import deque
from typing import Dict, List, Optional, Set

from .. import Event, Patient
from ..extension import datasets as extension_datasets
from .core import (
    Label,
    Labeler,
    LabelType,
    TimeHorizon,
    TimeHorizonEventLabeler,
)


def get_visit_concepts() -> List[str]:
    return ["Visit/IP"]


def get_inpatient_admission_concepts() -> List[str]:
    return ["Visit/IP"]


def get_death_concepts() -> List[str]:
    return [
        "Death Type/OMOP generated",
        "Condition Type/OMOP4822053",
    ]


def get_inpatient_admission_events(
    patient: Patient, ontology: extension_datasets.Ontology
) -> List[Event]:
    dictionary = ontology.get_dictionary()
    admission_codes: List[int] = [
        dictionary.index(x) for x in get_inpatient_admission_concepts()
    ]
    admissions: List[Event] = []
    for e in patient.events:
        if e.code in admission_codes:
            admissions.append(e)
    return admissions


def map_omop_concept_ids_to_piton_codes(
    ontology: extension_datasets.Ontology, omop_concept_ids: List[int]
) -> List[int]:
    codes: List[int] = []
    for omop_concept_id in omop_concept_ids:
        try:
            piton_code: int = ontology.get_code_from_concept_id(  # type:ignore
                omop_concept_id
            )
            codes.append(piton_code)
        except Exception as e:
            print(f"code {omop_concept_id} not found", e)
    return codes


def _get_all_children(
    ontology: extension_datasets.Ontology, code: int
) -> Set[int]:
    children_code_set = set([code])
    parent_deque = deque([code])

    while len(parent_deque) > 0:
        temp_parent_code: int = parent_deque.popleft()
        for temp_child_code in ontology.get_children(temp_parent_code):
            children_code_set.add(temp_child_code)
            parent_deque.append(temp_child_code)

    return children_code_set


##########################################################
##########################################################
# Abstract classes derived from Labeler
##########################################################
##########################################################


class WithinVisitLabeler(Labeler):
    # TODO - check
    """
    The `VisitLabeler` predicts whether or not a patient experiences a specific event (i.e. has a `code` within
    `self.outcome_codes` within each visit.

    The prediction time is the start of each visit (adjusted by `self.prediction_adjustment_timedelta` if provided)

    IMPORTANT: This labeler assumes that every event has a `event.visit_id` property.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        outcome_codes: List[int],
        prediction_adjustment_timedelta: Optional[datetime.timedelta] = None,
    ):
        dictionary = ontology.get_dictionary()
        self.visit_codes: List[int] = [
            dictionary.index(x) for x in get_visit_concepts()
        ]
        self.outcome_codes: List[int] = outcome_codes
        self.prediction_adjustment_timedelta: Optional[
            datetime.timedelta
        ] = prediction_adjustment_timedelta

    def label(self, patient: Patient) -> List[Label]:
        """Label all visits with whether the patient experiences outcomes
        in `self.outcome_codes` during each visit."""
        # Loop through all visits in patient, check if outcome, if so, mark
        # that it occurred in `visit_to_outcome_count`.
        # NOTE: `visit_to_outcome_count` and `visits` are kept in sync with each other
        # Contains all events whose `event.visit_id`` is referenced in `visit_to_outcome_count`
        visit_to_outcome_count: Dict[int, int] = {}
        visits: List[Event] = []
        for event in patient.events:
            if event.visit_id is None:
                # Ignore events without a `visit_id`
                continue
                # raise RuntimeError(
                #     f"Event with code={event.code} at time={event.start} for patient id={patient.patient_id}"
                #     f" does not have a `visit_id`"
                # )
            if (
                event.visit_id not in visit_to_outcome_count
                and event.code in self.visit_codes
            ):
                visit_to_outcome_count[event.visit_id] = 0
                visits.append(event)
            elif event.code in self.outcome_codes:
                if not event.visit_id in visit_to_outcome_count:
                    raise RuntimeError(
                        f"Outcome event with code={event.code} at time={event.start} for patient id={patient.patient_id}"
                        f" occurred before its corresponding admission event with visit_id {event.visit_id}"
                    )
                visit_to_outcome_count[event.visit_id] += 1

        # Generate labels
        labels: List[Label] = []
        for event in visits:
            is_outcome_occurs: bool = visit_to_outcome_count[event.visit_id] > 0
            prediction_time: datetime.datetime = (
                (event.start + self.prediction_adjustment_timedelta)
                if self.prediction_adjustment_timedelta is not None
                else event.start
            )
            labels.append(Label(prediction_time, is_outcome_occurs))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


##########################################################
##########################################################
# Abstract classes derived from TimeHorizonEventLabeler
##########################################################
##########################################################


class CodeLabeler(TimeHorizonEventLabeler):
    """Apply a label based on 1+ outcome_codes' occurrence(s) over a fixed time horizon."""

    def __init__(
        self,
        outcome_codes: List[int],
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[int]] = None,
    ):
        """Create a CodeLabeler, which labels events whose index in your Ontology is in `self.outcome_codes`

        Args:
            prediction_codes (List[int]): Events that count as an occurrence of the outcome.
            time_horizon (TimeHorizon): An interval of time. If the event occurs during this time horizon, then
                the label is TRUE. Otherwise, FALSE.
            prediction_codes (Optional[List[int]]): If not None, limit events at which you make predictions to
                only events with an `event.code` in these codes.

        Raises:
            ValueError: Raised if there are multiple unique codes that map to the death code
        """
        self.outcome_codes: List[int] = outcome_codes
        self.time_horizon: TimeHorizon = time_horizon
        self.prediction_codes: Optional[List[int]] = prediction_codes

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return each event's start time as the time to make a prediction.
        Default to all events whose `code` is in `self.prediction_codes`."""
        return [
            # datetime.datetime.strptime(
            #     # TODO - Why add 23:59:00?
            #     str(e.start)[:10] + " 23:59:00", "%Y-%m-%d %H:%M:%S"
            # )
            e.start
            for e in patient.events
            if (self.prediction_codes is None)
            or (e.code in self.prediction_codes)
        ]

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of this patient's events whose `code` is in `self.outcome_codes`."""
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.code in self.outcome_codes:
                times.append(event.start)
        return times


class OMOPConceptCodeLabeler(CodeLabeler):
    # TODO - check
    """Same as CodeLabeler, but add the extra step of mapping OMOP concept IDs
    (stored in `omop_concept_ids`) to Piton codes (stored in `codes`)."""

    original_omop_concept_ids: List[int] = []
    omop_concept_ids: List[int] = []

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[int]] = None,
    ):
        outcome_codes: List[int] = map_omop_concept_ids_to_piton_codes(
            ontology, self.omop_concept_ids
        )
        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
            prediction_codes=prediction_codes,
        )


##########################################################
##########################################################
# Labeling functions derived from CodeLabeler
##########################################################
##########################################################


class MortalityCodeLabeler(CodeLabeler):
    """Apply a label for whether or not a patient dies within the `time_horizon`.
    Make prediction at admission time.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[int]] = None,
    ):
        """Create a Mortality labeler."""
        dictionary = ontology.get_dictionary()
        outcome_codes = [dictionary.index(x) for x in get_death_concepts()]

        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
            prediction_codes=prediction_codes,
        )


class LupusCodeLabeler(CodeLabeler):
    """
    Label if patient is diagnosed with Lupus.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[int]] = None,
    ):
        dictionary = ontology.get_dictionary()
        codes = set()
        snomed_codes: List[str] = ["55464009", "201436003"]
        for code in snomed_codes:
            codes |= _get_all_children(
                ontology, dictionary.index("SNOMED/" + code)
            )

        super().__init__(
            outcome_codes=list(codes),
            time_horizon=time_horizon,
            prediction_codes=prediction_codes,
        )


class HighHbA1cCodeLabeler(Labeler):
    """
    The high HbA1c labeler tries to predict whether a non-diabetic patient will test as diabetic.
    Note: This labeler will only trigger at most once every 6 months.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        last_trigger_timedelta: datetime.timedelta = datetime.timedelta(
            days=180
        ),
    ):
        """Create a High HbA1c (i.e. diabetes) labeler."""
        self.last_trigger_timedelta = last_trigger_timedelta

        HbA1c_str: str = "LOINC/4548-4"
        self.hba1c_lab_code = ontology.get_dictionary().index(HbA1c_str)

        diabetes_str: str = "SNOMED/44054006"
        diabetes_code = ontology.get_dictionary().index(diabetes_str)
        self.diabetes_codes = _get_all_children(ontology, diabetes_code)

    def label(self, patient: Patient) -> List[Label]:
        if len(patient.events) == 0:
            return []

        high_cutoff_threshold: float = 6.5
        labels: List[Label] = []
        last_trigger: Optional[datetime.datetime] = None

        first_diabetes_code_date = None
        for event in patient.events:
            if event.code in self.diabetes_codes:
                first_diabetes_code_date = event.start
                break

        for event in patient.events:

            if (
                first_diabetes_code_date is not None
                and event.start > first_diabetes_code_date
            ):
                break

            if event.value is None or type(event.value) is memoryview:
                continue

            if event.code == self.hba1c_lab_code:
                is_diabetes = float(event.value) > high_cutoff_threshold
                if last_trigger is None or (
                    event.start - last_trigger > self.last_trigger_timedelta
                ):
                    labels.append(
                        Label(
                            time=event.start - datetime.timedelta(minutes=1),
                            value=is_diabetes,
                        )
                    )
                    last_trigger = event.start

                if is_diabetes:
                    break

        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


##########################################################
##########################################################
# Labeling functions derived from OMOPConceptCodeLabeler
##########################################################
##########################################################


class HypoglycemiaCodeLabeler(OMOPConceptCodeLabeler):
    # TODO - check
    """Apply a label for whether a patient has at 1+ explicitly
    coded occurrence(s) of Hypoglycemia in `time_horizon`."""
    # fmt: off
    original_omop_concept_ids = [
        380688, 4226798, 36714116, 24609, 4029423, 45757363, 4096804, 4048805, 4228112, 23034, 4029424, 45769876,
    ]
    omop_concept_ids = [
        45769876, 23034, 4029423, 4226798, 24609, 380688, 4228112, 36714116, 4048805, 44789318,
        4030181, 4016047, 42536567, 4129522, 4029426, 44789319, 45769875, 4171099, 4056338, 36676692,
        4008578, 4173186, 4030070, 45757363, 4320478, 4029422, 4029424, 3169474, 45772060, 4096804,
        3185010, 4029425, 42536568, 4129520, 4008577, 4034969, 42572837, 4340777, 44809809, 45757362,
        4030182,
    ]
    # fmt: on


class AKICodeLabeler(OMOPConceptCodeLabeler):
    # TODO - check
    """Apply a label for whether a patient has at 1+ explicitly
    coded occurrence(s) of AKI in `time_horizon`."""
    # fmt: off
    original_omop_concept_ids = [
        197320, 432961, 444044
    ]
    omop_concept_ids = [
        444044, 197320, 432961, 43530914, 44809173, 4143190, 196455, 37395520, 37395518, 44809170,
        4030519, 4215648, 37016366, 197329, 4180453, 4160274, 45757466, 4126424, 37116834, 4139414,
        44809063, 37116432, 4125969, 4066005, 37116430, 37116431, 4119093, 4195297, 44808340, 4200639,
        44808822, 36716182, 44808338, 44808823, 37395514, 44813790, 45757442, 4128029, 4151112, 4265212,
        4070939, 44809061, 44809286, 4232873, 44808744, 37395516, 4125968, 4149888, 43530928, 3180540,
        4306513, 44813789, 4126426, 4126427, 44808128, 45757398, 37395519, 4308408, 4128067, 37395517,
        36716312, 4228827, 4181114, 44809062, 36716183, 4311129, 44808821, 37395521, 4264681, 43530935,
        4066405,
    ]
    # fmt: on


class AnemiaCodeLabeler(OMOPConceptCodeLabeler):
    # TODO - check
    """Apply a label for whether a patient has at 1+ explicitly
    coded occurrence(s) of Anemia in `time_horizon`."""
    # fmt: off
    original_omop_concept_ids = [
        439777, 37018722, 37017132, 35624756, 4006467, 37398911, 37395652,
    ]
    omop_concept_ids = [
        4144077, 4312021, 42537687, 37399453, 4097961, 4225810, 45757092, 4203291, 37018722, 4190190,
        4098013, 4125491, 4085853, 444289, 4121114, 4173192, 4242755, 4032006, 4035282, 434701,
        4323223, 36715492, 4234973, 4120454, 4099224, 4034711, 3193947, 4131915, 4115393, 4221567,
        37399441, 4146088, 4247416, 4151502, 4009645, 4219359, 434622, 4254380, 4098027, 4308125,
        4178731, 4290596, 4254249, 4135931, 4173191, 37016151, 4281394, 4130193, 4215791, 22281,
        45773534, 4116343, 4241982, 4311676, 4159966, 4058246, 4145893, 4200254, 4280070, 4244129,
        4039536, 4121109, 4238731, 4159651, 37397537, 4297537, 4185253, 4098754, 4189321, 35623407,
        4090690, 4312853, 4190771, 37019055, 4121107, 4098019, 437834, 4196102, 4250028, 434156,
        439777, 42872405, 4146771, 4184758, 4171026, 443738, 435789, 4139550, 37019193, 4099508,
        45768813, 4338976, 4303199, 4157495, 4009787, 4101582, 4098760, 4302298, 37312032, 77662,
        37116298, 4269764, 4101573, 4035988, 4063216, 4209139, 4009785, 37017607, 4174376, 4269919,
        4150499, 4143895, 4149681, 4125628, 136949, 4009788, 4291002, 4184603, 4298690, 4315393,
        4149042, 36674478, 37110336, 45757189, 4209085, 37397197, 37116297, 433603, 4098753, 4098749,
        4218974, 4198102, 4206913, 36715494, 436659, 4311538, 4098007, 4008273, 4228681, 4168911,
        4181143, 4008663, 4009304, 4101001, 4201444, 4327045, 36713573, 36715351, 36716460, 36713763,
        4101578, 4214023, 4223031, 45768941, 4031699, 37110923, 4098758, 26942, 4101458, 4125496,
        4125493, 4327824, 37395652, 35624756, 37398911, 37017132, 4347537, 4160238, 4104541, 4160887,
        4278920, 4284415, 4102056, 4131919, 36713571, 432967, 4093515, 432452, 4177177, 4297017,
        22288, 4278816, 4009786, 4039785, 37204551, 4171201, 4131128, 4130191, 4018378, 4122932,
        4339113, 4079852, 4235788, 4313581, 4188474, 4098145, 36713572, 765176, 4034963, 4172446,
        4213893, 40599994, 4207240, 4187768, 4286660, 4313413, 437090, 37117164, 4312008, 4099889,
        4232474, 4105919, 36713112, 4082253, 4195171, 4100991, 4319914, 4130189, 4209094, 4024671,
        4035974, 4143351, 137829, 440977, 4138560, 4143167, 4298813, 440218, 4280354, 4031662,
        4031827, 4132431, 4184200, 36674474, 45773071, 4139549, 4035316, 4222774, 4195271, 40485018,
        36715580, 4099005, 4258685, 4130068, 4243831, 441269, 4098017, 4122928, 4308062, 42599242,
        4173028, 4098740, 4268894, 432875, 4196206, 37110070, 4287844, 4100962, 438722, 36674519,
        4130067, 45768812, 4130195, 23988, 4261354, 4097214, 4100987, 4156842, 4228194, 37204236,
        4168441, 432295, 4130678, 4098009, 36676579, 4219253, 4211348, 4306199, 4009306, 4098008,
        437247, 4267646, 4329173, 4228444, 30978, 4264909, 4131127, 4330322, 36713168, 4243676,
        4082917, 36715009, 4320651, 4147911, 4130192, 37016121, 4139775, 434894, 4120450, 4159967,
        40481595, 4088386, 4278669, 4121110, 4307799, 4121106, 4130194, 4021911, 4071444, 4155187,
        432968, 4146086, 35623048, 4143629, 4188208, 4314111, 4198185, 42536531, 4031204, 321263,
        4187773, 4122930, 760845, 4098131, 36715493, 30683, 4100998, 4060269, 432588, 4311397,
        4022198, 443961, 4301602, 4307469, 4231453, 37204308, 4228805, 40491316, 36714533, 4215784,
        4092893, 4145277, 4194519, 436820, 4122929, 4150547, 4218794, 4081520, 42596488, 4298975,
        4098762, 43022052, 442923, 4101917, 4338370, 4122924, 28396, 440979, 4038772, 4271197,
        4019001, 4096927, 4300295, 36714288, 4120451, 36714258, 4071073, 4101583, 4098627, 138723,
        42536530, 4100985, 37119138, 4336555, 4267282, 4122931, 4144956, 4098028, 4139552, 4178677,
        435503, 4139556, 140681, 4147600, 37111627, 4297024, 4290233, 4262948, 4003185, 4158891,
        4260689, 4204062, 4186108, 4002495, 4148471, 4231887, 4122923, 45766392, 442922, 4223896,
        37312165, 4203904, 3194085, 197253, 4147491, 4176884, 4144811, 4206007, 4218100, 4044728,
        4098746, 4287402, 44810002, 37396761, 4230266, 42599372, 4130680, 4287574, 42572621, 4098759,
        4114026, 35624317, 441258, 4121112, 4187355, 36716203, 4185887, 37116300, 4122079, 4219853,
        4168286, 4271753, 4282785, 4023612, 45772084, 315523, 4045142, 42539413, 4120448, 4264017,
        4101000, 4063215, 4003186, 37311954, 4242111, 4046563, 4146936, 4246105, 37116301, 4273847,
        4149183, 42596489, 433168, 42596487, 760844, 4147365, 434616, 4311673, 4098747, 4140101,
        4006468, 37110727, 4099603, 4131914, 4181743, 4122926, 4131917, 4062928, 4204900, 4263315,
        4248254, 36716126, 4098018, 4032352, 4042927, 45757146, 432282, 4318674, 36713970, 36716259,
        4139551, 4131913, 4100992, 36714965, 4122927, 4168772, 4258261, 4135713, 4021466, 4079181,
        436956, 4174412, 4213820, 37117740, 42596486, 4121113, 4015896, 4052711, 4132085, 42596091,
        4174701, 4101584, 4247579, 4230471, 4006467, 4220697, 4121115, 37397036, 4175331, 4243950,
        4238904, 46272744, 4061151, 24909, 4265915, 4009784, 36717581, 4345236, 4193514, 444238,
        4339722, 4125633, 37017285, 4187256, 37109777, 4187034,
    ]
    # fmt: on


class HyperkalemiaCodeLabeler(OMOPConceptCodeLabeler):
    # TODO - check
    """Apply a label for whether a patient has at 1+ explicitly
    coded occurrence(s) of Hyperkalemia in `time_horizon`."""
    # fmt: off
    original_omop_concept_ids = [
        434610
    ]
    omop_concept_ids = [
        434610, 4030355, 4185833, 4183002, 4028948, 4029592, 4201725, 4029591, 4071744, 4236458, 4146120,
    ]
    # fmt: on


class HyponatremiaCodeLabeler(OMOPConceptCodeLabeler):
    # TODO - check
    """Apply a label for whether a patient has at 1+ explicitly
    coded occurrence(s) of Hyponatremia in `time_horizon`."""
    # fmt: off
    original_omop_concept_ids = [
        435515, 4232311
    ]
    omop_concept_ids: List[int] = [
        435515, 4232311, 4029590, 4028947, 4225276, 4227093, 4252414, 4164126, 4175012, 4177324, 4048926,
        4253214, 4139394, 4215499, 4028946,
    ]
    # fmt: on


class ThrombocytopeniaCodeLabeler(OMOPConceptCodeLabeler):
    # TODO - check
    """Apply a label for whether a patient has at 1+ explicitly
    coded occurrence(s) of Thrombocytopenia in `time_horizon`."""
    # fmt: off
    original_omop_concept_ids = [
        432870,
    ]
    omop_concept_ids = [
        432870, 4100998, 4156233, 42572969, 4301602, 4300464, 35625536, 4101583, 138723, 4139555,
        440372, 36674972, 4098028, 4219476, 46272950, 140681, 437242, 4186108, 4148471, 4121265,
        4098148, 42536958, 37312165, 4345345, 4230266, 4301128, 4272928, 4027374, 4133981, 440982,
        4173278, 4235220, 37018663, 4123075, 4204900, 4226905, 36713970, 4258261, 436956, 4239484,
        36715586, 4137430, 4345236, 4121264, 37396502, 4159736, 4133984, 4225810, 4159749, 435076,
        40321716, 4234973, 4119134, 4292425, 4140545, 4146088, 4221109, 37110394, 4098027, 4305588,
        37016151, 4123076, 4233407, 4159966, 4049028, 37204548, 4197574, 35623407, 37209558, 4185078,
        36716406, 4028065, 4234257, 37019055, 4184758, 4125494, 44782445, 4101582, 432881, 36715053,
        37017607, 37204520, 433749, 4133983, 4298690, 37116398, 4338386, 42537688, 4147049, 4172999,
        4207877, 36716460, 4230228, 4031699, 4125496, 4101603, 37204478, 318397, 36713443, 4177177,
        4103532, 37204551, 4314802, 4311682, 4299560, 36717326, 4098145, 4120620, 4048742, 37117164,
        4214947, 441264, 4048930, 36713112, 137829, 4247776, 4195579, 4184200, 36674474, 4123074,
        37017165, 4172008, 4009307, 4218171, 4000065, 4211348, 36716047, 37016797, 4316372, 4292531,
        4102469, 4264464, 42598452, 4145458, 4077348, 4082738, 4166754, 4146086, 4264166, 4188208,
        4187773,
    ]
    # fmt: on


class NeutropeniaCodeLabeler(OMOPConceptCodeLabeler):
    """Apply a label for whether a patient has at 1+ explicitly
    coded occurrence(s) of Neutkropenia in `time_horizon`."""

    # fmt: off
    original_omop_concept_ids = [
        301794, 320073,
    ]
    omop_concept_ids = [
        37117238, 37205095, 36716754, 36716753, 4095623, 320073, 42596532, 42536512, 42539535, 4101126,
    ]
    # fmt: on


##########################################################
##########################################################
# Other labeling functions
##########################################################
##########################################################


class OpioidOverdoseLabeler(TimeHorizonEventLabeler):
    """
    TODO - check
    The opioid overdose labeler predicts whether or not an opioid overdose will occur in the time horizon
    after being prescribed opioids.
    It is conditioned on the patient being prescribed opioids.
    """

    def __init__(
        self, ontology: extension_datasets.Ontology, time_horizon: TimeHorizon
    ):
        self.time_horizon: TimeHorizon = time_horizon

        dictionary = ontology.get_dictionary()
        icd9_codes: List[str] = [
            "E850.0",
            "E850.1",
            "E850.2",
            "965.00",
            "965.01",
            "965.02",
            "965.09",
        ]
        icd10_codes: List[str] = ["T40.0", "T40.1", "T40.2", "T40.3", "T40.4"]

        self.overdose_codes: Set[int] = set()
        for code in icd9_codes:
            self.overdose_codes |= _get_all_children(
                ontology, dictionary.index("ICD9CM/" + code)
            )
        for code in icd10_codes:
            self.overdose_codes |= _get_all_children(
                ontology, dictionary.index("ICD10CM/" + code)
            )

        self.opioid_codes = _get_all_children(
            ontology, dictionary.index("ATC/N02A")
        )

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of this patient's events whose `code` is in `self.overdose_codes`."""
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.code in self.overdose_codes:
                times.append(event.start)
        return times

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a sorted list containing the datetimes at which we'll make a prediction."""
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.code in self.opioid_codes:
                times.append(event.start)
        return times

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class CeliacTestLabeler(Labeler):
    # TODO - check
    """
    The Celiac test labeler predicts whether or not a celiac test will be positive or negative.
    The prediction time is 24 hours before the lab results come in.
    Note: This labeler excludes patients who either already had a celiac test or were previously diagnosed.
    """

    def __init__(
        self, ontology: extension_datasets.Ontology, time_horizon: TimeHorizon
    ):
        dictionary = ontology.get_dictionary()
        self.lab_codes = _get_all_children(
            ontology, dictionary.index("LNC/31017-7")
        )
        self.celiac_codes = _get_all_children(
            ontology, dictionary.index("ICD9CM/579.0")
        ) | _get_all_children(ontology, dictionary.index("ICD10CM/K90.0"))

        self.pos_value = "Positive"
        self.neg_value = "Negative"

    def label(self, patient: Patient) -> List[Label]:
        if len(patient.events) == 0:
            return []

        for event in patient.events:
            if event.code in self.celiac_codes:
                # This patient already has Celiacs
                return []
            if event.code in self.lab_codes and event.value in [
                self.pos_value,
                self.neg_value,
            ]:
                # This patient got a Celiac lab test result
                # We'll return the Label 24 hours prior
                return [
                    Label(
                        event.start - datetime.timedelta(hours=24),
                        event.value == self.pos_value,
                    )
                ]
        return []

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class IsMaleLabeler(Labeler):
    """Apply a label for whether or not a patient is male or not.

    The prediction time is on admission.

    This is primarily intended as a "debugging" labeler that should be "trivial" and get 1.0 AUROC.

    """

    def __init__(self, ontology: extension_datasets.Ontology):
        self.male_code: int = ontology.get_dictionary().index("Gender/M")

    def label(self, patient: Patient) -> List[Label]:
        """Label this patient as Male (TRUE) or not (FALSE)."""
        # Determine if patient is male
        is_male: bool = self.male_code in [e.code for e in patient.events]

        # Apply `is_male` label to every admission
        labels: List[Label] = []
        for event in patient.events:
            if event.code in get_inpatient_admission_concepts():
                labels.append(Label(time=event.start, value=is_male))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


if __name__ == "__main__":
    pass

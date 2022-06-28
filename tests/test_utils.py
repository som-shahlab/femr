import pytest
import datetime

import ehr_ml.timeline
import ehr_ml.utils

from collections import namedtuple
from dataclasses import dataclass

from typing import List


@dataclass
class DummyObservationWithValue:
    code: int
    numeric_value: float
    is_text: bool


@dataclass
class DummyDay:
    """Class for keeping track of an item in inventory."""

    age: int
    date: datetime.date
    observations: List[int]
    observations_with_values: List[DummyObservationWithValue]


@dataclass
class DummyPatient:
    patient_id: int
    days: List[DummyDay]


def create_dummy_patient(patient_id, days_info):
    def init_date(date, year, month, day):
        date.year = year
        date.month = month
        date.day = day

    days = []

    for date, observations, observations_with_values in days_info:
        days.append(
            DummyDay(
                age=(
                    datetime.date(*date) - datetime.date(*days_info[0][0])
                ).days,
                date=datetime.date(*date),
                observations=observations,
                observations_with_values=[
                    DummyObservationWithValue(
                        code=c, numeric_value=v, is_text=False
                    )
                    for c, v in observations_with_values.items()
                ],
            )
        )
    return DummyPatient(patient_id=patient_id, days=days)


@pytest.fixture(scope="session")
def dummy_patient():
    return create_dummy_patient(
        234123,
        [
            ((1994, 2, 9), [0, 4, 12], {}),
            ((1997, 2, 9), [0, 34, 129], {909: 0}),
            ((2000, 2, 9), [0, 34, 129], {909: 2}),
        ],
    )


class DummyDictionary:
    def map(self, code):
        return None


@pytest.fixture(scope="session")
def dummy_ontologies():
    class DummyOntologies:
        def get_recorded_date_codes(self):
            return range(1000)

    return DummyOntologies()


@pytest.fixture(scope="session")
def dummy_timeline():
    dictionary = DummyDictionary()

    class DummyTimeline:
        def get_dictionary(self):
            return dictionary

        def get_patient_ids(self):
            return [234123]

        def get_patient(self, patient_id, end_date=None):
            return create_dummy_patient(
                patient_id,
                [
                    ((1994, 2, 9), [0, 4, 12], {}),
                    ((1997, 2, 9), [0, 34, 129], {909: 0}),
                    ((2000, 2, 9), [0, 34, 129], {909: 2}),
                ],
            )

    return DummyTimeline()

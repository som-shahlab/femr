import datetime

from femr_test_tools import create_patients_dataset

import femr.models.processor
import femr.models.tasks


class DummyTokenizer:
    def __init__(self, is_hierarchical: bool = False):
        self.is_hierarchical = is_hierarchical
        self.ontology = None
        self.vocab_size = 100

    def start_patient(self):
        pass

    def get_feature_codes(self, time, measurement):
        if measurement["code"] == "SNOMED/184099003":
            return [1], None
        else:
            return [int(measurement["code"])], None

    def normalize_age(self, age):
        return 0.5


def assert_two_batches_equal_third(batch1, batch2, batch3):
    """This asserts that batch1 + batch2 = batchs3"""
    assert batch3["patient_ids"].tolist() == batch1["patient_ids"].tolist() + batch2["patient_ids"].tolist()

    assert (
        batch3["transformer"]["ages"].tolist()
        == batch1["transformer"]["ages"].tolist() + batch2["transformer"]["ages"].tolist()
    )
    assert (
        batch3["transformer"]["timestamps"].tolist()
        == batch1["transformer"]["timestamps"].tolist() + batch2["transformer"]["timestamps"].tolist()
    )

    # Checking the label indices is a bit more involved as we have to map to age/patient id and then check that
    target_label_ages = []
    target_label_patient_ids = []

    for label_index in batch1["transformer"]["label_indices"].tolist():
        target_label_ages.append(batch1["transformer"]["ages"][label_index])
        target_label_patient_ids.append(batch1["patient_ids"][label_index])

    for label_index in batch2["transformer"]["label_indices"].tolist():
        target_label_ages.append(batch2["transformer"]["ages"][label_index])
        target_label_patient_ids.append(batch2["patient_ids"][label_index])

    actual_label_ages = []
    actual_label_patient_ids = []

    for label_index in batch3["transformer"]["label_indices"].tolist():
        actual_label_ages.append(batch3["transformer"]["ages"][label_index])
        actual_label_patient_ids.append(batch3["patient_ids"][label_index])

    assert target_label_ages == actual_label_ages
    assert target_label_patient_ids == actual_label_patient_ids

    if "tokens" in batch3["transformer"]:
        assert (
            batch3["transformer"]["tokens"].tolist()
            == batch1["transformer"]["tokens"].tolist() + batch2["transformer"]["tokens"].tolist()
        )


def test_two_patients_concat_no_task():
    tokenizer = DummyTokenizer()

    fake_patients = create_patients_dataset(10)

    fake_patient1 = fake_patients[1]
    fake_patient2 = fake_patients[5]

    creator = femr.models.processor.BatchCreator(tokenizer)

    creator.start_batch()
    creator.add_patient(fake_patient1)

    data_for_patient1 = creator.get_batch_data()

    creator.start_batch()
    creator.add_patient(fake_patient2)

    data_for_patient2 = creator.get_batch_data()

    creator.start_batch()
    creator.add_patient(fake_patient1)
    creator.add_patient(fake_patient2)

    data_for_patients = creator.get_batch_data()

    assert_two_batches_equal_third(data_for_patient1, data_for_patient2, data_for_patients)


def test_split_patients_concat_no_task():
    tokenizer = DummyTokenizer()

    fake_patients = create_patients_dataset(10)

    fake_patient = fake_patients[1]

    creator = femr.models.processor.BatchCreator(tokenizer)

    creator.start_batch()
    creator.add_patient(fake_patient)

    data_for_patient = creator.get_batch_data()

    length = len(data_for_patient["transformer"]["timestamps"])

    creator.start_batch()
    creator.add_patient(fake_patient, offset=0, max_length=length // 2)

    data_for_part1 = creator.get_batch_data()

    creator.start_batch()
    creator.add_patient(fake_patient, offset=length // 2, max_length=None)

    data_for_part2 = creator.get_batch_data()

    assert_two_batches_equal_third(data_for_part1, data_for_part2, data_for_patient)


def test_two_patients_concat_task():
    tokenizer = DummyTokenizer()

    fake_patients = create_patients_dataset(10)

    task = femr.models.tasks.LabeledPatientTask(
        [
            {"patient_id": 1, "prediction_time": datetime.datetime(2011, 7, 6)},
            {"patient_id": 1, "prediction_time": datetime.datetime(2017, 1, 1)},
            {"patient_id": 5, "prediction_time": datetime.datetime(2011, 11, 6)},
            {"patient_id": 5, "prediction_time": datetime.datetime(2017, 2, 1)},
        ]
    )

    fake_patient1 = fake_patients[1]
    fake_patient2 = fake_patients[5]

    creator = femr.models.processor.BatchCreator(tokenizer, task=task)

    creator.start_batch()
    creator.add_patient(fake_patient1)

    data_for_patient1 = creator.get_batch_data()

    creator.start_batch()
    creator.add_patient(fake_patient2)

    data_for_patient2 = creator.get_batch_data()

    creator.start_batch()
    creator.add_patient(fake_patient1)
    creator.add_patient(fake_patient2)

    data_for_patients = creator.get_batch_data()

    assert_two_batches_equal_third(data_for_patient1, data_for_patient2, data_for_patients)


def test_split_patients_concat_task():
    tokenizer = DummyTokenizer()

    fake_patients = create_patients_dataset(10)

    fake_patient = fake_patients[1]

    task = femr.models.tasks.LabeledPatientTask(
        [
            {"patient_id": 1, "prediction_time": datetime.datetime(2011, 7, 6)},
            {"patient_id": 1, "prediction_time": datetime.datetime(2017, 1, 1)},
        ]
    )

    creator = femr.models.processor.BatchCreator(tokenizer, task=task)

    creator.start_batch()
    creator.add_patient(fake_patient)

    data_for_patient = creator.get_batch_data()

    length = len(data_for_patient["transformer"]["timestamps"])

    creator.start_batch()
    creator.add_patient(fake_patient, offset=0, max_length=length // 2)

    data_for_part1 = creator.get_batch_data()

    creator.start_batch()
    creator.add_patient(fake_patient, offset=length // 2, max_length=None)

    data_for_part2 = creator.get_batch_data()

    assert_two_batches_equal_third(data_for_part1, data_for_part2, data_for_patient)

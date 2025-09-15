import datetime

import meds
from femr_test_tools import create_subjects_dataset

import femr.models.processor
import femr.models.tasks
import femr.models.tokenizer


class DummyTokenizer(femr.models.tokenizer.HierarchicalTokenizer):
    def __init__(self, is_hierarchical: bool = True):
        self.is_hierarchical = is_hierarchical
        self.ontology = None
        self.vocab_size = 100

    def start_subject(self):
        pass

    def get_feature_codes(self, event):
        if event.code == meds.birth_code:
            return [1], [1]
        else:
            return [int(event.code)], [1]

    def get_time_data(self, age: datetime.timedelta, delta: datetime.timedelta) -> float:
        return [1, 1, 1, 1]

    def normalize_age(self, age):
        return 0.5


def assert_two_batches_equal_third(batch1, batch2, batch3):
    """This asserts that batch1 + batch2 = batchs3"""
    assert batch3["subject_ids"].tolist() == batch1["subject_ids"].tolist() + batch2["subject_ids"].tolist()

    batch3["transformer"]["ages"][len(batch1["transformer"]["ages"])] = 0
    batch3["transformer"]["timestamps"][len(batch1["transformer"]["ages"])] = batch3["transformer"]["timestamps"][0]

    assert (
        batch3["transformer"]["ages"].tolist()
        == batch1["transformer"]["ages"].tolist() + batch2["transformer"]["ages"].tolist()
    )
    assert (
        batch3["transformer"]["timestamps"].tolist()
        == batch1["transformer"]["timestamps"].tolist() + batch2["transformer"]["timestamps"].tolist()
    )

    # Checking the label indices is a bit more involved as we have to map to age/subject id and then check that
    target_label_ages = []
    target_label_subject_ids = []

    for label_index in batch1["transformer"]["label_indices"].tolist():
        target_label_ages.append(batch1["transformer"]["ages"][label_index])
        target_label_subject_ids.append(batch1["subject_ids"][label_index])

    for label_index in batch2["transformer"]["label_indices"].tolist():
        target_label_ages.append(batch2["transformer"]["ages"][label_index])
        target_label_subject_ids.append(batch2["subject_ids"][label_index])

    actual_label_ages = []
    actual_label_subject_ids = []

    for label_index in batch3["transformer"]["label_indices"].tolist():
        actual_label_ages.append(batch3["transformer"]["ages"][label_index])
        actual_label_subject_ids.append(batch3["subject_ids"][label_index])

    assert target_label_ages == actual_label_ages
    assert target_label_subject_ids == actual_label_subject_ids

    batch3["transformer"]["hierarchical_tokens"][len(batch1["transformer"]["hierarchical_tokens"])] = batch3[
        "transformer"
    ]["hierarchical_tokens"][0]

    assert (
        batch3["transformer"]["hierarchical_tokens"].tolist()
        == batch1["transformer"]["hierarchical_tokens"].tolist() + batch2["transformer"]["hierarchical_tokens"].tolist()
    )


def test_two_subjects_concat_no_task():
    tokenizer = DummyTokenizer()

    fake_subjects = create_subjects_dataset(10)

    fake_subject1 = fake_subjects[1]
    fake_subject2 = fake_subjects[5]

    creator = femr.models.processor.BatchCreator(tokenizer)

    creator.start_batch()
    creator.add_subject(fake_subject1)

    data_for_subject1 = creator.get_batch_data()

    creator.start_batch()
    creator.add_subject(fake_subject2)

    data_for_subject2 = creator.get_batch_data()

    creator.start_batch()
    creator.add_subject(fake_subject1)
    creator.add_subject(fake_subject2)

    data_for_subjects = creator.get_batch_data()

    assert_two_batches_equal_third(data_for_subject1, data_for_subject2, data_for_subjects)


def test_split_subjects_concat_no_task():
    tokenizer = DummyTokenizer()

    fake_subjects = create_subjects_dataset(10)

    fake_subject = fake_subjects[1]

    creator = femr.models.processor.BatchCreator(tokenizer)

    creator.start_batch()
    creator.add_subject(fake_subject)

    data_for_subject = creator.get_batch_data()

    length = len(data_for_subject["transformer"]["timestamps"])

    creator.start_batch()
    creator.add_subject(fake_subject, offset=0, max_length=length // 2)

    data_for_part1 = creator.get_batch_data()

    creator.start_batch()
    creator.add_subject(fake_subject, offset=length // 2, max_length=None)

    data_for_part2 = creator.get_batch_data()

    assert_two_batches_equal_third(data_for_part1, data_for_part2, data_for_subject)


def test_two_subjects_concat_task():
    tokenizer = DummyTokenizer()

    fake_subjects = create_subjects_dataset(10)

    labels = [
        {"subject_id": 1, "prediction_time": datetime.datetime(2011, 7, 6)},
        {"subject_id": 1, "prediction_time": datetime.datetime(2017, 1, 1)},
        {"subject_id": 5, "prediction_time": datetime.datetime(2011, 11, 6)},
        {"subject_id": 5, "prediction_time": datetime.datetime(2017, 2, 1)},
    ]
    labels = [meds.Label(**label) for label in labels]

    task = femr.models.tasks.LabeledSubjectTask(labels)

    fake_subject1 = fake_subjects[1]
    fake_subject2 = fake_subjects[5]

    creator = femr.models.processor.BatchCreator(tokenizer, task=task)

    creator.start_batch()
    creator.add_subject(fake_subject1)

    data_for_subject1 = creator.get_batch_data()

    creator.start_batch()
    creator.add_subject(fake_subject2)

    data_for_subject2 = creator.get_batch_data()

    creator.start_batch()
    creator.add_subject(fake_subject1)
    creator.add_subject(fake_subject2)

    data_for_subjects = creator.get_batch_data()

    assert_two_batches_equal_third(data_for_subject1, data_for_subject2, data_for_subjects)


def test_split_subjects_concat_task():
    tokenizer = DummyTokenizer()

    fake_subjects = create_subjects_dataset(10)

    fake_subject = fake_subjects[1]

    task = femr.models.tasks.LabeledSubjectTask(
        [
            {"subject_id": 1, "prediction_time": datetime.datetime(2010, 8, 6)},
            {"subject_id": 1, "prediction_time": datetime.datetime(2017, 1, 1)},
        ]
    )

    creator = femr.models.processor.BatchCreator(tokenizer, task=task)

    creator.start_batch()
    creator.add_subject(fake_subject)

    data_for_subject = creator.get_batch_data()

    length = len(data_for_subject["transformer"]["timestamps"])

    creator.start_batch()
    creator.add_subject(fake_subject, offset=0, max_length=length // 2)

    data_for_part1 = creator.get_batch_data()

    creator.start_batch()
    creator.add_subject(fake_subject, offset=length // 2, max_length=None)

    data_for_part2 = creator.get_batch_data()

    assert_two_batches_equal_third(data_for_part1, data_for_part2, data_for_subject)

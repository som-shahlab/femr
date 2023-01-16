from __future__ import annotations

import datetime

import extension.datasets as m
import numpy as np
import pytest

import piton

def test_helper(tmp_path, capsys):
    with capsys.disabled():
        print("Starting")
        #    m.sort_and_join_csvs("foo", "bar", ["a", "b"], ",", 3)
        fancy_type = np.dtype(
            [
                ("patient_id", np.uint64),
                ("start", np.datetime64),
                ("code", np.uint64),
            ]
        )
        #   m.sort_and_join_csvs("foo_fancy", "bar_fancy", fancy_type, ",", 3)

        print("Done")

        concept_root = tmp_path / "concepts"
        m.test.create_ontology_files(str(concept_root))

        patients = tmp_path / "patients"
        m.test.create_database_files(str(patients))

        target = tmp_path / "database"

        database = m.convert_patient_collection_to_patient_database(
            str(patients), str(concept_root), str(target), ",", 1
        )

        print(database.version_id())
        print(database.database_id())

        assert len(database) == 3

        def f(a):
            return datetime.datetime.fromisoformat(a)

        patient_id = database.get_patient_id_from_original(30)
        assert database.get_original_patient_id(patient_id) == 30

        with pytest.raises(ValueError):
            database.get_code_dictionary().index("not in there")

        assert database.get_code_dictionary().index("bar/foo") is not None
        assert (
            database.get_code_count(
                database.get_code_dictionary().index("bar/foo")
            )
            == 4
        )
        assert database.get_text_count("Short Text") == 2
        assert database.get_text_count("Long Text") == 1
        assert database.get_text_count("Missing Text") == 0

        patient = database[patient_id]

        assert patient.patient_id == patient_id
        assert patient.events == (
            piton.Event(start=f("1990-03-08 09:30:00"), code=0, value=None),
            piton.Event(
                start=f("1990-03-08 10:30:00"),
                end=f("1990-03-18 10:50:00"),
                code=0,
                value=None,
            ),
            piton.Event(
                start=f("1990-03-11 14:30:00"),
                code=2,
                value="Long Text",
                visit_id=0,
            ),
            piton.Event(
                start=f("1990-03-11 14:30:00"),
                code=1,
                value="Short Text",
            ),
            piton.Event(
                start=f("1990-03-14 14:30:00"), code=1, value=34.0, visit_id=1
            ),
            piton.Event(
                start=f("1990-03-15 14:30:00"), code=1, value=34.5, visit_id=0
            ),
        )

        total = 0
        for patient in database:
            total += len(patient.events)
        assert total == 9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

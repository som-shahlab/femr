from __future__ import annotations

import datetime

import extension.datasets as m
import numpy as np
import pytest

import femr


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
        m.test.create_ontology_files(str(concept_root), True)

        patients = tmp_path / "patients"
        m.test.create_database_files(str(patients))

        target = tmp_path / "database"

        m.convert_patient_collection_to_patient_database(
            str(patients), str(concept_root), str(target), ",", 1
        )

        database = m.PatientDatabase(str(target), False)

        print(database.version_id())
        print(database.database_id())

        assert len(database) == 3

        def f(a):
            return datetime.datetime.fromisoformat(a)

        patient_id = 30
        patient = database[patient_id]

        assert patient.patient_id == patient_id

        assert patient.events == (
            femr.Event(start=f("1990-03-08 09:30:00"), code='bar/foo', value=None),
            femr.Event(
                start=f("1990-03-08 10:30:00"),
                code='bar/foo',
                value=None,
            ),
            femr.Event(
                start=f("1990-03-11 14:30:00"),
                code='bar/parent of foo',
                value="Long Text",
            ),
            femr.Event(
                start=f("1990-03-11 14:30:00"),
                code='lol/lmao',
                value="Short Text",
            ),
            femr.Event(start=f("1990-03-14 14:30:00"), code='lol/lmao', value=34.0),
            femr.Event(start=f("1990-03-15 14:30:00"), code='lol/lmao', value=34.5),
        )

        assert set(database.get_ontology().get_all_parents('bar/foo')) == {'bar/foo', 'bar/parent of foo', 'bar/grandparent of foo'}

        total = 0
        for patient_id in database:
            total += len(database[patient_id].events)
        assert total == 9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

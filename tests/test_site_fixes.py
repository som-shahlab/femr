from __future__ import annotations

import datetime
import functools
import json
import pickle
import sys

import meds
import meds_reader.transform
import pytest
from femr_test_tools import DummyEvent, DummySubject

from femr.post_etl_pipelines import site_fixes, stanford
from femr.post_etl_pipelines.site_fixes import (
    SITE_PROFILES,
    FixSpec,
    available_fixes,
    build_pipeline,
    get_fix,
    load_config,
    register_fix,
)
from femr.transforms import delta_encode, remove_nones
from femr.transforms.stanford import (
    make_move_billing_codes,
    move_billing_codes,
    move_pre_birth,
    move_to_day_end,
    move_visit_start_to_first_event_start,
    switch_to_icd10cm,
)


def _stanford_like_subject() -> DummySubject:
    """A subject exercising every step of the Stanford pipeline."""
    return DummySubject(
        subject_id=123,
        events=[
            # Pre-birth: one within 30 days (moved to birth), one older (dropped)
            DummyEvent(time=datetime.datetime(1999, 7, 2), code="early_code"),
            DummyEvent(time=datetime.datetime(1999, 5, 1), code="ancient_code"),
            DummyEvent(time=datetime.datetime(1999, 7, 9), code=meds.birth_code),
            # Midnight timestamp -> moved to end of day
            DummyEvent(time=datetime.datetime(1999, 7, 10), code="midnight_code"),
            # ICD10 prefix -> ICD10CM
            DummyEvent(time=datetime.datetime(1999, 7, 10, 12), code="ICD10/E11"),
            # Visit start -> moved to first non-visit event start
            DummyEvent(time=datetime.datetime(1999, 7, 11), code="visit_code", visit_id=10, table="visit"),
            DummyEvent(time=datetime.datetime(1999, 7, 11, 8), code="morning_rounds", visit_id=10),
            # Billing code -> moved to visit end
            DummyEvent(
                time=datetime.datetime(1999, 7, 12),
                code="encounter",
                visit_id=20,
                clarity_table="lpch_pat_enc",
                end=datetime.datetime(1999, 7, 15),
            ),
            DummyEvent(
                time=datetime.datetime(1999, 7, 12),
                code="SNOMED/123",
                visit_id=20,
                clarity_table="shc_pat_enc_dx",
            ),
            # Flowsheet -> removed
            DummyEvent(time=datetime.datetime(1999, 7, 13), code="STANFORD_OBS/Flowsheet"),
            # Valueless duplicate -> removed
            DummyEvent(time=datetime.datetime(1999, 7, 14, 9), code="dup_code"),
            DummyEvent(time=datetime.datetime(1999, 7, 14, 10), code="dup_code", numeric_value=5),
            # Sequential duplicate value -> removed
            DummyEvent(time=datetime.datetime(1999, 7, 15, 9), code="ddup_code", numeric_value=1),
            DummyEvent(time=datetime.datetime(1999, 7, 15, 10), code="ddup_code", numeric_value=1),
        ],
    )


# ---------------------------------------------------------------------------
# Snapshot of the pre-refactor Stanford pipeline (used to prove the refactor
# is behavior-preserving). Copied verbatim from
# src/femr/post_etl_pipelines/stanford.py before the plugin-system refactor.
# ---------------------------------------------------------------------------


def _legacy_is_visit_measurement(e) -> bool:
    return e.table == "visit"


def _legacy_apply_transformations(subject, *, transforms):
    for transform in transforms:
        subject = transform(subject)
    return subject


def _legacy_remove_flowsheets(subject):
    new_events = []
    for event in subject.events:
        if event.code != "STANFORD_OBS/Flowsheet":
            new_events.append(event)
    subject.events = new_events
    return subject


def _legacy_get_stanford_transformations():
    transforms = [
        move_pre_birth,
        move_visit_start_to_first_event_start,
        move_to_day_end,
        switch_to_icd10cm,
        move_billing_codes,
        functools.partial(
            remove_nones,
            do_not_apply_to_filter=_legacy_is_visit_measurement,
        ),
        functools.partial(
            delta_encode,
            do_not_apply_to_filter=_legacy_is_visit_measurement,
        ),
        _legacy_remove_flowsheets,
    ]
    return functools.partial(_legacy_apply_transformations, transforms=transforms)


def test_stanford_profile_matches_legacy_pipeline() -> None:
    """The refactored Stanford path must be behavior-identical to the legacy one."""
    legacy_pipeline = _legacy_get_stanford_transformations()
    new_pipeline = build_pipeline(SITE_PROFILES["stanford"])
    wrapper_pipeline = stanford._get_stanford_transformations()

    legacy_result = legacy_pipeline(_stanford_like_subject())
    new_result = new_pipeline(_stanford_like_subject())
    wrapper_result = wrapper_pipeline(_stanford_like_subject())

    assert new_result == legacy_result
    assert wrapper_result == legacy_result
    # Sanity: the pipeline actually did something (flowsheet gone, ICD10 switched)
    codes = [e.code for e in new_result.events]
    assert "STANFORD_OBS/Flowsheet" not in codes
    assert "ICD10CM/E11" in codes
    assert "ancient_code" not in codes


def test_built_pipelines_survive_pickle_round_trip() -> None:
    """transform_meds_dataset pickles the pipeline for multiprocessing workers."""
    for profile in SITE_PROFILES:
        pipeline = build_pipeline(SITE_PROFILES[profile])
        restored = pickle.loads(pickle.dumps(pipeline))
        assert restored(_stanford_like_subject()) == pipeline(_stanford_like_subject())

    custom = build_pipeline(
        [
            FixSpec("remove_codes", {"codes": ["X/1"]}),
            FixSpec(
                "move_billing_codes",
                {"billing_code_tables": ["mysite_dx"], "encounter_tables": ["mysite_enc"]},
            ),
        ]
    )
    restored_custom = pickle.loads(pickle.dumps(custom))
    subject = DummySubject(
        subject_id=1,
        events=[
            DummyEvent(time=datetime.datetime(2020, 1, 1), code="X/1"),
            DummyEvent(time=datetime.datetime(2020, 1, 2), code="keep"),
        ],
    )
    assert [e.code for e in restored_custom(subject).events] == ["keep"]


def test_registered_fixes() -> None:
    expected = {
        "move_pre_birth",
        "move_visit_start_to_first_event_start",
        "move_to_day_end",
        "switch_to_icd10cm",
        "move_billing_codes",
        "remove_nones",
        "delta_encode",
        "remove_codes",
        "fix_events",
    }
    assert expected.issubset(set(available_fixes()))


def test_unknown_fix_raises() -> None:
    with pytest.raises(ValueError, match="Unknown fix 'nope'"):
        get_fix("nope")


def test_invalid_fix_params_raise() -> None:
    with pytest.raises(ValueError, match="Invalid params for fix 'remove_codes'"):
        get_fix("remove_codes", {"bogus_param": 1})


def test_duplicate_registration_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_fix("move_pre_birth")(lambda: move_pre_birth)


def test_build_pipeline_applies_in_order() -> None:
    def _marker(code: str):
        def mark(subject):
            subject.events.append(DummyEvent(time=datetime.datetime(2000, 1, 1), code=code))
            return subject

        return mark

    register_fix("test_site_fixes_marker_a")(lambda: _marker("A"))
    register_fix("test_site_fixes_marker_b")(lambda: _marker("B"))

    pipeline = build_pipeline([FixSpec("test_site_fixes_marker_a"), FixSpec("test_site_fixes_marker_b")])
    subject = DummySubject(subject_id=1, events=[])
    result = pipeline(subject)
    assert [e.code for e in result.events] == ["A", "B"]


def test_move_billing_codes_custom_tables() -> None:
    fix = get_fix(
        "move_billing_codes",
        {"billing_code_tables": ["mysite_dx"], "encounter_tables": ["mysite_enc"]},
    )
    subject = DummySubject(
        subject_id=123,
        events=[
            DummyEvent(
                time=datetime.datetime(2020, 1, 2),
                code="enc",
                visit_id=5,
                clarity_table="mysite_enc",
                end=datetime.datetime(2020, 1, 10),
            ),
            DummyEvent(
                time=datetime.datetime(2020, 1, 2),
                code="SNOMED/1",
                visit_id=5,
                clarity_table="mysite_dx",
            ),
        ],
    )
    result = fix(subject)
    billing = [e for e in result.events if e.clarity_table == "mysite_dx"][0]
    assert billing.time == datetime.datetime(2020, 1, 10)


def _billing_subject() -> DummySubject:
    return DummySubject(
        subject_id=123,
        events=[
            DummyEvent(
                time=datetime.datetime(1999, 7, 2),
                code=1234,
                visit_id=10,
                clarity_table="lpch_pat_enc",
                end=datetime.datetime(1999, 7, 20),
            ),
            DummyEvent(
                time=datetime.datetime(1999, 7, 9),
                code="SNOMED/184099003",
                visit_id=10,
                clarity_table="lpch_pat_enc_dx",
            ),
        ],
    )


def test_make_move_billing_codes_defaults_match_wrapper() -> None:
    assert make_move_billing_codes()(_billing_subject()) == move_billing_codes(_billing_subject())


def test_remove_codes_custom() -> None:
    fix = get_fix("remove_codes", {"codes": ["MY_OBS/Flowsheet"]})
    subject = DummySubject(
        subject_id=123,
        events=[
            DummyEvent(time=datetime.datetime(2020, 1, 1), code="MY_OBS/Flowsheet"),
            DummyEvent(time=datetime.datetime(2020, 1, 2), code="OTHER/1"),
        ],
    )
    result = fix(subject)
    assert [e.code for e in result.events] == ["OTHER/1"]


def test_load_config_profile() -> None:
    assert load_config({"profile": "stanford"}) == SITE_PROFILES["stanford"]
    assert load_config({}) == SITE_PROFILES["generic"]


def test_load_config_fixes_override_profile() -> None:
    config = {
        "profile": "stanford",
        "fixes": [
            {"name": "move_to_day_end"},
            {"name": "remove_codes", "params": {"codes": ["X/1", "X/2"]}},
        ],
    }
    assert load_config(config) == [
        FixSpec("move_to_day_end"),
        FixSpec("remove_codes", {"codes": ["X/1", "X/2"]}),
    ]


def test_load_config_errors() -> None:
    with pytest.raises(ValueError, match="Unknown profile"):
        load_config({"profile": "nope"})
    with pytest.raises(ValueError, match="'name'"):
        load_config({"fixes": [{"params": {}}]})
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config({"fixes": [{"name": "move_to_day_end", "params": ["not-a-mapping"]}]})
    with pytest.raises(ValueError, match="must be a list"):
        load_config({"fixes": "move_to_day_end"})


def test_fix_spec_from_dict_errors() -> None:
    with pytest.raises(ValueError, match="'name'"):
        FixSpec.from_dict({})
    with pytest.raises(ValueError, match="must be a mapping"):
        FixSpec.from_dict("move_to_day_end")  # type: ignore[arg-type]


def test_generic_profile_runs() -> None:
    pipeline = build_pipeline(SITE_PROFILES["generic"])
    subject = DummySubject(
        subject_id=123,
        events=[
            DummyEvent(time=datetime.datetime(1999, 7, 9), code=meds.birth_code),
            DummyEvent(time=datetime.datetime(1999, 7, 10), code="ICD10/E11"),
            DummyEvent(time=datetime.datetime(1999, 7, 10, 9), code="dup", numeric_value=1),
            DummyEvent(time=datetime.datetime(1999, 7, 10, 10), code="dup", numeric_value=1),
        ],
    )
    result = pipeline(subject)
    codes = [e.code for e in result.events]
    assert "ICD10CM/E11" in codes
    assert "ICD10/E11" not in codes
    assert len([e for e in result.events if e.code == "dup"]) == 1
    # Events re-sorted by time
    times = [e.time for e in result.events]
    assert times == sorted(times)


def _run_program(monkeypatch, tmp_path, program, argv):
    captured = {}

    def fake_transform(source, target, transform_fn, num_threads=1):
        captured["source"] = source
        captured["target"] = target
        captured["num_threads"] = num_threads
        captured["transform_fn"] = transform_fn

    monkeypatch.setattr(meds_reader.transform, "transform_meds_dataset", fake_transform)
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "dataset.json").write_text(json.dumps({"dataset_name": "test"}))
    monkeypatch.setattr(sys, "argv", argv)
    program()
    return captured


def test_femr_omop_fixer_program_with_config(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "fixes": [
                    {"name": "switch_to_icd10cm"},
                    {"name": "remove_codes", "params": {"codes": ["X/bad"]}},
                ]
            }
        )
    )
    target = tmp_path / "target"
    captured = _run_program(
        monkeypatch,
        target,
        site_fixes.femr_omop_fixer_program,
        ["femr_omop_fixer", "source_ds", str(target), "--config", str(config_path)],
    )
    assert captured["num_threads"] == 1

    subject = DummySubject(
        subject_id=1,
        events=[
            DummyEvent(time=datetime.datetime(2020, 1, 1), code="ICD10/E11"),
            DummyEvent(time=datetime.datetime(2020, 1, 2), code="X/bad"),
        ],
    )
    result = captured["transform_fn"](subject)
    assert [e.code for e in result.events] == ["ICD10CM/E11"]

    metadata = json.loads((target / "metadata" / "dataset.json").read_text())
    assert metadata["post_etl_name"] == "femr_omop_fixer"
    assert metadata["post_etl_version"] == "0.1"
    assert metadata["post_etl_profile"] == "custom"


def test_femr_omop_fixer_program_with_profile(monkeypatch, tmp_path) -> None:
    target = tmp_path / "target"
    captured = _run_program(
        monkeypatch,
        target,
        site_fixes.femr_omop_fixer_program,
        ["femr_omop_fixer", "source_ds", str(target), "--profile", "stanford"],
    )
    assert captured["transform_fn"](_stanford_like_subject()) == build_pipeline(SITE_PROFILES["stanford"])(
        _stanford_like_subject()
    )
    metadata = json.loads((target / "metadata" / "dataset.json").read_text())
    assert metadata["post_etl_profile"] == "stanford"


def test_stanford_program_still_works(monkeypatch, tmp_path) -> None:
    target = tmp_path / "target"
    captured = _run_program(
        monkeypatch,
        target,
        stanford.femr_stanford_omop_fixer_program,
        ["femr_stanford_omop_fixer", "source_ds", str(target), "--num_proc", "4"],
    )
    assert captured["num_threads"] == 4
    metadata = json.loads((target / "metadata" / "dataset.json").read_text())
    assert metadata["post_etl_name"] == "femr_stanford_omop_fixer"
    assert metadata["post_etl_version"] == "0.1"
    assert "post_etl_profile" not in metadata

"""A configurable plugin system for site-specific post-ETL fixes.

OMOP extracts from different sites have different data-quality quirks (midnight timestamps,
billing codes stamped at visit start, unreliable flowsheet measurements, ...). Historically
FEMR handled this with a single hardcoded Stanford pipeline
(:func:`femr.post_etl_pipelines.stanford.femr_stanford_omop_fixer_program`).

This module generalizes that into small, named, parameterizable "fixes" that can be composed
into a per-site pipeline, either from a built-in profile or from a JSON config file, e.g.::

    {
        "profile": "generic",
        "fixes": [
            {"name": "move_pre_birth"},
            {"name": "move_to_day_end"},
            {"name": "remove_codes", "params": {"codes": ["MY_OBS/Flowsheet"]}},
            {
                "name": "move_billing_codes",
                "params": {
                    "billing_code_tables": ["mysite_pat_enc_dx"],
                    "encounter_tables": ["mysite_pat_enc"]
                }
            }
        ]
    }

If ``"fixes"`` is present it replaces the profile's fix list; otherwise the profile's
built-in list is used.

Custom fixes can be added without forking FEMR internals::

    from femr.post_etl_pipelines import site_fixes

    @site_fixes.register_fix("drop_test_patients")
    def _make_drop_test_patients() -> site_fixes.SubjectTransform:
        def drop_test_patients(subject):
            ...
            return subject
        return drop_test_patients
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import inspect
import json
import os
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence

import meds_reader
import meds_reader.transform

from femr.transforms import delta_encode, fix_events, remove_nones
from femr.transforms.stanford import (
    make_move_billing_codes,
    move_pre_birth,
    move_to_day_end,
    move_visit_start_to_first_event_start,
    switch_to_icd10cm,
)

SubjectTransform = Callable[[meds_reader.transform.MutableSubject], meds_reader.transform.MutableSubject]
FixFactory = Callable[..., SubjectTransform]

POST_ETL_NAME = "femr_omop_fixer"
POST_ETL_VERSION = "0.1"

_FIX_FACTORIES: Dict[str, FixFactory] = {}


def register_fix(name: str) -> Callable[[FixFactory], FixFactory]:
    """Register a fix factory under ``name`` so it can be referenced from configs.

    A fix factory is a callable taking keyword params (all JSON-serializable) and
    returning a :data:`SubjectTransform`. Param-less fixes can ignore arguments.
    """

    def decorator(factory: FixFactory) -> FixFactory:
        if name in _FIX_FACTORIES:
            raise ValueError(f"A fix named {name!r} is already registered")
        _FIX_FACTORIES[name] = factory
        return factory

    return decorator


def available_fixes() -> List[str]:
    """Return the sorted names of all registered fixes."""
    return sorted(_FIX_FACTORIES)


def get_fix(name: str, params: Optional[Mapping[str, Any]] = None) -> SubjectTransform:
    """Build the named fix with the given params.

    Raises:
        ValueError: If the fix name is unknown or the params don't match the factory signature.
    """
    try:
        factory = _FIX_FACTORIES[name]
    except KeyError:
        raise ValueError(f"Unknown fix {name!r}. Available fixes: {available_fixes()}") from None
    try:
        return factory(**dict(params or {}))
    except TypeError as e:
        valid_params = list(inspect.signature(factory).parameters)
        raise ValueError(f"Invalid params for fix {name!r}: {e}. Valid params: {valid_params}") from None


def _is_visit_event(event: meds_reader.Event) -> bool:
    return event.table == "visit"


@register_fix("move_pre_birth")
def _fix_move_pre_birth() -> SubjectTransform:
    """Move events dated before birth to the birth date (dropping those >30 days early)."""
    return move_pre_birth


@register_fix("move_visit_start_to_first_event_start")
def _fix_move_visit_start() -> SubjectTransform:
    """Set each visit's start time to the start of its first non-visit event."""
    return move_visit_start_to_first_event_start


@register_fix("move_to_day_end")
def _fix_move_to_day_end() -> SubjectTransform:
    """Move midnight timestamps to the end of the day (23:59)."""
    return move_to_day_end


@register_fix("switch_to_icd10cm")
def _fix_switch_to_icd10cm() -> SubjectTransform:
    """Rewrite ``ICD10/`` code prefixes to ``ICD10CM/``."""
    return switch_to_icd10cm


@register_fix("move_billing_codes")
def _fix_move_billing_codes(
    billing_code_tables: Optional[Sequence[str]] = None,
    encounter_tables: Optional[Sequence[str]] = None,
) -> SubjectTransform:
    """Move billing codes to the end of each visit.

    Args:
        billing_code_tables: Source tables holding billing codes.
            Defaults to the Stanford Clarity billing tables.
        encounter_tables: Source tables holding encounter records, used to find each
            visit's end time. Defaults to the Stanford Clarity encounter tables.
    """
    return make_move_billing_codes(
        billing_code_tables=billing_code_tables,
        encounter_tables=encounter_tables,
    )


@register_fix("remove_nones")
def _fix_remove_nones(*, exclude_visit_events: bool = True) -> SubjectTransform:
    """Drop valueless duplicate codes when a valued copy exists on the same day.

    Args:
        exclude_visit_events: If True (default), never drop visit events, so that
            ``visit_id`` linkage stays intact for downstream steps.
    """
    do_not_apply_to_filter = _is_visit_event if exclude_visit_events else None
    return functools.partial(remove_nones, do_not_apply_to_filter=do_not_apply_to_filter)


@register_fix("delta_encode")
def _fix_delta_encode(*, exclude_visit_events: bool = True) -> SubjectTransform:
    """Drop sequential duplicate (code, day, value) events within a day.

    Args:
        exclude_visit_events: If True (default), never drop visit events, so that
            ``visit_id`` linkage stays intact for downstream steps.
    """
    do_not_apply_to_filter = _is_visit_event if exclude_visit_events else None
    return functools.partial(delta_encode, do_not_apply_to_filter=do_not_apply_to_filter)


@register_fix("remove_codes")
def _fix_remove_codes(codes: Sequence[str]) -> SubjectTransform:
    """Drop all events whose code is in ``codes``.

    Used for measurements with known timing bugs (e.g. STARR-OMOP flowsheets);
    configure per site instead of hardcoding.
    """
    return functools.partial(_remove_codes_impl, codes=frozenset(codes))


def _remove_codes_impl(
    subject: meds_reader.transform.MutableSubject, codes: FrozenSet[str]
) -> meds_reader.transform.MutableSubject:
    subject.events = [event for event in subject.events if event.code not in codes]
    return subject


@register_fix("fix_events")
def _fix_fix_events() -> SubjectTransform:
    """Final cleanup pass: re-sort events by time to meet MEDS requirements."""
    return fix_events


@dataclasses.dataclass(frozen=True)
class FixSpec:
    """A single configured fix step: a registered fix name plus its params."""

    name: str
    params: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "FixSpec":
        """Build a FixSpec from a ``{"name": ..., "params": {...}}`` mapping."""
        if not isinstance(data, Mapping) or "name" not in data:
            raise ValueError(f"Each fix must be a mapping with a 'name' key, got: {data!r}")
        params = data.get("params", {})
        if not isinstance(params, Mapping):
            raise ValueError(f"'params' for fix {data.get('name')!r} must be a mapping, got: {params!r}")
        return FixSpec(name=str(data["name"]), params=dict(params))


# Built-in site profiles. "stanford" reproduces the historical
# femr_stanford_omop_fixer transform list (and order) exactly; "generic" applies
# only the site-agnostic timing/code fixes suitable for any OMOP -> MEDS output.
SITE_PROFILES: Dict[str, List[FixSpec]] = {
    "stanford": [
        FixSpec("move_pre_birth"),
        FixSpec("move_visit_start_to_first_event_start"),
        FixSpec("move_to_day_end"),
        FixSpec("switch_to_icd10cm"),
        FixSpec("move_billing_codes"),
        FixSpec("remove_nones", {"exclude_visit_events": True}),
        FixSpec("delta_encode", {"exclude_visit_events": True}),
        FixSpec("remove_codes", {"codes": ["STANFORD_OBS/Flowsheet"]}),
    ],
    "generic": [
        FixSpec("move_pre_birth"),
        FixSpec("move_visit_start_to_first_event_start"),
        FixSpec("move_to_day_end"),
        FixSpec("switch_to_icd10cm"),
        FixSpec("remove_nones", {"exclude_visit_events": True}),
        FixSpec("delta_encode", {"exclude_visit_events": True}),
        FixSpec("fix_events"),
    ],
}


def _apply_fixes(
    subject: meds_reader.transform.MutableSubject, transforms: Sequence[SubjectTransform]
) -> meds_reader.transform.MutableSubject:
    for transform in transforms:
        subject = transform(subject)
    return subject


def build_pipeline(fixes: Sequence[FixSpec]) -> SubjectTransform:
    """Compose configured fixes into a single subject transform, applied in order.

    The returned transform is picklable, so it can be used with multiprocessing-based
    dataset transforms such as ``meds_reader.transform.transform_meds_dataset``.
    """
    transforms = tuple(get_fix(spec.name, spec.params) for spec in fixes)
    return functools.partial(_apply_fixes, transforms=transforms)


def load_config(config: Mapping[str, Any]) -> List[FixSpec]:
    """Build a fix list from a config mapping.

    Recognized keys:

    - ``profile``: name of a built-in profile in :data:`SITE_PROFILES` (default ``"generic"``).
    - ``fixes``: list of ``{"name": ..., "params": {...}}`` mappings. When present,
      it replaces the profile's fix list.

    Raises:
        ValueError: If the config is malformed or references an unknown profile/fix.
    """
    if not isinstance(config, Mapping):
        raise ValueError(f"Config must be a mapping, got: {config!r}")
    profile_name = config.get("profile", "generic")
    if profile_name not in SITE_PROFILES:
        raise ValueError(f"Unknown profile {profile_name!r}. Available profiles: {sorted(SITE_PROFILES)}")
    if "fixes" in config:
        raw_fixes = config["fixes"]
        if not isinstance(raw_fixes, Sequence) or isinstance(raw_fixes, (str, bytes)):
            raise ValueError(f"'fixes' must be a list of fix mappings, got: {raw_fixes!r}")
        # Validate eagerly so config errors surface before any data is touched.
        return [FixSpec.from_dict(item) for item in raw_fixes]
    return list(SITE_PROFILES[profile_name])


def _resolve_config(args: argparse.Namespace) -> tuple[List[FixSpec], str]:
    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)
        profile_label = "custom" if "fixes" in config else str(config.get("profile", "generic"))
        return load_config(config), profile_label
    return list(SITE_PROFILES[args.profile]), args.profile


def femr_omop_fixer_program() -> None:
    """Apply a configurable set of site-specific fixes to a MEDS dataset from an OMOP ETL."""
    parser = argparse.ArgumentParser(description="Apply configurable site-specific fixes to a MEDS dataset")

    parser.add_argument(
        "source_dataset",
        type=str,
        help="Path of the folder to source dataset",
    )

    parser.add_argument(
        "target_dataset",
        type=str,
        help="The place to store the extract",
    )

    parser.add_argument(
        "--num_proc",
        type=int,
        help="The number of threads to use",
        default=1,
    )

    parser.add_argument(
        "--profile",
        type=str,
        choices=sorted(SITE_PROFILES),
        default="generic",
        help="Built-in site profile to apply (ignored if --config is given)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file describing the fixes to apply",
    )

    args = parser.parse_args()

    fixes, profile_label = _resolve_config(args)
    pipeline = build_pipeline(fixes)

    meds_reader.transform.transform_meds_dataset(
        args.source_dataset, args.target_dataset, pipeline, num_threads=args.num_proc
    )

    metadata_path = os.path.join(args.target_dataset, "metadata/dataset.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Let's mark that we modified this dataset
    metadata["post_etl_name"] = POST_ETL_NAME
    metadata["post_etl_version"] = POST_ETL_VERSION
    metadata["post_etl_profile"] = profile_label

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

from __future__ import annotations

import collections
import os
from typing import Dict, Iterable, Optional, Set

import meds
import polars as pl


class Ontology:
    def __init__(self, athena_path: str, code_metadata: meds.CodeMetadata = {}):
        """Create an Ontology from an Athena download and an optional meds Code Metadata structure.

        NOTE: This is an expensive operation.
        It is recommended to create an ontology once and then save/load it as necessary.
        """
        # Load from code metadata
        self.description_map = {}
        self.parents_map: Dict[str, Set[str]] = collections.defaultdict(set)

        for code, code_info in code_metadata.items():
            if code_info.get("description") is not None:
                self.description_map[code] = code_info["description"]
            if code_info.get("parent_codes") is not None:
                self.parents_map[code] |= set(code_info["parent_codes"])

        # Load from the athena path ...
        concept = pl.scan_csv(os.path.join(athena_path, "CONCEPT.csv"), separator="\t", infer_schema_length=0)
        code_col = pl.col("vocabulary_id") + "/" + pl.col("concept_code")
        description_col = pl.col("concept_name")
        concept_id_col = pl.col("concept_id").cast(pl.Int64)

        processed_concepts = (
            concept.select(code_col, concept_id_col, description_col, pl.col("standard_concept").is_null())
            .collect()
            .rows()
        )

        self.concept_id_to_code_map = {}
        self.code_to_concept_id_map = {}

        non_standard_concepts = set()

        for code, concept_id, description, is_non_standard in processed_concepts:
            self.concept_id_to_code_map[concept_id] = code
            self.code_to_concept_id_map[code] = concept_id

            # We don't want to override code metadata
            if code not in self.description_map:
                self.description_map[code] = description

            if is_non_standard:
                non_standard_concepts.add(concept_id)

        relationship = pl.scan_csv(
            os.path.join(athena_path, "CONCEPT_RELATIONSHIP.csv"), separator="\t", infer_schema_length=0
        )
        relationship_id = pl.col("relationship_id")
        relationship = relationship.filter(
            relationship_id == "Maps to", pl.col("concept_id_1") != pl.col("concept_id_2")
        )
        for concept_id_1, concept_id_2 in (
            relationship.select(pl.col("concept_id_1").cast(pl.Int64), pl.col("concept_id_2").cast(pl.Int64))
            .collect()
            .rows()
        ):
            if concept_id_1 in non_standard_concepts:
                self.parents_map[self.concept_id_to_code_map[concept_id_1]].add(
                    self.concept_id_to_code_map[concept_id_2]
                )

        ancestor = pl.scan_csv(os.path.join(athena_path, "CONCEPT_ANCESTOR.csv"), separator="\t", infer_schema_length=0)
        ancestor = ancestor.filter(pl.col("min_levels_of_separation") == "1")
        for concept_id, parent_concept_id in (
            ancestor.select(
                pl.col("descendant_concept_id").cast(pl.Int64), pl.col("ancestor_concept_id").cast(pl.Int64)
            )
            .collect()
            .rows()
        ):
            self.parents_map[self.concept_id_to_code_map[concept_id]].add(
                self.concept_id_to_code_map[parent_concept_id]
            )

        self.children_map = collections.defaultdict(set)
        for code, parents in self.parents_map.items():
            for parent in parents:
                self.children_map[parent].add(code)

        self.all_parents_map: Dict[str, Set[str]] = {}
        self.all_children_map: Dict[str, Set[str]] = {}

    def get_description(self, code: str) -> Optional[str]:
        """Get a description of a code."""
        return self.description_map.get(code)

    def get_children(self, code: str) -> Iterable[str]:
        """Get the children for a given code."""
        return self.children_map.get(code, set())

    def get_parents(self, code: str) -> Iterable[str]:
        """Get the parents for a given code."""
        return self.parents_map.get(code, set())

    def get_all_children(self, code: str) -> Set[str]:
        """Get all children, including through the ontology."""
        if code not in self.all_children_map:
            result = {code}
            for child in self.children_map.get(code, set()):
                result |= self.get_all_children(child)
            self.all_children_map[code] = result
        return self.all_children_map[code]

    def get_all_parents(self, code: str) -> Set[str]:
        """Get all parents, including through the ontology."""
        if code not in self.all_parents_map:
            result = {code}
            for parent in self.parents_map.get(code, set()):
                result |= self.get_all_parents(parent)
            self.all_parents_map[code] = result

        return self.all_parents_map[code]

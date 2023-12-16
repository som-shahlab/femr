import meds
import polars as pl


class Ontology:
    @classmethod
    def extract_from_athena(cls, athena_path: str) -> Ontology:
        """Create an ontology given an Athena download."""
        concept = pl.scan_csv(os.path.join(athena_path, "CONCEPT.csv"))

        concept = pl.scan_csv(os.path.join(athena_path, "CONCEPT.csv"))
        code = pl.col("vocabulary_id") + "/" + pl.col("concept_code")
        description = pl.col("concet_name")
        concept_id = pl.col("concept_id")

        processed_concept = concept.select(code=code, description=description, concept_id=concept_id)

    def __init__(self, ontology_data: str):
        # Load from the athena path ...
        pass

    def encorperate_code_metadata(self, code_metadata: meds.CodeMetadata):
        """Add the provided code metadata to the given ontology definitions."""

    def get_description(self, code: str) -> str:
        """Get a description of a code."""

    def get_children(self, code: str) -> Iterable[str]:
        """Get the children for a given code."""

    def get_parents(self, code) -> Iterable[str]:
        """Get the parents for a given code."""

    def get_all_children(self, code) -> Iterable[str]:
        """Get all children, including through the ontology."""

    def get_all_parents(self, code) -> Iterable[str]:
        """Get all parents, including through the ontology."""

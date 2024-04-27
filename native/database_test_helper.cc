#include "database_test_helper.hh"

#include "csv.hh"

template <typename Writer>
void ontology_helper(const boost::filesystem::path& concept_root,
                     const char* pattern) {
    boost::filesystem::path concept = concept_root / "concept";
    boost::filesystem::create_directory(concept);
    {
        CSVWriter<Writer> writer(
            (concept / boost::filesystem::unique_path(pattern)).string(),
            {"concept_id", "concept_code", "concept_name", "vocabulary_id",
             "standard_concept"},
            ',');

        writer.add_row({"32", "foo", "foo name", "bar", ""});
        writer.add_row({"323", "parent of foo", "parent foo name", "bar", ""});
        writer.add_row(
            {"3235", "grandparent of foo", "grandparent", "bar", "S"});
        writer.add_row(
            {"32356", "bad grandparent of foo", "bad grandparent", "bar", ""});
        writer.add_row({"326", "lmao", "test the descr", "lol", ""});
    }

    boost::filesystem::path relationship = concept_root / "relationship";
    boost::filesystem::create_directory(relationship);
    {
        CSVWriter<Writer> writer(
            (relationship / boost::filesystem::unique_path(pattern)).string(),
            {"reverse_relationship_id", "defines_ancestry"}, ',');

        writer.add_row({"Is a", "1"});
    }

    boost::filesystem::path concept_relationship =
        concept_root / "concept_relationship";
    boost::filesystem::create_directory(concept_relationship);
    {
        CSVWriter<Writer> writer(
            (concept_relationship / boost::filesystem::unique_path(pattern))
                .string(),
            {"concept_id_1", "concept_id_2", "relationship_id"}, ',');

        writer.add_row({"32", "323", "Is a"});
        writer.add_row({"323", "3235", "Is a"});
        writer.add_row({"323", "32356", "Is a"});
    }
}

void create_ontology_files(const boost::filesystem::path& concept_root,
                           bool compressed) {
    boost::filesystem::create_directory(concept_root);

    const char* pattern;

    if (compressed) {
        pattern = "%%%%%%%%%%.csv.zst";
        ontology_helper<ZstdWriter>(concept_root, pattern);
    } else {
        pattern = "%%%%%%%%%%.csv";
        ontology_helper<TextWriter>(concept_root, pattern);
    }
}

void create_database_files(const boost::filesystem::path& patients) {
    const char* pattern = "%%%%%%%%%%.csv.zst";

    boost::filesystem::create_directory(patients);

    {
        CSVWriter<ZstdWriter> writer(
            (patients / boost::filesystem::unique_path(pattern)).string(),
            {"patient_id", "concept_id", "start", "value", "metadata"}, ',');

        writer.add_row({"30", "32", "1990-03-08T09:30:00", "", ""});
        writer.add_row({"30", "32", "1990-03-08T10:30:00", "", ""});
        writer.add_row({"30", "323", "1990-03-11T14:30:00", "Long Text",
                        "gASVEQAAAAAAAAB9lIwIdmlzaXRfaWSUSwFzLg=="});
        writer.add_row({"30", "326", "1990-03-11T14:30:00", "Short Text", ""});
        writer.add_row({"30", "326", "1990-03-14T14:30:00", "34",
                        "gASVEQAAAAAAAAB9lIwIdmlzaXRfaWSUSwFzLg=="});
        writer.add_row({"30", "326", "1990-03-15T14:30:00", "34.5",
                        "gASVEQAAAAAAAAB9lIwIdmlzaXRfaWSUSwFzLg=="});
    }
    {
        CSVWriter<ZstdWriter> writer(
            (patients / boost::filesystem::unique_path(pattern)).string(),
            {"patient_id", "concept_id", "start", "value", "metadata"}, ',');

        writer.add_row({"70", "32", "1990-03-08T09:30:00", "", ""});
        writer.add_row({"70", "323", "1990-03-08T14:30:00", "Short Text", ""});
        writer.add_row({"80", "32", "1990-03-08T14:30:00", "", ""});
    }
}

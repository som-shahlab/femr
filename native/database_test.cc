#include "database.hh"

#include <boost/filesystem.hpp>
#include <random>

#include "csv.hh"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace testing;

boost::filesystem::path create_ontology(const boost::filesystem::path& root) {
    boost::filesystem::path concept_root =
        root / boost::filesystem::unique_path();
    boost::filesystem::create_directory(concept_root);

    boost::filesystem::path concept = concept_root / "concept";
    boost::filesystem::create_directory(concept);
    {
        CSVWriter writer((concept / boost::filesystem::unique_path()).string(),
                         {"concept_id", "concept_code", "vocabulary_id"}, ',');

        writer.add_row({"32", "foo", "bar"});
        writer.add_row({"323", "parent of foo", "bar"});
        writer.add_row({"3235", "grandparent of foo", "bar"});
        writer.add_row({"326", "lmao", "lol"});
    }

    boost::filesystem::path concept_relationship =
        concept_root / "concept_relationship";
    boost::filesystem::create_directory(concept_relationship);
    {
        CSVWriter writer(
            (concept_relationship / boost::filesystem::unique_path()).string(),
            {"concept_id_1", "concept_id_2", "relationship_id"}, ',');

        writer.add_row({"32", "323", "Is a"});
        writer.add_row({"323", "3235", "Is a"});
    }
    return concept_root;
}

TEST(Database, CreateOntology) {
    boost::filesystem::path root = boost::filesystem::temp_directory_path() /
                                   boost::filesystem::unique_path();
    boost::filesystem::create_directory(root);

    boost::filesystem::path concept_root = create_ontology(root);

    boost::filesystem::path destination =
        root / boost::filesystem::unique_path();

    std::vector<uint64_t> concepts_to_map = {32, 326};

    create_ontology(concepts_to_map, concept_root, destination, ',', 3);

    {
        Ontology ontology(destination);

        auto helper = [&](absl::Span<const uint32_t> indices) {
            std::vector<std::string> results;
            for (auto index : indices) {
                results.push_back(
                    std::string(ontology.get_dictionary().get_text(index)));
            }
            return results;
        };

        EXPECT_EQ(ontology.get_dictionary().get_text(0), "bar/foo");
        EXPECT_EQ(ontology.get_dictionary().get_text(1), "lol/lmao");
        EXPECT_EQ(ontology.get_dictionary().get_text(2), "bar/parent of foo");
        EXPECT_EQ(ontology.get_dictionary().get_text(3),
                  "bar/grandparent of foo");

        EXPECT_THAT(helper(ontology.get_children(1)), ElementsAre());
        EXPECT_THAT(helper(ontology.get_parents(1)), ElementsAre());
        EXPECT_THAT(helper(ontology.get_all_parents(1)),
                    ElementsAre("lol/lmao"));

        EXPECT_THAT(helper(ontology.get_children(3)),
                    ElementsAre("bar/parent of foo"));
        EXPECT_THAT(helper(ontology.get_parents(0)),
                    ElementsAre("bar/parent of foo"));
        EXPECT_THAT(helper(ontology.get_all_parents(0)),
                    UnorderedElementsAre("bar/foo", "bar/parent of foo",
                                         "bar/grandparent of foo"));
    }

    boost::filesystem::remove_all(root);
}

TEST(Database, CreateDatabase) {
    boost::filesystem::path root = boost::filesystem::temp_directory_path() /
                                   boost::filesystem::unique_path();
    boost::filesystem::create_directory(root);

    std::cout << root << std::endl;

    boost::filesystem::path patients = root / boost::filesystem::unique_path();
    boost::filesystem::create_directory(patients);

    {
        CSVWriter writer((patients / boost::filesystem::unique_path()).string(),
                         {"patient_id", "code", "start", "value", "value_type"},
                         ',');

        writer.add_row(
            {"30", "32", "1990-03-08T09:30:00", "", "ValueType.NONE"});
        writer.add_row(
            {"30", "32", "1990-03-08T10:30:00", "", "ValueType.NONE"});
        writer.add_row({"30", "323", "1990-03-11T14:30:00", "Long Text",
                        "ValueType.TEXT"});
        writer.add_row({"30", "326", "1990-03-11T14:30:00", "Short Text",
                        "ValueType.TEXT"});
        writer.add_row(
            {"30", "326", "1990-03-14T14:30:00", "34", "ValueType.NUMERIC"});
        writer.add_row(
            {"30", "326", "1990-03-15T14:30:00", "34.5", "ValueType.NUMERIC"});
    }
    {
        CSVWriter writer((patients / boost::filesystem::unique_path()).string(),
                         {"patient_id", "code", "start", "value", "value_type"},
                         ',');

        writer.add_row(
            {"70", "32", "1990-03-08T09:30:00", "", "ValueType.NONE"});
        writer.add_row({"70", "323", "1990-03-08T14:30:00", "Short Text",
                        "ValueType.TEXT"});
        writer.add_row(
            {"80", "32", "1990-03-08T14:30:00", "", "ValueType.NONE"});
    }
    boost::filesystem::path concept_root = create_ontology(root);

    boost::filesystem::path destination =
        root / boost::filesystem::unique_path();

    convert_patient_collection_to_patient_database(patients, concept_root,
                                                   destination, ',', 3);

    PatientDatabase database(destination, true);

    EXPECT_EQ(database.get_num_patients(), 3);
    EXPECT_EQ(database.get_patient_id_from_original(30).has_value(), true);
    EXPECT_EQ(database.get_patient_id_from_original(70).has_value(), true);
    uint32_t patient_id = *database.get_patient_id_from_original(30);
    EXPECT_EQ(database.get_original_patient_id(patient_id), 30);
    EXPECT_EQ(database.get_patient_id_from_original(1235).has_value(), false);
    EXPECT_EQ(database.get_patient_id_from_original(31).has_value(), false);

    EXPECT_EQ(database.get_code_count(0), 4);
    EXPECT_EQ(database.get_short_text_count(0), 2);

    PatientDatabaseIterator iterator = database.iterator();

    const Patient& patient = iterator.get_patient(patient_id);

    EXPECT_EQ(patient.patient_id, patient_id);
    EXPECT_EQ(patient.birth_date, absl::CivilDay(1990, 3, 8));

    Event a = {
        .age_in_days = 3,
        .minutes_offset = 14 * 60 + 30,
        .code = *database.get_code_dictionary().get_code("bar/parent of foo"),
        .value_type = ValueType::LONG_TEXT};
    EXPECT_EQ(
        database.get_long_text_dictionary().get_code("Long Text").has_value(),
        true);
    a.text_value = *database.get_long_text_dictionary().get_code("Long Text");
    std::cout << *database.get_long_text_dictionary().get_code("Long Text")
              << std::endl;

    Event b = {.age_in_days = 3,
               .minutes_offset = 14 * 60 + 30,
               .code = *database.get_code_dictionary().get_code("lol/lmao"),
               .value_type = ValueType::SHORT_TEXT};
    b.text_value = *database.get_short_text_dictionary().get_code("Short Text");

    Event c = {.age_in_days = 6,
               .minutes_offset = 14 * 60 + 30,
               .code = *database.get_code_dictionary().get_code("lol/lmao"),
               .value_type = ValueType::NUMERIC};
    c.numeric_value = 34;

    Event d = {.age_in_days = 7,
               .minutes_offset = 14 * 60 + 30,
               .code = *database.get_code_dictionary().get_code("lol/lmao"),
               .value_type = ValueType::NUMERIC};
    d.numeric_value = 34.5;

    EXPECT_THAT(
        patient.events,
        ElementsAre(
            Event{.age_in_days = 0,
                  .minutes_offset = 9 * 60 + 30,
                  .code = *database.get_code_dictionary().get_code("bar/foo"),
                  .value_type = ValueType::NONE},
            Event{.age_in_days = 0,
                  .minutes_offset = 10 * 60 + 30,
                  .code = *database.get_code_dictionary().get_code("bar/foo"),
                  .value_type = ValueType::NONE},
            a, b, c, d));

    boost::filesystem::remove_all(root);
}

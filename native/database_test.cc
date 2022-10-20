#include "database.hh"

#include <boost/filesystem.hpp>
#include <random>

#include "csv.hh"
#include "database_test_helper.hh"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace testing;

TEST(Database, CreateOntology) {
    boost::filesystem::path root = boost::filesystem::temp_directory_path() /
                                   boost::filesystem::unique_path();
    boost::filesystem::create_directory(root);

    boost::filesystem::path concept_root =
        root / boost::filesystem::unique_path();
    create_ontology_files(concept_root);

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
                    std::string(ontology.get_dictionary()[index]));
            }
            return results;
        };

        EXPECT_EQ(ontology.get_dictionary()[0], "bar/foo");
        EXPECT_EQ(ontology.get_dictionary()[1], "lol/lmao");
        EXPECT_EQ(ontology.get_dictionary()[2], "bar/parent of foo");
        EXPECT_EQ(ontology.get_dictionary()[3], "bar/grandparent of foo");

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

    boost::filesystem::path concept_root =
        root / boost::filesystem::unique_path();
    create_ontology_files(concept_root);

    boost::filesystem::path patients = root / boost::filesystem::unique_path();
    create_database_files(patients);

    boost::filesystem::path destination =
        root / boost::filesystem::unique_path();

    convert_patient_collection_to_patient_database(patients, concept_root,
                                                   destination, ',', 3);

    PatientDatabase database(destination, true);

    EXPECT_EQ(database.size(), 3);
    EXPECT_EQ(database.get_patient_id_from_original(30).has_value(), true);
    EXPECT_EQ(database.get_patient_id_from_original(70).has_value(), true);
    uint32_t patient_id = *database.get_patient_id_from_original(30);
    EXPECT_EQ(database.get_original_patient_id(patient_id), 30);
    EXPECT_EQ(database.get_patient_id_from_original(1235).has_value(), false);
    EXPECT_EQ(database.get_patient_id_from_original(31).has_value(), false);

    EXPECT_EQ(database.get_code_count(0), 4);
    EXPECT_EQ(database.get_shared_text_count(0), 2);

    PatientDatabaseIterator iterator = database.iterator();

    const Patient& patient = iterator.get_patient(patient_id);

    EXPECT_EQ(patient.patient_id, patient_id);
    EXPECT_EQ(patient.birth_date, absl::CivilDay(1990, 3, 8));

    Event a = {
        .age_in_days = 3,
        .minutes_offset = 14 * 60 + 30,
        .code = *database.get_code_dictionary().find("bar/parent of foo"),
        .value_type = ValueType::UNIQUE_TEXT};
    EXPECT_EQ(
        database.get_unique_text_dictionary().find("Long Text").has_value(),
        true);
    a.text_value = *database.get_unique_text_dictionary().find("Long Text");
    std::cout << *database.get_unique_text_dictionary().find("Long Text")
              << std::endl;

    Event b = {.age_in_days = 3,
               .minutes_offset = 14 * 60 + 30,
               .code = *database.get_code_dictionary().find("lol/lmao"),
               .value_type = ValueType::SHARED_TEXT};
    b.text_value = *database.get_shared_text_dictionary().find("Short Text");

    Event c = {.age_in_days = 6,
               .minutes_offset = 14 * 60 + 30,
               .code = *database.get_code_dictionary().find("lol/lmao"),
               .value_type = ValueType::NUMERIC};
    c.numeric_value = 34;

    Event d = {.age_in_days = 7,
               .minutes_offset = 14 * 60 + 30,
               .code = *database.get_code_dictionary().find("lol/lmao"),
               .value_type = ValueType::NUMERIC};
    d.numeric_value = 34.5;

    EXPECT_THAT(
        patient.events,
        ElementsAre(
            Event{.age_in_days = 0,
                  .minutes_offset = 9 * 60 + 30,
                  .code = *database.get_code_dictionary().find("bar/foo"),
                  .value_type = ValueType::NONE},
            Event{.age_in_days = 0,
                  .minutes_offset = 10 * 60 + 30,
                  .code = *database.get_code_dictionary().find("bar/foo"),
                  .value_type = ValueType::NONE},
            a, b, c, d));

    // boost::filesystem::remove_all(root);
}

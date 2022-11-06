#pragma once

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <cstdint>
#include <string_view>
#include <vector>

#include "absl/time/civil_time.h"
#include "absl/types/span.h"
#include "dictionary.hh"

class LazyDictionary {
   public:
    LazyDictionary(const boost::filesystem::path& _path, bool _read_all)
        : path(_path), read_all(_read_all), value(boost::none) {}

    Dictionary* operator->() {
        init();
        return &(*value);
    }

    Dictionary& operator*() {
        init();
        return *value;
    }

   private:
    void init() {
        if (!value) {
            value.emplace(path, read_all);
        }
    }

    boost::filesystem::path path;
    bool read_all;
    boost::optional<Dictionary> value;
};

class Ontology {
   public:
    Ontology(const boost::filesystem::path& path);
    Ontology(Ontology&&) = default;

    absl::Span<const uint32_t> get_parents(uint32_t code);
    absl::Span<const uint32_t> get_children(uint32_t code);
    absl::Span<const uint32_t> get_all_parents(uint32_t code);
    Dictionary& get_dictionary();

   private:
    LazyDictionary main_dictionary;
    LazyDictionary parent_dict;
    LazyDictionary children_dict;
    LazyDictionary all_parents_dict;
};

enum class ValueType {
    NONE,
    NUMERIC,
    SHARED_TEXT,
    UNIQUE_TEXT,
};

struct Event {
    uint16_t age_in_days;
    uint16_t minutes_offset;

    uint32_t code;
    ValueType value_type;

    union {
        float numeric_value;
        uint32_t text_value;
    };

    bool operator==(const Event& other) const {
        return (age_in_days == other.age_in_days &&
                minutes_offset == other.minutes_offset && code == other.concept_id &&
                value_type == other.value_type &&
                text_value == other.text_value);
    }
};

struct Patient {
    uint32_t patient_id;
    absl::CivilDay birth_date;
    std::vector<Event> events;
};

class PatientDatabase;

class PatientDatabaseIterator {
   public:
    Patient& get_patient(uint32_t patient_id);

   private:
    PatientDatabaseIterator(const Dictionary* d);
    friend PatientDatabase;

    const Dictionary* const parent_dictionary;

    Patient current_patient;
    std::vector<uint32_t> buffer;
};

class PatientDatabase {
   public:
    PatientDatabase(const boost::filesystem::path& path, bool read_all);
    PatientDatabase(PatientDatabase&&) = default;

    PatientDatabaseIterator iterator();
    friend PatientDatabaseIterator;

    Patient get_patient(uint32_t patient_id);
    uint32_t size();

    // Dictionary handling
    Dictionary& get_code_dictionary();
    Dictionary& get_unique_text_dictionary();
    Dictionary& get_shared_text_dictionary();

    // Ontology
    Ontology& get_ontology();

    // Indexing
    absl::Span<const uint32_t> get_patient_ids_with_code(uint32_t code);
    absl::Span<const uint32_t> get_patient_ids_with_codes(
        absl::Span<const uint32_t> codes);

    absl::Span<const uint32_t> get_patient_ids_with_shared_text(
        uint32_t text_value);
    absl::Span<const uint32_t> get_patient_ids_with_shared_text(
        absl::Span<const uint32_t> text_values);

    // Map back to original patient ids
    boost::optional<uint32_t> get_patient_id_from_original(
        uint64_t original_patient_id);
    uint64_t get_original_patient_id(uint32_t patient_id);

    // Count information
    uint32_t get_code_count(uint32_t code);
    uint32_t get_shared_text_count(uint32_t text_value);

   private:
    LazyDictionary patients;

    Ontology ontology;

    LazyDictionary shared_text_dictionary;
    LazyDictionary unique_text_dictionary;

    LazyDictionary code_index_dictionary;
    LazyDictionary value_index_dictionary;

    // 0 original_patient_ids
    // 1 sorted_original_patient_ids
    // 2 code counts
    // 3 text value counts
    // 4 original_code_ids
    LazyDictionary meta_dictionary;
};

PatientDatabase convert_patient_collection_to_patient_database(
    const boost::filesystem::path& patient_root,
    const boost::filesystem::path& concept,
    const boost::filesystem::path& target, char delimiter, size_t num_threads);

Ontology create_ontology(const std::vector<uint64_t>& raw_codes,
                         const boost::filesystem::path& concept,
                         const boost::filesystem::path& target, char delimiter,
                         size_t num_threads);

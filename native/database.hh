#pragma once

#include <arpa/inet.h>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <cstdint>
#include <string_view>
#include <thread>
#include <vector>

#include "absl/time/civil_time.h"
#include "absl/types/span.h"
#include "dictionary.hh"
#include "picosha2.h"

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

    operator bool() const { return boost::filesystem::exists(path); }

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

    std::string_view get_text_description(uint32_t code);

    boost::optional<uint32_t> get_code_from_concept_id(uint64_t concept_id);
    uint64_t get_concept_id_from_code(uint32_t code);

   private:
    LazyDictionary main_dictionary;
    LazyDictionary parent_dict;
    LazyDictionary children_dict;
    LazyDictionary all_parents_dict;

    LazyDictionary text_description;
    LazyDictionary concept_ids;
};

enum class ValueType {
    NONE,
    NUMERIC,
    SHARED_TEXT,
    UNIQUE_TEXT,
};

struct Event {
    uint32_t start_age_in_minutes;
    uint32_t code;
    ValueType value_type;

    union {
        float numeric_value;
        uint32_t text_value;
    };

    bool operator==(const Event& other) const {
        return (start_age_in_minutes == other.start_age_in_minutes &&
                value_type == other.value_type &&
                text_value == other.text_value);
    }
};

struct Patient {
    uint32_t patient_offset;
    absl::CivilDay birth_date;
    std::vector<Event> events;
};

class PatientDatabase;

class PatientDatabaseIterator {
   public:
    Patient& get_patient(uint32_t patient_offset);

   private:
    PatientDatabaseIterator(PatientDatabase* d);
    friend PatientDatabase;

    PatientDatabase* const parent_database;

    Patient current_patient;
    std::vector<uint32_t> buffer;
};

class PatientDatabase {
   public:
    PatientDatabase(const boost::filesystem::path& path, bool read_all,
                    bool read_all_unique_text = false);
    PatientDatabase(PatientDatabase&&) = default;

    PatientDatabaseIterator iterator();
    friend PatientDatabaseIterator;

    Patient get_patient(uint32_t patient_offset);
    uint32_t size();

    // Dictionary handling
    Dictionary& get_code_dictionary();
    Dictionary* get_unique_text_dictionary();
    Dictionary& get_shared_text_dictionary();

    // Ontology
    Ontology& get_ontology();

    // Indexing
    absl::Span<const uint32_t> get_patient_offsets_with_code(uint32_t code);
    absl::Span<const uint32_t> get_patient_offsets_with_codes(
        absl::Span<const uint32_t> codes);

    absl::Span<const uint32_t> get_patient_offsets_with_shared_text(
        uint32_t text_value);
    absl::Span<const uint32_t> get_patient_offsets_with_shared_text(
        absl::Span<const uint32_t> text_values);

    // Map back to original patient ids
    boost::optional<uint32_t> get_patient_offset_from_patient_id(
        uint64_t patient_id);
    uint64_t get_patient_id(uint32_t patient_offset);
    absl::Span<const uint64_t> get_patient_ids();

    // Count information
    uint32_t get_code_count(uint32_t code);
    uint32_t get_shared_text_count(uint32_t text_value);

    uint32_t compute_split(uint32_t seed, uint32_t patient_offset) {
        uint32_t patient_id = get_patient_id(patient_offset);
        uint32_t network_patient_offset = htonl(patient_id);
        uint32_t network_seed = htonl(seed);

        char to_hash[sizeof(uint32_t) * 2];
        memcpy(to_hash, &network_seed, sizeof(uint32_t));
        memcpy(to_hash + sizeof(uint32_t), &network_patient_offset,
               sizeof(uint32_t));

        std::vector<unsigned char> hash(picosha2::k_digest_size);
        picosha2::hash256(std::begin(to_hash), std::end(to_hash),
                          std::begin(hash), std::end(hash));

        uint32_t result = 0;
        for (size_t i = 0; i < picosha2::k_digest_size; i++) {
            result = (result * 256 + hash[i]) % 100;
        }

        return result;
    }

    // Event metadata
    std::string_view get_event_metadata(uint32_t patient_offset,
                                        uint32_t event_index);

    // Metadata information
    uint32_t version_id();
    uint32_t database_id();

   private:
    LazyDictionary patients;

    Ontology ontology;

    LazyDictionary shared_text_dictionary;
    LazyDictionary unique_text_dictionary;
    bool has_unique_text_dictionary;

    LazyDictionary code_index_dictionary;
    LazyDictionary value_index_dictionary;

    LazyDictionary event_metadata_dictionary;
    bool has_event_metadata;

    // 0 patient_ids
    // 1 sorted_patient_offsets
    // 2 code counts
    // 3 text value counts
    // 4 original_code_ids
    // 5 version_id
    // 6 database_id
    Dictionary meta_dictionary;
};

PatientDatabase convert_patient_collection_to_patient_database(
    const boost::filesystem::path& patient_root,
    const boost::filesystem::path& concept,
    const boost::filesystem::path& target, char delimiter, size_t num_threads);

Ontology create_ontology(std::vector<uint64_t> raw_codes,
                         const boost::filesystem::path& concept,
                         const boost::filesystem::path& target, char delimiter,
                         size_t num_threads);

template <typename Ret, typename Arg, typename... Rest>
Arg first_argument_helper(Ret (*)(Arg, Rest...));

template <typename Ret, typename F, typename Arg, typename... Rest>
Arg first_argument_helper(Ret (F::*)(Arg, Rest...));

template <typename Ret, typename F, typename Arg, typename... Rest>
Arg first_argument_helper(Ret (F::*)(Arg, Rest...) const);

template <typename F>
decltype(first_argument_helper(&F::operator())) first_argument_helper(F);

template <typename T>
using first_argument =
    std::remove_reference_t<decltype(first_argument_helper(std::declval<T>()))>;

template <typename F, typename R>
first_argument<R> proccess_patients_in_parallel(PatientDatabase& database,
                                                size_t num_threads, F func,
                                                R reducer) {
    std::vector<std::thread> threads;

    uint32_t pids_per_thread =
        (database.size() + num_threads - 1) / num_threads;
    std::vector<first_argument<R>> temp_results(num_threads);

    // Got to prime the pump by triggering lazy loading
    database.iterator().get_patient(0);

    for (size_t i = 0; i < num_threads; i++) {
        threads.emplace_back(
            std::thread([i, &temp_results, &database, &func, &reducer,
                         &num_threads, &pids_per_thread]() {
                PatientDatabaseIterator iter = database.iterator();
                uint32_t start_pid = pids_per_thread * i;
                uint32_t end_pid = std::min(
                    database.size(), (uint32_t)(pids_per_thread * (i + 1)));
                for (uint32_t pat_id = start_pid; pat_id < end_pid; pat_id++) {
                    func(temp_results[i], iter.get_patient(pat_id));
                }
            }));
    }

    first_argument<R> final_result;
    for (size_t i = 0; i < num_threads; i++) {
        threads[i].join();
        reducer(final_result, temp_results[i]);
    }

    return final_result;
}

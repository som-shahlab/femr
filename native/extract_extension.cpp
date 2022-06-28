#include "extract_extension.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string_view>

namespace py = pybind11;

#include <sys/resource.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "concept.h"
#include "constdb.h"
#include "csv.h"
#include "gem.h"
#include "reader.h"
#include "rxnorm.h"
#include "sort_csv.h"
#include "umls.h"
#include "writer.h"

std::vector<std::string> map_terminology_type(std::string_view terminology) {
    if (terminology == "CPT4") {
        return {"CPT"};
    } else if (terminology == "LOINC") {
        return {"LNC"};
    } else if (terminology == "HCPCS") {
        return {"HCPCS", "CDT"};
    } else {
        return {std::string(terminology)};
    }
}

std::vector<uint32_t> compute_subwords(
    std::string aui, const UMLS& umls,
    absl::flat_hash_map<std::string, std::vector<uint32_t>>&
        aui_to_subwords_map,
    absl::flat_hash_map<uint32_t, std::vector<uint32_t>>& code_to_parents_map,
    TermDictionary& dictionary) {
    if (aui_to_subwords_map.find(aui) == std::end(aui_to_subwords_map)) {
        std::vector<uint32_t> results;
        auto parents = umls.get_parents(aui);

        auto info = umls.get_code(aui);

        if (!info) {
            std::cout << "Could not find " << aui << std::endl;
            abort();
        }

        std::string word = absl::Substitute("$0/$1", info->first, info->second);

        uint32_t aui_code = dictionary.map_or_add(word);

        results.push_back(aui_code);

        for (const auto& parent_aui : parents) {
            auto parent_info_iter = umls.get_code(parent_aui);

            if (!parent_info_iter) {
                std::cout << "Could not find " << parent_aui << std::endl;
                abort();
            }

            std::string parent_word = absl::Substitute(
                "$0/$1", parent_info_iter->first, parent_info_iter->second);

            code_to_parents_map[aui_code].push_back(
                dictionary.map_or_add(parent_word));

            for (uint32_t subword :
                 compute_subwords(parent_aui, umls, aui_to_subwords_map,
                                  code_to_parents_map, dictionary)) {
                results.push_back(subword);
            }
        }

        if (std::find(std::begin(results), std::end(results),
                      dictionary.map_or_add("SRC/V-SRC")) ==
            std::end(results)) {
            std::cout << "AUI " << aui << " has no root " << std::endl;

            for (const auto& item : results) {
                std::cout << "Got " << dictionary.get_word(item).value()
                          << std::endl;
            }

            return {};
        }

        std::sort(std::begin(results), std::end(results));
        results.erase(std::unique(std::begin(results), std::end(results)),
                      std::end(results));
        aui_to_subwords_map.insert(std::make_pair(aui, std::move(results)));
    }

    return aui_to_subwords_map.find(aui)->second;
}

void create_ontology(std::string_view root_path, std::string umls_path,
                     std::string cdm_location, const ConceptTable& table) {
    std::string source = absl::Substitute("$0/extract.db", root_path);
    ExtractReader extract(source.c_str(), false);

    const TermDictionary& dictionary = extract.get_dictionary();

    auto entries = dictionary.decompose();

    UMLS umls(absl::Substitute("$0/META", umls_path));

    TermDictionary ontology_dictionary;
    OntologyCodeDictionary aui_text_description_dictionary;

    std::string ontology_path = absl::Substitute("$0/ontology.db", root_path);
    ConstdbWriter ontology_writer(ontology_path.c_str());

    absl::flat_hash_map<std::string, std::vector<uint32_t>> aui_to_subwords_map;
    absl::flat_hash_map<uint32_t, std::vector<uint32_t>> code_to_parents_map;

    std::vector<uint32_t> words_with_subwords;
    std::vector<uint32_t> recorded_date_codes;

    std::set<std::string> code_types;

    std::set<std::string> recorded_date_code_types = {
        "ATC",   "CPT4",    "DRG",      "Gender",
        "HCPCS", "ICD10CM", "ICD10PCS", "LOINC"};

    for (uint32_t i = 0; i < entries.size(); i++) {
        const auto& word = entries[i].first;
        std::vector<uint32_t> subwords = {};

        std::vector<std::string_view> parts = absl::StrSplit(word, '/');

        if (parts.size() != 2) {
            std::cout << "Got weird vocab string " << word << std::endl;
            abort();
        }

        code_types.insert(std::string(parts[0]));

        if (recorded_date_code_types.find(std::string(parts[0])) !=
            std::end(recorded_date_code_types)) {
            recorded_date_codes.push_back(i);
        }

        std::optional<std::string> result = std::nullopt;

        for (std::string terminology : map_terminology_type(parts[0])) {
            auto res = umls.get_aui(terminology, std::string(parts[1]));
            if (res) {
                result = res;
            }
        }

        if (result == std::nullopt) {
            subwords = {ontology_dictionary.map_or_add(
                absl::Substitute("NO_MAP/$0", word))};
            aui_text_description_dictionary.add(word, "NO_DEF");
        } else {
            subwords =
                compute_subwords(*result, umls, aui_to_subwords_map,
                                 code_to_parents_map, ontology_dictionary);
            auto res = umls.get_definition(*result);
            if (res) {
                aui_text_description_dictionary.add(word, *res);
            } else {
                aui_text_description_dictionary.add(word, "NO_DEF");
            }
        }

        ontology_writer.add_int(i, (const char*)subwords.data(),
                                subwords.size() * sizeof(uint32_t));
        words_with_subwords.push_back(i);
    }

    for (auto& iter : code_to_parents_map) {
        auto& parent_codes = iter.second;
        std::sort(std::begin(parent_codes), std::end(parent_codes));

        int32_t subword_as_int = iter.first + 1;
        ontology_writer.add_int(-subword_as_int,
                                (const char*)parent_codes.data(),
                                parent_codes.size() * sizeof(uint32_t));
    }

    for (auto& type : code_types) {
        std::cout << "Got type " << type << std::endl;
    }

    std::string dictionary_str = ontology_dictionary.to_json();
    ontology_writer.add_str("dictionary", dictionary_str.data(),
                            dictionary_str.size());
    std::string description_dictionary_str =
        aui_text_description_dictionary.to_json();
    ontology_writer.add_str("text_description_dictionary",
                            description_dictionary_str.data(),
                            description_dictionary_str.size());
    ontology_writer.add_str("words_with_subwords",
                            (const char*)words_with_subwords.data(),
                            words_with_subwords.size() * sizeof(uint32_t));
    ontology_writer.add_str("recorded_date_codes",
                            (const char*)recorded_date_codes.data(),
                            recorded_date_codes.size() * sizeof(uint32_t));
    uint32_t root_node = *ontology_dictionary.map("SRC/V-SRC");
    ontology_writer.add_str("root", (const char*)&root_node, sizeof(uint32_t));
}

struct RawPatientRecord {
    uint64_t person_id;
    std::optional<absl::CivilDay> birth_date;
    std::vector<std::pair<absl::CivilDay, uint32_t>> observations;
    std::vector<std::pair<absl::CivilDay, std::pair<uint32_t, uint32_t>>>
        observations_with_values;
};

using QueueItem = std::variant<RawPatientRecord, Metadata>;
using Queue = BlockingQueue<QueueItem>;

absl::CivilDay parse_date(std::string_view datestr) {
    std::string_view time_column = datestr;
    auto location = time_column.find(' ');
    if (location != std::string_view::npos) {
        time_column = time_column.substr(0, location);
    }

    location = time_column.find('T');
    if (location != std::string_view::npos) {
        time_column = time_column.substr(0, location);
    }

    auto first_dash = time_column.find('-');
    int year;
    attempt_parse_or_die(time_column.substr(0, first_dash), year);
    time_column = time_column.substr(first_dash + 1, std::string_view::npos);

    auto second_dash = time_column.find('-');
    int month;
    attempt_parse_or_die(time_column.substr(0, second_dash), month);
    time_column = time_column.substr(second_dash + 1, std::string_view::npos);

    int day;
    attempt_parse_or_die(time_column, day);

    return absl::CivilDay(year, month, day);
}

class Converter {
   public:
    std::string_view get_file_prefix() const;
    std::vector<std::string_view> get_columns() const;

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<std::string_view>& row) const;
};

template <typename C>
void run_converter(C converter, Queue& queue, boost::filesystem::path file,
                   char delimiter, bool use_quotes) {
    Metadata meta;

    size_t num_rows = 0;

    RawPatientRecord current_record;
    current_record.person_id = 0;

    std::vector<std::string_view> columns = converter.get_columns();
    columns.push_back("person_id");

    csv_iterator(file.c_str(), columns, delimiter, {}, true, false,
                 [&](const auto& row) {
                     num_rows++;

                     if (num_rows % 100000000 == 0) {
                         std::cout
                             << absl::Substitute("Processed $0 rows for $1\n",
                                                 num_rows, file.string());
                     }

                     uint64_t person_id;
                     attempt_parse_or_die(row[columns.size() - 1], person_id);

                     if (person_id != current_record.person_id) {
                         if (current_record.person_id) {
                             queue.wait_enqueue({std::move(current_record)});
                         }

                         current_record = {};
                         current_record.person_id = person_id;
                     }

                     converter.augment_day(meta, current_record, row);
                 });

    if (current_record.person_id) {
        queue.wait_enqueue({std::move(current_record)});
    }

    std::cout << absl::Substitute("Done working on $0\n", file.string());

    queue.wait_enqueue({std::move(meta)});
}

class DemographicsConverter : public Converter {
   public:
    std::string_view get_file_prefix() const { return "person"; }

    std::vector<std::string_view> get_columns() const {
        return {"birth_DATETIME",
                "gender_source_concept_id",
                "ethnicity_source_concept_id",
                "race_source_concept_id",
                "year_of_birth",
                "month_of_birth",
                "day_of_birth"};
    }

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<std::string_view>& row) const {
        absl::CivilDay birth;
        if (!row[0].empty()) {
            birth = parse_date(row[0]);
            patient_record.birth_date = birth;
        } else {
            int year = 1900;
            int month = 1;
            int day = 1;

            if (!row[4].empty()) {
                attempt_parse_or_die(row[4], year);
            }
            if (!row[5].empty()) {
                attempt_parse_or_die(row[5], month);
            }
            if (!row[6].empty()) {
                attempt_parse_or_die(row[6], day);
            }
            birth = absl::CivilDay(year, month, day);
        }
        patient_record.birth_date = birth;

        uint32_t gender_code = meta.dictionary.map_or_add(row[1]);
        patient_record.observations.push_back(
            std::make_pair(birth, gender_code));

        uint32_t ethnicity_code = meta.dictionary.map_or_add(row[2]);
        patient_record.observations.push_back(
            std::make_pair(birth, ethnicity_code));

        uint32_t race_code = meta.dictionary.map_or_add(row[3]);
        patient_record.observations.push_back(std::make_pair(birth, race_code));
    }
};

class StandardConceptTableConverter : public Converter {
   public:
    StandardConceptTableConverter(std::string f, std::string d, std::string c)
        : prefix(f), date_field(d), concept_id_field(c) {}

    std::string_view get_file_prefix() const { return prefix; }

    std::vector<std::string_view> get_columns() const {
        return {date_field, concept_id_field};
    }

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<std::string_view>& row) const {
        if (row[1] == "") {
            std::cout << "Got invalid code " << std::endl;
            abort();
        }
        if (row[1] == "0") {
            return;
        }
        patient_record.observations.push_back(std::make_pair(
            parse_date(row[0]), meta.dictionary.map_or_add(row[1])));
    }

   private:
    std::string prefix;
    std::string date_field;
    std::string concept_id_field;
};

class VisitConverter : public Converter {
   public:
    std::string_view get_file_prefix() const { return "visit_occurrence"; }

    std::vector<std::string_view> get_columns() const {
        return {"visit_start_DATE", "visit_concept_id", "visit_end_DATE"};
    }

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<std::string_view>& row) const {
        std::string_view code = row[1];

        if (code == "0") {
            return;
        }

        auto start_day = parse_date(row[0]);
        auto end_day = parse_date(row[2]);

        int days = end_day - start_day;

        ObservationWithValue obs;
        obs.code = meta.dictionary.map_or_add(code);
        obs.is_text = false;
        obs.numeric_value = days;

        patient_record.observations_with_values.push_back(
            std::make_pair(start_day, obs.encode()));
    }
};

class MeasurementConverter : public Converter {
   public:
    std::string_view get_file_prefix() const { return "measurement"; }

    std::vector<std::string_view> get_columns() const {
        return {"measurement_DATE", "measurement_source_concept_id",
                "value_as_number", "value_source_value"};
    }

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<std::string_view>& row) const {
        std::string_view code = row[1];
        if (code == "") {
            std::cout << "Got invalid code" << std::endl;
            abort();
        }
        if (code == "0") {
            return;
        }
        std::string_view value;

        if (row[3] != "") {
            value = row[3];
        } else if (row[2] != "") {
            value = row[2];
        } else {
            value = "";
        }

        auto day = parse_date(row[0]);

        if (value == "") {
            patient_record.observations.push_back(
                std::make_pair(day, meta.dictionary.map_or_add(code)));
        } else {
            ObservationWithValue obs;
            obs.code = meta.dictionary.map_or_add(code);

            float numeric_value;
            bool is_valid_numeric = absl::SimpleAtof(value, &numeric_value);

            if (is_valid_numeric) {
                obs.is_text = false;
                obs.numeric_value = numeric_value;
            } else {
                obs.is_text = true;
                obs.text_value = meta.value_dictionary.map_or_add(value);
            }

            patient_record.observations_with_values.push_back(
                std::make_pair(day, obs.encode()));
        }
    }
};

template <typename C>
std::pair<std::thread, std::shared_ptr<Queue>> generate_converter_thread(
    const C& converter, boost::filesystem::path target, char delimiter,
    bool use_quotes) {
    std::shared_ptr<Queue> queue =
        std::make_shared<Queue>(10000);  // Ten thousand patient records
    std::thread thread([converter, queue, target, delimiter, use_quotes]() {
        std::string thread_name = target.string();
        thread_name = thread_name.substr(0, 15);
        std::string name_copy(std::begin(thread_name), std::end(thread_name));
        int error = pthread_setname_np(pthread_self(), name_copy.c_str());
        if (error != 0) {
            std::cout << "Could not set thread name to " << thread_name << " "
                      << error << std::endl;
            abort();
        }
        run_converter(std::move(converter), *queue, target, delimiter,
                      use_quotes);
    });

    return std::make_pair(std::move(thread), std::move(queue));
}

template <typename C>
std::vector<std::pair<std::thread, std::shared_ptr<Queue>>>
generate_converter_threads(
    const C& converter,
    const std::vector<boost::filesystem::path>& possible_targets,
    char delimiter, bool use_quotes) {
    std::vector<std::pair<std::thread, std::shared_ptr<Queue>>> results;

    std::vector<std::string> options = {
        absl::Substitute("/$0/", converter.get_file_prefix()),
        absl::Substitute("/$0.csv.gz", converter.get_file_prefix()),
        absl::Substitute("/$00", converter.get_file_prefix()),
    };

    for (const auto& target : possible_targets) {
        bool found = false;
        for (const auto& option : options) {
            if (target.string().find(option) != std::string::npos &&
                target.string().find(".csv.gz") != std::string::npos) {
                found = true;
                break;
            }
        }
        if (found) {
            results.push_back(generate_converter_thread(converter, target,
                                                        delimiter, use_quotes));
        }
    }

    return results;
}

class HeapItem {
   public:
    HeapItem(size_t _index, QueueItem _item)
        : index(_index), item(std::move(_item)) {}

    bool operator<(const HeapItem& second) const {
        std::optional<uint64_t> first_person_id = get_person_id();
        std::optional<uint64_t> second_person_id = second.get_person_id();

        uint64_t limit = std::numeric_limits<uint64_t>::max();

        return first_person_id.value_or(limit) >
               second_person_id.value_or(limit);
    }

    std::optional<uint64_t> get_person_id() const {
        return std::visit(
            [](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;

                if constexpr (std::is_same_v<T, RawPatientRecord>) {
                    return std::optional<uint64_t>(arg.person_id);
                } else {
                    return std::optional<uint64_t>();
                }
            },
            item);
    }

    size_t index;
    QueueItem item;
};

class Merger {
   public:
    Merger(std::vector<std::pair<std::thread, std::shared_ptr<Queue>>>
               _converter_threads)
        : converter_threads(std::move(_converter_threads)) {
        for (size_t i = 0; i < converter_threads.size(); i++) {
            const auto& entry = converter_threads[i];
            QueueItem nextItem;
            entry.second->wait_dequeue(nextItem);
            heap.push_back(HeapItem(i, std::move(nextItem)));
        }

        std::make_heap(std::begin(heap), std::end(heap));

        if (heap.size() == 0) {
            std::cout << "No converters in the heap?" << std::endl;
            abort();
        }
    }

    WriterItem operator()() {
        while (true) {
            std::optional<uint64_t> possible_person_id =
                heap.front().get_person_id();

            if (possible_person_id.has_value()) {
                RawPatientRecord total_record;
                total_record.person_id = possible_person_id.value();
                total_record.birth_date = {};

                contributing_indexes.clear();

                while (heap.front().get_person_id() == possible_person_id) {
                    std::pop_heap(std::begin(heap), std::end(heap));
                    HeapItem& heap_item = heap.back();
                    QueueItem& queue_item = heap_item.item;

                    size_t index = heap_item.index;
                    RawPatientRecord& record =
                        std::get<RawPatientRecord>(queue_item);

                    if (record.birth_date) {
                        total_record.birth_date = record.birth_date;
                    }

                    auto offset =
                        [&](uint32_t val,
                            absl::flat_hash_map<std::pair<size_t, uint32_t>,
                                                uint32_t>& m) {
                            auto iter = m.find(std::make_pair(index, val));
                            if (iter == std::end(m)) {
                                auto res = m.insert(std::make_pair(
                                    std::make_pair(index, val), m.size()));
                                iter = res.first;
                            }
                            return iter->second;
                        };

                    for (const auto& obs : record.observations) {
                        total_record.observations.push_back(std::make_pair(
                            obs.first, offset(obs.second, mapper)));
                    }

                    for (const auto& obs : record.observations_with_values) {
                        ObservationWithValue obs_with_value(obs.second.first,
                                                            obs.second.second);

                        obs_with_value.code =
                            offset(obs_with_value.code, mapper);

                        if (obs_with_value.is_text) {
                            obs_with_value.text_value =
                                offset(obs_with_value.text_value, value_mapper);
                        }

                        total_record.observations_with_values.push_back(
                            std::make_pair(obs.first, obs_with_value.encode()));
                    }

                    converter_threads[index].second->wait_dequeue(queue_item);
                    contributing_indexes.push_back(index);

                    std::push_heap(std::begin(heap), std::end(heap));
                }

                total_patients++;

                if (!total_record.birth_date) {
                    lost_patients++;

                    if (rand() % lost_patients == 0) {
                        std::cout
                            << "You have a patient without a birth date?? "
                            << total_record.person_id << " so far "
                            << lost_patients << " out of " << total_patients
                            << std::endl;
                        for (const auto& c_index : contributing_indexes) {
                            char thread_name[16];
                            pthread_getname_np(converter_threads[c_index]
                                                   .first.native_handle(),
                                               thread_name,
                                               sizeof(thread_name));
                            std::cout << "Thread: " << thread_name << std::endl;
                        }
                        std::cout << std::endl;
                    }

                    continue;
                }

                if (contributing_indexes.size() == 1) {
                    // Only got the person thread
                    ignored_patients++;

                    if (rand() % ignored_patients == 0) {
                        std::cout << "You are ignoring a patient "
                                  << total_record.person_id
                                  << ", you have ignored " << ignored_patients
                                  << " out of " << total_patients << std::endl;
                    }

                    continue;
                } else {
                    if (rand() % total_patients == 0) {
                        std::cout
                            << "You finished a patient "
                            << total_record.person_id
                            << ", total finished so far: " << total_patients
                            << std::endl;
                    }
                }

                PatientRecord final_record;
                final_record.person_id = total_record.person_id;

                final_record.birth_date = *total_record.birth_date;

                for (const auto& observ : total_record.observations) {
                    auto delta = observ.first - final_record.birth_date;
                    if (delta < 0) {
                        delta = 0;
                    }
                    final_record.observations.push_back(
                        std::make_pair(delta, observ.second));
                }

                for (const auto& observ :
                     total_record.observations_with_values) {
                    auto delta = observ.first - final_record.birth_date;
                    if (delta < 0) {
                        delta = 0;
                    }
                    final_record.observations_with_values.push_back(
                        std::make_pair(delta, observ.second));
                }

                return final_record;
            } else {
                Metadata total_metadata;

                std::vector<std::vector<std::pair<std::string, uint32_t>>>
                    dictionaries(heap.size());
                std::vector<std::vector<std::pair<std::string, uint32_t>>>
                    value_dictionaries(heap.size());

                std::vector<std::pair<std::string, uint32_t>> value_dictionary;

                for (auto& heap_item : heap) {
                    QueueItem& queue_item = heap_item.item;

                    size_t index = heap_item.index;
                    Metadata& meta = std::get<Metadata>(queue_item);
                    dictionaries[index] = meta.dictionary.decompose();
                    value_dictionaries[index] =
                        meta.value_dictionary.decompose();
                }

                auto process =
                    [&](const absl::flat_hash_map<std::pair<size_t, uint32_t>,
                                                  uint32_t>& m,
                        const std::vector<
                            std::vector<std::pair<std::string, uint32_t>>>&
                            dicts) {
                        std::vector<std::pair<std::string, uint32_t>> dict(
                            m.size());

                        for (const auto& item : m) {
                            dict[item.second] =
                                dicts[item.first.first][item.first.second];
                        }

                        return TermDictionary(dict);
                    };

                total_metadata.dictionary = process(mapper, dictionaries);
                total_metadata.value_dictionary =
                    process(value_mapper, value_dictionaries);

                std::cout << "Done with " << lost_patients
                          << " lost patients and " << ignored_patients
                          << " ignored patients out of " << total_patients
                          << std::endl;

                return total_metadata;
            }
        }
    }

    ~Merger() {
        std::cout << "Joining threads" << std::endl;

        for (auto& entry : converter_threads) {
            entry.first.join();
        }

        std::cout << "Done joining" << std::endl;
    }

   private:
    int lost_patients = 0;
    int ignored_patients = 0;
    int total_patients = 0;
    std::vector<size_t> contributing_indexes;
    std::vector<HeapItem> heap;
    std::vector<std::pair<std::thread, std::shared_ptr<Queue>>>
        converter_threads;
    absl::flat_hash_map<std::pair<size_t, uint32_t>, uint32_t> mapper;
    absl::flat_hash_map<std::pair<size_t, uint32_t>, uint32_t> value_mapper;
};

std::function<std::optional<PatientRecord>()> convert_vector_to_iter(
    std::vector<PatientRecord> r) {
    size_t index = 0;
    return [index, records = std::move(r)]() mutable {
        if (index == records.size()) {
            return std::optional<PatientRecord>();
        } else {
            return std::optional<PatientRecord>(std::move(records[index++]));
        }
    };
}

TermDictionary counts_to_dict(
    const absl::flat_hash_map<std::string, uint32_t>& counts) {
    TermDictionary result;

    std::vector<std::pair<uint32_t, std::string>> entries;

    for (const auto& iter : counts) {
        entries.push_back(std::make_pair(iter.second, iter.first));
    }

    std::sort(std::begin(entries), std::end(entries),
              std::greater<std::pair<uint32_t, std::string>>());

    for (const auto& row : entries) {
        result.map_or_add(row.second, row.first);
    }

    return result;
}

std::vector<std::string> normalize(std::string input_code,
                                   const ConceptTable& table,
                                   const GEMMapper& gem, const RxNorm& rxnorm) {
    if (input_code == "" || input_code == "0") {
        return {};
    }

    uint32_t concept_id;
    attempt_parse_or_die(input_code, concept_id);

    std::set<std::string> good_items = {"LOINC",
                                        "ICD10CM",
                                        "CPT4",
                                        "Gender",
                                        "HCPCS",
                                        "Ethnicity",
                                        "Race",
                                        "ICD10PCS",
                                        "Condition Type",
                                        "Visit",
                                        "CMS Place of Service",
                                        "SNOMED"};
    std::set<std::string> bad_items = {"NDC", "ICD10CN", "ICD10",
                                       "ICD9ProcCN"};

    std::vector<std::string> results;

    std::optional<ConceptInfo> info_ptr = table.get_info(concept_id);
    if (!info_ptr) {
        std::cout << "Could not find the concept id " << concept_id
                  << std::endl;
        abort();
    }

    ConceptInfo info = *info_ptr;

    if (info.vocabulary_id.rfind("STANFORD_", 0) == 0) {
        // Try to save the stanford concept
        auto possib = table.get_custom_map(concept_id);

        if (possib.has_value()) {
            concept_id = *possib;
            info_ptr = table.get_info(concept_id);
            if (!info_ptr) {
                std::cout << "Could not find the concept id " << concept_id
                          << std::endl;
                abort();
            }
            info = *info_ptr;
        } else {
            std::cout << "WAT " << concept_id << " " << info.vocabulary_id
                      << std::endl;
        }
    }

    if (info.vocabulary_id == "RxNorm" || info.vocabulary_id == "NDC" ||
        info.vocabulary_id == "HCPCS") {
        // Need to map over to ATC to avoid painful issues
        results = rxnorm.get_atc_codes(info.vocabulary_id, info.concept_code);

        if (results.empty() && info.vocabulary_id == "HCPCS") {
            // If we fail to map to a drug, take it normally
            results.push_back(absl::Substitute("$0/$1", info.vocabulary_id,
                                               info.concept_code));
        } else if (results.empty()) {
            std::cout << "Could not map ? " << info.vocabulary_id << " "
                      << info.concept_code << std::endl;
        }
    } else if (info.vocabulary_id == "ICD9Proc") {
        for (const auto& proc : gem.map_proc(info.concept_code)) {
            results.push_back(absl::Substitute("ICD10PCS/$0", proc));
        }
    } else if (info.vocabulary_id == "ICD9CM") {
        for (std::string diag : gem.map_diag(info.concept_code)) {
            results.push_back(absl::Substitute("ICD10CM/$0", diag));
        }
    } else if (good_items.find(info.vocabulary_id) != std::end(good_items)) {
        if (info.vocabulary_id == "Condition Type") {
            std::string final = absl::Substitute("$0/$1", info.concept_class_id,
                                                 info.concept_code);

            results.push_back(final);
        } else {
            std::string final = absl::Substitute("$0/$1", info.vocabulary_id,
                                                 info.concept_code);

            results.push_back(final);
        }
    } else if (bad_items.find(info.vocabulary_id) != std::end(bad_items)) {
        return {};
    } else {
        std::cout << "Could not handle '" << info.vocabulary_id << "' '"
                  << input_code << "'" << std::endl;
        return {};
    }

    return results;
}

class Cleaner {
   public:
    Cleaner(const ConceptTable& concepts, const GEMMapper& gem,
            const RxNorm& rxnorm, const char* path)
        : reader(path, false), iterator(reader.iter()) {
        patient_ids = reader.get_patient_ids();
        original_patient_ids = reader.get_original_patient_ids();
        current_index = 0;

        {
            TermDictionary temp_dictionary;
            std::vector<std::pair<std::string, uint32_t>> items =
                reader.get_dictionary().decompose();

            remap_dict.reserve(items.size());

            absl::flat_hash_map<std::string, uint32_t> lost_counts;

            uint32_t total_lost = 0;

            for (const auto& entry : items) {
                std::vector<std::string> terms =
                    normalize(entry.first, concepts, gem, rxnorm);
                if (terms.size() == 0) {
                    total_lost += entry.second;
                    lost_counts[entry.first] += entry.second;
                }

                std::vector<uint32_t> result;
                for (const auto& term : terms) {
                    result.push_back(
                        temp_dictionary.map_or_add(term, entry.second));
                }

                remap_dict.push_back(result);
            }

            std::cout << "Lost items " << total_lost << std::endl;

            std::vector<std::pair<int32_t, std::string>> lost_entries;

            for (const auto& entry : lost_counts) {
                lost_entries.push_back(
                    std::make_pair(-entry.second, entry.first));
            }
            std::sort(std::begin(lost_entries), std::end(lost_entries));

            for (size_t i = 0; i < 30 && i < lost_entries.size(); i++) {
                const auto& entry = lost_entries[i];
                std::cout << entry.second << " " << entry.first << std::endl;
            }

            auto [a, b] = temp_dictionary.optimize();
            final_dictionary = a;

            for (auto& entry : remap_dict) {
                for (auto& val : entry) {
                    val = b[val];
                }
            }
        }

        {
            TermDictionary temp_dictionary;

            std::vector<std::pair<std::string, uint32_t>> items =
                reader.get_value_dictionary().decompose();
            value_remap_dict.reserve(items.size());

            for (const auto& entry : items) {
                value_remap_dict.push_back(
                    temp_dictionary.map_or_add(entry.first, entry.second));
            }

            auto [a, b] = temp_dictionary.optimize();
            final_value_dictionary = a;

            for (auto& entry : value_remap_dict) {
                entry = b[entry];
            }
        }

        std::cout << "Dictionary size " << reader.get_dictionary().size()
                  << std::endl;
        std::cout << "Value dictionary size "
                  << reader.get_value_dictionary().size() << std::endl;
        std::cout << "Final Dictionary size " << final_dictionary.size()
                  << std::endl;
        std::cout << "Final Value dictionary size "
                  << final_value_dictionary.size() << std::endl;
        std::cout << "Num patients " << patient_ids.size() << std::endl;
    }

    WriterItem operator()() {
        if (current_index == patient_ids.size()) {
            Metadata meta;
            meta.dictionary = final_dictionary;
            meta.value_dictionary = final_value_dictionary;
            return meta;
        } else {
            uint32_t patient_id = patient_ids[current_index];

            PatientRecord record;
            record.person_id = original_patient_ids[current_index];
            current_index++;

            iterator.process_patient(
                patient_id, [&](absl::CivilDay birth_date, uint32_t age,
                                const std::vector<uint32_t>& observations,
                                const std::vector<ObservationWithValue>&
                                    observations_with_values) {
                    record.birth_date = birth_date;

                    for (const auto& obs : observations) {
                        for (const auto& remapped : remap_dict[obs]) {
                            record.observations.push_back(
                                std::make_pair(age, remapped));
                        }
                    }

                    for (const auto& obs_with_value :
                         observations_with_values) {
                        for (const auto& remapped_code :
                             remap_dict[obs_with_value.code]) {
                            ObservationWithValue new_obs;
                            new_obs.code = remapped_code;

                            if (obs_with_value.is_text) {
                                new_obs.is_text = true;
                                new_obs.text_value =
                                    value_remap_dict[obs_with_value.text_value];
                            } else {
                                new_obs.is_text = false;
                                new_obs.numeric_value =
                                    obs_with_value.numeric_value;
                            }

                            record.observations_with_values.push_back(
                                std::make_pair(age, new_obs.encode()));
                        }
                    }
                });

            return record;
        }
    }

   private:
    ExtractReader reader;

    std::vector<std::vector<uint32_t>> remap_dict;
    std::vector<uint32_t> value_remap_dict;

    TermDictionary final_dictionary;
    TermDictionary final_value_dictionary;

    ExtractReaderIterator iterator;
    absl::Span<const uint32_t> patient_ids;
    absl::Span<const uint64_t> original_patient_ids;
    size_t current_index;
};

void create_extract(std::string omop_source_dir, std::string target_directory,
                    const ConceptTable& concepts, const GEMMapper& gem,
                    const RxNorm& rxnorm, char delimiter, bool use_quotes) {
    std::vector<std::pair<std::thread, std::shared_ptr<Queue>>>
        converter_threads;

    std::vector<boost::filesystem::path> targets;

    for (const auto& p :
         boost::filesystem::recursive_directory_iterator(omop_source_dir)) {
        targets.push_back(p);
    }

    auto helper = [&](const auto& c) {
        auto results =
            generate_converter_threads(c, targets, delimiter, use_quotes);

        for (auto& result : results) {
            converter_threads.push_back(std::move(result));
        }
    };

    helper(DemographicsConverter());
    helper(VisitConverter());
    helper(MeasurementConverter());
    helper(StandardConceptTableConverter(
        "drug_exposure", "drug_exposure_start_date", "drug_source_concept_id"));
    helper(StandardConceptTableConverter("death", "death_date",
                                         "death_type_concept_id"));
    helper(StandardConceptTableConverter("condition_occurrence",
                                         "condition_start_date",
                                         "condition_source_concept_id"));
    helper(StandardConceptTableConverter("procedure_occurrence",
                                         "procedure_date",
                                         "procedure_source_concept_id"));
    helper(StandardConceptTableConverter("device_exposure",
                                         "device_exposure_start_date",
                                         "device_source_concept_id"));
    helper(StandardConceptTableConverter("observation", "observation_date",
                                         "observation_source_concept_id"));

    std::string tmp_extract = absl::Substitute("$0/tmp.db", target_directory);
    std::string final_extract =
        absl::Substitute("$0/extract.db", target_directory);

    write_timeline(tmp_extract.c_str(), Merger(std::move(converter_threads)));

    write_timeline(final_extract.c_str(),
                   Cleaner(concepts, gem, rxnorm, tmp_extract.c_str()));

    boost::filesystem::remove(tmp_extract);
}

void perform_omop_extraction(std::string omop_source_dir_str,
                             std::string umls_dir, std::string gem_dir,
                             std::string rxnorm_dir, std::string target_dir_str,
                             char delimiter, bool use_quotes) {
    struct rlimit current_limit;

    if (getrlimit(RLIMIT_NOFILE, &current_limit) != 0) {
        perror("Could not get the limit on the number of files");
        abort();
    }

    current_limit.rlim_cur = current_limit.rlim_max;

    if (setrlimit(RLIMIT_NOFILE, &current_limit) != 0) {
        perror("Could not set the limit on the number of files");
        abort();
    }

    boost::filesystem::path omop_source_dir =
        boost::filesystem::canonical(omop_source_dir_str);
    boost::filesystem::path target_dir =
        boost::filesystem::weakly_canonical(target_dir_str);

    if (!boost::filesystem::create_directory(target_dir)) {
        std::cout << absl::Substitute(
            "Could not make result directory $0, got error $1\n",
            target_dir.string(), std::strerror(errno));
        exit(-1);
    }

    boost::filesystem::path sorted_dir = target_dir / "sorted";
    boost::filesystem::create_directory(sorted_dir);

    sort_csvs(omop_source_dir, sorted_dir, delimiter, use_quotes);

    ConceptTable concepts = construct_concept_table(omop_source_dir.string(),
                                                    delimiter, use_quotes);

    GEMMapper gem(gem_dir);
    RxNorm rxnorm(rxnorm_dir);

    create_extract(sorted_dir.string(), target_dir.string(), concepts, gem,
                   rxnorm, delimiter, use_quotes);

    boost::filesystem::remove_all(sorted_dir);

    create_ontology(target_dir.string(), umls_dir, omop_source_dir.string(),
                    concepts);
}


void register_extract_extension(py::module& root) {
    py::module m = root.def_submodule("extract");

}

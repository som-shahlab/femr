#include "database.hh"

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <boost/range/iterator_range.hpp>
#include <deque>
#include <queue>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "blockingconcurrentqueue.h"
#include "count_codes_and_values.hh"
#include "csv.hh"
#include "parse_utils.hh"
#include "readerwritercircularbuffer.h"
#include "streamvbyte.h"
#include "thread_utils.hh"

constexpr int QUEUE_SIZE = 1000;
constexpr absl::CivilDay epoch(1900);

constexpr int seconds_per_minute = 60;
constexpr int minutes_per_hour = 60;
constexpr int hours_per_day = 24;

template <typename T>
std::string_view container_to_view(const T& data) {
    return std::string_view(reinterpret_cast<const char*>(data.data()),
                            data.size() * sizeof(data[0]));
}

template <typename T>
absl::Span<const T> read_span(const Dictionary& dict, uint32_t index) {
    std::string_view mapping = dict[index];
    return {reinterpret_cast<const T*>(mapping.data()),
            mapping.size() / sizeof(T)};
}

using UniqueValue = std::pair<uint32_t, std::string>;

struct Entry {
    uint64_t original_patient_id;
    std::string bytes;
    std::vector<UniqueValue> unique_values;
};

void write_patient_to_buffer(uint64_t original_patient_id,
                             const Patient& current_patient,
                             std::vector<uint32_t>& buffer) {
    if (current_patient.birth_date < epoch) {
        throw std::runtime_error(
            absl::StrCat("Cannot have a birth date before epoch (1900) ",
                         absl::FormatCivilTime(current_patient.birth_date),
                         " for ", original_patient_id));
    }
    buffer.clear();
    buffer.push_back(current_patient.birth_date - epoch);
    buffer.push_back(current_patient.events.size());

    ssize_t count_offset = -1;

    int32_t last_age = 0;
    int32_t last_minute_offset = 0;

    for (const Event& event : current_patient.events) {
        int32_t delta = static_cast<int32_t>(event.age_in_days) - last_age;
        if (delta < 0) {
            throw std::runtime_error(absl::StrCat(
                "Patient days are not sorted in order ", original_patient_id,
                " with ", event.age_in_days, " ", delta));
        }
        int32_t minute_delta =
            ((minutes_per_hour * hours_per_day) +
             static_cast<int32_t>(event.minutes_offset) - last_minute_offset) %
            (minutes_per_hour * hours_per_day);

        if (minute_delta < 0) {
            throw std::runtime_error(
                absl::StrCat("Should not have a ngeative minute delta?"));
        }
        last_minute_offset = event.minutes_offset;
        last_age = event.age_in_days;

        if (delta == 0 && minute_delta == 0 && count_offset != -1) {
            buffer[count_offset] += 1;
        } else {
            count_offset = buffer.size();
            buffer.push_back(0);
            buffer.push_back(delta);
            buffer.push_back(minute_delta);
        }

        switch (event.value_type) {
            case ValueType::NONE:
                buffer.push_back((event.code << 2));
                break;

            case ValueType::UNIQUE_TEXT:
            case ValueType::SHARED_TEXT: {
                buffer.push_back((event.code << 2) | 1);
                bool is_shared = event.value_type == ValueType::SHARED_TEXT;
                buffer.push_back((event.text_value << 1) |
                                 static_cast<uint32_t>(is_shared));
                break;
            }

            case ValueType::NUMERIC:
                if (static_cast<uint32_t>(event.numeric_value) ==
                    event.numeric_value) {
                    buffer.push_back((event.code << 2) | 2);
                    buffer.push_back(event.numeric_value);
                } else {
                    buffer.push_back((event.code << 2) | 3);
                    buffer.push_back(event.text_value);
                }
                break;

            default:
                throw std::runtime_error("Invalid value type?");
        }
    }
}

void read_patient_from_buffer(Patient& current_patient,
                              const std::vector<uint32_t>& buffer,
                              uint32_t count) {
    size_t index = 0;
    current_patient.birth_date = epoch + buffer[index++];
    current_patient.events.resize(buffer[index++]);

    uint32_t last_age = 0;
    uint32_t last_minutes = 0;

    uint32_t count_with_same = 0;
    for (Event& event : current_patient.events) {
        if (count_with_same == 0) {
            count_with_same = buffer[index++];
            last_age += buffer[index++];
            last_minutes += buffer[index++];
            last_minutes %= (minutes_per_hour * hours_per_day);
        } else {
            count_with_same--;
        }
        event.age_in_days = last_age;
        event.minutes_offset = last_minutes;

        uint32_t code_and_type = buffer[index++];
        event.code = code_and_type >> 2;
        uint32_t type = code_and_type & 3;

        switch (type) {
            case 0:
                event.value_type = ValueType::NONE;
                break;

            case 1: {
                uint32_t text_value = buffer[index++];
                bool is_shared = (text_value & 1) == 1;
                event.value_type =
                    is_shared ? ValueType::SHARED_TEXT : ValueType::UNIQUE_TEXT;
                event.text_value = text_value >> 1;
                break;
            }

            case 2:
            case 3: {
                event.value_type = ValueType::NUMERIC;
                if (type == 2) {
                    event.numeric_value = buffer[index++];
                } else {
                    event.text_value = buffer[index++];
                }
                break;
            }

            default:
                throw std::runtime_error("Invalid value type?");
        }
    }

    if (index != count) {
        throw std::runtime_error(
            absl::StrCat("Did not read through the entire patient record? ",
                         index, " ", buffer.size()));
    }
}

void reader_thread(
    const boost::filesystem::path& patient_file,
    moodycamel::BlockingReaderWriterCircularBuffer<boost::optional<Entry>>&
        queue,
    std::atomic<uint32_t>& unique_counter,
    const absl::flat_hash_map<uint64_t, uint32_t>& code_to_index,
    const absl::flat_hash_map<std::string, uint32_t>& text_value_to_index) {
    CSVReader reader(patient_file, {"patient_id", "code", "start", "value"},
                     ',');

    Entry current_entry;
    current_entry.original_patient_id = 0;

    Patient current_patient;

    absl::CivilSecond birth_date;

    std::vector<uint32_t> buffer;
    std::string byte_buffer;
    auto output_patient = [&]() {
        if (current_entry.original_patient_id == 0) {
            return;
        }

        write_patient_to_buffer(current_entry.original_patient_id,
                                current_patient, buffer);

        if (byte_buffer.size() <
            streamvbyte_max_compressedbytes(buffer.size())) {
            byte_buffer.resize(streamvbyte_max_compressedbytes(buffer.size()) *
                               2);
        }

        size_t num_bytes =
            streamvbyte_encode(buffer.data(), buffer.size(),
                               reinterpret_cast<uint8_t*>(byte_buffer.data()));

        if (buffer.size() >= std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(absl::StrCat(
                "Cannot create a patient with more than uint32_t events ... ",
                current_entry.original_patient_id, " ", buffer.size()));
        }
        uint32_t count = buffer.size();
        current_entry.bytes.resize(sizeof(count) + num_bytes);
        std::memcpy(current_entry.bytes.data(), &count, sizeof(count));
        std::memcpy(current_entry.bytes.data() + sizeof(count),
                    byte_buffer.data(), num_bytes);

        queue.wait_enqueue({std::move(current_entry)});
    };

    while (reader.next_row()) {
        uint64_t patient_id;
        attempt_parse_or_die(reader.get_row()[0], patient_id);
        uint64_t code;
        attempt_parse_or_die(reader.get_row()[1], code);
        absl::CivilSecond time;
        attempt_parse_time_or_die(reader.get_row()[2], time);

        if (patient_id != current_entry.original_patient_id) {
            output_patient();

            current_patient.birth_date = absl::CivilDay(time);
            birth_date = absl::CivilSecond(current_patient.birth_date);
            current_patient.events.clear();

            current_entry.original_patient_id = patient_id;
            current_entry.unique_values.clear();
        }

        Event next_event;
        uint64_t age_in_seconds = (time - birth_date);
        next_event.age_in_days =
            age_in_seconds /
            (seconds_per_minute * minutes_per_hour * hours_per_day);
        next_event.minutes_offset =
            (age_in_seconds / seconds_per_minute) -
            (next_event.age_in_days * hours_per_day * minutes_per_hour);
        next_event.code = code_to_index.find(code)->second;

        if (reader.get_row()[3].empty()) {
            next_event.value_type = ValueType::NONE;
        } else {
            bool parse_number = absl::SimpleAtof(reader.get_row()[3],
                                                 &next_event.numeric_value);
            if (parse_number) {
                next_event.value_type = ValueType::NUMERIC;
            } else {
                auto iter = text_value_to_index.find(reader.get_row()[3]);
                if (iter != std::end(text_value_to_index)) {
                    next_event.value_type = ValueType::SHARED_TEXT;
                    next_event.text_value = iter->second;
                } else {
                    next_event.value_type = ValueType::UNIQUE_TEXT;
                    next_event.text_value = unique_counter.fetch_add(1);
                    current_entry.unique_values.emplace_back(
                        next_event.text_value, std::move(reader.get_row()[3]));
                }
            }
        }

        current_patient.events.push_back(next_event);
    }

    output_patient();

    queue.wait_enqueue(boost::none);
}

PatientDatabase convert_patient_collection_to_patient_database(
    const boost::filesystem::path& patient_root,
    const boost::filesystem::path& concept_root,
    const boost::filesystem::path& target, char delimiter, size_t num_threads) {
    boost::filesystem::create_directories(target);
    std::cout << "Counting " << absl::Now() << std::endl;

    boost::filesystem::path temp_path =
        target / boost::filesystem::unique_path();
    boost::filesystem::create_directory(temp_path);
    auto codes_and_values =
        count_codes_and_values(patient_root, temp_path, num_threads);

    std::vector<uint64_t> codes;
    codes.reserve(codes_and_values.first.size());
    absl::flat_hash_map<uint64_t, uint32_t> code_to_index;
    code_to_index.reserve(codes_and_values.first.size());
    for (size_t i = 0; i < codes_and_values.first.size(); i++) {
        const auto& entry = codes_and_values.first[i];
        code_to_index[entry.first] = i;
        codes.push_back(entry.first);
    }

    std::cout << "Creating ontology " << absl::Now() << std::endl;

    create_ontology(codes, concept_root, target / "ontology", delimiter,
                    num_threads);
    std::cout << "Done with ontology " << absl::Now() << std::endl;

    absl::flat_hash_map<std::string, uint32_t> text_value_to_index;
    {
        DictionaryWriter writer(target / "shared_text");

        for (size_t i = 0; i < codes_and_values.second.size(); i++) {
            const auto& entry = codes_and_values.second[i];
            text_value_to_index[entry.first] = i;
            writer.add_value(entry.first);
        }
    }

    std::vector<uint64_t> original_patient_ids;
    {
        DictionaryWriter patients(target / "patients");

        std::vector<std::thread> threads;

        std::vector<boost::filesystem::path> files;

        for (auto& entry : boost::make_iterator_range(
                 boost::filesystem::directory_iterator(patient_root), {})) {
            files.emplace_back(entry.path());
        }

        std::vector<moodycamel::BlockingReaderWriterCircularBuffer<
            boost::optional<Entry>>>
            queues;
        queues.reserve(files.size());

        std::atomic<uint32_t> unique_counter(0);

        for (size_t i = 0; i < files.size(); i++) {
            queues.emplace_back(QUEUE_SIZE);
            threads.emplace_back([i, &files, &queues, &unique_counter,
                                  &text_value_to_index, &code_to_index]() {
                reader_thread(files[i], queues[i], unique_counter,
                              code_to_index, text_value_to_index);
            });
        }

        uint32_t next_write_unique = 0;
        DictionaryWriter unique_text(target / "unique_text");
        std::priority_queue<UniqueValue, std::vector<UniqueValue>,
                            std::greater<>>
            unique_value_heap;

        dequeue_many_loop(queues, [&](Entry& entry) {
            for (auto& unique_value : entry.unique_values) {
                if (unique_value.first == next_write_unique) {
                    unique_text.add_value(unique_value.second);
                    next_write_unique++;

                    while (!unique_value_heap.empty() &&
                           unique_value_heap.top().first == next_write_unique) {
                        unique_text.add_value(unique_value_heap.top().second);
                        next_write_unique++;
                        unique_value_heap.pop();
                    }
                } else {
                    unique_value_heap.emplace(std::move(unique_value));
                }
            }

            original_patient_ids.push_back(entry.original_patient_id);
            patients.add_value(entry.bytes);
        });

        if (!unique_value_heap.empty()) {
            throw std::runtime_error(
                "Should have an empty heap after done processing?");
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }

    std::cout << "Done with main " << absl::Now() << std::endl;

    {
        DictionaryWriter meta(target / "meta");

        meta.add_value(container_to_view(original_patient_ids));

        std::vector<uint32_t> sorted_indices;
        for (size_t i = 0; i < original_patient_ids.size(); i++) {
            sorted_indices.push_back(i);
        }
        std::sort(std::begin(sorted_indices), std::end(sorted_indices),
                  [&](uint32_t a, uint32_t b) {
                      return original_patient_ids[a] < original_patient_ids[b];
                  });

        meta.add_value(container_to_view(sorted_indices));

        auto add_counts = [&meta](const auto& a) {
            std::vector<uint32_t> result;
            result.reserve(a.size());
            for (auto& entry : a) {
                result.push_back(entry.second);
            }

            meta.add_value(container_to_view(result));
        };

        add_counts(codes_and_values.first);
        add_counts(codes_and_values.second);
        meta.add_value(container_to_view(codes));
    }
    std::cout << "Done with meta " << absl::Now() << std::endl;

    return PatientDatabase(target, false);
}

PatientDatabaseIterator::PatientDatabaseIterator(const Dictionary* d)
    : parent_dictionary(d) {}

Patient& PatientDatabaseIterator::get_patient(uint32_t patient_id) {
    std::string_view data = (*parent_dictionary)[patient_id];

    uint32_t count;
    std::memcpy(&count, data.data(), sizeof(count));
    if (buffer.size() < count) {
        buffer.resize(count * 2);
    }

    streamvbyte_decode(
        reinterpret_cast<const uint8_t*>(data.data() + sizeof(count)),
        buffer.data(), count);

    current_patient.patient_id = patient_id;
    read_patient_from_buffer(current_patient, buffer, count);

    return current_patient;
}

PatientDatabase::PatientDatabase(boost::filesystem::path const& path,
                                 bool read_all)
    : patients(path / "patients", read_all),
      ontology(path / "ontology"),
      shared_text_dictionary(path / "shared_text", read_all),
      unique_text_dictionary(path / "unique_text", read_all),
      code_index_dictionary(path / "code_index", read_all),
      value_index_dictionary(path / "value_index", read_all),
      meta_dictionary(path / "meta", read_all) {}

uint32_t PatientDatabase::size() { return patients->size(); }

PatientDatabaseIterator PatientDatabase::iterator() {
    return PatientDatabaseIterator(&(*patients));
}

Patient PatientDatabase::get_patient(uint32_t patient_id) {
    auto iter = iterator();
    return std::move(iter.get_patient(patient_id));
}

uint64_t PatientDatabase::get_original_patient_id(uint32_t patient_id) {
    return read_span<uint64_t>(*meta_dictionary, 0)[patient_id];
}

boost::optional<uint32_t> PatientDatabase::get_patient_id_from_original(
    uint64_t original_patient_id) {
    absl::Span<const uint32_t> sorted_span =
        read_span<uint32_t>(*meta_dictionary, 1);
    const auto* iter = std::lower_bound(
        std::begin(sorted_span), std::end(sorted_span), original_patient_id,
        [&](uint32_t index, uint64_t original) {
            return get_original_patient_id(index) < original;
        });
    if (iter == std::end(sorted_span) ||
        get_original_patient_id(*iter) != original_patient_id) {
        return {};
    } else {
        return *iter;
    }
}

uint32_t PatientDatabase::get_code_count(uint32_t code) {
    return read_span<uint32_t>(*meta_dictionary, 2)[code];
}

uint32_t PatientDatabase::get_shared_text_count(uint32_t value) {
    return read_span<uint32_t>(*meta_dictionary, 3)[value];
}

Ontology& PatientDatabase::get_ontology() { return ontology; }

Dictionary& PatientDatabase::get_code_dictionary() {
    return get_ontology().get_dictionary();
}

Dictionary& PatientDatabase::get_shared_text_dictionary() {
    return *shared_text_dictionary;
}

Dictionary& PatientDatabase::get_unique_text_dictionary() {
    return *unique_text_dictionary;
}

template <typename F>
void process_nested_helper(
    moodycamel::BlockingConcurrentQueue<
        boost::optional<boost::filesystem::path>>& queue,
    const F& f,
    std::vector<std::result_of_t<F(const boost::filesystem::path&)>>& result) {
    boost::optional<boost::filesystem::path> next_item;
    while (true) {
        queue.wait_dequeue(next_item);

        if (!next_item) {
            break;
        } else {
            result.push_back(f(*next_item));
        }
    }
}

template <typename F>
std::vector<std::result_of_t<F(const boost::filesystem::path)>> process_nested(
    const boost::filesystem::path& root, std::string_view prefix,
    size_t num_threads, F f) {
    boost::filesystem::path directory = root / std::string(prefix);
    boost::filesystem::path direct_file =
        root / (std::string(prefix) + ".csv.tsv");
    if (boost::filesystem::exists(direct_file)) {
        return {f(direct_file)};
    } else if (boost::filesystem::exists(directory)) {
        std::vector<std::thread> threads;
        moodycamel::BlockingConcurrentQueue<
            boost::optional<boost::filesystem::path>>
            queue;
        std::vector<
            std::vector<std::result_of_t<F(const boost::filesystem::path&)>>>
            result_queues(num_threads);

        for (auto& entry : boost::make_iterator_range(
                 boost::filesystem::directory_iterator(directory), {})) {
            boost::filesystem::path source = entry.path();
            queue.enqueue(source);
        }

        for (size_t i = 0; i < num_threads; i++) {
            threads.emplace_back([i, &queue, &f, &result_queues]() {
                process_nested_helper(queue, f, result_queues[i]);
            });
            queue.enqueue(boost::none);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        size_t size = 0;
        for (const auto& vec : result_queues) {
            size += vec.size();
        }
        std::vector<std::result_of_t<F(const boost::filesystem::path&)>>
            final_result;
        final_result.reserve(size);
        for (auto& vec : result_queues) {
            for (auto& entry : vec) {
                final_result.emplace_back(std::move(entry));
            }
        }
        return final_result;
    } else {
        throw std::runtime_error(absl::StrCat("Could not find directory ",
                                              root.string(), " , ", prefix));
    }
}

absl::flat_hash_set<uint64_t> get_standard_codes(
    const boost::filesystem::path& concept, char delimiter,
    size_t num_threads) {
    auto valid = process_nested(
        concept, "concept", num_threads,
        [&](const boost::filesystem::path& path) {
            std::vector<uint64_t> result;

            CSVReader reader(path, {"concept_id", "standard_concept"},
                             delimiter);

            while (reader.next_row()) {
                uint64_t concept_id;
                attempt_parse_or_die(reader.get_row()[0], concept_id);

                if (reader.get_row()[1] != "") {
                    result.push_back(concept_id);
                }
            }

            return result;
        });

    absl::flat_hash_set<uint64_t> result;
    for (const auto& entry : valid) {
        for (const auto& val : entry) {
            result.insert(val);
        }
    }

    return result;
}

std::pair<absl::flat_hash_map<uint64_t, uint32_t>,
          std::vector<std::vector<uint32_t>>>
get_parents(const std::vector<uint64_t>& raw_codes,
            const boost::filesystem::path& concept, char delimiter,
            size_t num_threads) {
    auto standard_code_map =
        get_standard_codes(concept, delimiter, num_threads);

    using ParentMap =
        absl::flat_hash_map<uint64_t,
                            std::vector<std::tuple<bool, size_t, uint64_t>>>;

    std::vector<std::string> valid_rels = {"Has precise ingredient",
                                           "RxNorm has ing",
                                           "Quantified form of",
                                           "Has ingredient",
                                           "Form of",
                                           "Consists of",
                                           "Is a"};

    auto parents = process_nested(
        concept, "concept_relationship", num_threads,
        [&](const boost::filesystem::path& path) {
            ParentMap result;

            CSVReader reader(
                path, {"concept_id_1", "concept_id_2", "relationship_id"},
                delimiter);

            while (reader.next_row()) {
                const auto& rel_id = reader.get_row()[2];
                uint64_t concept_id_1;
                attempt_parse_or_die(reader.get_row()[0], concept_id_1);
                uint64_t concept_id_2;
                attempt_parse_or_die(reader.get_row()[1], concept_id_2);

                bool is_non_standard =
                    standard_code_map.count(concept_id_2) == 0;

                size_t valid_rel_index;
                for (valid_rel_index = 0; valid_rel_index < valid_rels.size();
                     valid_rel_index++) {
                    if (valid_rels[valid_rel_index] == rel_id) {
                        break;
                    }
                }

                if (valid_rel_index < valid_rels.size()) {
                    result[concept_id_1].push_back(std::make_tuple(
                        is_non_standard, valid_rel_index, concept_id_2));
                }
            }

            return result;
        });

    ParentMap merged;

    for (const auto& parent : parents) {
        for (const auto& entry : parent) {
            auto& target_entry = merged[entry.first];
            target_entry.insert(std::end(target_entry),
                                std::begin(entry.second),
                                std::end(entry.second));
        }
    }

    for (auto& entry : merged) {
        // Only use the ideal mappings
        std::sort(std::begin(entry.second), std::end(entry.second));

        auto first = entry.second[0];

        auto desired = std::make_pair(std::get<0>(first), std::get<1>(first));

        auto invalid = std::remove_if(
            std::begin(entry.second), std::end(entry.second),
            [&](const auto& tup) {
                return std::make_pair(std::get<0>(tup), std::get<1>(tup)) !=
                       desired;
            });

        entry.second.erase(invalid, std::end(entry.second));
    }

    absl::flat_hash_map<uint64_t, uint32_t> index_map;

    std::deque<uint64_t> to_process;

    uint32_t next_index = 0;

    for (uint64_t code : raw_codes) {
        index_map[code] = next_index++;
        to_process.push_back(code);
    }

    auto get_index = [&](uint64_t target_index) {
        auto iter = index_map.find(target_index);
        if (iter == std::end(index_map)) {
            if (to_process.size() > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error(
                    "Cannot process more than uint32_t codes");
            }
            uint32_t index = next_index++;
            index_map[target_index] = index;
            to_process.push_back(target_index);
            return index;
        } else {
            return iter->second;
        }
    };

    std::vector<std::vector<uint32_t>> result;
    while (!to_process.empty()) {
        const auto& ps = merged[to_process.front()];
        to_process.pop_front();

        std::vector<uint32_t> indices;
        indices.reserve(merged.size());
        for (auto parent : ps) {
            indices.push_back(get_index(std::get<2>(parent)));
        }

        std::sort(std::begin(indices), std::end(indices));
        auto last = std::unique(std::begin(indices), std::end(indices));
        indices.erase(last, std::end(indices));
        result.emplace_back(std::move(indices));
    }

    return {std::move(index_map), std::move(result)};
}

std::vector<std::string> get_concept_text(
    const std::vector<uint64_t>& originals,
    const absl::flat_hash_map<uint64_t, uint32_t>& index_map,
    const boost::filesystem::path& concept, char delimiter,
    size_t num_threads) {
    auto texts = process_nested(
        concept, "concept", num_threads,
        [&](const boost::filesystem::path& path) {
            std::vector<std::pair<std::string, uint32_t>> result;

            CSVReader reader(path,
                             {"concept_id", "concept_code", "vocabulary_id"},
                             delimiter);

            while (reader.next_row()) {
                uint64_t concept_id;
                attempt_parse_or_die(reader.get_row()[0], concept_id);

                auto iter = index_map.find(concept_id);
                if (iter != std::end(index_map)) {
                    std::string text = absl::StrCat(reader.get_row()[2], "/",
                                                    reader.get_row()[1]);
                    result.emplace_back(std::move(text), iter->second);
                }
            }

            return result;
        });
    std::vector<std::string> result(index_map.size());
    for (auto& text : texts) {
        for (auto& entry : text) {
            result[entry.second] = std::move(entry.first);
        }
    }

    for (size_t i = 0; i < result.size(); i++) {
        if (result[i].empty()) {
            throw std::runtime_error(
                absl::StrCat("Could not map ", i, " ", originals[i]));
        }
    }
    return result;
}

const std::vector<uint32_t>& all_parents_helper(
    std::vector<boost::optional<std::vector<uint32_t>>>& all_parents,
    std::vector<std::vector<uint32_t>>& parents, uint32_t index) {
    auto& value = all_parents[index];
    if (!value) {
        std::vector<uint32_t> result;
        for (const auto& p : parents[index]) {
            const auto& p_res = all_parents_helper(all_parents, parents, p);
            result.insert(std::end(result), std::begin(p_res), std::end(p_res));
        }
        result.push_back(index);

        std::sort(std::begin(result), std::end(result));
        auto last = std::unique(std::begin(result), std::end(result));
        result.erase(last, std::end(result));
        value = result;
    }
    return *value;
}

Ontology create_ontology(const std::vector<uint64_t>& raw_codes,
                         const boost::filesystem::path& concept,
                         const boost::filesystem::path& target, char delimiter,
                         size_t num_threads) {
    boost::filesystem::create_directory(target);
    auto parent_info = get_parents(raw_codes, concept, delimiter, num_threads);
    auto text = get_concept_text(raw_codes, parent_info.first, concept,
                                 delimiter, num_threads);

    {
        DictionaryWriter main(target / "main");
        for (const auto& t : text) {
            main.add_value(t);
        }
    }
    {
        std::vector<std::vector<uint32_t>> children(parent_info.second.size());

        DictionaryWriter parent(target / "parent");
        for (size_t i = 0; i < parent_info.second.size(); i++) {
            const auto& parents = parent_info.second[i];

            for (uint32_t p : parents) {
                children[p].push_back(i);
            }

            parent.add_value(container_to_view(parents));
        }

        DictionaryWriter child(target / "children");
        for (const auto& ch : children) {
            child.add_value(container_to_view(ch));
        }
    }
    {
        std::vector<boost::optional<std::vector<uint32_t>>> all_parents(
            parent_info.second.size());

        DictionaryWriter all_parent(target / "all_parents");
        for (size_t i = 0; i < parent_info.second.size(); i++) {
            const auto& parents =
                all_parents_helper(all_parents, parent_info.second, i);

            all_parent.add_value(container_to_view(parents));
        }
    }
    return Ontology(target);
}

Ontology::Ontology(const boost::filesystem::path& path)
    : main_dictionary(path / "main", true),
      parent_dict(path / "parent", true),
      children_dict(path / "children", true),
      all_parents_dict(path / "all_parents", true) {}

absl::Span<const uint32_t> Ontology::get_parents(uint32_t code) {
    return read_span<uint32_t>(*parent_dict, code);
}
absl::Span<const uint32_t> Ontology::get_children(uint32_t code) {
    return read_span<uint32_t>(*children_dict, code);
}
absl::Span<const uint32_t> Ontology::get_all_parents(uint32_t code) {
    return read_span<uint32_t>(*all_parents_dict, code);
}
Dictionary& Ontology::get_dictionary() { return *main_dictionary; }

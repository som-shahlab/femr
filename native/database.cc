#include "database.hh"

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <boost/range/iterator_range.hpp>
#include <deque>
#include <queue>
#include <random>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "base64.h"
#include "blockingconcurrentqueue.h"
#include "count_codes_and_values.hh"
#include "csv.hh"
#include "parse_utils.hh"
#include "readerwritercircularbuffer.h"
#include "streamvbyte.h"
#include "thread_utils.hh"

constexpr int QUEUE_SIZE = 1000;
constexpr absl::CivilDay epoch(1800);
constexpr absl::CivilDay legacy_epoch(1900);

constexpr int seconds_per_minute = 60;
constexpr int minutes_per_hour = 60;
constexpr int hours_per_day = 24;

constexpr uint32_t current_version = 3;

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

template <typename T>
std::string_view element_to_view(const T* data) {
    return absl::string_view(reinterpret_cast<const char*>(data),
                             sizeof(*data));
}

template <typename T>
T read_element(const Dictionary& dict, uint32_t index) {
    absl::Span<const T> span = read_span<T>(dict, index);
    assert(span.size() == 1);
    return span[0];
}

struct Entry {
    uint32_t patient_offset;
    uint64_t patient_id;

    uint32_t num_unique;
    uint32_t num_metadata;

    std::vector<std::string>
        data;  // of size 1 + num_unique + num_event
               // stores data, then unique values, then event metadata

    bool operator<(const Entry& r) const {
        return patient_offset > r.patient_offset;
    }
};

void write_patient_to_buffer(uint32_t start_unique, uint64_t patient_id,
                             const Patient& current_patient,
                             std::vector<uint32_t>& buffer) {
    if (current_patient.birth_date < epoch) {
        throw std::runtime_error(
            absl::StrCat("Cannot have a birth date before epoch (1800) ",
                         absl::FormatCivilTime(current_patient.birth_date),
                         " for ", patient_id));
    }
    buffer.clear();
    buffer.push_back(start_unique);
    buffer.push_back(current_patient.birth_date - epoch);
    buffer.push_back(current_patient.events.size());

    ssize_t count_offset = -1;

    int64_t last_age = 0;

    for (const Event& event : current_patient.events) {
        int64_t delta =
            static_cast<int64_t>(event.start_age_in_minutes) - last_age;
        if (delta < 0) {
            throw std::runtime_error(absl::StrCat(
                "Patient days are not sorted in order ", patient_id, " with ",
                event.start_age_in_minutes, " ", delta));
        }
        if (delta > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Out of bounds error?");
        }
        last_age = event.start_age_in_minutes;

        if (delta == 0 && count_offset != -1) {
            buffer[count_offset] += 1;
        } else {
            count_offset = buffer.size();
            buffer.push_back(0);
            buffer.push_back((uint32_t)delta);
        }

        if ((((uint64_t)event.code) << 2) >
            std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Numeric limits error");
        }

        uint32_t mask = (event.code << 2);

        switch (event.value_type) {
            case ValueType::NONE:
                buffer.push_back(mask | 0);
                break;

            case ValueType::UNIQUE_TEXT:
            case ValueType::SHARED_TEXT: {
                uint32_t text_value = event.text_value;

                if (event.value_type == ValueType::UNIQUE_TEXT) {
                    buffer.push_back((mask << 1) | 5);
                } else {
                    buffer.push_back((mask << 1) | 1);
                    buffer.push_back(text_value);
                }

                break;
            }

            case ValueType::NUMERIC:
                if (static_cast<uint32_t>(event.numeric_value) ==
                    event.numeric_value) {
                    buffer.push_back(mask | 2);
                    buffer.push_back(event.numeric_value);
                } else {
                    buffer.push_back(mask | 3);
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
                              uint32_t count, uint32_t version) {
    if (version == 0) {
        size_t index = 0;
        current_patient.birth_date = legacy_epoch + buffer[index++];
        current_patient.events.resize(buffer[index++]);

        uint64_t last_age = 0;
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
            uint64_t darn =
                last_age * minutes_per_hour * hours_per_day + last_minutes;
            if (darn > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("Invalid range ...");
            }
            event.start_age_in_minutes = darn;

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
                    event.value_type = is_shared ? ValueType::SHARED_TEXT
                                                 : ValueType::UNIQUE_TEXT;
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
            throw std::runtime_error(absl::StrCat(
                "v0 - Did not read through the entire patient record? ", index,
                " ", count, " ", buffer.size()));
        }
    } else if (version == 1) {
        size_t index = 0;
        current_patient.birth_date = legacy_epoch + buffer[index++];
        current_patient.events.resize(buffer[index++]);

        uint32_t last_age = 0;

        uint32_t count_with_same = 0;
        for (Event& event : current_patient.events) {
            if (count_with_same == 0) {
                count_with_same = buffer[index++];
                last_age += buffer[index++];
            } else {
                count_with_same--;
            }
            event.start_age_in_minutes = last_age;

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
                    event.value_type = is_shared ? ValueType::SHARED_TEXT
                                                 : ValueType::UNIQUE_TEXT;
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
            throw std::runtime_error(absl::StrCat(
                "v1 - Did not read through the entire patient record? ", index,
                " ", count, " ", buffer.size()));
        }
    } else if (version == 2) {
        size_t index = 0;
        uint32_t current_unique = buffer[index++];
        current_patient.birth_date = legacy_epoch + buffer[index++];
        current_patient.events.resize(buffer[index++]);

        uint32_t last_age = 0;

        uint32_t count_with_same = 0;
        for (Event& event : current_patient.events) {
            if (count_with_same == 0) {
                count_with_same = buffer[index++];
                last_age += buffer[index++];
            } else {
                count_with_same--;
            }
            event.start_age_in_minutes = last_age;

            uint32_t code_and_type = buffer[index++];
            event.code = code_and_type >> 2;
            uint32_t type = code_and_type & 3;

            switch (type) {
                case 0:
                    event.value_type = ValueType::NONE;
                    break;

                case 1: {
                    event.code = event.code >> 1;
                    if (code_and_type & 4) {
                        // Is unique
                        event.value_type = ValueType::UNIQUE_TEXT;
                        event.text_value = current_unique++;
                    } else {
                        event.value_type = ValueType::SHARED_TEXT;
                        event.text_value = buffer[index++];
                    }
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
            throw std::runtime_error(absl::StrCat(
                "v2 - Did not read through the entire patient record? ", index,
                " ", count, " ", buffer.size()));
        }
    } else if (version == 3) {
        size_t index = 0;
        uint32_t current_unique = buffer[index++];
        current_patient.birth_date = epoch + buffer[index++];
        current_patient.events.resize(buffer[index++]);

        uint32_t last_age = 0;

        uint32_t count_with_same = 0;
        for (Event& event : current_patient.events) {
            if (count_with_same == 0) {
                count_with_same = buffer[index++];
                last_age += buffer[index++];
            } else {
                count_with_same--;
            }
            event.start_age_in_minutes = last_age;

            uint32_t code_and_type = buffer[index++];
            event.code = code_and_type >> 2;
            uint32_t type = code_and_type & 3;

            switch (type) {
                case 0:
                    event.value_type = ValueType::NONE;
                    break;

                case 1: {
                    event.code = event.code >> 1;
                    if (code_and_type & 4) {
                        // Is unique
                        event.value_type = ValueType::UNIQUE_TEXT;
                        event.text_value = current_unique++;
                    } else {
                        event.value_type = ValueType::SHARED_TEXT;
                        event.text_value = buffer[index++];
                    }
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
            throw std::runtime_error(absl::StrCat(
                "v2 - Did not read through the entire patient record? ", index,
                " ", count, " ", buffer.size()));
        }
    } else {
        throw std::runtime_error(absl::StrCat(
            "Does not support reading femr databases of version ", version));
    }
}

void reader_thread(
    const boost::filesystem::path& patient_file,
    moodycamel::BlockingReaderWriterCircularBuffer<boost::optional<Entry>>&
        queue,
    std::atomic<uint64_t>& offset_and_unique_counter,
    const absl::flat_hash_map<uint64_t, uint32_t>& code_to_index,
    const absl::flat_hash_map<std::string, uint32_t>& text_value_to_index) {
    CSVReader<ZstdReader> reader(
        patient_file, {"patient_id", "concept_id", "start", "value", "metadata"},
        ',');

    uint64_t patient_id = 0;
    Patient current_patient;
    std::vector<std::string> current_unique;
    std::vector<std::string> current_metadata;

    absl::CivilSecond birth_date;

    std::vector<uint32_t> buffer;
    std::string byte_buffer;
    auto output_patient = [&]() {
        if (patient_id == 0) {
            return;
        }

        uint64_t offset_and_unique = offset_and_unique_counter.fetch_add(
            1ul << 32 | current_unique.size(),
            std::memory_order::memory_order_relaxed);

        uint32_t offset = offset_and_unique >> 32;
        uint32_t start_unique = offset_and_unique & 0xfffffffful;

        write_patient_to_buffer(start_unique, patient_id, current_patient,
                                buffer);

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
                patient_id, " ", buffer.size()));
        }

        uint32_t count = buffer.size();

        std::string bytes(sizeof(count) + num_bytes, 0);
        std::memcpy(bytes.data(), &count, sizeof(count));
        std::memcpy(bytes.data() + sizeof(count), byte_buffer.data(),
                    num_bytes);

        Entry next_entry;
        next_entry.patient_offset = offset;
        next_entry.patient_id = patient_id;
        next_entry.num_unique = current_unique.size();
        next_entry.num_metadata = current_metadata.size();

        next_entry.data.reserve(1 + current_unique.size() +
                                current_metadata.size());

        next_entry.data.emplace_back(std::move(bytes));
        std::move(std::begin(current_unique), std::end(current_unique),
                  std::back_inserter(next_entry.data));
        std::move(std::begin(current_metadata), std::end(current_metadata),
                  std::back_inserter(next_entry.data));

        queue.wait_enqueue({std::move(next_entry)});
    };

    while (reader.next_row()) {
        uint64_t patient_offset;
        attempt_parse_or_die(reader.get_row()[0], patient_offset);
        uint64_t code;
        attempt_parse_or_die(reader.get_row()[1], code);
        absl::CivilSecond start;
        attempt_parse_time_or_die(reader.get_row()[2], start);

        if (patient_offset != patient_id) {
            output_patient();

            current_patient.birth_date = absl::CivilDay(start);
            birth_date = absl::CivilSecond(current_patient.birth_date);
            current_patient.events.clear();

            patient_id = patient_offset;
            current_unique.clear();
            current_metadata.clear();
        }

        Event next_event;
        uint64_t start_age_in_seconds = (start - birth_date);
        next_event.start_age_in_minutes =
            start_age_in_seconds / seconds_per_minute;

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
                    current_unique.emplace_back(std::move(reader.get_row()[3]));
                }
            }
        }

        current_metadata.emplace_back(base64_decode(reader.get_row()[4]));

        current_patient.events.push_back(next_event);
    }

    output_patient();

    queue.wait_enqueue(boost::none);
}

void convert_patient_collection_to_patient_database(
    const boost::filesystem::path& patient_root,
    const boost::filesystem::path& concept_root,
    const boost::filesystem::path& target, char delimiter, size_t num_threads) {
    boost::filesystem::create_directories(target);

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

    create_ontology(codes, concept_root, target / "ontology", delimiter,
                    num_threads);

    absl::flat_hash_map<std::string, uint32_t> text_value_to_index;
    {
        DictionaryWriter writer(target / "shared_text");

        for (size_t i = 0; i < codes_and_values.second.size(); i++) {
            const auto& entry = codes_and_values.second[i];
            text_value_to_index[entry.first] = i;
            writer.add_value(entry.first);
        }
    }

    std::vector<uint64_t> patient_ids;
    {
        DictionaryWriter patients(target / "patients");
        DictionaryWriter event_metadata(target / "event_metadata");

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

        std::atomic<uint64_t> offset_and_unique_counter(0);

        for (size_t i = 0; i < files.size(); i++) {
            queues.emplace_back(QUEUE_SIZE);
            threads.emplace_back([i, &files, &queues,
                                  &offset_and_unique_counter,
                                  &text_value_to_index, &code_to_index]() {
                reader_thread(files[i], queues[i], offset_and_unique_counter,
                              code_to_index, text_value_to_index);
            });
        }

        uint32_t next_write_patient = 0;
        DictionaryWriter unique_text(target / "unique_text");
        std::priority_queue<Entry> entry_heap;

        auto write_patient = [&](const Entry& entry) {
            patient_ids.push_back(entry.patient_id);

            patients.add_value(entry.data[0]);

            for (uint32_t i = 0; i < entry.num_unique; i++) {
                unique_text.add_value(entry.data[i + 1]);
            }

            uint32_t total_length = 0;

            for (uint32_t i = 0; i < entry.num_metadata; i++) {
                total_length += entry.data[i + entry.num_unique + 1].size();
            }

            std::vector<uint32_t> event_metadata_offsets;
            std::string event_metadata_value;
            event_metadata_value.reserve(total_length);
            event_metadata_offsets.reserve(entry.num_metadata);

            for (uint32_t i = 0; i < entry.num_metadata; i++) {
                event_metadata_offsets.push_back(event_metadata_value.size());
                event_metadata_value.append(
                    entry.data[i + entry.num_unique + 1]);
            }

            event_metadata.add_value(container_to_view(event_metadata_offsets));
            event_metadata.add_value(event_metadata_value);

            next_write_patient++;
        };

        dequeue_many_loop(queues, [&](Entry& entry) {
            if (entry.patient_offset == next_write_patient) {
                write_patient(entry);

                while (!entry_heap.empty() &&
                       entry_heap.top().patient_offset == next_write_patient) {
                    write_patient(entry_heap.top());
                    entry_heap.pop();
                }
            } else {
                entry_heap.emplace(std::move(entry));
            }
        });

        if (!entry_heap.empty()) {
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

        meta.add_value(container_to_view(patient_ids));

        std::vector<uint32_t> sorted_indices;
        for (size_t i = 0; i < patient_ids.size(); i++) {
            sorted_indices.push_back(i);
        }
        std::sort(std::begin(sorted_indices), std::end(sorted_indices),
                  [&](uint32_t a, uint32_t b) {
                      return patient_ids[a] < patient_ids[b];
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

        meta.add_value(element_to_view(&current_version));

        std::random_device device;
        uint32_t extract_id = device();
        meta.add_value(element_to_view(&extract_id));
    }
    std::cout << "Done with meta " << absl::Now() << std::endl;
}

PatientDatabaseIterator::PatientDatabaseIterator(PatientDatabase* d)
    : parent_database(d) {
    (void)parent_database->patients->size();
}

Patient& PatientDatabaseIterator::get_patient(uint32_t patient_offset) {
    std::string_view data = (*(parent_database->patients))[patient_offset];

    uint32_t count;
    std::memcpy(&count, data.data(), sizeof(count));
    if (buffer.size() < count) {
        buffer.resize(count * 2);
    }

    if (parent_database->version_id() > 0) {
        streamvbyte_decode(
            reinterpret_cast<const uint8_t*>(data.data() + sizeof(count)),
            buffer.data(), count);
    } else {
        streamvbyte_decode(
            reinterpret_cast<const uint8_t*>(data.data() + sizeof(count)),
            buffer.data(), count);
    }

    current_patient.patient_offset = patient_offset;
    read_patient_from_buffer(current_patient, buffer, count,
                             parent_database->version_id());

    return current_patient;
}

PatientDatabase::PatientDatabase(boost::filesystem::path const& path,
                                 bool read_all, bool read_all_unique_text)
    : patients(path / "patients", read_all),
      ontology(path / "ontology"),
      shared_text_dictionary(path / "shared_text", read_all),
      unique_text_dictionary(path / "unique_text", read_all_unique_text),
      code_index_dictionary(path / "code_index", read_all),
      value_index_dictionary(path / "value_index", read_all),
      event_metadata_dictionary(path / "event_metadata", read_all),
      meta_dictionary(path / "meta", read_all) {
    (void)version_id();
}

uint32_t PatientDatabase::size() { return patients->size(); }

uint32_t PatientDatabase::version_id() {
    if (meta_dictionary.size() <= 5) {
        // Needed to handle very old extracts
        return 0;
    } else {
        return read_element<uint32_t>(meta_dictionary, 5);
    }
}

uint32_t PatientDatabase::database_id() {
    if (version_id() == 0) {
        return 0;
    } else {
        return read_element<uint32_t>(meta_dictionary, 6);
    }
}

PatientDatabaseIterator PatientDatabase::iterator() {
    return PatientDatabaseIterator(this);
}

Patient PatientDatabase::get_patient(uint32_t patient_offset) {
    auto iter = iterator();
    return std::move(iter.get_patient(patient_offset));
}

uint64_t PatientDatabase::get_patient_id(uint32_t patient_offset) {
    return read_span<uint64_t>(meta_dictionary, 0)[patient_offset];
}

absl::Span<const uint64_t> PatientDatabase::get_patient_ids() {
    return read_span<uint64_t>(meta_dictionary, 0);
}

boost::optional<uint32_t> PatientDatabase::get_patient_offset(
    uint64_t patient_id) {
    absl::Span<const uint32_t> sorted_span =
        read_span<uint32_t>(meta_dictionary, 1);
    const auto* iter =
        std::lower_bound(std::begin(sorted_span), std::end(sorted_span),
                         patient_id, [&](uint32_t index, uint64_t original) {
                             return get_patient_id(index) < original;
                         });
    if (iter == std::end(sorted_span) || get_patient_id(*iter) != patient_id) {
        return {};
    } else {
        return *iter;
    }
}

uint32_t PatientDatabase::get_code_count(uint32_t code) {
    return read_span<uint32_t>(meta_dictionary, 2)[code];
}

uint32_t PatientDatabase::get_shared_text_count(uint32_t value) {
    return read_span<uint32_t>(meta_dictionary, 3)[value];
}

Ontology& PatientDatabase::get_ontology() { return ontology; }

Dictionary& PatientDatabase::get_code_dictionary() {
    return get_ontology().get_dictionary();
}

Dictionary& PatientDatabase::get_shared_text_dictionary() {
    return *shared_text_dictionary;
}

Dictionary* PatientDatabase::get_unique_text_dictionary() {
    if (unique_text_dictionary) {
        return &(*unique_text_dictionary);
    } else {
        return nullptr;
    }
}

std::string_view PatientDatabase::get_event_metadata(uint32_t patient_offset,
                                                     uint32_t event_index) {
    if (!event_metadata_dictionary) {
        return std::string_view(nullptr, 0);
    }

    absl::Span<const uint32_t> event_offsets =
        read_span<uint32_t>(*event_metadata_dictionary, patient_offset * 2);

    std::string_view data =
        (*event_metadata_dictionary)[patient_offset * 2 + 1];
    uint32_t end;
    if (event_index == event_offsets.size() - 1) {
        end = data.size();
    } else {
        end = event_offsets[event_index + 1];
    }

    return data.substr(event_offsets[event_index],
                       end - event_offsets[event_index]);
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
std::vector<std::result_of_t<F(CSVReader<TextReader>&)>> process_nested(
    const boost::filesystem::path& root, std::string_view prefix,
    size_t num_threads, bool actually_necessary,
    const std::vector<std::string>& columns, char delimiter, F f) {
    boost::filesystem::path directory = root / std::string(prefix);

    boost::filesystem::path direct_file_uncompressed =
        root / (std::string(prefix) + ".csv");
    boost::filesystem::path direct_file_compressed =
        root / (std::string(prefix) + ".csv.zst");

    using R = std::result_of_t<F(CSVReader<TextReader>&)>;

    auto helper = [&f, &columns,
                   &delimiter](const boost::filesystem::path& path) -> R {
        if (!boost::filesystem::exists(path)) {
            return {};
        } else {
            if (path.extension() == ".zst") {
                CSVReader<ZstdReader> reader(path, columns, delimiter);
                return f(reader);
            } else {
                CSVReader<TextReader> reader(path, columns, delimiter);
                return f(reader);
            }
        }
    };

    if (boost::filesystem::exists(direct_file_compressed)) {
        return {helper(direct_file_compressed)};
    } else if (boost::filesystem::exists(direct_file_uncompressed)) {
        return {helper(direct_file_uncompressed)};
    } else if (boost::filesystem::exists(directory)) {
        std::vector<std::thread> threads;
        moodycamel::BlockingConcurrentQueue<
            boost::optional<boost::filesystem::path>>
            queue;
        std::vector<std::vector<R>> result_queues(num_threads);

        for (auto& entry : boost::make_iterator_range(
                 boost::filesystem::directory_iterator(directory), {})) {
            boost::filesystem::path source = entry.path();
            queue.enqueue(source);
        }

        for (size_t i = 0; i < num_threads; i++) {
            threads.emplace_back([i, &queue, &helper, &result_queues]() {
                process_nested_helper(queue, helper, result_queues[i]);
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
        std::vector<R> final_result;
        final_result.reserve(size);
        for (auto& vec : result_queues) {
            for (auto& entry : vec) {
                final_result.emplace_back(std::move(entry));
            }
        }
        return final_result;
    } else {
        if (actually_necessary) {
            throw std::runtime_error(absl::StrCat(
                "Could not find directory ", root.string(), " , ", prefix));
        } else {
            std::vector<R> final_result;
            return final_result;
        }
    }
}

absl::flat_hash_set<uint64_t> get_standard_codes(
    const boost::filesystem::path& concept, char delimiter,
    size_t num_threads) {
    auto valid = process_nested(
        concept, "concept", num_threads, true,
        {"concept_id", "standard_concept"}, delimiter, [&](auto& reader) {
            std::vector<uint64_t> result;

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
get_parents(std::vector<uint64_t>& raw_codes,
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
        concept, "concept_relationship", num_threads, false,
        {"concept_id_1", "concept_id_2", "relationship_id"}, delimiter,
        [&](auto& reader) {
            ParentMap result;

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
            raw_codes.push_back(target_index);
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

std::vector<std::pair<std::string, std::string>> get_concept_text(
    const std::vector<uint64_t>& originals,
    const absl::flat_hash_map<uint64_t, uint32_t>& index_map,
    const boost::filesystem::path& concept, char delimiter,
    size_t num_threads) {
    auto texts = process_nested(
        concept, "concept", num_threads, true,
        {"concept_id", "concept_code", "vocabulary_id", "concept_name"},
        delimiter, [&](auto& reader) {
            std::vector<
                std::pair<std::pair<std::string, std::string>, uint32_t>>
                result;

            while (reader.next_row()) {
                uint64_t concept_id;
                attempt_parse_or_die(reader.get_row()[0], concept_id);

                auto iter = index_map.find(concept_id);
                if (iter != std::end(index_map)) {
                    std::string text = absl::StrCat(reader.get_row()[2], "/",
                                                    reader.get_row()[1]);

                    std::string description = reader.get_row()[3];
                    result.emplace_back(
                        std::make_pair(std::move(text), std::move(description)),
                        iter->second);
                }
            }

            return result;
        });
    std::vector<std::pair<std::string, std::string>> result(index_map.size());
    for (auto& text : texts) {
        for (auto& entry : text) {
            result[entry.second] = std::move(entry.first);
        }
    }

    for (size_t i = 0; i < result.size(); i++) {
        if (result[i].first.empty()) {
            std::cout << absl::StrCat(
                "Could not find the following concept_id in any of the concept "
                "tables",
                originals[i]);

            result[i].first = absl::StrCat("INVALID/", originals[i]);
            result[i].second = absl::StrCat("INVALID/", originals[i]);
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

Ontology create_ontology(std::vector<uint64_t> raw_codes,
                         const boost::filesystem::path& concept,
                         const boost::filesystem::path& target, char delimiter,
                         size_t num_threads) {
    boost::filesystem::create_directory(target);
    auto parent_info = get_parents(raw_codes, concept, delimiter, num_threads);
    auto text = get_concept_text(raw_codes, parent_info.first, concept,
                                 delimiter, num_threads);

    {
        boost::filesystem::create_directory(target);
        DictionaryWriter main(target / "main");
        for (const auto& t : text) {
            main.add_value(t.first);
        }
    }
    {
        DictionaryWriter text_description(target / "text_description");
        for (const auto& t : text) {
            text_description.add_value(t.second);
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
    {
        DictionaryWriter concept_ids(target / "concept_id");
        for (size_t i = 0; i < raw_codes.size(); i++) {
            concept_ids.add_value(container_to_view(
                absl::Span<const uint64_t>(raw_codes.data() + i, 1)));
        }
    }
    return Ontology(target);
}

Ontology::Ontology(const boost::filesystem::path& path)
    : main_dictionary(path / "main", true),
      parent_dict(path / "parent", true),
      children_dict(path / "children", true),
      all_parents_dict(path / "all_parents", true),
      text_description(path / "text_description", true),
      concept_ids(path / "concept_id", true) {}

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

std::string_view Ontology::get_text_description(uint32_t code) {
    if (!text_description) {
        return std::string_view(nullptr, 0);
    } else {
        return (*text_description)[code];
    }
}

boost::optional<uint32_t> Ontology::get_code_from_concept_id(
    uint64_t concept_id) {
    std::string_view target =
        container_to_view(absl::Span<uint64_t>(&concept_id, 1));
    return concept_ids->find(target);
}

uint64_t Ontology::get_concept_id_from_code(uint32_t code) {
    return read_span<uint64_t>(*concept_ids, code)[0];
}

#ifndef WRITER_H_INCLUDED
#define WRITER_H_INCLUDED

#include <variant>
#include <vector>

#include "atomicops.h"
#include "constdb.h"
#include "reader.h"
#include "readerwriterqueue.h"

struct Metadata {
    TermDictionary dictionary;
    TermDictionary value_dictionary;
};

struct PatientRecord {
    uint64_t person_id;
    absl::CivilDay birth_date;
    // <age, obs code>
    std::vector<std::pair<uint32_t, uint32_t>> observations;
    // <age, <obs code, value>>
    std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>>
        observations_with_values;
};

template <typename QueueItem>
class BlockingQueue {
   public:
    BlockingQueue(size_t max_size)
        : inner_queue(max_size), capacity(max_size), count(0) {}

    void wait_enqueue(QueueItem&& item) {
        while (!capacity.wait())
            ;
        bool value = inner_queue.try_enqueue(item);
        if (!value) {
            std::cout << "Invariant failed in queue enqueue" << std::endl;
            abort();
        }
        count.signal();
    }

    void wait_dequeue(QueueItem& item) {
        while (!count.wait())
            ;
        bool value = inner_queue.try_dequeue(item);
        if (!value) {
            std::cout << "Invariant failed in queue dequeue" << std::endl;
            abort();
        }
        capacity.signal();
    }

   private:
    moodycamel::ReaderWriterQueue<QueueItem> inner_queue;
    moodycamel::spsc_sema::LightweightSemaphore capacity;
    moodycamel::spsc_sema::LightweightSemaphore count;
};

using WriterItem = std::variant<PatientRecord, Metadata>;

template <typename F>
void write_timeline(const char* filename, F get_next) {
    ConstdbWriter writer(filename);

    std::vector<uint64_t> original_ids;
    std::vector<uint32_t> patient_ids;

    std::vector<uint32_t> buffer;
    std::vector<uint8_t> compressed_buffer;
    std::vector<uint32_t> ages;

    uint32_t current_index = 0;

    while (true) {
        WriterItem next_item = get_next();

        PatientRecord* possible_record = std::get_if<PatientRecord>(&next_item);

        if (possible_record != nullptr) {
            PatientRecord& record = *possible_record;

            buffer.clear();
            compressed_buffer.clear();
            ages.clear();

            uint32_t index = current_index++;

            patient_ids.push_back(index);
            original_ids.push_back(record.person_id);

            buffer.push_back(record.birth_date.year());
            buffer.push_back(record.birth_date.month());
            buffer.push_back(record.birth_date.day());

            std::sort(std::begin(record.observations),
                      std::end(record.observations));
            std::sort(std::begin(record.observations_with_values),
                      std::end(record.observations_with_values));

            record.observations.erase(
                std::unique(std::begin(record.observations),
                            std::end(record.observations)),
                std::end(record.observations));
            record.observations_with_values.erase(
                std::unique(std::begin(record.observations_with_values),
                            std::end(record.observations_with_values)),
                std::end(record.observations_with_values));

            for (const auto& elem : record.observations) {
                ages.push_back(elem.first);
            }
            for (const auto& elem : record.observations_with_values) {
                ages.push_back(elem.first);
            }

            std::sort(std::begin(ages), std::end(ages));
            ages.erase(std::unique(std::begin(ages), std::end(ages)),
                       std::end(ages));

            buffer.push_back(ages.size());

            uint32_t last_age = 0;

            size_t current_observation_index = 0;
            size_t current_observation_with_values_index = 0;

            for (uint32_t age : ages) {
                uint32_t delta = age - last_age;
                last_age = age;

                buffer.push_back(delta);

                size_t num_obs_index = buffer.size();
                buffer.push_back(1 << 30);  // Use a high value to force crashes

                size_t starting_observation_index = current_observation_index;
                uint32_t last_observation = 0;

                while (current_observation_index < record.observations.size() &&
                       record.observations[current_observation_index].first ==
                           age) {
                    uint32_t current_obs =
                        record.observations[current_observation_index].second;
                    uint32_t delta = current_obs - last_observation;
                    last_observation = current_obs;
                    buffer.push_back(delta);
                    current_observation_index++;
                }

                buffer[num_obs_index] =
                    current_observation_index - starting_observation_index;

                size_t num_obs_with_value_index = buffer.size();
                buffer.push_back(1 << 30);  // Use a high value to force crashes

                size_t starting_observation_value_index =
                    current_observation_with_values_index;
                uint32_t last_observation_with_value = 0;

                while (current_observation_with_values_index <
                           record.observations_with_values.size() &&
                       record.observations_with_values
                               [current_observation_with_values_index]
                                   .first == age) {
                    auto [code, value] =
                        record
                            .observations_with_values
                                [current_observation_with_values_index]
                            .second;
                    uint32_t delta = code - last_observation_with_value;
                    last_observation_with_value = code;
                    buffer.push_back(delta);
                    buffer.push_back(value);
                    current_observation_with_values_index++;
                }

                buffer[num_obs_with_value_index] =
                    current_observation_with_values_index -
                    starting_observation_value_index;
            }

            size_t max_needed_size =
                streamvbyte_max_compressedbytes(buffer.size()) +
                sizeof(uint32_t);

            if (compressed_buffer.size() < max_needed_size) {
                compressed_buffer.resize(max_needed_size * 2 + 1);
            }

            size_t actual_size =
                streamvbyte_encode(buffer.data(), buffer.size(),
                                   compressed_buffer.data() + sizeof(uint32_t));

            uint32_t* start_of_compressed_buffer =
                reinterpret_cast<uint32_t*>(compressed_buffer.data());
            *start_of_compressed_buffer = buffer.size();

            writer.add_int(index, (const char*)compressed_buffer.data(),
                           actual_size + sizeof(uint32_t));
        } else {
            const Metadata& meta = std::get<Metadata>(next_item);

            uint32_t num_patients = original_ids.size();

            writer.add_str("num_patients", (const char*)&num_patients,
                           sizeof(uint32_t));
            writer.add_str("original_ids", (const char*)original_ids.data(),
                           sizeof(uint64_t) * original_ids.size());
            writer.add_str("patient_ids", (const char*)patient_ids.data(),
                           sizeof(uint32_t) * patient_ids.size());

            std::string dictionary_str = meta.dictionary.to_json();
            std::string value_dictionary_str = meta.value_dictionary.to_json();

            writer.add_str("dictionary", dictionary_str.data(),
                           dictionary_str.size());
            writer.add_str("value_dictionary", value_dictionary_str.data(),
                           value_dictionary_str.size());

            return;
        }
    }
}

#endif

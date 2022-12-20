#include "count_codes_and_values.hh"

#include <boost/optional.hpp>
#include <boost/range/iterator_range.hpp>
#include <thread>

#include "absl/container/flat_hash_map.h"
#include "blockingconcurrentqueue.h"
#include "csv.hh"
#include "join_csvs.hh"
#include "parse_utils.hh"
#include "picosha2.h"
#include "readerwritercircularbuffer.h"
#include "thread_utils.hh"

using CodeCounter = absl::flat_hash_map<uint64_t, size_t>;

template <typename T>
void sort_by_count(T& result) {
    std::sort(std::begin(result), std::end(result),
              [](const auto& a, const auto& b) { return a.second > b.second; });
}

template <typename T>
std::vector<std::pair<typename T::value_type::key_type, size_t>>
convert_to_vector(const T& container) {
    absl::flat_hash_map<typename T::value_type::key_type, size_t> temp;
    for (const auto& code : container) {
        for (const auto& entry : code) {
            temp[entry.first] += entry.second;
        }
    }
    std::vector<std::pair<typename T::value_type::key_type, size_t>> result;
    result.reserve(temp.size());

    for (auto& entry : temp) {
        result.emplace_back(std::move(entry.first), entry.second);
    }

    sort_by_count(result);

    return result;
}

void clean_thread(const boost::filesystem::path& in_path,
                  CodeCounter& code_counts,
                  const boost::filesystem::path& out_path) {
    CSVReader<ZstdReader> reader(in_path, {"value", "code"}, ',');
    CSVWriter<ZstdWriter> writer(out_path, {"value"}, ',');

    while (reader.next_row()) {
        uint64_t code;
        attempt_parse_or_die(reader.get_row()[1], code);
        code_counts[code] += 1;

        auto& text_value = reader.get_row()[0];

        if (text_value.empty()) {
            continue;
        }

        double value;
        if (absl::SimpleAtod(text_value, &value)) {
            continue;
        }

        reader.get_row().resize(1);
        writer.add_row(reader.get_row());
    }
}

void process_thread(
    boost::filesystem::path& in_path,
    moodycamel::BlockingReaderWriterCircularBuffer<
        boost::optional<std::pair<std::string, size_t>>>& out_queue) {
    CSVReader<ZstdReader> reader(in_path, {"value"}, ',');

    std::string current_value = "";
    size_t current_count = 0;

    auto flush = [&]() {
        if (current_count > 1) {
            out_queue.wait_enqueue(
                std::make_pair(std::move(current_value), current_count));
        }
    };

    while (reader.next_row()) {
        auto& text_value = reader.get_row()[0];

        if (text_value != current_value) {
            flush();
            current_count = 1;
            current_value = std::move(text_value);
        } else {
            current_count += 1;
        }
    }

    flush();

    out_queue.wait_enqueue(boost::none);
}

std::pair<std::vector<std::pair<uint64_t, size_t>>,
          std::vector<std::pair<std::string, size_t>>>
count_codes_and_values(const boost::filesystem::path& path,
                       const boost::filesystem::path& temp_path,
                       size_t num_threads) {
    boost::filesystem::path only_values = temp_path / "only_values";

    boost::filesystem::create_directory(only_values);

    std::vector<std::pair<uint64_t, size_t>> code_result;

    {
        std::vector<boost::filesystem::path> source_files;
        for (auto& entry : boost::make_iterator_range(
                 boost::filesystem::directory_iterator(path), {})) {
            source_files.push_back(entry.path());
        }

        std::vector<CodeCounter> code_counts(source_files.size());
        std::vector<std::thread> threads;

        for (size_t i = 0; i < source_files.size(); i++) {
            threads.emplace_back(
                [i, &only_values, &source_files, &code_counts]() {
                    auto target = only_values / absl::StrCat(i, ".csv.zst");
                    clean_thread(source_files[i], code_counts[i], target);
                });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        code_result = convert_to_vector(code_counts);
    }

    boost::filesystem::path joined = temp_path / "joined";

    sort_and_join_csvs(only_values, joined,
                       {std::make_pair("value", ColumnValueType::STRING)}, ',',
                       num_threads);

    std::vector<std::pair<std::string, size_t>> string_result;

    {
        std::vector<boost::filesystem::path> source_files;
        for (auto& entry : boost::make_iterator_range(
                 boost::filesystem::directory_iterator(joined), {})) {
            source_files.push_back(entry.path());
        }

        std::vector<moodycamel::BlockingReaderWriterCircularBuffer<
            boost::optional<std::pair<std::string, size_t>>>>
            out_queues;
        out_queues.reserve(source_files.size());

        std::vector<std::thread> threads;

        for (size_t i = 0; i < source_files.size(); i++) {
            out_queues.emplace_back(1000);
            threads.emplace_back([i, &source_files, &out_queues]() {
                process_thread(source_files[i], out_queues[i]);
            });
        }

        dequeue_many_loop(
            out_queues,
            [&string_result](std::pair<std::string, size_t>& next_entry) {
                string_result.push_back(std::move(next_entry));
            });

        for (auto& thread : threads) {
            thread.join();
        }

        sort_by_count(string_result);
    }

    boost::filesystem::remove_all(temp_path);

    return std::make_pair(std::move(code_result), std::move(string_result));
}

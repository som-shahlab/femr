#include "count_codes_and_values.hh"

#include <boost/range/iterator_range.hpp>
#include <thread>

#include "absl/container/flat_hash_map.h"
#include "blockingconcurrentqueue.h"
#include "csv.hh"
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

void reader_thread(CodeCounter& code_counts,
                   moodycamel::BlockingReaderWriterCircularBuffer<
                       std::optional<std::pair<size_t, std::string>>>& queue,
                   const boost::filesystem::path& source) {
    CSVReader reader(source.string(), {"value", "value_type", "code"}, ',');

    while (reader.next_row()) {
        uint64_t code;
        attempt_parse_or_die(reader.get_row()[2], code);
        code_counts[code] += 1;

        if (reader.get_row()[1] == "ValueType.TEXT") {
            size_t string_hash = std::hash<std::string>()(reader.get_row()[0]);
            queue.wait_enqueue(
                std::make_pair(string_hash, std::move(reader.get_row()[0])));
        }
    }

    queue.wait_enqueue(std::nullopt);
}

void multiplex_thread(
    std::vector<moodycamel::BlockingReaderWriterCircularBuffer<
        std::optional<std::pair<size_t, std::string>>>>& in_queues,
    std::vector<moodycamel::BlockingReaderWriterCircularBuffer<
        std::optional<std::string>>>& out_queues) {
    dequeue_many_loop(
        in_queues, [&out_queues](std::pair<size_t, std::string>& next_entry) {
            out_queues[next_entry.first % out_queues.size()].wait_enqueue(
                std::move(next_entry.second));
        });

    for (auto& out_queue : out_queues) {
        out_queue.wait_enqueue(std::nullopt);
    }
}

void writer_thread(moodycamel::BlockingReaderWriterCircularBuffer<
                       std::optional<std::string>>& queue,
                   const boost::filesystem::path& target) {
    CSVWriter writer(target.string(), {"value"}, ',');

    std::optional<std::string> next_item;
    std::vector<std::string> next_row(1);

    while (true) {
        queue.wait_dequeue(next_item);
        if (!next_item) {
            break;
        } else {
            next_row[0] = std::move(*next_item);
            writer.add_row(next_row);
        }
    }
}

void process_thread(
    moodycamel::BlockingConcurrentQueue<std::optional<boost::filesystem::path>>&
        in_queue,
    moodycamel::BlockingReaderWriterCircularBuffer<
        std ::optional<std::pair<std::string, size_t>>>& out_queue) {
    std::optional<boost::filesystem::path> next_item;
    std::vector<std::string> rows;
    while (true) {
        in_queue.wait_dequeue(next_item);
        if (!next_item) {
            break;
        } else {
            CSVReader reader(next_item->string(), {"value"}, ',');
            rows.clear();
            while (reader.next_row()) {
                rows.push_back(std::move(reader.get_row()[0]));
            }

            std::sort(std::begin(rows), std::end(rows));

            size_t start_index = 0;
            for (size_t index = 0; index <= rows.size(); index++) {
                if (index == rows.size() || rows[index] != rows[start_index]) {
                    size_t count = index - start_index;
                    if (count > 1) {
                        out_queue.wait_enqueue(std::make_pair(
                            std::move(rows[start_index]), count));
                    }
                    start_index = index;
                }
            }
        }
    }
    out_queue.wait_enqueue(std::nullopt);
}

std::pair<std::vector<std::pair<uint64_t, size_t>>,
          std::vector<std::pair<std::string, size_t>>>
count_codes_and_values(const boost::filesystem::path& path,
                       const boost::filesystem::path& temp_path,
                       size_t num_threads) {
    std::vector<boost::filesystem::path> files;
    for (auto& entry : boost::make_iterator_range(
             boost::filesystem::directory_iterator(path), {})) {
        files.push_back(entry.path());
    }

    // Assume a worst case 10x blowup
    size_t data_size =
        boost::filesystem::file_size(files[0]) * files.size() * 10;

    size_t desired_data_size = 100000000;  // 100 megabytes

    size_t num_pieces = (data_size + desired_data_size - 1) / desired_data_size;

    boost::filesystem::create_directory(temp_path);

    std::vector<std::pair<uint64_t, size_t>> code_result;

    std::vector<boost::filesystem::path> sort_files;
    sort_files.reserve(num_pieces);
    std::cout << "Starting phase 1 " << absl::Now() << std::endl;

    {
        std::vector<CodeCounter> code_counts(files.size());
        std::vector<moodycamel::BlockingReaderWriterCircularBuffer<
            std::optional<std::string>>>
            write_queues;
        write_queues.reserve(num_pieces);
        std::vector<moodycamel::BlockingReaderWriterCircularBuffer<
            std::optional<std::pair<size_t, std::string>>>>
            read_queues;
        read_queues.reserve(files.size());

        std::vector<std::thread> threads;

        for (size_t i = 0; i < num_pieces; i++) {
            write_queues.emplace_back(1000);
            sort_files.emplace_back(temp_path /
                                    boost::filesystem::unique_path());
            threads.emplace_back([i, &write_queues, &sort_files]() {
                writer_thread(write_queues[i], sort_files[i]);
            });
        }

        for (size_t i = 0; i < files.size(); i++) {
            read_queues.emplace_back(1000);
            threads.emplace_back([i, &code_counts, &read_queues, &files]() {
                reader_thread(code_counts[i], read_queues[i], files[i]);
            });
        }

        threads.emplace_back([&read_queues, &write_queues]() {
            multiplex_thread(read_queues, write_queues);
        });

        for (auto& thread : threads) {
            thread.join();
        }

        code_result = convert_to_vector(code_counts);
    }
    std::cout << "Starting phase 2 " << absl::Now() << std::endl;

    std::vector<std::pair<std::string, size_t>> string_result;
    {
        std::vector<moodycamel::BlockingReaderWriterCircularBuffer<
            std::optional<std::pair<std::string, size_t>>>>
            out_queues;
        out_queues.reserve(num_threads);

        moodycamel::BlockingConcurrentQueue<
            std::optional<boost::filesystem::path>>
            in_queue;

        for (size_t i = 0; i < num_pieces; i++) {
            in_queue.enqueue({sort_files[i]});
        }

        std::vector<std::thread> threads;

        for (size_t i = 0; i < num_threads; i++) {
            out_queues.emplace_back(1000);
            threads.emplace_back([i, &in_queue, &out_queues]() {
                process_thread(in_queue, out_queues[i]);
            });
            in_queue.enqueue(std::nullopt);
        }

        dequeue_many_loop(
            out_queues,
            [&string_result](std::pair<std::string, size_t>& next_entry) {
                string_result.push_back(std::move(next_entry));
            });

        sort_by_count(string_result);

        for (auto& thread : threads) {
            thread.join();
        }
    }
    std::cout << "Done " << absl::Now() << std::endl;

    boost::filesystem::remove_all(temp_path);

    return std::make_pair(std::move(code_result), std::move(string_result));
}
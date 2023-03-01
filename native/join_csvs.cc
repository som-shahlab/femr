#include "join_csvs.hh"

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <boost/range/iterator_range.hpp>
#include <iostream>
#include <queue>

#include "absl/time/civil_time.h"
#include "blockingconcurrentqueue.h"
#include "csv.hh"
#include "parse_utils.hh"
#include "readerwritercircularbuffer.h"
#include "thread_utils.hh"

constexpr int QUEUE_SIZE = 1000;

union ColumnValue {
    std::string_view string;
    uint64_t integer;
    absl::CivilSecond time;

    ColumnValue() {}
};

constexpr size_t MAX_SIZE = 1000000000;
// constexpr size_t MAX_SIZE = 10000000;

using Row = std::vector<std::string>;
using QueueItem = boost::optional<std::vector<std::string>>;
using QueueType = moodycamel::BlockingReaderWriterCircularBuffer<QueueItem>;

bool compare_rows_using_indices(
    const std::vector<std::pair<std::string, ColumnValueType>>& sort_keys,
    const std::vector<ColumnValue>& column_values, size_t a, size_t b) {
    for (size_t i = 0; i < sort_keys.size(); i++) {
        const auto& val_a = column_values[a * sort_keys.size() + i];
        const auto& val_b = column_values[b * sort_keys.size() + i];
        switch (sort_keys[i].second) {
            case ColumnValueType::STRING:
                if (val_a.string < val_b.string) {
                    return true;
                } else if (val_a.string > val_b.string) {
                    return false;
                }
                break;

            case ColumnValueType::UINT64_T:
                if (val_a.integer < val_b.integer) {
                    return true;
                } else if (val_a.integer > val_b.integer) {
                    return false;
                }
                break;

            case ColumnValueType::DATETIME:
                if (val_a.time < val_b.time) {
                    return true;
                } else if (val_a.time > val_b.time) {
                    return false;
                }
                break;
        }
    }
    return false;
}

void convert_to_column_values(
    const std::vector<std::pair<std::string, ColumnValueType>>& sort_keys,
    const std::vector<size_t>& sort_indices,
    const std::vector<std::string>& row,
    std::vector<ColumnValue>& column_values, ssize_t start_location = -1) {
    for (size_t i = 0; i < sort_indices.size(); i++) {
        const std::string& val = row[sort_indices[i]];
        ColumnValue column_val;
        switch (sort_keys[i].second) {
            case ColumnValueType::STRING:
                column_val.string = val;
                break;

            case ColumnValueType::UINT64_T:
                attempt_parse_or_die(val, column_val.integer);
                break;

            case ColumnValueType::DATETIME:
                attempt_parse_time_or_die(val, column_val.time);
                break;

            default:
                throw std::runtime_error("Unexpected column value type?");
        }
        if (start_location == -1) {
            column_values.emplace_back(column_val);
        } else {
            column_values[start_location * sort_indices.size() + i] =
                column_val;
        }
    }
}

std::vector<size_t> get_sort_indices(
    const std::vector<std::string>& columns,
    const std::vector<std::pair<std::string, ColumnValueType>>& sort_keys) {
    std::vector<size_t> indices;
    for (const auto& sort_key : sort_keys) {
        auto iter =
            std::find(std::begin(columns), std::end(columns), sort_key.first);
        if (iter == std::end(columns)) {
            throw std::runtime_error("Could not find the sort key " +
                                     sort_key.first + " in " +
                                     absl::StrJoin(columns, ","));
        } else {
            indices.push_back(iter - std::begin(columns));
        }
    }
    return indices;
}

void sort_reader(
    size_t i, size_t num_shards,
    moodycamel::BlockingConcurrentQueue<
        boost::optional<boost::filesystem::path>>& file_queue,
    std::vector<
        std::vector<moodycamel::BlockingReaderWriterCircularBuffer<QueueItem>>>&
        all_write_queues,
    const std::vector<std::string>& columns,
    const std::vector<std::pair<std::string, ColumnValueType>>& sort_keys,
    char delimiter) {
    boost::optional<boost::filesystem::path> item;
    auto sort_indices = get_sort_indices(columns, sort_keys);
    while (true) {
        file_queue.wait_dequeue(item);

        if (!item) {
            break;
        } else {
            auto source = *item;

            CSVReader<ZstdReader> reader(source.string(), delimiter);
            if (reader.columns != columns) {
                throw std::runtime_error("Columns of input don't match " +
                                         source.string());
            }
            while (reader.next_row()) {
                Row r = std::move(reader.get_row());
                size_t index =
                    std::hash<std::string>()(r[sort_indices[0]]) % (num_shards);
                all_write_queues[index][i].wait_enqueue(std::move(r));
            }
        }
    }

    for (size_t j = 0; j < num_shards; j++) {
        all_write_queues[j][i].wait_enqueue(boost::none);
    }
}

void sort_writer(
    size_t j, size_t num_shards,
    std::vector<moodycamel::BlockingReaderWriterCircularBuffer<QueueItem>>&
        write_queues,
    const boost::filesystem::path& target_dir,
    const std::vector<std::string>& columns,
    const std::vector<std::pair<std::string, ColumnValueType>>& sort_keys,
    char delimiter) {
    boost::filesystem::create_directory(target_dir);
    std::vector<Row> rows;
    std::vector<ColumnValue> row_values;
    std::vector<size_t> row_indices;
    size_t current_size = 0;

    auto sort_indices = get_sort_indices(columns, sort_keys);

    auto flush = [&]() {
        auto target_file = target_dir / boost::filesystem::unique_path(
                                            "%%%%%%%%%%%%%%.csv.zst");

        for (const auto& row : rows) {
            convert_to_column_values(sort_keys, sort_indices, row, row_values);
        }

        std::sort(std::begin(row_indices), std::end(row_indices),
                  [&](size_t a, size_t b) {
                      return compare_rows_using_indices(sort_keys, row_values,
                                                        a, b);
                  });

        CSVWriter<ZstdWriter> writer(target_file.string(), columns, delimiter);
        for (const auto& row_index : row_indices) {
            writer.add_row(rows[row_index]);
        }

        rows.clear();
        row_values.clear();
        row_indices.clear();
        current_size = 0;
    };

    dequeue_many_loop(write_queues, [&](Row& r) {
        for (const auto& column : r) {
            current_size += column.size();
        }
        row_indices.emplace_back(rows.size());
        rows.emplace_back(std::move(r));
        if (current_size > MAX_SIZE) {
            flush();
        }
    });

    if (current_size > 0) {
        flush();
    }
}

void sort_csvs(
    const boost::filesystem::path& source_directory,
    const boost::filesystem::path& target_directory,
    const std::vector<std::pair<std::string, ColumnValueType>>& sort_keys,
    char delimiter, size_t num_shards) {
    boost::filesystem::create_directory(target_directory);

    std::vector<boost::filesystem::path> target_shards;

    for (size_t i = 0; i < num_shards; i++) {
        target_shards.push_back(target_directory / std::to_string(i));
    }

    moodycamel::BlockingConcurrentQueue<
        boost::optional<boost::filesystem::path>>
        file_queue;

    std::vector<std::string> columns;
    for (auto& entry : boost::make_iterator_range(
             boost::filesystem::directory_iterator(source_directory), {})) {
        boost::filesystem::path source = entry.path();
        if (columns.empty()) {
            columns = get_csv_columns(source, delimiter);
        }
        file_queue.enqueue(source);
    }

    for (size_t i = 0; i < num_shards; i++) {
        file_queue.enqueue({});
    }

    std::vector<
        std::vector<moodycamel::BlockingReaderWriterCircularBuffer<QueueItem>>>
        write_queues(num_shards);

    for (size_t i = 0; i < num_shards; i++) {
        for (size_t j = 0; j < num_shards; j++) {
            write_queues[i].emplace_back(QUEUE_SIZE);
        }
    }

    std::vector<std::thread> threads;

    for (size_t i = 0; i < num_shards; i++) {
        threads.emplace_back([i, &file_queue, &write_queues, num_shards,
                              &columns, &sort_keys, delimiter]() {
            sort_reader(i, num_shards, file_queue, write_queues, columns,
                        sort_keys, delimiter);
        });

        threads.emplace_back([i, &write_queues, &target_shards, num_shards,
                              &columns, &sort_keys, delimiter]() {
            sort_writer(i, num_shards, write_queues[i], target_shards[i],
                        columns, sort_keys, delimiter);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

void join_csvs(
    const boost::filesystem::path& source_directory,
    const boost::filesystem::path& target_file,
    const std::vector<std::pair<std::string, ColumnValueType>>& sort_keys,
    char delimiter) {
    std::vector<CSVReader<ZstdReader>> source_files;

    std::vector<std::string> columns;

    for (auto& entry : boost::make_iterator_range(
             boost::filesystem::directory_iterator(source_directory), {})) {
        const boost::filesystem::path& source = entry.path();
        if (columns.empty()) {
            columns = get_csv_columns(source, delimiter);
        }
        source_files.emplace_back(source, delimiter);
        if (source_files.size() == 1) {
            columns = source_files.back().columns;
        } else {
            if (source_files.back().columns != columns) {
                throw std::runtime_error("Invalid columns");
            }
        }
    }

    if (columns.empty()) {
        // No data to join
        return;
    }

    CSVWriter<ZstdWriter> target(target_file, columns, delimiter);

    auto sort_indices = get_sort_indices(columns, sort_keys);

    std::vector<Row> rows(source_files.size());
    std::vector<ColumnValue> column_vals;

    auto comp = [&](size_t a, size_t b) {
        return compare_rows_using_indices(sort_keys, column_vals, b, a);
    };

    std::priority_queue<size_t, std::vector<size_t>, decltype(comp)> queue(
        comp);

    for (size_t i = 0; i < source_files.size(); i++) {
        auto& source_file = source_files[i];
        if (source_file.next_row()) {
            rows[i] = std::move(source_file.get_row());
            convert_to_column_values(sort_keys, sort_indices, rows[i],
                                     column_vals);
            queue.push(i);
        }
    }

    while (!queue.empty()) {
        size_t read_index = queue.top();
        queue.pop();

        Row r = std::move(rows[read_index]);
        target.add_row(r);

        auto& source_file = source_files[read_index];
        if (source_file.next_row()) {
            rows[read_index] = std::move(source_file.get_row());
            convert_to_column_values(sort_keys, sort_indices, rows[read_index],
                                     column_vals, read_index);
            queue.push(read_index);
        }
    }
}

void sort_and_join_csvs(
    const boost::filesystem::path& source_directory,
    const boost::filesystem::path& target_directory,
    const std::vector<std::pair<std::string, ColumnValueType>>& sort_keys,
    char delimiter, size_t num_shards) {

    boost::filesystem::create_directory(target_directory);
    boost::filesystem::path sorted_dir =
        target_directory / boost::filesystem::unique_path();
    sort_csvs(source_directory, sorted_dir, sort_keys, delimiter, num_shards);
    std::vector<std::thread> threads;

    for (size_t i = 0; i < num_shards; i++) {
        threads.emplace_back(
            [i, &sorted_dir, &target_directory, &sort_keys, delimiter]() {
                join_csvs(sorted_dir / std::to_string(i),
                          target_directory / (std::to_string(i) + ".csv.zst"),
                          sort_keys, delimiter);
            });
    }

    for (size_t i = 0; i < num_shards; i++) {
        threads[i].join();
    }
    boost::filesystem::remove_all(sorted_dir);
}

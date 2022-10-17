#include "join_csvs.hh"

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <iostream>
#include <queue>

#include "blockingconcurrentqueue.h"
#include "csv.hh"
#include "readerwritercircularbuffer.h"

constexpr int QUEUE_SIZE = 100;

using Row = std::vector<std::string>;
using QueueItem = boost::optional<std::vector<std::string>>;
using QueueType = moodycamel::BlockingReaderWriterCircularBuffer<QueueItem>;

bool compare_rows_using_indices(const std::vector<size_t>& sort_indices,
                                const Row& a, const Row& b) {
    for (size_t sort_index : sort_indices) {
        if (a[sort_index] < b[sort_index]) {
            return true;
        } else if (a[sort_index] > b[sort_index]) {
            return false;
        }
    }
    return false;
}

std::vector<size_t> get_sort_indices(
    const std::vector<std::string>& columns,
    const std::vector<std::string>& sort_keys) {
    std::vector<size_t> indices;
    for (const auto& sort_key : sort_keys) {
        auto iter = std::find(std::begin(columns), std::end(columns), sort_key);
        if (iter == std::end(columns)) {
            throw std::runtime_error("Could not find the sort key " + sort_key);
        } else {
            indices.push_back(iter - std::begin(columns));
        }
    }
    for (size_t i = 0; i < columns.size(); i++) {
        if (std::find(std::begin(sort_keys), std::end(sort_keys), columns[i]) ==
            std::end(sort_keys)) {
            indices.push_back(i);
        }
    }
    return indices;
}

void sort_processor(
    moodycamel::BlockingConcurrentQueue<boost::optional<
        std::pair<boost::filesystem::path, boost::filesystem::path>>>& queue,
    const std::vector<std::string>& sort_keys, char delimiter) {
    boost::optional<std::pair<boost::filesystem::path, boost::filesystem::path>>
        item;
    while (true) {
        queue.wait_dequeue(item);

        if (!item) {
            break;
        } else {
            auto source = item->first;
            auto destination = item->second;

            auto columns = get_csv_columns(source.string(), delimiter);

            auto sort_indices = get_sort_indices(columns, sort_keys);

            CSVReader reader(source.string(), columns, delimiter);

            std::vector<std::vector<std::string>> rows;

            while (reader.next_row()) {
                rows.push_back(reader.get_row());
            }

            std::sort(std::begin(rows), std::end(rows),
                      [&](const Row& a, const Row& b) {
                          return compare_rows_using_indices(sort_indices, a, b);
                      });

            CSVWriter writer(destination.string(), columns, delimiter);
            for (const auto& row : rows) {
                writer.add_row(row);
            }
        }
    }
}

void sort_csvs(const boost::filesystem::path& source_directory,
               const boost::filesystem::path& target_directory,
               const std::vector<std::string>& sort_keys, char delimiter,
               size_t num_threads) {
    boost::filesystem::create_directory(target_directory);
    moodycamel::BlockingConcurrentQueue<boost::optional<
        std::pair<boost::filesystem::path, boost::filesystem::path>>>
        queue;

    std::vector<std::thread> threads;

    for (size_t i = 0; i < num_threads; i++) {
        threads.emplace_back(
            [&]() { sort_processor(queue, sort_keys, delimiter); });
    }

    for (auto& entry : boost::make_iterator_range(
             boost::filesystem::directory_iterator(source_directory), {})) {
        boost::filesystem::path source = entry.path();
        boost::filesystem::path target =
            target_directory /
            boost::filesystem::relative(source, source_directory);
        queue.enqueue(std::make_pair(source, target));
    }

    for (size_t i = 0; i < num_threads; i++) {
        queue.enqueue({});
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

void read_join_processor(QueueType& read_queue,
                         const boost::filesystem::path& source,
                         const std::vector<std::string>& columns,
                         char delimiter) {
    CSVReader reader(source.string(), columns, delimiter);

    while (reader.next_row()) {
        read_queue.wait_enqueue(reader.get_row());
    }
    read_queue.wait_enqueue(boost::none);
}

void multiplex_join_processor(std::vector<QueueType>& read_queues,
                              std::vector<QueueType>& write_queues,
                              const std::vector<size_t>& sort_indices) {
    auto comp = [sort_indices](const std::pair<Row, size_t>& a,
                               const std::pair<Row, size_t>& b) {
        return compare_rows_using_indices(sort_indices, b.first, a.first);
    };
    std::priority_queue<std::pair<Row, size_t>,
                        std::vector<std::pair<Row, size_t>>, decltype(comp)>
        queue(comp);

    Row last_row;
    QueueItem next_item;
    for (size_t i = 0; i < read_queues.size(); i++) {
        read_queues[i].wait_dequeue(next_item);

        if (next_item) {
            queue.push(std::make_pair(std::move(*next_item), i));
        }
    }

    while (!queue.empty()) {
        Row r = std::move(const_cast<Row&>(queue.top().first));
        size_t read_index = queue.top().second;
        queue.pop();
        size_t index = std::hash<std::string>()(r[sort_indices[0]]) %
                       (write_queues.size());
        if (r != last_row) {
            last_row = r;
            write_queues[index].wait_enqueue({std::move(r)});
        }
        read_queues[read_index].wait_dequeue(next_item);
        if (next_item) {
            queue.emplace(std::move(*next_item), read_index);
        }
    }

    next_item = boost::nullopt;
    for (auto& write_queue : write_queues) {
        write_queue.wait_enqueue(next_item);
    }
}

void write_join_processor(QueueType& write_queue,
                          const boost::filesystem::path& target,
                          const std::vector<std::string>& columns,
                          char delimiter) {
    CSVWriter writer(target.string(), columns, delimiter);

    QueueItem item;
    while (true) {
        write_queue.wait_dequeue(item);
        if (!item) {
            break;
        }
        writer.add_row(*item);
    }
}

void join_csvs(const boost::filesystem::path& source_directory,
               const boost::filesystem::path& target_directory,
               const std::vector<std::string>& sort_keys, char delimiter,
               size_t num_shards) {
    boost::filesystem::create_directory(target_directory);
    std::vector<std::thread> threads;

    std::vector<boost::filesystem::path> source_paths;

    std::vector<std::string> columns;

    for (auto& entry : boost::make_iterator_range(
             boost::filesystem::directory_iterator(source_directory), {})) {
        const boost::filesystem::path& source = entry.path();
        if (source_paths.empty()) {
            columns = get_csv_columns(source.string(), delimiter);
        } else {
            if (get_csv_columns(source.string(), delimiter) != columns) {
                throw std::runtime_error("Invalid columns");
            }
        }
        source_paths.push_back(source);
    }

    auto sort_indices = get_sort_indices(columns, sort_keys);

    std::vector<QueueType> read_queues;
    read_queues.reserve(source_paths.size());
    std::vector<QueueType> write_queues;
    write_queues.reserve(num_shards);

    for (size_t i = 0; i < source_paths.size(); i++) {
        read_queues.emplace_back(QUEUE_SIZE);
    }

    for (size_t i = 0; i < num_shards; i++) {
        write_queues.emplace_back(QUEUE_SIZE);
    }

    for (size_t i = 0; i < source_paths.size(); i++) {
        threads.emplace_back(
            [i, &read_queues, &source_paths, &columns, &delimiter]() {
                read_join_processor(read_queues[i], source_paths[i], columns,
                                    delimiter);
            });
    }

    for (size_t i = 0; i < num_shards; i++) {
        threads.emplace_back(
            [i, &write_queues, &target_directory, &columns, &delimiter]() {
                write_join_processor(
                    write_queues[i],
                    target_directory / boost::filesystem::unique_path(
                                           "%%%%-%%%%-%%%%-%%%%.csv.zst"),
                    columns, delimiter);
            });
    }
    threads.emplace_back([&]() {
        multiplex_join_processor(read_queues, write_queues, sort_indices);
    });

    for (auto& thread : threads) {
        thread.join();
    }
}

void sort_and_join_csvs(const boost::filesystem::path& source_directory,
                        const boost::filesystem::path& target_directory,
                        const std::vector<std::string>& sort_keys,
                        char delimiter, size_t num_shards) {
    boost::filesystem::create_directory(target_directory);
    boost::filesystem::path sorted_dir = boost::filesystem::unique_path();
    sort_csvs(source_directory, target_directory / sorted_dir, sort_keys,
              delimiter, num_shards);
    join_csvs(target_directory / sorted_dir, target_directory, sort_keys,
              delimiter, num_shards);
    boost::filesystem::remove_all(target_directory / sorted_dir);
}
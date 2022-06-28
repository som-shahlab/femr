#include "sort_csv.h"

#include <boost/filesystem.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>

#include "blockingconcurrentqueue.h"
#include "csv.h"

void sort_csv_file(boost::filesystem::path source_file,
                   boost::filesystem::path target_file, char delimiter,
                   bool use_quotes) {
    gzFile file = gzopen(source_file.c_str(), "r");
    if (file == nullptr) {
        std::cout << absl::Substitute("Could not open $0 due to $1",
                                      source_file.string(),
                                      std::strerror(errno))
                  << std::endl;
        ;
        abort();
    }

    gzbuffer(file, BUFFER_SIZE);

    std::vector<char> buffer(BUFFER_SIZE);
    char* first_line = gzgets(file, buffer.data(), BUFFER_SIZE);

    if (first_line == nullptr) {
        std::cout << absl::Substitute("Could read header on $0 due to $1\n",
                                      file, std::strerror(errno));
        abort();
    }

    std::string header_copy = first_line;

    int person_id_index = -1;

    line_iter(first_line, delimiter, use_quotes,
              [&](int index, std::string_view column) {
                  if (column == "person_id") {
                      person_id_index = index;
                  }
              });

    if (person_id_index == -1) {
        boost::filesystem::copy_file(source_file, target_file);

        gzclose(file);

        return;
    }

    std::vector<std::pair<uint64_t, std::string>> data_elements;

    while (true) {
        char* next_line = gzgets(file, buffer.data(), BUFFER_SIZE);
        if (next_line == nullptr) {
            break;
        }

        uint64_t person_id;

        line_iter(next_line, delimiter, true,
                  [&](int index, std::string_view column) {
                      if (index == person_id_index) {
                          if (!absl::SimpleAtoi(column, &person_id)) {
                              std::cout << "Could not parse person id "
                                        << column << " " << source_file
                                        << std::endl;
                              abort();
                          }
                      }
                  });

        data_elements.push_back(std::make_pair(person_id, next_line));
    }

    gzclose(file);

    std::sort(std::begin(data_elements), std::end(data_elements),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    gzFile targetFile = gzopen(target_file.c_str(), "w");

    gzwrite(targetFile, header_copy.data(), header_copy.size());

    for (auto& pid_and_value : data_elements) {
        auto& value = pid_and_value.second;

        gzwrite(targetFile, value.data(), value.size());
    }

    gzclose(targetFile);
}

using WorkItem = std::pair<boost::filesystem::path, boost::filesystem::path>;
using WorkQueue = moodycamel::BlockingConcurrentQueue<std::optional<WorkItem>>;

void worker_thread(std::shared_ptr<WorkQueue> work_queue, char delimiter,
                   bool use_quotes) {
    while (true) {
        std::optional<WorkItem> result;
        work_queue->wait_dequeue(result);

        if (!result) {
            break;
        } else {
            auto& source = result->first;
            auto& target = result->second;

            sort_csv_file(source, target, delimiter, use_quotes);
        }
    }
}

void sort_csvs(boost::filesystem::path source_dir,
               boost::filesystem::path target_dir, char delimiter,
               bool use_quotes) {
    std::vector<std::pair<boost::filesystem::path, boost::filesystem::path>>
        files_to_sort;

    std::shared_ptr<WorkQueue> work_queue = std::make_shared<WorkQueue>();

    for (auto&& file :
         boost::filesystem::recursive_directory_iterator(source_dir)) {
        auto path = file.path();

        if (boost::filesystem::is_directory(path)) {
            continue;
        }

        boost::filesystem::path target =
            target_dir / boost::filesystem::relative(file, source_dir);
        boost::filesystem::create_directories(target.parent_path());
        work_queue->enqueue(std::make_pair(path, target));
    }

    int num_threads = 10;

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; i++) {
        std::thread thread([work_queue, delimiter, use_quotes]() {
            worker_thread(work_queue, delimiter, use_quotes);
        });

        threads.push_back(std::move(thread));

        work_queue->enqueue(std::nullopt);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}
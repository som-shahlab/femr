
#include "join_csvs.hh"

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <random>

#include "csv.hh"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "parse_utils.hh"

TEST(JoinCsvTest, TestSort) {
    boost::filesystem::path root = boost::filesystem::temp_directory_path() /
                                   boost::filesystem::unique_path();
    boost::filesystem::create_directory(root);
    boost::filesystem::path source = root / "sorted_source_dir";
    boost::filesystem::path target = root / "sorted_target_dir";
    boost::filesystem::create_directory(source);
    std::vector<std::string> columns = {"col1", "col2"};

    std::vector<std::vector<std::string>> entries;

    for (int i = 1; i <= 100; i++) {
        for (int j = 1; j <= 3; j++) {
            entries.push_back({std::to_string(i), std::to_string(j * 100)});
        }
    }

    std::shuffle(std::begin(entries), std::end(entries),
                 std::default_random_engine(1235423));

    size_t num_chunks = 7;

    size_t entries_per_chunk = (entries.size() + num_chunks - 1) / num_chunks;

    for (size_t i = 0; i < num_chunks; i++) {
        CSVWriter<ZstdWriter> writer(
            (source / absl::StrCat(i, ".csv.zst")).string(), columns, ',');
        for (size_t j = 0; j < entries_per_chunk; j++) {
            size_t index = i * entries_per_chunk + j;
            if (index < entries.size()) {
                writer.add_row(entries[index]);
            }
        }
    }

    size_t num_shards = 1;

    sort_csvs(source.string(), target.string(),
              {{"col1", ColumnValueType::UINT64_T},
               {"col2", ColumnValueType::STRING}},
              ',', num_shards);

    std::set<std::vector<std::string>> found_entries;
    std::vector<std::set<std::string>> found_initial;
    for (size_t i = 0; i < num_shards; i++) {
        found_initial.emplace_back();
        for (auto& entry :
             boost::make_iterator_range(boost::filesystem::directory_iterator(
                 target / std::to_string(i), {}))) {
            CSVReader<ZstdReader> reader(entry, columns, ',');

            std::vector<std::string> last_row;
            while (reader.next_row()) {
                if (!last_row.empty()) {
                    uint64_t last_row_int;
                    uint64_t current_row_int;
                    attempt_parse_or_die(reader.get_row()[0], current_row_int);
                    attempt_parse_or_die(last_row[0], last_row_int);
                    auto current =
                        std::make_pair(current_row_int, reader.get_row()[1]);
                    auto last = std::make_pair(last_row_int, last_row[1]);
                    EXPECT_EQ(current > last, true);
                }
                last_row = reader.get_row();
                found_entries.insert(last_row);
                found_initial[i].insert(last_row[0]);

                for (size_t j = 0; j < i; j++) {
                    EXPECT_EQ(found_initial[j].count(last_row[0]), 0);
                }
            }
        }
    }

    EXPECT_EQ(found_entries.size(), entries.size());

    boost::filesystem::remove_all(root);
}

TEST(JoinCsvTest, TestSortAndJoin) {
    boost::filesystem::path root = boost::filesystem::temp_directory_path() /
                                   boost::filesystem::unique_path();
    boost::filesystem::create_directory(root);
    boost::filesystem::path source = root / "source_dir";
    boost::filesystem::path target = root / "target_dir";
    boost::filesystem::create_directory(source);
    std::vector<std::string> columns = {"col1", "col2"};

    std::vector<std::vector<std::string>> entries;

    for (int i = 1; i <= 100; i++) {
        for (int j = 1; j <= 3; j++) {
            entries.push_back({std::to_string(i), std::to_string(j * 100)});
        }
    }

    std::shuffle(std::begin(entries), std::end(entries),
                 std::default_random_engine(1235423));

    size_t num_chunks = 7;

    size_t entries_per_chunk = (entries.size() + num_chunks - 1) / num_chunks;

    size_t num_shards = 5;

    for (size_t i = 0; i < num_chunks; i++) {
        CSVWriter<ZstdWriter> writer(
            (source / absl::StrCat(i, ".csv.zst")).string(), columns, ',');
        for (size_t j = 0; j < entries_per_chunk; j++) {
            size_t index = i * entries_per_chunk + j;
            if (index < entries.size()) {
                writer.add_row(entries[index]);
            }
        }
    }

    std::vector<std::string> additional_sort = {"col2"};
    sort_and_join_csvs(source.string(), target.string(),
                       {{"col1", ColumnValueType::UINT64_T},
                        {"col2", ColumnValueType::STRING}},
                       ',', num_shards);

    int num_keys = 0;
    for (size_t i = 0; i < num_shards; i++) {
        CSVReader<ZstdReader> reader(target / absl::StrCat(i, ".csv.zst"),
                                     columns, ',');
        while (reader.next_row()) {
            num_keys += 1;
            std::string key = reader.get_row()[0];
            EXPECT_EQ(reader.get_row()[0], key);
            EXPECT_EQ(reader.get_row()[1], "100");
            EXPECT_EQ(reader.next_row(), true);
            EXPECT_EQ(reader.get_row()[0], key);
            EXPECT_EQ(reader.get_row()[1], "200");
            EXPECT_EQ(reader.next_row(), true);
            EXPECT_EQ(reader.get_row()[0], key);
            EXPECT_EQ(reader.get_row()[1], "300");
        }
    }

    EXPECT_EQ(num_keys, 100);
    boost::filesystem::remove_all(root);
}

#include "csv.hh"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

TEST(CsvTest, TestCsvWriter) {
    boost::filesystem::path root = boost::filesystem::temp_directory_path() /
                                   boost::filesystem::unique_path();
    boost::filesystem::create_directories(root);

    std::vector<std::string> columns = {"col", "col1", "col\"2"};

    boost::filesystem::path path = root / "test_csv";
    {
        CSVWriter writer(path, columns, ',');
        writer.add_row({"foo\"whaw\nt", "lol", "bar"});
        writer.add_row({"1", "2", "3"});
    }
    {
        auto read_columns = get_csv_columns(path, ',');
        EXPECT_EQ(columns, read_columns);
    }
    std::vector<std::string> remapped_columns = {"col\"2", "col"};
    std::vector<std::string> expected_row = {"bar", "foo\"whaw\nt"};
    std::vector<std::string> expected_ro2 = {"3", "1"};

    {
        CSVReader reader(path, remapped_columns, ',');
        EXPECT_EQ(true, reader.next_row());
        EXPECT_EQ(expected_row, reader.get_row());
        EXPECT_EQ(true, reader.next_row());
        EXPECT_EQ(expected_ro2, reader.get_row());
        EXPECT_EQ(false, reader.next_row());
    }

    boost::filesystem::remove_all(root);
}

TEST(CsvTest, TestZstdWriter) {
    boost::filesystem::path root = boost::filesystem::temp_directory_path() /
                                   boost::filesystem::unique_path();
    boost::filesystem::create_directories(root);

    boost::filesystem::path path = root / "test_csv";
    {
        ZstdWriter writer(path);
        for (int i = 0; i < 1000000; i++) {
            writer.add_data(std::to_string(i) + ",");
            if ((i + 1) % 1000 == 0) {
                writer.add_data("\n");
            }
        }
    }
    {
        int next_integer = 0;
        ZstdReader reader(path);
        while (true) {
            auto data = reader.get_data();
            std::string temp = "";
            size_t last_good = 0;
            for (size_t i = 0; i < data.size(); i++) {
                char next = data[i];
                if (next == ',') {
                    last_good = i + 1;
                    EXPECT_EQ(temp, std::to_string(next_integer));
                    next_integer++;
                    temp = "";
                } else if (next != '\n') {
                    temp.push_back(next);
                }
            }

            if (reader.eof()) {
                break;
            }
            reader.seek(last_good);
        }
    }
    boost::filesystem::remove_all(root);
}

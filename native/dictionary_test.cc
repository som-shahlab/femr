#include "dictionary.hh"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

TEST(DictionaryTest, WriteDictionary) {
    boost::filesystem::path root = boost::filesystem::temp_directory_path() /
                                   boost::filesystem::unique_path();
    boost::filesystem::create_directory(root);

    std::cout << root << std::endl;

    boost::filesystem::path dictionary_path =
        root / boost::filesystem::unique_path();

    {
        DictionaryWriter writer(dictionary_path);
        writer.add_value("foo");
        writer.add_value("bar");
        writer.add_value("zoo");
    }

    {
        Dictionary dict(dictionary_path, true);

        std::vector<std::string_view> expected = {"foo", "bar", "zoo"};
        EXPECT_EQ(dict.get_all_text(), expected);
        EXPECT_EQ(dict.get_text(1), "bar");
        EXPECT_EQ(dict.get_code("zoo"), std::optional<uint32_t>(2));
        EXPECT_EQ(dict.get_code("bar"), std::optional<uint32_t>(1));
        EXPECT_EQ(dict.get_code("foo"), std::optional<uint32_t>(0));
        EXPECT_EQ(dict.get_code("missing"), std::nullopt);
        EXPECT_EQ(dict.get_code("zzzzzmissing"), std::nullopt);
        EXPECT_EQ(dict.get_num_entries(), 3);

        for (size_t i = 0; i < dict.get_num_entries(); i++) {
            std::string text = std::string(dict.get_text(i));
            auto other_i = dict.get_code(text);
            EXPECT_EQ(other_i.has_value(), true);
            EXPECT_EQ(*other_i, i);
        }
    }

    boost::filesystem::remove_all(root);
}
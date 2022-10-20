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
        EXPECT_EQ(dict.values(), expected);
        EXPECT_EQ(dict[1], "bar");
        EXPECT_EQ(dict.find("zoo"), boost::optional<uint32_t>(2));
        EXPECT_EQ(dict.find("bar"), boost::optional<uint32_t>(1));
        EXPECT_EQ(dict.find("foo"), boost::optional<uint32_t>(0));
        EXPECT_EQ(dict.find("missing"), boost::none);
        EXPECT_EQ(dict.find("zzzzzmissing"), boost::none);
        EXPECT_EQ(dict.size(), 3);

        for (size_t i = 0; i < dict.size(); i++) {
            std::string text = std::string(dict[i]);
            auto other_i = dict.find(text);
            auto alternative_i = dict.find(dict[i]);
            // auto alternative_i =
            //    dict.find(std::string_view(dict[i].data(), 300));
            EXPECT_EQ(other_i.has_value(), true);
            EXPECT_EQ(alternative_i.has_value(), true);
            EXPECT_EQ(*other_i, i);
            EXPECT_EQ(*alternative_i, i);
        }
    }

    boost::filesystem::remove_all(root);
}
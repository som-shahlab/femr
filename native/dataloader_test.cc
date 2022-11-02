#include "dataloader.hh"

#include <boost/optional.hpp>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

TEST(DataloaderTest, TestInitial) {
    auto process = [](size_t batch, int& item) { item = batch * 7; };

    size_t current_batch = 0;
    auto get_batch = [&current_batch]() -> boost::optional<size_t> {
        if (current_batch == 20) {
            return boost::none;
        } else {
            return current_batch++;
        }
    };

    auto loader = Dataloader(get_batch, process, 5);

    std::vector<int> results;
    while (true) {
        std::unique_ptr<int> next = loader.get_next();
        if (!next) {
            break;
        }
        results.push_back(*next);
        loader.recycle(std::move(next));
    }

    std::sort(std::begin(results), std::end(results));

    EXPECT_EQ(results.size(), 20);

    for (size_t i = 0; i < results.size(); i++) {
        EXPECT_EQ(results[i], i * 7);
    }
}

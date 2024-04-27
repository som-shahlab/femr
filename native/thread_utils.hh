#pragma once

#include <vector>

template <typename T, typename F>
void dequeue_many_loop(T& in_queues, F f) {
    std::vector<size_t> good_indices;
    good_indices.reserve(in_queues.size());
    for (size_t i = 0; i < in_queues.size(); i++) {
        good_indices.push_back(i);
    }

    typename T::value_type::value_type next_entry;

    while (good_indices.size() > 0) {
        for (size_t i = 1; i <= good_indices.size(); i++) {
            size_t index = good_indices[i - 1];
            while (true) {
                bool found = in_queues[index].try_dequeue(next_entry);

                if (!found) {
                    break;
                }

                if (!next_entry) {
                    std::swap(good_indices[i - 1], good_indices.back());
                    good_indices.pop_back();
                    i -= 1;
                    break;
                } else {
                    f(*next_entry);
                }
            }
        }
    }
}

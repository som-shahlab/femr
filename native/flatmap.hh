#pragma once

#include <boost/optional.hpp>
#include <vector>

template <typename T>
struct FlatMap {
   public:
    const T* find(size_t index) const {
        if (index < data.size()) {
            if (data[index].has_value()) {
                return &data[index].value();
            } else {
                return nullptr;
            }
        } else {
            return nullptr;
        }
    }

    T* find(size_t index) {
        if (index < data.size()) {
            if (data[index].has_value()) {
                return &data[index].value();
            } else {
                return nullptr;
            }
        } else {
            return nullptr;
        }
    }

    T* find_or_insert(size_t index, T value) {
        T* ptr = find(index);
        if (ptr == nullptr) {
            insert(index, value);
            ptr = find(index);
        }
        return ptr;
    }

    void insert(size_t index, T value) {
        if (index >= data.size()) {
            data.resize(index * 2 + 1);
        }
        if (!data[index]) {
            active_keys.push_back(index);
        }
        data[index] = boost::optional<T>(std::move(value));
    }

    const std::vector<uint32_t>& keys() const { return active_keys; }

    size_t size() const { return data.size(); }

    void clear() {
        data.clear();
        active_keys.clear();
    }

   private:
    std::vector<boost::optional<T>> data;
    std::vector<uint32_t> active_keys;
};

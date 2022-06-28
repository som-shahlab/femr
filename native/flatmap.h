#ifndef FLATMAP_H_INCLUDED
#define FLATMAP_H_INCLUDED

#include <optional>
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
        data[index] = std::optional<T>(std::move(value));
    }

    size_t size() const { return data.size(); }

    void clear() { data.clear(); }

   private:
    std::vector<std::optional<T>> data;
};

#endif
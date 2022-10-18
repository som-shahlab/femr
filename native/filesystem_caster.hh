#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

#include "absl/time/civil_time.h"

namespace pybind11 {
namespace detail {
template <>
struct type_caster<boost::filesystem::path> {
   public:
    PYBIND11_TYPE_CASTER(boost::filesystem::path, _("boost::filesystem::path"));

    bool load(handle src, bool) {
        /* Extract PyObject from handle */
        value = boost::filesystem::path(src.cast<std::string>());
        return true;
    }

    static handle cast(boost::filesystem::path src,
                       return_value_policy /* policy */, handle /* parent */) {
        return pybind11::cast(src.string()).release();
    }
};

template <typename T>
struct type_caster<boost::optional<T>> : optional_caster<boost::optional<T>> {};

template <>
struct type_caster<boost::none_t> : void_caster<boost::none_t> {};

}  // namespace detail
}  // namespace pybind11

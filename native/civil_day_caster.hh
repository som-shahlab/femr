#pragma once

#include <pybind11/pybind11.h>

#include "absl/time/civil_time.h"

namespace pybind11 {
namespace detail {
template <>
struct type_caster<absl::CivilDay> {
   public:
    PYBIND11_TYPE_CASTER(absl::CivilDay, _("absl::CivilDay"));

    bool load(handle src, bool) {
        /* Extract PyObject from handle */
        value = absl::CivilDay(src.attr("year").cast<int>(),
                               src.attr("month").cast<int>(),
                               src.attr("day").cast<int>());
        return true;
    }

    static handle cast(absl::CivilDay src, return_value_policy /* policy */,
                       handle /* parent */) {
        object date = module::import("datetime").attr("date");
        object date_obj = date(src.year(), src.month(), src.day());
        date_obj.inc_ref();
        return std::move(date_obj);
    }
};

template <>
struct type_caster<absl::CivilSecond> {
   public:
    PYBIND11_TYPE_CASTER(absl::CivilSecond, _("absl::CivilSecond"));

    bool load(handle src, bool) {
        /* Extract PyObject from handle */
        value = absl::CivilSecond(
            src.attr("year").cast<int>(), src.attr("month").cast<int>(),
            src.attr("day").cast<int>(), src.attr("hour").cast<int>(),
            src.attr("minute").cast<int>(), src.attr("second").cast<int>());
        return true;
    }

    static handle cast(absl::CivilSecond src, return_value_policy /* policy */,
                       handle /* parent */) {
        object datetime = module::import("datetime").attr("datetime");
        object date_obj = datetime(src.year(), src.month(), src.day(),
                                   src.hour(), src.minute(), src.second());
        date_obj.inc_ref();
        return std::move(date_obj);
    }
};

}  // namespace detail
}  // namespace pybind11

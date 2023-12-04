#include "jax_extension.hh"
#include "common_jax_extension.hh"

#include <pybind11/stl.h>

namespace py = pybind11;

template <class To, class From>
std::enable_if_t<sizeof(To) == sizeof(From) &&
                     std::is_trivially_copyable_v<From> &&
                     std::is_trivially_copyable_v<To>,
                 To>
// constexpr support needs compiler magic
bit_cast(const From &src) noexcept {
    static_assert(std::is_trivially_constructible_v<To>,
                  "This implementation additionally requires "
                  "destination type to be trivially constructible");

    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}

template <typename T>
py::capsule convert_to_capsule(T *fn) {
    return py::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

std::vector<std::tuple<std::string, py::capsule, std::string>> get_kernels() {
    std::vector<std::tuple<std::string, py::capsule, std::string>> result;

#define WRAP(result, A, B) result.emplace_back(#A, convert_to_capsule(A), B)

    WRAP(result, convert_to_dense, "cpu");

#undef WRAP

    return result;
}

void register_jax_extension(pybind11::module &root) {
    py::module m = root.def_submodule("jax");
    m.def("get_kernels", get_kernels);
    m.def("get_local_attention_shape", get_attention_shape);
}

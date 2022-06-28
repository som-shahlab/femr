#ifndef REGISTER_ITERABLE_H_INCLUDED
#define REGISTER_ITERABLE_H_INCLUDED

#include <pybind11/pybind11.h>
#include <string_view>

namespace py = pybind11;

template <typename T>
constexpr auto type_name() {
    std::string_view name, prefix, suffix;
#ifdef __clang__
    name = __PRETTY_FUNCTION__;
    prefix = "auto type_name() [T = ";
    suffix = "]";
#elif defined(__GNUC__)
    name = __PRETTY_FUNCTION__;
    prefix = "constexpr auto type_name() [with T = ";
    suffix = "]";
#elif defined(_MSC_VER)
    name = __FUNCSIG__;
    prefix = "auto __cdecl type_name<";
    suffix = ">(void)";
#endif
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    return name;
}

namespace detail {
template <typename L, typename R>
struct has_operator_equals_impl {
    template <typename T = L,
              typename U = R>  // template parameters here to enable SFINAE
    static auto test(T&& t, U&& u)
        -> decltype(t == u, void(), std::true_type{});
    static auto test(...) -> std::false_type;
    using type = decltype(test(std::declval<L>(), std::declval<R>()));
};
}  // namespace detail

template <typename L, typename R = L>
struct has_operator_equals : detail::has_operator_equals_impl<L, R>::type {};

template <typename T, typename std::enable_if<has_operator_equals<
                          typename T::value_type>::value>::type* = nullptr>
void register_iterable(py::module& m) {
    py::class_<T>(m, std::string(type_name<T>()).c_str())
        .def(
            "__iter__",
            [](const T& span) {
                return py::make_iterator(std::begin(span), std::end(span));
            },
            py::keep_alive<0, 1>())
        .def("__len__", [](const T& span) { return span.size(); })
        .def("__getitem__",
             [](const T& span, ssize_t index) {
                 if (index < 0) {
                     index = span.size() + index;
                 }
                 return span[index];
             })
        .def("__contains__",
             [](const T& span, const typename T::value_type& value) {
                 return std::find(std::begin(span), std::end(span), value) !=
                        std::end(span);
             });
}

template <typename T, typename std::enable_if<!has_operator_equals<
                          typename T::value_type>::value>::type* = nullptr>
void register_iterable(py::module& m) {
    py::class_<T>(m, std::string(type_name<T>()).c_str())
        .def(
            "__iter__",
            [](const T& span) {
                return py::make_iterator(std::begin(span), std::end(span));
            },
            py::keep_alive<0, 1>())
        .def("__len__", [](const T& span) { return span.size(); })
        .def("__getitem__", [](const T& span, ssize_t index) {
            if (index < 0) {
                index = span.size() + index;
            }
            return span[index];
        });
}

#endif

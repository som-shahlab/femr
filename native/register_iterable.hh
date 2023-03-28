#ifndef REGISTER_ITERABLE_H_INCLUDED
#define REGISTER_ITERABLE_H_INCLUDED

#include <pybind11/pybind11.h>

#include <string_view>

namespace py = pybind11;

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
void register_iterable(py::module& m, const char* name) {
    py::class_<T>(m, name)
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
                 if (index >= (ssize_t) span.size() || index < 0) {
                     throw py::index_error();
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
void register_iterable(py::module& m, const char* name) {
    py::class_<T>(m, name)
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

            if (index >= span.size() || index < 0) {
                throw py::index_error();
            }
            return span[index];
        });
}

#endif

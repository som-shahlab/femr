#include "jax_extension.hh"

namespace py = pybind11;

void register_jax_extension(pybind11::module &root) {
    py::module m = root.def_submodule("jax");
}
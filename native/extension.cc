#include <pybind11/pybind11.h>

#include "datasets_extension.hh"

PYBIND11_MODULE(extension, m) { register_datasets_extension(m); }

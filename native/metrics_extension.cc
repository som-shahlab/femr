#include "metrics_extension.hh"

#include <pybind11/eigen/tensor.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>

#include "survival_metrics.hh"

namespace py = pybind11;

void register_metrics_extension(pybind11::module& root) {
    py::module m = root.def_submodule("metrics");

    m.def("compute_c_statistic", compute_c_statistic);
    m.def("compute_calibration", compute_calibration);
}

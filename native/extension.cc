#include <pybind11/pybind11.h>

#include "dataloader_extension.hh"
#include "datasets_extension.hh"
#include "jax_extension.hh"

PYBIND11_MODULE(extension, m) {
    register_datasets_extension(m);
    register_dataloader_extension(m);
    register_jax_extension(m);
}

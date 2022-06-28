#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// #include "clmbr_extension.h"
// #include "extract_extension.h"
// #include "extract_subset_extension.h"
// #include "index_extension.h"
// #include "ontology_extension.h"
// #include "patient2vec_extension.h"
#include "patient_collection_extension.h"

PYBIND11_MODULE(extension, m) {
    register_patient_collection_extension(m);
    // register_index_extension(m);
    // register_ontology_extension(m);
    // register_patient2vec_extension(m);
    // register_clmbr_extension(m);
    // register_extract_extension(m);
    // register_subset_extension(m);
}

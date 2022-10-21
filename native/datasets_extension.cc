#include "datasets_extension.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

#include "absl/strings/str_cat.h"
#include "absl/time/civil_time.h"
#include "civil_day_caster.hh"
#include "database.hh"
#include "database_test_helper.hh"
#include "filesystem_caster.hh"
#include "join_csvs.hh"
#include "register_iterable.hh"

namespace py = pybind11;

void keep_alive(py::handle nurse, py::handle patient) {
    /* Fall back to clever approach based on weak references taken from
     * Boost.Python. This is not used for pybind-registered types because
     * the objects can be destroyed out-of-order in a GC pass. */
    py::cpp_function disable_lifesupport([patient](py::handle weakref) {
        patient.dec_ref();
        weakref.dec_ref();
    });

    py::weakref wr(nurse, disable_lifesupport);

    patient.inc_ref(); /* reference patient and leak the weak reference */
    (void)wr.release();
}

void register_datasets_extension(py::module& root) {
    register_iterable<absl::Span<const uint32_t>>(root, "IntSpan");

    py::module m = root.def_submodule("datasets");
    py::object piton_root = py::module_::import("piton");
    if (!piton_root) {
        throw std::runtime_error("Could not import root");
    }
    py::object python_patient = piton_root.attr("Patient");
    if (!python_patient) {
        throw std::runtime_error("Could not import python patient");
    }
    py::object python_event = piton_root.attr("Event");
    if (!python_event) {
        throw std::runtime_error("Could not import python event");
    }

    {
        py::module test_module = m.def_submodule("test");
        test_module.def("create_ontology_files", create_ontology_files);
        test_module.def("create_database_files", create_database_files);
    }

    m.def("sort_and_join_csvs", sort_and_join_csvs);

    m.def("convert_patient_collection_to_patient_database",
          convert_patient_collection_to_patient_database,
          py::return_value_policy::move);

    py::class_<PatientDatabase>(m, "PatientDatabase")
        .def(py::init<const char*, bool>(), py::arg("filename"),
             py::arg("read_all") = false)
        .def("__len__", [](PatientDatabase& self) { return self.size(); })
        .def(
            "__getitem__",
            [python_patient, python_event](py::object self_object,
                                           uint32_t index) {
                using namespace pybind11::literals;

                PatientDatabase& self = self_object.cast<PatientDatabase&>();

                Patient p = self.get_patient(index);
                py::tuple events(p.events.size());

                for (size_t i = 0; i < p.events.size(); i++) {
                    const Event& event = p.events[i];
                    absl::CivilSecond event_time = p.birth_date;
                    uint32_t minutes = event.minutes_offset;
                    minutes += 24 * 60 * event.age_in_days;
                    event_time += 60 * minutes;
                    py::object value;
                    py::object value_type;
                    switch (event.value_type) {
                        case ValueType::NONE:
                            value = py::none();
                            break;

                        case ValueType::NUMERIC:
                            value = py::cast(event.numeric_value);
                            break;

                        case ValueType::UNIQUE_TEXT:
                        case ValueType::SHARED_TEXT: {
                            std::string_view data;

                            if (event.value_type == ValueType::UNIQUE_TEXT) {
                                data = self.get_unique_text_dictionary()
                                           [event.text_value];
                            } else {
                                data = self.get_shared_text_dictionary()
                                           [event.text_value];
                            }

                            value = py::memoryview::from_memory(
                                (void*)data.data(), data.size(), true);

                            keep_alive(value, self_object);

                            break;
                        }
                    }
                    events[i] =
                        python_event("start"_a = event_time,
                                     "code"_a = event.code, "value"_a = value);
                }

                return python_patient("patient_id"_a = p.patient_id,
                                      "events"_a = events);
            },
            py::return_value_policy::reference_internal)

        .def("get_code_dictionary", &PatientDatabase::get_code_dictionary,
             py::return_value_policy::reference_internal)

        .def("get_shared_text_dictionary",
             &PatientDatabase::get_shared_text_dictionary,
             py::return_value_policy::reference_internal)

        .def("get_unique_text_dictionary",
             &PatientDatabase::get_unique_text_dictionary,
             py::return_value_policy::reference_internal)

        .def("get_ontology", &PatientDatabase::get_ontology,
             py::return_value_policy::reference_internal)

        .def("get_patient_id_from_original",
             &PatientDatabase::get_patient_id_from_original)
        .def("get_original_patient_id",
             &PatientDatabase::get_original_patient_id)

        .def("get_code_count", &PatientDatabase::get_code_count)
        .def("get_shared_text_count", &PatientDatabase::get_shared_text_count)
        .def("close", [](const PatientDatabase& self) {
            // TODO: Implement this to save memory and file pointers
        });

    py::class_<Ontology>(m, "Ontology")
        .def("get_parents", &Ontology::get_parents,
             py::return_value_policy::reference_internal)
        .def("get_children", &Ontology::get_children,
             py::return_value_policy::reference_internal)
        .def("get_all_parents", &Ontology::get_all_parents,
             py::return_value_policy::reference_internal)
        .def("get_dictionary", &Ontology::get_dictionary,
             py::return_value_policy::reference_internal);

    py::class_<Dictionary>(m, "Dictionary")
        .def("__len__", [](Dictionary& self) { return self.size(); })
        .def(
            "__getitem__",
            [](Dictionary& self, uint32_t index) {
                auto data = self[index];
                return py::memoryview::from_memory((void*)data.data(),
                                                   data.size(), true);
            },
            py::return_value_policy::reference_internal)
        .def("find", &Dictionary::find,
             py::return_value_policy::reference_internal)
        .def("values", &Dictionary::values,
             py::return_value_policy::reference_internal);
}

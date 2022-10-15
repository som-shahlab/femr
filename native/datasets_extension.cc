#include "datasets_extension.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "absl/strings/str_cat.h"
#include "absl/time/civil_time.h"
#include "civil_day_caster.hh"
#include "database.hh"
#include "filesystem_caster.hh"
#include "join_csvs.hh"
#include "register_iterable.hh"

namespace py = pybind11;

void register_datasets_extension(py::module& root) {
    register_iterable<absl::Span<const Event>>(root);
    register_iterable<absl::Span<const uint32_t>>(root);

    py::module m = root.def_submodule("datasets");

    m.def("sort_and_join_csvs", sort_and_join_csvs);

    m.def("convert_patient_collection_to_patient_database",
          convert_patient_collection_to_patient_database);

    py::class_<PatientDatabase>(m, "PatientDatabase")
        .def(py::init<const char*, bool>(), py::arg("filename"),
             py::arg("read_all") = false)
        .def("get_patient", &PatientDatabase::get_patient)
        .def("get_num_patients", &PatientDatabase::get_num_patients)

        .def("get_code_dictionary", &PatientDatabase::get_code_dictionary)
        .def("get_short_text_dictionary",
             &PatientDatabase::get_short_text_dictionary)
        .def("get_long_text_dictionary",
             &PatientDatabase::get_long_text_dictionary)

        .def("get_ontology", &PatientDatabase::get_ontology)

        .def("get_patient_id_from_original",
             &PatientDatabase::get_patient_id_from_original)
        .def("get_original_patient_id",
             &PatientDatabase::get_original_patient_id)

        .def("get_code_count", &PatientDatabase::get_code_count)
        .def("get_short_text_count", &PatientDatabase::get_short_text_count)
        .def("close", [](const PatientDatabase& self) {});

    py::class_<Patient>(m, "Patient")
        .def_readonly("patient_id", &Patient::patient_id)
        .def_readonly("birth_date", &Patient::birth_date)
        .def_readonly("events", &Patient::events)
        .def("__repr__", [](const Patient& p) {
            return absl::StrCat("<Patient patient_id=", p.patient_id, ">");
        });

    py::class_<Event>(m, "Event")
        .def_readonly("age_in_days", &Event::age_in_days)
        .def_readonly("minutes_offset", &Event::minutes_offset)
        .def_readonly("code", &Event::code)
        .def_readonly("value_type", &Event::value_type)
        .def_readonly("numeric_value", &Event::numeric_value)
        .def_readonly("text_value", &Event::text_value)
        .def("__repr__", [](const Event& e) {
            std::string value;
            switch (e.value_type) {
                case ValueType::NONE:
                    value = "";
                    break;

                case ValueType::NUMERIC:
                    value = " value=" + std::to_string(e.numeric_value);
                    break;

                case ValueType::SHORT_TEXT:
                case ValueType::LONG_TEXT:
                    value = " value=" + std::to_string(e.text_value);
                    break;
            }
            float age =
                (e.age_in_days + e.minutes_offset / (24.0 * 60.0)) / 365.0;
            return absl::StrCat("<Event code=", e.code, " age=", age, value,
                                ">");
        });

    py::enum_<ValueType>(m, "ValueType")
        .value("NONE", ValueType::NONE)
        .value("SHORT_TEXT", ValueType::SHORT_TEXT)
        .value("LONG_TEXT", ValueType::LONG_TEXT)
        .value("NUMERIC", ValueType::NUMERIC)
        .export_values();
}

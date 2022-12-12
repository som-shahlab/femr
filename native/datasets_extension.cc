#include "datasets_extension.hh"

#include <pybind11/numpy.h>
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

namespace {
class EventWrapper {
   public:
    EventWrapper(py::module pickle, PatientDatabase* database,
                 uint32_t patient_id, absl::CivilSecond birth_date,
                 uint32_t event_index, const Event& event)
        : m_pickle(pickle),
          m_database(database),
          m_patient_id(patient_id),
          m_birth_date(birth_date),
          m_event_index(event_index),
          m_event(event) {}

    py::object code() {
        if (!m_code) {
            m_code.emplace(py::cast(m_event.code));
        }
        return *m_code;
    }

    py::object start() {
        if (!m_start) {
            absl::CivilSecond start_time =
                m_birth_date + 60 * m_event.start_age_in_minutes;

            m_start.emplace(py::cast(start_time));
        }
        return *m_start;
    }

    py::object value() {
        if (!m_value) {
            switch (m_event.value_type) {
                case ValueType::NONE:
                    m_value.emplace(py::none());
                    break;

                case ValueType::NUMERIC:
                    m_value.emplace(py::cast(m_event.numeric_value));
                    break;

                case ValueType::UNIQUE_TEXT:
                case ValueType::SHARED_TEXT: {
                    std::string_view data;

                    if (m_event.value_type == ValueType::UNIQUE_TEXT) {
                        auto dict = m_database->get_unique_text_dictionary();
                        if (dict == nullptr) {
                            data = "";
                        } else {
                            data = (*dict)[m_event.text_value];
                        }
                    } else {
                        data = m_database->get_shared_text_dictionary()
                                   [m_event.text_value];
                    }

                    m_value.emplace(py::str(data.data(), data.size()));
                    break;
                }

                default:
                    throw std::runtime_error("Invalid value?");
            }
        }

        return *m_value;
    }

    py::object metadata() {
        if (!m_metadata) {
            std::string_view metadata_str =
                m_database->get_event_metadata(m_patient_id, m_event_index);
            if (metadata_str.data() != nullptr) {
                py::object bytes =
                    py::bytes(metadata_str.data(), metadata_str.size());
                m_metadata.emplace(m_pickle.attr("loads")(bytes));
            } else {
                m_metadata.emplace(py::dict());
            }
        }
        return *m_metadata;
    }

   private:
    py::module m_pickle;

    PatientDatabase* m_database;
    uint32_t m_patient_id;
    absl::CivilSecond m_birth_date;
    uint32_t m_event_index;
    Event m_event;

    boost::optional<py::object> m_start;
    boost::optional<py::object> m_code;
    boost::optional<py::object> m_value;
    boost::optional<py::object> m_metadata;
};
}  // namespace

void register_datasets_extension(py::module& root) {
    py::module pickle = py::module::import("pickle");

    py::module abc = py::module::import("collections.abc");
    py::object abc_sequence = abc.attr("Sequence");

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

    m.def("sort_and_join_csvs", [](std::string source_path,
                                   std::string target_path, py::object fields,
                                   char delimiter, int num_threads) {
        std::vector<std::pair<std::string, ColumnValueType>> column_types;

        if (py::isinstance<py::list>(fields)) {
            // Assume all string fields
            py::list list = py::reinterpret_borrow<py::list>(fields);
            for (auto item : list) {
                column_types.emplace_back(item.cast<std::string>(),
                                          ColumnValueType::STRING);
            }
        } else if (py::isinstance<py::dtype>(fields)) {
            py::dict fields_dict = fields.attr("fields");
            for (const auto& entry : fields_dict) {
                std::string name = entry.first.cast<std::string>();
                py::tuple type_and_offset = entry.second.cast<py::tuple>();
                py::dtype type = type_and_offset[0].cast<py::dtype>();
                ColumnValueType our_type;
                switch (type.kind()) {
                    case 'M':
                        our_type = ColumnValueType::DATETIME;
                        break;

                    case 'S':
                        our_type = ColumnValueType::STRING;
                        break;

                    case 'u':
                        our_type = ColumnValueType::UINT64_T;
                        break;
                    default:
                        throw std::runtime_error(absl::StrCat(
                            "Invalid kind ", std::to_string(type.kind())));
                }
                column_types.emplace_back(name, our_type);
            }
        } else {
            throw std::runtime_error(
                "Invalid type passed as fields to sort_and_join_csvs");
        }
        sort_and_join_csvs(source_path, target_path, column_types, delimiter,
                           num_threads);
    });

    m.def("convert_patient_collection_to_patient_database",
          convert_patient_collection_to_patient_database,
          py::return_value_policy::move);

    py::class_<PatientDatabase> database_binding(m, "PatientDatabase");

    database_binding
        .def(py::init<const char*, bool, bool>(), py::arg("filename"),
             py::arg("read_all") = false,
             py::arg("read_all_unique_text") = false)
        .def("__len__", [](PatientDatabase& self) { return self.size(); })
        .def(
            "__getitem__",
            [python_patient, pickle](py::object self_object,
                                     uint32_t patient_id) {
                using namespace pybind11::literals;

                PatientDatabase& self = self_object.cast<PatientDatabase&>();
                if (patient_id >= self.size()) {
                    throw py::index_error();
                }

                Patient p = self.get_patient(patient_id);
                py::tuple events(p.events.size());

                absl::CivilSecond birth_date = p.birth_date;

                for (size_t i = 0; i < p.events.size(); i++) {
                    const Event& event = p.events[i];
                    events[i] = EventWrapper(pickle, &self, patient_id,
                                             birth_date, i, event);
                }

                return python_patient("patient_id"_a = p.patient_id,
                                      "events"_a = events);
            },
            py::return_value_policy::reference_internal)
        .def("get_patient_birth_date",
             [](PatientDatabase& self, uint32_t patient_id) {
                 Patient p = self.get_patient(patient_id);
                 return p.birth_date;
             })
        .def("get_code_dictionary", &PatientDatabase::get_code_dictionary,
             py::return_value_policy::reference_internal)
        .def("get_ontology", &PatientDatabase::get_ontology,
             py::return_value_policy::reference_internal)
        .def("get_patient_id_from_original",
             &PatientDatabase::get_patient_id_from_original)
        .def("get_original_patient_id",
             &PatientDatabase::get_original_patient_id)
        .def("get_code_count", &PatientDatabase::get_code_count)
        .def("get_text_count",
             [](PatientDatabase& self, std::string data) -> uint32_t {
                 auto iter = self.get_shared_text_dictionary().find(data);
                 if (iter) {
                     return self.get_shared_text_count(*iter);
                 }

                 auto dict = self.get_unique_text_dictionary();
                 if (dict != nullptr && dict->find(data)) {
                     return 1;
                 } else {
                     return 0;
                 }
             })
        .def("compute_split", &PatientDatabase::compute_split)
        .def("version_id", &PatientDatabase::version_id)
        .def("database_id", &PatientDatabase::database_id)
        .def("close",
             [](const PatientDatabase& self) {
                 // TODO: Implement this to save memory and file pointers
             })
        .attr("__bases__") =
        py::make_tuple(abc_sequence) + database_binding.attr("__bases__");

    py::class_<Ontology>(m, "Ontology")
        .def("get_parents", &Ontology::get_parents,
             py::return_value_policy::reference_internal)
        .def("get_children", &Ontology::get_children,
             py::return_value_policy::reference_internal)
        .def("get_all_parents", &Ontology::get_all_parents,
             py::return_value_policy::reference_internal)
        .def("get_dictionary", &Ontology::get_dictionary,
             py::return_value_policy::reference_internal)
        .def("get_text_description", [](Ontology& self, uint32_t index) {
            if (index >= self.get_dictionary().size()) {
                throw py::index_error();
            }
            auto descr = self.get_text_description(index);
            return py::str(descr.data(), descr.size());
        });

    py::class_<EventWrapper>(m, "EventWrapper")
        .def_property_readonly("code", &EventWrapper::code)
        .def_property_readonly("start", &EventWrapper::start)
        .def_property_readonly("value", &EventWrapper::value)
        .def(
            "__getattr__",
            [](EventWrapper& wrapper, const std::string& attr) {
                return wrapper.metadata().attr("get")(attr, py::none());
            })
        .def("__repr__", [python_event](EventWrapper& wrapper) {
            using namespace pybind11::literals;
            return py::str(python_event(
                "code"_a = wrapper.code(), "start"_a = wrapper.start(),
                "value"_a = wrapper.value(), **wrapper.metadata()));
        });

    py::class_<Dictionary> dictionary_binding(m, "Dictionary");
    dictionary_binding
        .def("__len__", [](Dictionary& self) { return self.size(); })
        .def("__getitem__",
             [](Dictionary& self, uint32_t index) {
                 if (index >= self.size()) {
                     throw py::index_error();
                 }
                 auto data = self[index];
                 return py::str(data.data(), data.size());
             })
        .def("index",
             [](Dictionary& self, std::string data) {
                 auto iter = self.find(data);
                 if (!iter) {
                     throw py::value_error();
                 }
                 return *iter;
             })
        .attr("__bases__") =
        py::make_tuple(abc_sequence) + dictionary_binding.attr("__bases__");
}

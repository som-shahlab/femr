#pragma once

#include <boost/filesystem.hpp>

void create_ontology_files(const boost::filesystem::path& concept_root);
void create_database_files(const boost::filesystem::path& patients);

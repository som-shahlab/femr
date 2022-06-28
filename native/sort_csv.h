#ifndef SORT_CSV_H_INCLUDED
#define SORT_CSV_H_INCLUDED

#include <boost/filesystem.hpp>

void sort_csvs(boost::filesystem::path source_dir,
               boost::filesystem::path target_dir, char delimiter,
               bool use_quotes);

#endif
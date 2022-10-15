load(
    "@rules_foreign_cc//foreign_cc:defs.bzl", "cmake", "configure_make", "make"
)

load("@pybind11_bazel//:build_defs.bzl", "pybind_library", "pybind_extension")

make(
    name="zstd",
    lib_source="@source_zstd//:all",
    lib_name="libzstd",
    copts=["-fPIC"],
)

filegroup(
    name="clang_tidy_config",
    srcs=[".clang-tidy"],
    visibility=["//visibility:public"],
)

cc_library(
    name="parse_utils",
    hdrs=[
        "parse_utils.hh",
    ],
    srcs=[
        "parse_utils.cc",
    ],
    deps=[
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name="register_iterable",
    hdrs=[
        "register_iterable.hh",
    ],
    deps=[
        "@pybind11",
        "@com_google_absl//absl/strings",
    ],
)


cc_library(
    name="dictionary",
    hdrs=[
        "dictionary.hh",
    ],
    srcs=[
        "dictionary.cc",
    ],
    deps=[
        "@streamvbyte",
        "@com_google_absl//absl/strings",
        "@boost//:filesystem",
    ],
)

cc_test(
    name="dictionary_test",
    srcs=[
        "dictionary_test.cc",
    ],
    deps=[
        ":dictionary",
        "@gtest//:gtest_main",
    ],
)


cc_library(
    name="csv",
    hdrs=[
        "csv.hh",
    ],
    srcs=[
        "csv.cc",
    ],
    deps=[
        ":zstd",
        "@com_google_absl//absl/strings",
        "@boost//:filesystem",
    ],
)

cc_test(
    name="csv_test",
    srcs=[
        "csv_test.cc",
    ],
    deps=[
        ":csv",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name="join_csvs",
    hdrs=[
        "join_csvs.hh",
    ],
    srcs=[
        "join_csvs.cc",
    ],
    deps=[
        ":csv",
        "@readerwriterqueue",
        "@concurrentqueue",
        "@com_google_absl//absl/strings",
        "@boost//:filesystem",
    ],
)

cc_test(
    name="join_csvs_test",
    srcs=[
        "join_csvs_test.cc",
    ],
    deps=[
        ":join_csvs",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name="thread_utils",
    hdrs=[
        "thread_utils.hh",
    ],
    srcs=[],
    deps=[],
)

cc_library(
    name="count_codes_and_values",
    hdrs=[
        "count_codes_and_values.hh",
    ],
    srcs=[
        "count_codes_and_values.cc",
    ],
    deps=[
        "@picosha2",
        ":csv",
        ":parse_utils",
        ":thread_utils",
        "@com_google_absl//absl/strings",
        "@readerwriterqueue",
        "@concurrentqueue",
        "@boost//:filesystem",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_test(
    name="count_codes_and_values_test",
    srcs=[
        "count_codes_and_values_test.cc",
    ],
    deps=[
        ":count_codes_and_values",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name="database",
    hdrs=[
        "database.hh",
    ],
    srcs=[
        "database.cc",
    ],
    deps=[
        "@picosha2",
        ":dictionary",
        ":csv",
        ":parse_utils",
        ":thread_utils",
        ":count_codes_and_values",
        "@com_google_absl//absl/strings",
        "@readerwriterqueue",
        "@concurrentqueue",
        "@boost//:filesystem",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_test(
    name="database_test",
    srcs=[
        "database_test.cc",
    ],
    deps=[
        ":database",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name="civil_day_caster",
    hdrs=["civil_day_caster.hh"],
)

cc_library(
    name="filesystem_caster",
    hdrs=["filesystem_caster.hh"],
)

pybind_library(
    name="datasets_extension",
    hdrs=[
        "datasets_extension.hh",
    ],
    srcs=[
        "datasets_extension.cc",
    ],
    deps=[
        ":database",
        ":filesystem_caster",
        ":civil_day_caster",
        ":register_iterable",
        ":join_csvs",
    ],
)

pybind_extension(
    name="extension",
    srcs=[
        "extension.cc",
    ],
    deps=[
        ":datasets_extension",
    ],
)

cc_binary(
    name="compute_statistics",
    srcs=[
        "compute_statistics.cc",
    ],
    deps=[
        "@json",
        ":database",
    ],
)

load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake", "configure_make")

cmake(
    name = "zlib_ng",
    # Values to be passed as -Dkey=value on the CMake command line;
    # here are serving to provide some CMake script configuration options
    cache_entries = {
        "WITH_NATIVE_INSTRUCTIONS": "on",
        "WITH_GZFILEOP": "on",
        "WITH_SSE4": "off",
        "ZLIB_COMPAT": "on",
        "CMAKE_POSITION_INDEPENDENT_CODE": "on",
    },
    lib_source = "@source_zlib_ng//:all",

   # We are selecting the resulting static library to be passed in C/C++ provider
   # as the result of the build;
   # However, the cmake_external dependants could use other artefacts provided by the build,
   # according to their CMake script
    out_static_libs = ["libz.a"],
)

cc_library(
    name = "flatmap",
    hdrs = ["flatmap.h"],
    deps = []
)

cc_library (
    name="parse_utils",
    hdrs = [
        "parse_utils.h",
    ],
    srcs = [
        "parse_utils.cpp",
    ],
    deps = [
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/strings",
    ],
)

cc_library (
    name="register_iterable",
    hdrs = [
        "register_iterable.h",
    ],
    deps = [
        "@pybind11",
        "@com_google_absl//absl/strings",
    ],
)

cc_library (
    name="umls",
    hdrs = [
        "umls.h",
    ],
    deps = [
        "@boost//:filesystem",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_test (
    name="umls_test",
    srcs = [
        "umls_test.cpp",
    ],
    deps = [
        ":umls",
        "@gtest//:gtest_main",
    ],
)

cc_library (
    name="rxnorm",
    hdrs = [
        "rxnorm.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test (
    name="rxnorm_test",
    srcs = [
        "rxnorm_test.cpp",
    ],
    deps = [
        ":rxnorm",
        "@gtest//:gtest_main",
    ],
)


cc_library (
    name="gem",
    hdrs = [
        "gem.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)


cc_library (
    name="csv",
    hdrs = [
        "csv.h",
    ],
    deps = [
        ":zlib_ng",
        "@com_google_absl//absl/strings",
    ],
)

cc_library (
    name="concept",
    hdrs = [
        "concept.h",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@boost//:filesystem",
        ":parse_utils",
    ],
)

cc_library (
    name="reader",
    hdrs = [
        "reader.h",
    ],
    deps = [
        ":constdb",
        "@json",
        "@streamvbyte",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library (
    name="writer",
    hdrs = [
        "writer.h",
    ],
    deps = [
        ":constdb",
        ":reader",
        "@readerwriterqueue",
    ],
)

cc_library (
    name="constdb",
    hdrs = [
        "constdb.h"
    ],
    srcs = [
        "constdb.cpp",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
    ]
)

cc_library(
    name = "python",
    hdrs = glob(["python/*.h", "python/*/*.h"]),
    includes = ["python"],
)

cc_library(
    name = "numpy",
    hdrs = glob(["numpy/numpy/*.h"]),
)

cc_binary(
    name="extract_omop",
    srcs=[
        "extract_omop.cpp"
    ],
    deps = [
        ":csv",
        ":constdb",
        ":reader",
        ":concept",
        "@streamvbyte",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ]
)

cc_binary(
    name="omop_ontology",
    srcs=[
        "omop_ontology.cpp"
    ],
    deps = [
        ":csv",
        ":constdb",
        ":reader",
        ":concept",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name="extract_optum",
    srcs=[
        "extract_optum.cpp"
    ],
    deps = [
        ":csv",    
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        ":writer",
        ":parse_utils",
        "@readerwriterqueue",
    ]
)

cc_binary(
    name="clean_optum",
    srcs=[
        "clean_optum.cpp"
    ],
    deps = [    
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        ":reader",
        ":writer",
        ":umls",
        ":rxnorm",
        ":gem",
    ]
)

cc_binary(
    name="ontology_optum",
    srcs=[
        "ontology_optum.cpp"
    ],
    deps = [
        ":constdb",
        ":reader",
        ":umls",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],

)

cc_binary(
    name="create_index",
    srcs=[
        "create_index.cpp"
    ],
    deps = [
        ":constdb",
        ":reader",
        "@streamvbyte",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@boost//:filesystem",
    ]
)

cc_binary(
    name="featurize",
    srcs=[
        "featurize.cpp"
    ],
    deps = [
        ":constdb",
        ":reader",
        ":flatmap",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@boost//:filesystem",
        ":python",
        ":npy",
    ]
)


cc_library(
    name="npy",
    hdrs = [
        "npy.hpp"
    ],
)

cc_binary(
    name="extension.so",
    srcs = [
        "extension.cpp",
        "patient_collection_extension.cpp",
        "patient_collection_extension.h",
        "civil_day_caster.h",
    ],
    deps = [
	":register_iterable",
        ":parse_utils",
        ":concept",
        ":python",
        "@pybind11",
        ":reader",
        ":writer",
        ":numpy",
        "@concurrentqueue",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@picosha2",
        ":flatmap",
        ":gem",
        ":umls",
        ":csv",
        ":rxnorm",
        "@boost//:filesystem",
        "@boost//:algorithm",
    ],
    linkstatic=1, 
    linkshared=1
)


load(
    "@rules_foreign_cc//foreign_cc:defs.bzl", "cmake", "configure_make", "make"
)

load("@pybind11_bazel//:build_defs.bzl", "pybind_library", "pybind_extension")

load("@rules_cuda//cuda:defs.bzl", "cuda_library")

load("@bazel_skylib//rules:common_settings.bzl", "string_flag")

cuda_library(
    name="local_attention",
    hdrs=["local_attention.hh"],
    srcs=["local_attention.cu"],
)

cuda_library(
    name="embedding_dot",
    hdrs=["embedding_dot.hh"],
    srcs=["embedding_dot.cu"],
)

cuda_library(
    name="gather_scatter",
    hdrs=["gather_scatter.hh"],
    srcs=["gather_scatter.cu"],
)

cuda_library(
    name="exp_mean",
    hdrs=["exp_mean.hh"],
    srcs=["exp_mean.cu"],
)

pybind_library(
    name="jax_extension",
    hdrs=[
        "jax_extension.hh",
    ],
    srcs=[
        "jax_extension.cc",
    ],
    deps=[
        ":embedding_dot",
        ":exp_mean",
        ":local_attention",
        ":gather_scatter",
    ],
)

pybind_library(
    name="cpu_jax_extension",
    hdrs=[
        "jax_extension.hh",
    ],
    srcs=[
        "cpu_jax_extension.cc",
    ],
)

string_flag(name="cuda", build_setting_default="disabled")

config_setting(name="cuda_enabled", flag_values={":cuda": "enabled"})

jax_extension = select(
    {
        ":cuda_enabled": [":jax_extension"],
        "//conditions:default": [":cpu_jax_extension"],
    }
)

pybind_library(
    name="dataloader_extension",
    hdrs=[
        "dataloader_extension.hh",
    ],
    srcs=[
        "dataloader_extension.cc",
    ],
    deps=[
        ":create_dictionary",
        ":create_survival_dictionary",
        ":constdb",
        ":npy",
        ":database",
        ":clmbr_dictionary",
        ":flatmap",
        ":survival",
        "@eigen",
    ],
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
    name="survival_metrics",
    hdrs=[
        "survival_metrics.hh",
    ],
    srcs=[
        "survival_metrics.cc",
    ],
    deps=[
        "@eigen",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/strings",
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
        "@zstd//:everything",
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
        ":parse_utils",
        ":csv",
        "@readerwriterqueue",
        ":thread_utils",
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
    name="npy",
    hdrs=["npy.hh"],
)

cc_library(
    name="constdb",
    hdrs=["constdb.hh"],
    srcs=[
        "constdb.cc",
    ],
    deps=[
        "@com_google_absl//absl/container:flat_hash_map",
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
    name="stat_utils",
    hdrs=[
        "stat_utils.hh",
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
        ":join_csvs",
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
        "database_test_helper.hh",
    ],
    srcs=[
        "database.cc",
        "database_test_helper.cc",
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
        "@com_google_absl//absl/container:flat_hash_set",
        "@cpp-base64",
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

pybind_library(
    name="metrics_extension",
    hdrs=[
        "metrics_extension.hh",
    ],
    srcs=[
        "metrics_extension.cc",
    ],
    deps=[
        ":survival_metrics",
    ],
)


pybind_extension(
    name="extension",
    srcs=[
        "extension.cc",
    ],
    deps=jax_extension
    + [
        ":datasets_extension",
        ":dataloader_extension",
        ":metrics_extension",
    ],
)

py_test(
    name="extension_test",
    srcs=["extension_test.py"],
    data=[
        ":extension.so",
    ],
    deps=[],
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

cc_binary(
    name="prepare_notes_for_tokenization",
    srcs=[
        "prepare_notes_for_tokenization.cc",
    ],
    deps=[
        ":database",
    ],
)

cc_library(
    name="create_dictionary",
    hdrs=["create_dictionary.hh"],
    srcs=[
        "create_dictionary.cc",
    ],
    deps=[
        "@json",
        ":database",
        ":flatmap",
        ":clmbr_dictionary",
        ":stat_utils",
    ],
)

cc_library(
    name="create_survival_dictionary",
    hdrs=["create_survival_dictionary.hh"],
    srcs=[
        "create_survival_dictionary.cc",
    ],
    deps=[
        "@json",
        ":database",
        ":flatmap",
        ":survival",
        ":stat_utils",
    ],
)

cc_library(
    name="flatmap",
    hdrs=[
        "flatmap.hh",
    ],
    deps=[
        "@boost//:optional",
    ],
)

cc_library(
    name="survival",
    hdrs=[
        "survival.hh",
    ],
    deps=[
        ":flatmap",
    ],
)

cc_library(
    name="clmbr_dictionary",
    hdrs=[
        "clmbr_dictionary.hh",
    ],
    deps=[
        "@json",
    ],
)


cc_binary(
    name="convert_dictionary_formats",
    srcs = ["convert_dictionary_formats.cc"],
    deps = [
        ":database",
        "@json",
    ]
)

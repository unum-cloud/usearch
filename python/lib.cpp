/**
 *  @file python.cpp
 *  @author Ash Vardanian
 *  @brief Python bindings for Unum USearch.
 *  @date 2023-04-26
 *
 *
 *  https://pythoncapi.readthedocs.io/type_object.html
 *  https://numpy.org/doc/stable/reference/c-api/types-and-structures.html
 *  https://pythonextensionpatterns.readthedocs.io/en/latest/refcount.html
 *  https://docs.python.org/3/extending/newtypes_tutorial.html#adding-data-and-methods-to-the-basic-example
 *
 *  @copyright Copyright (c) 2023
 */
#define __cpp_exceptions 1
#include <limits> // `std::numeric_limits`
#include <thread> // `std::thread`

#define _CRT_SECURE_NO_WARNINGS
#define PY_SSIZE_T_CLEAN
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <usearch/index_dense.hpp>
#include <usearch/index_plugins.hpp>

using namespace unum::usearch;
using namespace unum;

/**
 *  @brief  The signature of the user-defined function.
 *          Can be just two array pointers, precompiled for a specific array length,
 *          or include one or two array sizes as 64-bit unsigned integers.
 */
enum class metric_signature_t {
    array_array_k = 0,
    array_array_size_k,
    array_size_array_size_k,
};

namespace py = pybind11;
using py_shape_t = py::array::ShapeContainer;
#define key_t typename index_dense_t::key_t
using metric_t = metric_punned_t;
using distance_t = distance_punned_t;

using dense_add_result_t = typename index_dense_t::add_result_t;
using dense_search_result_t = typename index_dense_t::search_result_t;
using dense_labeling_result_t = typename index_dense_t::labeling_result_t;

struct dense_index_py_t : public index_dense_t {
    using native_t = index_dense_t;
    using native_t::add;
    using native_t::capacity;
    using native_t::reserve;
    using native_t::search;
    using native_t::size;

    dense_index_py_t(native_t&& base) : index_dense_t(std::move(base)) {}
};

struct dense_indexes_py_t {
    std::vector<std::shared_ptr<dense_index_py_t>> shards_;

    void add(std::shared_ptr<dense_index_py_t> shard) { shards_.push_back(shard); }
    std::size_t bytes_per_vector() const noexcept { return shards_.empty() ? 0 : shards_[0]->bytes_per_vector(); }
    std::size_t scalar_words() const noexcept { return shards_.empty() ? 0 : shards_[0]->scalar_words(); }
    index_limits_t limits() const noexcept { return {size(), std::numeric_limits<std::size_t>::max()}; }

    std::size_t size() const noexcept {
        std::size_t result = 0;
        for (auto const& shard : shards_)
            result += shard->size();
        return result;
    }

    void reserve(index_limits_t) {
        for (auto const& shard : shards_)
            shard->reserve({shard->size(), 1});
    }
};

template <typename scalar_at>
metric_t typed_udf(                                                                  //
    metric_kind_t kind, metric_signature_t signature, std::uintptr_t metric_uintptr, //
    scalar_kind_t scalar_kind, std::size_t dimensions) {
    //
    using stl_function_t = metric_t::stl_function_t;
    stl_function_t stl_function;
    std::size_t scalar_words =
        divide_round_up(dimensions * bits_per_scalar(scalar_kind), bits_per_scalar_word(scalar_kind));
    switch (signature) {
    case metric_signature_t::array_array_k:
        stl_function = [metric_uintptr](byte_t const* a, byte_t const* b) -> distance_t {
            using metric_raw_t = distance_punned_t (*)(scalar_at const*, scalar_at const*);
            metric_raw_t metric_ptr = reinterpret_cast<metric_raw_t>(metric_uintptr);
            return metric_ptr((scalar_at const*)a, (scalar_at const*)b);
        };
        break;
    case metric_signature_t::array_array_size_k:
        stl_function = [metric_uintptr, scalar_words](byte_t const* a, byte_t const* b) -> distance_t {
            using metric_raw_t = distance_punned_t (*)(scalar_at const*, scalar_at const*, size_t);
            metric_raw_t metric_ptr = reinterpret_cast<metric_raw_t>(metric_uintptr);
            return metric_ptr((scalar_at const*)a, (scalar_at const*)b, scalar_words);
        };
        break;
    case metric_signature_t::array_size_array_size_k:
        stl_function = [metric_uintptr, scalar_words](byte_t const* a, byte_t const* b) -> distance_t {
            using metric_raw_t = distance_punned_t (*)(scalar_at const*, size_t, scalar_at const*, size_t);
            metric_raw_t metric_ptr = reinterpret_cast<metric_raw_t>(metric_uintptr);
            return metric_ptr((scalar_at const*)a, scalar_words, (scalar_at const*)b, scalar_words);
        };
        break;
    }
    return metric_t(stl_function, dimensions, kind, scalar_kind);
}

metric_t udf(metric_kind_t kind, metric_signature_t signature, std::uintptr_t metric_uintptr, //
             scalar_kind_t scalar_kind, std::size_t dimensions) {
    switch (scalar_kind) {
    case scalar_kind_t::b1x8_k: return typed_udf<b1x8_t>(kind, signature, metric_uintptr, scalar_kind, dimensions);
    case scalar_kind_t::f8_k: return typed_udf<f8_bits_t>(kind, signature, metric_uintptr, scalar_kind, dimensions);
    case scalar_kind_t::f16_k: return typed_udf<f16_t>(kind, signature, metric_uintptr, scalar_kind, dimensions);
    case scalar_kind_t::f32_k: return typed_udf<f32_t>(kind, signature, metric_uintptr, scalar_kind, dimensions);
    case scalar_kind_t::f64_k: return typed_udf<f64_t>(kind, signature, metric_uintptr, scalar_kind, dimensions);
    default: return {};
    }
}

static dense_index_py_t make_index(      //
    std::size_t dimensions,              //
    scalar_kind_t scalar_kind,           //
    metric_kind_t metric_kind,           //
    std::size_t connectivity,            //
    std::size_t expansion_add,           //
    std::size_t expansion_search,        //
    metric_signature_t metric_signature, //
    std::uintptr_t metric_uintptr) {

    index_dense_config_t config(connectivity, expansion_add, expansion_search);
    metric_t metric =  //
        metric_uintptr //
            ? udf(metric_kind, metric_signature, metric_uintptr, scalar_kind, dimensions)
            : metric_t(dimensions, metric_kind, scalar_kind);
    return index_dense_t::make(metric, config);
}

scalar_kind_t numpy_string_to_kind(std::string const& name) {
    // https://docs.python.org/3/library/struct.html#format-characters
    if (name == "B" || name == "<B" || name == "u1" || name == "|u1")
        return scalar_kind_t::b1x8_k;
    else if (name == "b" || name == "<b" || name == "i1" || name == "|i1")
        return scalar_kind_t::f8_k;
    else if (name == "e" || name == "<e" || name == "f2" || name == "<f2")
        return scalar_kind_t::f16_k;
    else if (name == "f" || name == "<f" || name == "f4" || name == "<f4")
        return scalar_kind_t::f32_k;
    else if (name == "d" || name == "<d" || name == "f8" || name == "<f8")
        return scalar_kind_t::f64_k;
    else
        return scalar_kind_t::unknown_k;
}

template <typename scalar_at>
static void add_typed_to_index(                                            //
    dense_index_py_t& index,                                               //
    py::buffer_info const& keys_info, py::buffer_info const& vectors_info, //
    bool force_copy, std::size_t threads) {

    Py_ssize_t vectors_count = vectors_info.shape[0];
    byte_t const* vectors_data = reinterpret_cast<byte_t const*>(vectors_info.ptr);
    byte_t const* keys_data = reinterpret_cast<byte_t const*>(keys_info.ptr);

    executor_default_t{threads}.execute_bulk(vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
        index_dense_add_config_t config;
        config.force_vector_copy = force_copy;
        config.thread = thread_idx;
        key_t key = *reinterpret_cast<key_t const*>(keys_data + task_idx * keys_info.strides[0]);
        scalar_at const* vector = reinterpret_cast<scalar_at const*>(vectors_data + task_idx * vectors_info.strides[0]);
        index.add(key, vector, config).error.raise();
        if (PyErr_CheckSignals() != 0)
            throw py::error_already_set();
    });
}

template <typename index_at>
static void add_many_to_index(                            //
    index_at& index, py::buffer keys, py::buffer vectors, //
    bool force_copy, std::size_t threads) {

    py::buffer_info keys_info = keys.request();
    py::buffer_info vectors_info = vectors.request();

    if (keys_info.itemsize != sizeof(key_t))
        throw std::invalid_argument("Incompatible key type!");

    if (keys_info.ndim != 1)
        throw std::invalid_argument("Labels must be placed in a single-dimensional array!");

    if (vectors_info.ndim != 2)
        throw std::invalid_argument("Expects a matrix of vectors to add!");

    Py_ssize_t labels_count = keys_info.shape[0];
    Py_ssize_t vectors_count = vectors_info.shape[0];
    Py_ssize_t vectors_dimensions = vectors_info.shape[1];
    if (vectors_dimensions != static_cast<Py_ssize_t>(index.scalar_words()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    if (labels_count != vectors_count)
        throw std::invalid_argument("Number of keys and vectors must match!");

    if (!threads)
        threads = std::thread::hardware_concurrency();
    if (!index.reserve(index_limits_t(ceil2(index.size() + vectors_count), threads)))
        throw std::invalid_argument("Out of memory!");

    // clang-format off
    switch (numpy_string_to_kind(vectors_info.format)) {
    case scalar_kind_t::b1x8_k: add_typed_to_index<b1x8_t>(index, keys_info, vectors_info, force_copy, threads); break;
    case scalar_kind_t::f8_k: add_typed_to_index<f8_bits_t>(index, keys_info, vectors_info, force_copy, threads); break;
    case scalar_kind_t::f16_k: add_typed_to_index<f16_t>(index, keys_info, vectors_info, force_copy, threads); break;
    case scalar_kind_t::f32_k: add_typed_to_index<f32_t>(index, keys_info, vectors_info, force_copy, threads); break;
    case scalar_kind_t::f64_k: add_typed_to_index<f64_t>(index, keys_info, vectors_info, force_copy, threads); break;
    default: throw std::invalid_argument("Incompatible scalars in the vectors matrix: " + vectors_info.format);
    }
    // clang-format on
}

template <typename scalar_at>
static void search_typed(                                   //
    dense_index_py_t& index, py::buffer_info& vectors_info, //
    std::size_t wanted, bool exact, std::size_t threads,    //
    py::array_t<key_t>& labels_py, py::array_t<distance_t>& distances_py, py::array_t<Py_ssize_t>& counts_py) {

    auto labels_py2d = labels_py.template mutable_unchecked<2>();
    auto distances_py2d = distances_py.template mutable_unchecked<2>();
    auto counts_py1d = counts_py.template mutable_unchecked<1>();

    Py_ssize_t vectors_count = vectors_info.shape[0];
    byte_t const* vectors_data = reinterpret_cast<byte_t const*>(vectors_info.ptr);

    if (!threads)
        threads = std::thread::hardware_concurrency();
    if (!index.reserve(index_limits_t(index.size(), threads)))
        throw std::invalid_argument("Out of memory!");

    executor_default_t{threads}.execute_bulk(vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
        index_search_config_t config;
        config.thread = thread_idx;
        config.exact = exact;
        scalar_at const* vector = (scalar_at const*)(vectors_data + task_idx * vectors_info.strides[0]);
        dense_search_result_t result = index.search(vector, wanted, config);
        result.error.raise();
        counts_py1d(task_idx) =
            static_cast<Py_ssize_t>(result.dump_to(&labels_py2d(task_idx, 0), &distances_py2d(task_idx, 0)));
        if (PyErr_CheckSignals() != 0)
            throw py::error_already_set();
    });
}

template <typename scalar_at>
static void search_typed(                                             //
    dense_indexes_py_t const& indexes, py::buffer_info& vectors_info, //
    std::size_t wanted, bool exact, std::size_t threads,              //
    py::array_t<key_t>& labels_py, py::array_t<distance_t>& distances_py, py::array_t<Py_ssize_t>& counts_py) {

    auto labels_py2d = labels_py.template mutable_unchecked<2>();
    auto distances_py2d = distances_py.template mutable_unchecked<2>();
    auto counts_py1d = counts_py.template mutable_unchecked<1>();

    Py_ssize_t vectors_count = vectors_info.shape[0];
    byte_t const* vectors_data = reinterpret_cast<byte_t const*>(vectors_info.ptr);
    for (std::size_t vector_idx = 0; vector_idx != static_cast<std::size_t>(vectors_count); ++vector_idx)
        counts_py1d(vector_idx) = 0;

    if (!threads)
        threads = std::thread::hardware_concurrency();

    std::vector<std::mutex> vectors_mutexes(static_cast<std::size_t>(vectors_count));
    executor_default_t{threads}.execute_bulk(indexes.size(), [&](std::size_t, std::size_t task_idx) {
        dense_index_py_t const& index = *indexes.shards_[task_idx].get();

        index_search_config_t config;
        config.thread = 0;
        config.exact = exact;
        for (std::size_t vector_idx = 0; vector_idx != static_cast<std::size_t>(vectors_count); ++vector_idx) {
            scalar_at const* vector = (scalar_at const*)(vectors_data + vector_idx * vectors_info.strides[0]);
            dense_search_result_t result = index.search(vector, wanted, config);
            result.error.raise();
            {
                std::unique_lock<std::mutex> lock(vectors_mutexes[vector_idx]);
                counts_py1d(vector_idx) = static_cast<Py_ssize_t>(result.merge_into( //
                    &labels_py2d(vector_idx, 0),                                     //
                    &distances_py2d(vector_idx, 0),                                  //
                    static_cast<std::size_t>(counts_py1d(vector_idx)),               //
                    wanted));
            }
            if (PyErr_CheckSignals() != 0)
                throw py::error_already_set();
        }
    });
}

/**
 *  @param vectors Matrix of vectors to search for.
 *  @param wanted Number of matches per request.
 *
 *  @return Tuple with:
 *      1. matrix of neighbors,
 *      2. matrix of distances,
 *      3. array with match counts.
 */
template <typename index_at>
static py::tuple search_many_in_index( //
    index_at& index, py::buffer vectors, std::size_t wanted, bool exact, std::size_t threads) {

    if (wanted == 0)
        return py::tuple(3);

    if (index.limits().threads_search < threads)
        throw std::invalid_argument("Can't use that many threads!");

    py::buffer_info vectors_info = vectors.request();
    if (vectors_info.ndim != 2)
        throw std::invalid_argument("Expects a matrix of vectors to add!");

    Py_ssize_t vectors_count = vectors_info.shape[0];
    Py_ssize_t vectors_dimensions = vectors_info.shape[1];
    if (vectors_dimensions != static_cast<Py_ssize_t>(index.scalar_words()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    py::array_t<key_t> ls({vectors_count, static_cast<Py_ssize_t>(wanted)});
    py::array_t<distance_t> ds({vectors_count, static_cast<Py_ssize_t>(wanted)});
    py::array_t<Py_ssize_t> cs(vectors_count);

    switch (numpy_string_to_kind(vectors_info.format)) {
    case scalar_kind_t::b1x8_k: search_typed<b1x8_t>(index, vectors_info, wanted, exact, threads, ls, ds, cs); break;
    case scalar_kind_t::f8_k: search_typed<f8_bits_t>(index, vectors_info, wanted, exact, threads, ls, ds, cs); break;
    case scalar_kind_t::f16_k: search_typed<f16_t>(index, vectors_info, wanted, exact, threads, ls, ds, cs); break;
    case scalar_kind_t::f32_k: search_typed<f32_t>(index, vectors_info, wanted, exact, threads, ls, ds, cs); break;
    case scalar_kind_t::f64_k: search_typed<f64_t>(index, vectors_info, wanted, exact, threads, ls, ds, cs); break;
    default: throw std::invalid_argument("Incompatible scalars in the query matrix: " + vectors_info.format);
    }

    py::tuple results(3);
    results[0] = ls;
    results[1] = ds;
    results[2] = cs;
    return results;
}

static std::unordered_map<key_t, key_t> join_index(       //
    dense_index_py_t const& a, dense_index_py_t const& b, //
    std::size_t max_proposals, bool exact) {

    std::unordered_map<key_t, key_t> a_to_b;
    a_to_b.reserve((std::min)(a.size(), b.size()));

    // index_join_config_t config;

    // config.max_proposals = max_proposals;
    // config.exact = exact;
    // config.expansion = (std::max)(a.expansion_search(), b.expansion_search());
    // std::size_t threads = (std::min)(a.limits().threads(), b.limits().threads());
    // executor_default_t executor{threads};
    // join_result_t result = dense_index_py_t::join( //
    //     a, b, config,                              //
    //     a_to_b,                                    //
    //     dummy_label_to_label_mapping_t{},          //
    //     executor);
    // result.error.raise();
    return a_to_b;
}

static dense_index_py_t copy_index(dense_index_py_t const& index) {

    using copy_result_t = typename dense_index_py_t::copy_result_t;
    index_copy_config_t config;
    copy_result_t result = index.copy(config);
    result.error.raise();
    return std::move(result.index);
}

// clang-format off
template <typename index_at> void save_index(index_at const& index, std::string const& path) { index.save(path.c_str()).error.raise(); }
template <typename index_at> void load_index(index_at& index, std::string const& path) { index.load(path.c_str()).error.raise(); }
template <typename index_at> void view_index(index_at& index, std::string const& path) { index.view(path.c_str()).error.raise(); }
template <typename index_at> void clear_index(index_at& index) { index.clear(); }
template <typename index_at> std::size_t max_level(index_at const &index) { return index.max_level(); }
template <typename index_at> typename index_at::stats_t compute_stats(index_at const &index) { return index.stats(); }
template <typename index_at> typename index_at::stats_t compute_level_stats(index_at const &index, std::size_t level) { return index.stats(level); }
// clang-format on

template <typename internal_at, typename external_at = internal_at, typename index_at = void>
py::object get_typed_member(index_at const& index, key_t key) {
    py::array_t<external_at> result_py(static_cast<Py_ssize_t>(index.scalar_words()));
    auto result_py1d = result_py.template mutable_unchecked<1>();
    if (!index.get(key, (internal_at*)&result_py1d(0)))
        return py::none();
    return py::object(result_py);
}

template <typename index_at> py::object get_member(index_at const& index, key_t key, scalar_kind_t scalar_kind) {
    if (scalar_kind == scalar_kind_t::f32_k)
        return get_typed_member<f32_t>(index, key);
    else if (scalar_kind == scalar_kind_t::f64_k)
        return get_typed_member<f64_t>(index, key);
    else if (scalar_kind == scalar_kind_t::f16_k)
        return get_typed_member<f16_t, std::uint16_t>(index, key);
    else if (scalar_kind == scalar_kind_t::f8_k)
        return get_typed_member<f8_bits_t, std::int8_t>(index, key);
    else if (scalar_kind == scalar_kind_t::b1x8_k)
        return get_typed_member<b1x8_t, std::uint8_t>(index, key);
    else
        throw std::invalid_argument("Incompatible scalars in the query matrix!");
}

template <typename index_at>
py::array_t<key_t> get_labels(index_at const& index, std::size_t offset, std::size_t limit) {
    limit = std::min(index.size(), limit);
    py::array_t<key_t> result_py(static_cast<Py_ssize_t>(limit));
    auto result_py1d = result_py.template mutable_unchecked<1>();
    index.export_labels(&result_py1d(0), offset, limit);
    return result_py;
}

template <typename index_at> py::array_t<key_t> get_all_labels(index_at const& index) {
    return get_labels(index, 0, index.size());
}

template <typename element_at> bool has_duplicates(element_at const* begin, element_at const* end) {
    if (begin == end)
        return false;
    element_at const* last = begin;
    begin++;
    for (; begin != end; ++begin, ++last) {
        if (*begin == *last)
            return true;
    }
    return false;
}

PYBIND11_MODULE(compiled, m) {
    m.doc() = "Smaller & Faster Single-File Vector Search Engine from Unum";

    m.attr("DEFAULT_CONNECTIVITY") = py::int_(default_connectivity());
    m.attr("DEFAULT_EXPANSION_ADD") = py::int_(default_expansion_add());
    m.attr("DEFAULT_EXPANSION_SEARCH") = py::int_(default_expansion_search());

    m.attr("USES_OPENMP") = py::int_(USEARCH_USE_OPENMP);
    m.attr("USES_SIMSIMD") = py::int_(USEARCH_USE_SIMSIMD);
    m.attr("USES_NATIVE_F16") = py::int_(USEARCH_USE_NATIVE_F16);

    py::enum_<metric_signature_t>(m, "MetricSignature")
        .value("ArrayArray", metric_signature_t::array_array_k)
        .value("ArrayArraySize", metric_signature_t::array_array_size_k)
        .value("ArraySizeArraySize", metric_signature_t::array_size_array_size_k);

    py::enum_<metric_kind_t>(m, "MetricKind")
        .value("Unknown", metric_kind_t::unknown_k)
        .value("IP", metric_kind_t::ip_k)
        .value("Cos", metric_kind_t::cos_k)
        .value("L2sq", metric_kind_t::l2sq_k)
        .value("Haversine", metric_kind_t::haversine_k)
        .value("Pearson", metric_kind_t::pearson_k)
        .value("Jaccard", metric_kind_t::jaccard_k)
        .value("Hamming", metric_kind_t::hamming_k)
        .value("Tanimoto", metric_kind_t::tanimoto_k)
        .value("Sorensen", metric_kind_t::sorensen_k);

    py::enum_<scalar_kind_t>(m, "ScalarKind")
        .value("Unknown", scalar_kind_t::unknown_k)
        .value("B1", scalar_kind_t::b1x8_k)
        .value("U40", scalar_kind_t::u40_k)
        .value("UUID", scalar_kind_t::uuid_k)
        .value("F64", scalar_kind_t::f64_k)
        .value("F32", scalar_kind_t::f32_k)
        .value("F16", scalar_kind_t::f16_k)
        .value("F8", scalar_kind_t::f8_k)
        .value("U64", scalar_kind_t::u64_k)
        .value("U32", scalar_kind_t::u32_k)
        .value("U16", scalar_kind_t::u16_k)
        .value("U8", scalar_kind_t::u8_k)
        .value("I64", scalar_kind_t::i64_k)
        .value("I32", scalar_kind_t::i32_k)
        .value("I16", scalar_kind_t::i16_k)
        .value("I8", scalar_kind_t::i8_k);

    m.def("index_dense_metadata", [](std::string const& path) -> py::dict {
        index_dense_metadata_result_t meta = index_dense_metadata(path.c_str());
        meta.error.raise();
        index_dense_head_t const& head = meta.head;

        py::dict result;
        result["matrix_included"] = !meta.config.exclude_vectors;
        result["matrix_uses_64_bit_dimensions"] = meta.config.use_64_bit_dimensions;

        result["version"] = std::to_string(head.version_major) + "." + //
                            std::to_string(head.version_minor) + "." + //
                            std::to_string(head.version_patch);

        result["kind_metric"] = metric_kind_t(head.kind_metric);
        result["kind_scalar"] = scalar_kind_t(head.kind_scalar);
        result["kind_key"] = scalar_kind_t(head.kind_key);
        result["kind_compressed_slot"] = scalar_kind_t(head.kind_compressed_slot);

        result["count_present"] = std::uint64_t(head.count_present);
        result["count_deleted"] = std::uint64_t(head.count_deleted);
        result["dimensions"] = std::uint64_t(head.dimensions);

        return result;
    });

    auto i = py::class_<dense_index_py_t, std::shared_ptr<dense_index_py_t>>(m, "Index");

    i.def(py::init(&make_index),                                           //
          py::kw_only(),                                                   //
          py::arg("ndim") = 0,                                             //
          py::arg("dtype") = scalar_kind_t::f32_k,                         //
          py::arg("metric_kind") = metric_kind_t::ip_k,                    //
          py::arg("connectivity") = default_connectivity(),                //
          py::arg("expansion_add") = default_expansion_add(),              //
          py::arg("expansion_search") = default_expansion_search(),        //
          py::arg("metric_signature") = metric_signature_t::array_array_k, //
          py::arg("metric_pointer") = 0                                    //
    );

    i.def(                                           //
        "add", &add_many_to_index<dense_index_py_t>, //
        py::arg("keys"),                             //
        py::arg("vectors"),                          //
        py::kw_only(),                               //
        py::arg("copy") = true,                      //
        py::arg("threads") = 0                       //
    );

    i.def(                                                 //
        "search", &search_many_in_index<dense_index_py_t>, //
        py::arg("query"),                                  //
        py::arg("count") = 10,                             //
        py::arg("exact") = false,                          //
        py::arg("threads") = 0                             //
    );

    i.def(
        "rename",
        [](dense_index_py_t& index, key_t from, key_t to) -> bool {
            dense_labeling_result_t result = index.rename(from, to);
            result.error.raise();
            return result.completed;
        },
        py::arg("from"), py::arg("to"));

    i.def(
        "remove",
        [](dense_index_py_t& index, key_t key, bool compact, std::size_t threads) -> bool {
            dense_labeling_result_t result = index.remove(key);
            result.error.raise();
            if (!compact)
                return result.completed;

            if (!threads)
                threads = std::thread::hardware_concurrency();
            if (!index.reserve(index_limits_t(index.size(), threads)))
                throw std::invalid_argument("Out of memory!");

            index.compact(executor_default_t{threads});
            return result.completed;
        },
        py::arg("key"), py::arg("compact"), py::arg("threads"));

    i.def(
        "remove",
        [](dense_index_py_t& index, std::vector<key_t> const& keys, bool compact, std::size_t threads) -> std::size_t {
            dense_labeling_result_t result = index.remove(keys.begin(), keys.end());
            result.error.raise();
            if (!compact)
                return result.completed;

            if (!threads)
                threads = std::thread::hardware_concurrency();
            if (!index.reserve(index_limits_t(index.size(), threads)))
                throw std::invalid_argument("Out of memory!");

            index.compact(executor_default_t{threads});
            return result.completed;
        },
        py::arg("key"), py::arg("compact"), py::arg("threads"));

    i.def("__len__", &dense_index_py_t::size);
    i.def_property_readonly("size", &dense_index_py_t::size);
    i.def_property_readonly("connectivity", &dense_index_py_t::connectivity);
    i.def_property_readonly("capacity", &dense_index_py_t::capacity);
    i.def_property_readonly("ndim",
                            [](dense_index_py_t const& index) -> std::size_t { return index.metric().dimensions(); });
    i.def_property_readonly( //
        "dtype", [](dense_index_py_t const& index) -> scalar_kind_t { return index.scalar_kind(); });
    i.def_property_readonly( //
        "memory_usage", [](dense_index_py_t const& index) -> std::size_t { return index.memory_usage(); },
        py::call_guard<py::gil_scoped_release>());

    i.def_property("expansion_add", &dense_index_py_t::expansion_add, &dense_index_py_t::change_expansion_add);
    i.def_property("expansion_search", &dense_index_py_t::expansion_search, &dense_index_py_t::change_expansion_search);

    i.def_property_readonly("keys", &get_all_labels<dense_index_py_t>);
    i.def("get_labels", &get_labels<dense_index_py_t>, py::arg("offset") = 0,
          py::arg("limit") = std::numeric_limits<std::size_t>::max());
    i.def("__contains__", &dense_index_py_t::contains);
    i.def("__getitem__", &get_member<dense_index_py_t>, py::arg("key"), py::arg("dtype") = scalar_kind_t::f32_k);

    i.def("save", &save_index<dense_index_py_t>, py::arg("path"), py::call_guard<py::gil_scoped_release>());
    i.def("load", &load_index<dense_index_py_t>, py::arg("path"), py::call_guard<py::gil_scoped_release>());
    i.def("view", &view_index<dense_index_py_t>, py::arg("path"), py::call_guard<py::gil_scoped_release>());
    i.def("clear", &clear_index<dense_index_py_t>, py::call_guard<py::gil_scoped_release>());
    i.def("copy", &copy_index, py::call_guard<py::gil_scoped_release>());
    i.def("join", &join_index, py::arg("other"), py::arg("max_proposals") = 0, py::arg("exact") = false,
          py::call_guard<py::gil_scoped_release>());

    using punned_index_stats_t = typename dense_index_py_t::stats_t;
    auto i_stats = py::class_<punned_index_stats_t>(m, "IndexStats");
    i_stats.def_readonly("nodes", &punned_index_stats_t::nodes);
    i_stats.def_readonly("edges", &punned_index_stats_t::edges);
    i_stats.def_readonly("max_edges", &punned_index_stats_t::max_edges);
    i_stats.def_readonly("allocated_bytes", &punned_index_stats_t::allocated_bytes);

    i.def_property_readonly("max_level", &max_level<dense_index_py_t>);
    i.def_property_readonly("levels_stats", &compute_stats<dense_index_py_t>);
    i.def("level_stats", &compute_level_stats<dense_index_py_t>, py::arg("level"));

    auto is = py::class_<dense_indexes_py_t>(m, "Indexes");
    is.def(py::init());
    is.def("add", &dense_indexes_py_t::add);
    is.def(                                                  //
        "search", &search_many_in_index<dense_indexes_py_t>, //
        py::arg("query"),                                    //
        py::arg("count") = 10,                               //
        py::arg("exact") = false,                            //
        py::arg("threads") = 0                               //
    );
}

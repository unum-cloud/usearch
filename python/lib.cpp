/**
 *  @file python.cpp
 *  @author Ashot Vardanian
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
#include <limits> // `std::numeric_limits`
#include <thread> // `std::thread`

#define _CRT_SECURE_NO_WARNINGS
#define PY_SSIZE_T_CLEAN
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <usearch/index_punned_dense.hpp>

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
using label_t = typename punned_small_t::label_t;
using id_t = typename punned_small_t::id_t;
using metric_t = index_punned_dense_metric_t;
using distance_t = punned_distance_t;
using dense_index_t = punned_small_t;
using dense_add_result_t = typename dense_index_t::add_result_t;
using dense_search_result_t = typename dense_index_t::search_result_t;
using dense_labeling_result_t = typename dense_index_t::labeling_result_t;

struct dense_index_py_t : public dense_index_t {
    using native_t = dense_index_t;
    using native_t::add;
    using native_t::capacity;
    using native_t::reserve;
    using native_t::search;
    using native_t::size;

    dense_index_py_t(native_t&& base) : native_t(std::move(base)) {}
};

using set_member_t = std::uint32_t;
using set_view_t = span_gt<set_member_t const>;
using sparse_index_t = index_gt<jaccard_gt<set_member_t>, label_t, id_t>;

struct sparse_index_py_t : public sparse_index_t {
    using native_t = sparse_index_t;
    using native_t::add;
    using native_t::capacity;
    using native_t::reserve;
    using native_t::search;
    using native_t::size;

    sparse_index_py_t(native_t&& base) : native_t(std::move(base)) {}
};

template <typename scalar_at>
metric_t typed_udf( //
    metric_kind_t kind, metric_signature_t signature, std::uintptr_t metric_uintptr, scalar_kind_t accuracy) {
    //
    metric_t result;
    result.kind_ = kind;
    result.scalar_kind_ = accuracy;
    switch (signature) {
    case metric_signature_t::array_array_k:
        result.func_ = [metric_uintptr](punned_vector_view_t a, punned_vector_view_t b) -> distance_t {
            using metric_raw_t = punned_distance_t (*)(scalar_at const*, scalar_at const*);
            metric_raw_t metric_ptr = reinterpret_cast<metric_raw_t>(metric_uintptr);
            return metric_ptr((scalar_at const*)a.data(), (scalar_at const*)b.data());
        };
        break;
    case metric_signature_t::array_array_size_k:
        result.func_ = [metric_uintptr](punned_vector_view_t a, punned_vector_view_t b) -> distance_t {
            using metric_raw_t = punned_distance_t (*)(scalar_at const*, scalar_at const*, size_t);
            metric_raw_t metric_ptr = reinterpret_cast<metric_raw_t>(metric_uintptr);
            return metric_ptr((scalar_at const*)a.data(), (scalar_at const*)b.data(), a.size() / sizeof(scalar_at));
        };
        break;
    case metric_signature_t::array_size_array_size_k:
        result.func_ = [metric_uintptr](punned_vector_view_t a, punned_vector_view_t b) -> distance_t {
            using metric_raw_t = punned_distance_t (*)(scalar_at const*, size_t, scalar_at const*, size_t);
            metric_raw_t metric_ptr = reinterpret_cast<metric_raw_t>(metric_uintptr);
            return metric_ptr(                                            //
                (scalar_at const*)a.data(), a.size() / sizeof(scalar_at), //
                (scalar_at const*)b.data(), b.size() / sizeof(scalar_at));
        };
        break;
    }
    return result;
}

metric_t udf(metric_kind_t kind, metric_signature_t signature, std::uintptr_t metric_uintptr, scalar_kind_t accuracy) {
    switch (accuracy) {
    case scalar_kind_t::b1x8_k: return typed_udf<b1x8_t>(kind, signature, metric_uintptr, accuracy);
    case scalar_kind_t::f8_k: return typed_udf<f8_bits_t>(kind, signature, metric_uintptr, accuracy);
    case scalar_kind_t::f16_k: return typed_udf<f16_t>(kind, signature, metric_uintptr, accuracy);
    case scalar_kind_t::f32_k: return typed_udf<f32_t>(kind, signature, metric_uintptr, accuracy);
    case scalar_kind_t::f64_k: return typed_udf<f64_t>(kind, signature, metric_uintptr, accuracy);
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
    std::uintptr_t metric_uintptr,       //
    bool tune) {

    index_config_t config;
    config.connectivity = connectivity;

    if (tune)
        config = dense_index_t::optimize(config);

    if (metric_uintptr)
        return dense_index_t::make( //
            dimensions, udf(metric_kind, metric_signature, metric_uintptr, scalar_kind), config, scalar_kind,
            expansion_add, expansion_search);
    else
        return dense_index_t::make(dimensions, metric_kind, config, scalar_kind, expansion_add, expansion_search);
}

static std::unique_ptr<sparse_index_py_t> make_sparse_index( //
    std::size_t connectivity,                                //
    std::size_t expansion_add,                               //
    std::size_t expansion_search                             //
) {
    index_config_t config;
    config.connectivity = connectivity;

    return std::unique_ptr<sparse_index_py_t>(new sparse_index_py_t(sparse_index_t(config)));
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

static void add_one_to_index(dense_index_py_t& index, label_t label, py::buffer vector, bool copy, std::size_t) {

    py::buffer_info vector_info = vector.request();
    if (vector_info.ndim != 1)
        throw std::invalid_argument("Expects a vector, not a higher-rank tensor!");

    Py_ssize_t vector_dimensions = vector_info.shape[0];
    char const* vector_data = reinterpret_cast<char const*>(vector_info.ptr);
    if (vector_dimensions != static_cast<Py_ssize_t>(index.scalar_words()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    if (!index.reserve(ceil2(index.size() + 1)))
        throw std::invalid_argument("Out of memory!");

    add_config_t config;
    config.store_vector = copy;

    switch (numpy_string_to_kind(vector_info.format)) {
    case scalar_kind_t::b1x8_k: index.add(label, (b1x8_t const*)(vector_data), config).error.raise(); break;
    case scalar_kind_t::f8_k: index.add(label, (f8_bits_t const*)(vector_data), config).error.raise(); break;
    case scalar_kind_t::f16_k: index.add(label, (f16_t const*)(vector_data), config).error.raise(); break;
    case scalar_kind_t::f32_k: index.add(label, (f32_t const*)(vector_data), config).error.raise(); break;
    case scalar_kind_t::f64_k: index.add(label, (f64_t const*)(vector_data), config).error.raise(); break;
    case scalar_kind_t::unknown_k:
        throw std::invalid_argument("Incompatible scalars in the vector: " + vector_info.format);
    }
}

template <typename scalar_at>
static void add_typed_to_index(                                              //
    dense_index_py_t& index,                                                 //
    py::buffer_info const& labels_info, py::buffer_info const& vectors_info, //
    bool copy, std::size_t threads) {

    Py_ssize_t vectors_count = vectors_info.shape[0];
    char const* vectors_data = reinterpret_cast<char const*>(vectors_info.ptr);
    char const* labels_data = reinterpret_cast<char const*>(labels_info.ptr);

    executor_default_t{threads}.execute_bulk(vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
        add_config_t config;
        config.store_vector = copy;
        config.thread = thread_idx;
        label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
        scalar_at const* vector = reinterpret_cast<scalar_at const*>(vectors_data + task_idx * vectors_info.strides[0]);
        index.add(label, vector, config).error.raise();
        if (PyErr_CheckSignals() != 0)
            throw py::error_already_set();
    });
}

static void add_many_to_index(                                      //
    dense_index_py_t& index, py::buffer labels, py::buffer vectors, //
    bool copy, std::size_t threads) {

    py::buffer_info labels_info = labels.request();
    py::buffer_info vectors_info = vectors.request();

    if (labels_info.itemsize != sizeof(label_t))
        throw std::invalid_argument("Incompatible label type!");

    if (labels_info.ndim != 1)
        throw std::invalid_argument("Labels must be placed in a single-dimensional array!");

    if (vectors_info.ndim != 2)
        throw std::invalid_argument("Expects a matrix of vectors to add!");

    Py_ssize_t labels_count = labels_info.shape[0];
    Py_ssize_t vectors_count = vectors_info.shape[0];
    Py_ssize_t vectors_dimensions = vectors_info.shape[1];
    if (vectors_dimensions != static_cast<Py_ssize_t>(index.scalar_words()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    if (labels_count != vectors_count)
        throw std::invalid_argument("Number of labels and vectors must match!");

    if (!threads)
        threads = std::thread::hardware_concurrency();
    if (!index.reserve(index_limits_t(ceil2(index.size() + vectors_count), threads)))
        throw std::invalid_argument("Out of memory!");

    switch (numpy_string_to_kind(vectors_info.format)) {
    case scalar_kind_t::b1x8_k: add_typed_to_index<b1x8_t>(index, labels_info, vectors_info, copy, threads); break;
    case scalar_kind_t::f8_k: add_typed_to_index<f8_bits_t>(index, labels_info, vectors_info, copy, threads); break;
    case scalar_kind_t::f16_k: add_typed_to_index<f16_t>(index, labels_info, vectors_info, copy, threads); break;
    case scalar_kind_t::f32_k: add_typed_to_index<f32_t>(index, labels_info, vectors_info, copy, threads); break;
    case scalar_kind_t::f64_k: add_typed_to_index<f64_t>(index, labels_info, vectors_info, copy, threads); break;
    case scalar_kind_t::unknown_k:
        throw std::invalid_argument("Incompatible scalars in the vectors matrix: " + vectors_info.format);
    }
}

static py::tuple search_one_in_index(dense_index_py_t& index, py::buffer vector, std::size_t wanted, bool exact) {

    py::buffer_info vector_info = vector.request();
    Py_ssize_t vector_dimensions = vector_info.shape[0];
    char const* vector_data = reinterpret_cast<char const*>(vector_info.ptr);
    if (vector_dimensions != static_cast<Py_ssize_t>(index.scalar_words()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    py::array_t<label_t> labels_py(static_cast<Py_ssize_t>(wanted));
    py::array_t<distance_t> distances_py(static_cast<Py_ssize_t>(wanted));
    std::size_t count{};
    auto labels_py1d = labels_py.template mutable_unchecked<1>();
    auto distances_py1d = distances_py.template mutable_unchecked<1>();

    search_config_t config;
    config.exact = exact;

    auto raise_and_dump = [&](dense_search_result_t result) {
        result.error.raise();
        count = result.dump_to(&labels_py1d(0), &distances_py1d(0));
    };

    switch (numpy_string_to_kind(vector_info.format)) {
    case scalar_kind_t::b1x8_k: raise_and_dump(index.search((b1x8_t const*)(vector_data), wanted, config)); break;
    case scalar_kind_t::f8_k: raise_and_dump(index.search((f8_bits_t const*)(vector_data), wanted, config)); break;
    case scalar_kind_t::f16_k: raise_and_dump(index.search((f16_t const*)(vector_data), wanted, config)); break;
    case scalar_kind_t::f32_k: raise_and_dump(index.search((f32_t const*)(vector_data), wanted, config)); break;
    case scalar_kind_t::f64_k: raise_and_dump(index.search((f64_t const*)(vector_data), wanted, config)); break;
    case scalar_kind_t::unknown_k:
        throw std::invalid_argument("Incompatible scalars in the query vector: " + vector_info.format);
    }

    labels_py.resize(py_shape_t{static_cast<Py_ssize_t>(count)});
    distances_py.resize(py_shape_t{static_cast<Py_ssize_t>(count)});

    py::tuple results(3);
    results[0] = labels_py;
    results[1] = distances_py;
    results[2] = static_cast<Py_ssize_t>(count);
    return results;
}

template <typename scalar_at>
static void search_typed(                                   //
    dense_index_py_t& index, py::buffer_info& vectors_info, //
    std::size_t wanted, bool exact, std::size_t threads,    //
    py::array_t<label_t>& labels_py, py::array_t<distance_t>& distances_py, py::array_t<Py_ssize_t>& counts_py) {

    auto labels_py2d = labels_py.template mutable_unchecked<2>();
    auto distances_py2d = distances_py.template mutable_unchecked<2>();
    auto counts_py1d = counts_py.template mutable_unchecked<1>();

    Py_ssize_t vectors_count = vectors_info.shape[0];
    char const* vectors_data = reinterpret_cast<char const*>(vectors_info.ptr);

    if (!threads)
        threads = std::thread::hardware_concurrency();
    if (!index.reserve(index_limits_t(index.size(), threads)))
        throw std::invalid_argument("Out of memory!");

    executor_default_t{threads}.execute_bulk(vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
        search_config_t config;
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

/**
 *  @param vectors Matrix of vectors to search for.
 *  @param wanted Number of matches per request.
 *
 *  @return Tuple with:
 *      1. matrix of neighbors,
 *      2. matrix of distances,
 *      3. array with match counts.
 */
static py::tuple search_many_in_index( //
    dense_index_py_t& index, py::buffer vectors, std::size_t wanted, bool exact, std::size_t threads) {

    if (wanted == 0)
        return py::tuple(3);

    if (index.limits().threads_search < threads)
        throw std::invalid_argument("Can't use that many threads!");

    py::buffer_info vectors_info = vectors.request();
    if (vectors_info.ndim == 1)
        return search_one_in_index(index, vectors, wanted, exact);
    if (vectors_info.ndim != 2)
        throw std::invalid_argument("Expects a matrix of vectors to add!");

    Py_ssize_t vectors_count = vectors_info.shape[0];
    Py_ssize_t vectors_dimensions = vectors_info.shape[1];
    if (vectors_dimensions != static_cast<Py_ssize_t>(index.scalar_words()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    py::array_t<label_t> ls({vectors_count, static_cast<Py_ssize_t>(wanted)});
    py::array_t<distance_t> ds({vectors_count, static_cast<Py_ssize_t>(wanted)});
    py::array_t<Py_ssize_t> cs(vectors_count);

    switch (numpy_string_to_kind(vectors_info.format)) {
    case scalar_kind_t::b1x8_k: search_typed<b1x8_t>(index, vectors_info, wanted, exact, threads, ls, ds, cs); break;
    case scalar_kind_t::f8_k: search_typed<f8_bits_t>(index, vectors_info, wanted, exact, threads, ls, ds, cs); break;
    case scalar_kind_t::f16_k: search_typed<f16_t>(index, vectors_info, wanted, exact, threads, ls, ds, cs); break;
    case scalar_kind_t::f32_k: search_typed<f32_t>(index, vectors_info, wanted, exact, threads, ls, ds, cs); break;
    case scalar_kind_t::f64_k: search_typed<f64_t>(index, vectors_info, wanted, exact, threads, ls, ds, cs); break;
    case scalar_kind_t::unknown_k:
        throw std::invalid_argument("Incompatible scalars in the query matrix: " + vectors_info.format);
    }

    py::tuple results(3);
    results[0] = ls;
    results[1] = ds;
    results[2] = cs;
    return results;
}

static std::unordered_map<label_t, label_t> join_index(   //
    dense_index_py_t const& a, dense_index_py_t const& b, //
    std::size_t max_proposals, bool exact) {

    std::unordered_map<label_t, label_t> a_to_b;
    a_to_b.reserve((std::min)(a.size(), b.size()));

    using join_result_t = typename dense_index_py_t::join_result_t;
    join_config_t config;

    config.max_proposals = max_proposals;
    config.exact = exact;
    config.expansion = (std::max)(a.expansion_search(), b.expansion_search());
    std::size_t threads = (std::min)(a.limits().threads(), b.limits().threads());
    executor_default_t executor{threads};
    join_result_t result = dense_index_py_t::join( //
        a, b, config,                              //
        a_to_b,                                    //
        dummy_label_to_label_mapping_t{},          //
        executor);
    result.error.raise();
    return a_to_b;
}

static dense_index_py_t copy_index(dense_index_py_t const& index) {

    using copy_result_t = typename dense_index_py_t::copy_result_t;
    copy_config_t config;

    copy_result_t result = index.copy();
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
py::object get_typed_member(index_at const& index, label_t label) {
    std::size_t result_slots = std::is_same<internal_at, b1x8_t>() ? index.scalar_words() : index.dimensions();
    py::array_t<external_at> result_py(static_cast<Py_ssize_t>(result_slots));
    auto result_py1d = result_py.template mutable_unchecked<1>();
    if (!index.get(label, (internal_at*)&result_py1d(0)))
        return py::none();
    return py::object(result_py);
}

template <typename index_at> py::object get_member(index_at const& index, label_t label, scalar_kind_t scalar_kind) {
    if (scalar_kind == scalar_kind_t::f32_k)
        return get_typed_member<f32_t>(index, label);
    else if (scalar_kind == scalar_kind_t::f64_k)
        return get_typed_member<f64_t>(index, label);
    else if (scalar_kind == scalar_kind_t::f16_k)
        return get_typed_member<f16_t, std::uint16_t>(index, label);
    else if (scalar_kind == scalar_kind_t::f8_k)
        return get_typed_member<f8_bits_t, std::int8_t>(index, label);
    else if (scalar_kind == scalar_kind_t::b1x8_k)
        return get_typed_member<b1x8_t, std::uint8_t>(index, label);
    else
        throw std::invalid_argument("Incompatible scalars in the query matrix!");
}

template <typename index_at>
py::array_t<label_t> get_labels(index_at const& index, std::size_t offset, std::size_t limit) {
    limit = std::min(index.size(), limit);
    py::array_t<label_t> result_py(static_cast<Py_ssize_t>(limit));
    auto result_py1d = result_py.template mutable_unchecked<1>();
    index.export_labels(&result_py1d(0), offset, limit);
    return result_py;
}

template <typename index_at> py::array_t<label_t> get_all_labels(index_at const& index) {
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

void validate_set(py::array_t<set_member_t> const& set) {
    if (set.ndim() != 1)
        throw std::invalid_argument("Set can't be multi-dimensional!");
    if (set.strides(0) != sizeof(set_member_t))
        throw std::invalid_argument("Set can't be strided!");
    auto proxy = set.unchecked<1>();
    set_member_t const* begin = &proxy(0);
    set_member_t const* end = begin + proxy.size();
    if (!std::is_sorted(begin, end))
        throw std::invalid_argument("Set must be sorted!");
    if (has_duplicates(begin, end))
        throw std::invalid_argument("Set must be deduplicated!");
}
#if 0

inline std::size_t hash_ror64(std::size_t v, int r) noexcept { return (v >> r) | (v << (64 - r)); }

inline std::size_t hash(std::size_t v) noexcept {
    v ^= hash_ror64(v, 25) ^ hash_ror64(v, 50);
    v *= 0xA24BAED4963EE407UL;
    v ^= hash_ror64(v, 24) ^ hash_ror64(v, 49);
    v *= 0x9FB21C651E98DF25UL;
    return v ^ v >> 28;
}

template <typename scalar_at>
void hash_typed_row(                                                             //
    byte_t const* scalars, std::size_t scalar_stride, std::size_t scalars_count, //
    b1x8_t* hashes, std::size_t bits) noexcept {

    std::size_t words = divide_round_up<8>(bits);
    for (std::size_t i = 0; i != scalars_count; ++i) {
        scalar_at scalar;
        std::memcpy(&scalar, scalars + i * scalar_stride, sizeof(scalar_at));
        std::size_t scalar_hash = hash(scalar);
        hashes[scalar_hash & (words - 1)] |= std::uint8_t(1) << (scalar_hash % 8);
    }
}

template <typename scalar_at>
py::array_t<bits_numpy_word_t> hash_typed_buffer(py::buffer_info const& info, std::size_t bits) {
    byte_t const* data = reinterpret_cast<byte_t const*>(info.ptr);
    Py_ssize_t elements_per_row = info.shape[0];
    Py_ssize_t elements_stride = info.strides[0];
    Py_ssize_t words_per_row = divide_round_up<std::size_t>(bits, 8);

    if (info.ndim == 2) {
        Py_ssize_t rows_count = info.shape[1];
        Py_ssize_t rows_stride = info.strides[1];
        auto hashes_shape = py_shape_t{rows_count, words_per_row};
        auto hashes_py = py::array_t<bits_numpy_word_t>(hashes_shape);
        auto hashes_proxy = hashes_py.template mutable_unchecked<2>();

        for (Py_ssize_t i = 0; i != rows_count; ++i)
            hash_typed_row(                                                                                 //
                data + i * rows_stride, hashes_proxy.stride(1), static_cast<std::size_t>(elements_per_row), //
                &hashes_proxy(i, 0), bits);

        return hashes_py;
    } else {
        auto hashes_shape = py_shape_t{words_per_row};
        auto hashes_py = py::array_t<bits_numpy_word_t>(hashes_shape);
        auto hashes_proxy = hashes_py.template mutable_unchecked<1>();

        hash_typed_row(                                                               //
            data, hashes_info.strides[0], static_cast<std::size_t>(elements_per_row), //
            &hashes_proxy(0), bits);

        return hashes_py;
    }
}

py::array_t<bits_numpy_word_t> hash_buffer(py::buffer vector, std::size_t bits) {
    if (std::bitset<64>(bits).count() == 1)
        throw std::invalid_argument("Number of bits must be a power of two!");
    py::buffer_info info = vector.request();
    if (info.ndim > 2)
        throw std::invalid_argument("Only one or two-dimensional arrays are supported!");

    // https://docs.python.org/3/library/struct.html#format-characters
    if (info.format == "h" || info.format == "H")
        return hash_typed_buffer<std::uint16_t>(info, bits);
    else if (info.format == "i" || info.format == "I" || info.format == "l" || info.format == "L")
        return hash_typed_buffer<std::uint32_t>(info, bits);
    else if (info.format == "q" || info.format == "Q" || info.format == "n" || info.format == "N")
        return hash_typed_buffer<std::uint64_t>(info, bits);
    else
        throw std::invalid_argument("Array elements must be 16, 32, or 64 bit hashable integers!");
}
#endif

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
        .value("F64", scalar_kind_t::f64_k)
        .value("F32", scalar_kind_t::f32_k)
        .value("F16", scalar_kind_t::f16_k)
        .value("F8", scalar_kind_t::f8_k)
        .value("B1", scalar_kind_t::b1x8_k);

    auto h = py::class_<file_head_result_t>(m, "IndexMetadata");
    h.def(py::init([](std::string const& path) {
        file_head_result_t h = index_metadata(path.c_str());
        h.error.raise();
        return h;
    }));
    h.def_property_readonly("version", [](file_head_result_t const& h) {
        return                                      //
            std::to_string(h.version_major) + "." + //
            std::to_string(h.version_minor) + "." + //
            std::to_string(h.version_patch);
    });
    h.def_readonly("metric", &file_head_result_t::metric);
    h.def_readonly("connectivity", &file_head_result_t::connectivity);
    h.def_readonly("max_level", &file_head_result_t::max_level);
    h.def_readonly("vector_alignment", &file_head_result_t::vector_alignment);
    h.def_readonly("bytes_per_label", &file_head_result_t::bytes_per_label);
    h.def_readonly("bytes_per_id", &file_head_result_t::bytes_per_id);
    h.def_readonly("scalar_kind", &file_head_result_t::scalar_kind);
    h.def_readonly("size", &file_head_result_t::size);
    h.def_readonly("entry_idx", &file_head_result_t::entry_idx);
    h.def_readonly("bytes_for_graphs", &file_head_result_t::bytes_for_graphs);
    h.def_readonly("bytes_for_vectors", &file_head_result_t::bytes_for_vectors);
    h.def_readonly("bytes_checksum", &file_head_result_t::bytes_checksum);

    auto i = py::class_<dense_index_py_t>(m, "Index");

    i.def(py::init(&make_index),                                           //
          py::kw_only(),                                                   //
          py::arg("ndim") = 0,                                             //
          py::arg("dtype") = scalar_kind_t::f32_k,                         //
          py::arg("metric") = metric_kind_t::ip_k,                         //
          py::arg("connectivity") = default_connectivity(),                //
          py::arg("expansion_add") = default_expansion_add(),              //
          py::arg("expansion_search") = default_expansion_search(),        //
          py::arg("metric_signature") = metric_signature_t::array_array_k, //
          py::arg("metric_pointer") = 0,                                   //
          py::arg("tune") = false                                          //
    );

    i.def(                         //
        "add", &add_many_to_index, //
        py::arg("labels"),         //
        py::arg("vectors"),        //
        py::kw_only(),             //
        py::arg("copy") = true,    //
        py::arg("threads") = 0     //
    );

    i.def(                        //
        "add", &add_one_to_index, //
        py::arg("label"),         //
        py::arg("vector"),        //
        py::kw_only(),            //
        py::arg("copy") = true,   //
        py::arg("threads") = 0    //
    );

    i.def(                               //
        "search", &search_many_in_index, //
        py::arg("query"),                //
        py::arg("count") = 10,           //
        py::arg("exact") = false,        //
        py::arg("threads") = 0           //
    );

    i.def(
        "rename",
        [](dense_index_py_t& index, label_t from, label_t to) -> bool {
            dense_labeling_result_t result = index.rename(from, to);
            result.error.raise();
            return result.completed;
        },
        py::arg("from"), py::arg("to"));

    i.def(
        "remove",
        [](dense_index_py_t& index, label_t label, bool compact, std::size_t threads) -> bool {
            dense_labeling_result_t result = index.remove(label);
            result.error.raise();

            if (!threads)
                threads = std::thread::hardware_concurrency();
            if (!index.reserve(index_limits_t(index.size(), threads)))
                throw std::invalid_argument("Out of memory!");

            index.compact(executor_default_t{threads});
            return result.completed;
        },
        py::arg("label"), py::arg("compact"), py::arg("threads"));

    i.def(
        "remove",
        [](dense_index_py_t& index, std::vector<label_t> const& labels, bool compact,
           std::size_t threads) -> std::size_t {
            dense_labeling_result_t result = index.remove(labels.begin(), labels.end());
            result.error.raise();

            if (!threads)
                threads = std::thread::hardware_concurrency();
            if (!index.reserve(index_limits_t(index.size(), threads)))
                throw std::invalid_argument("Out of memory!");

            index.compact(executor_default_t{threads});
            return result.completed;
        },
        py::arg("label"), py::arg("compact"), py::arg("threads"));

    i.def("__len__", &dense_index_py_t::size);
    i.def_property_readonly("size", &dense_index_py_t::size);
    i.def_property_readonly("ndim", &dense_index_py_t::dimensions);
    i.def_property_readonly("connectivity", &dense_index_py_t::connectivity);
    i.def_property_readonly("capacity", &dense_index_py_t::capacity);
    i.def_property_readonly( //
        "dtype", [](dense_index_py_t const& index) -> scalar_kind_t { return index.metric().scalar_kind_; });
    i.def_property_readonly( //
        "memory_usage", [](dense_index_py_t const& index) -> std::size_t { return index.memory_usage(); },
        py::call_guard<py::gil_scoped_release>());

    i.def_property("expansion_add", &dense_index_py_t::expansion_add, &dense_index_py_t::change_expansion_add);
    i.def_property("expansion_search", &dense_index_py_t::expansion_search, &dense_index_py_t::change_expansion_search);

    i.def_property_readonly("labels", &get_all_labels<dense_index_py_t>);
    i.def("get_labels", &get_labels<dense_index_py_t>, py::arg("offset") = 0,
          py::arg("limit") = std::numeric_limits<std::size_t>::max());
    i.def("__contains__", &dense_index_py_t::contains);
    i.def("__getitem__", &get_member<dense_index_py_t>, py::arg("label"), py::arg("dtype") = scalar_kind_t::f32_k);

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

    auto si = py::class_<sparse_index_py_t>(m, "SparseIndex");

    si.def(                                                      //
        py::init(&make_sparse_index),                            //
        py::kw_only(),                                           //
        py::arg("connectivity") = default_connectivity(),        //
        py::arg("expansion_add") = default_expansion_add(),      //
        py::arg("expansion_search") = default_expansion_search() //
    );

    si.def( //
        "add",
        [](sparse_index_py_t& index, label_t label, py::array_t<set_member_t> set, bool copy) {
            validate_set(set);
            if (index.size() + 1 >= index.capacity())
                index.reserve(ceil2(index.size() + 1));
            auto proxy = set.unchecked<1>();
            auto view = set_view_t{&proxy(0), static_cast<std::size_t>(proxy.shape(0))};

            add_config_t config;
            config.store_vector = copy;
            index.add(label, view, config).error.raise();
        },                     //
        py::arg("label"),      //
        py::arg("set"),        //
        py::kw_only(),         //
        py::arg("copy") = true //
    );

    si.def( //
        "search",
        [](sparse_index_py_t& index, py::array_t<set_member_t> set, std::size_t count) -> py::array_t<label_t> {
            validate_set(set);
            auto proxy = set.unchecked<1>();
            auto view = set_view_t{&proxy(0), static_cast<std::size_t>(proxy.shape(0))};
            auto labels_py = py::array_t<label_t>(py_shape_t{static_cast<Py_ssize_t>(count)});
            auto labels_proxy = labels_py.template mutable_unchecked<1>();
            auto result = index.search(view, count);
            result.error.raise();
            auto found = result.dump_to(&labels_proxy(0));
            labels_py.resize(py_shape_t{static_cast<Py_ssize_t>(found)});
            return labels_py;
        },
        py::arg("set"),       //
        py::arg("count") = 10 //
    );

    si.def("__len__", &sparse_index_py_t::size);
    si.def_property_readonly("size", &sparse_index_py_t::size);
    si.def_property_readonly("connectivity", &sparse_index_py_t::connectivity);
    si.def_property_readonly("capacity", &sparse_index_py_t::capacity);

    si.def("save", &save_index<sparse_index_py_t>, py::arg("path"));
    si.def("load", &load_index<sparse_index_py_t>, py::arg("path"));
    si.def("view", &view_index<sparse_index_py_t>, py::arg("path"));
    si.def("clear", &sparse_index_py_t::clear);
}

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
#if !defined(__cpp_exceptions)
#define __cpp_exceptions 1
#endif

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

    void add_paths(std::vector<std::string> const& paths, bool view = true, std::size_t threads = 0) {
        if (!threads)
            threads = std::thread::hardware_concurrency();

        shards_.reserve(shards_.size() + paths.size());
        std::mutex shards_mutex;
        executor_default_t{threads}.execute_bulk(paths.size(), [&](std::size_t, std::size_t task_idx) {
            index_dense_t index = index_dense_t::make(paths[task_idx].c_str(), view);
            if (!index)
                return;
            auto shared_index = std::make_shared<dense_index_py_t>(std::move(index));
            std::unique_lock<std::mutex> lock(shards_mutex);
            shards_.push_back(shared_index);
            if (PyErr_CheckSignals() != 0)
                throw py::error_already_set();
        });
    }

    std::size_t size() const noexcept {
        std::size_t result = 0;
        for (auto const& shard : shards_)
            result += shard->size();
        return result;
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

metric_t udf(                                                                        //
    metric_kind_t kind, metric_signature_t signature, std::uintptr_t metric_uintptr, //
    scalar_kind_t scalar_kind, std::size_t dimensions) {

    switch (scalar_kind) {
    case scalar_kind_t::b1x8_k: return typed_udf<b1x8_t>(kind, signature, metric_uintptr, scalar_kind, dimensions);
    case scalar_kind_t::i8_k: return typed_udf<i8_bits_t>(kind, signature, metric_uintptr, scalar_kind, dimensions);
    case scalar_kind_t::f16_k: return typed_udf<f16_t>(kind, signature, metric_uintptr, scalar_kind, dimensions);
    case scalar_kind_t::f32_k: return typed_udf<f32_t>(kind, signature, metric_uintptr, scalar_kind, dimensions);
    case scalar_kind_t::f64_k: return typed_udf<f64_t>(kind, signature, metric_uintptr, scalar_kind, dimensions);
    default: return {};
    }
}

static dense_index_py_t make_index(      //
    std::size_t dimensions,              //
    scalar_kind_t scalar_kind,           //
    std::size_t connectivity,            //
    std::size_t expansion_add,           //
    std::size_t expansion_search,        //
    metric_kind_t metric_kind,           //
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
        return scalar_kind_t::i8_k;
    else if (name == "e" || name == "<e" || name == "f2" || name == "<f2")
        return scalar_kind_t::f16_k;
    else if (name == "f" || name == "<f" || name == "f4" || name == "<f4")
        return scalar_kind_t::f32_k;
    else if (name == "d" || name == "<d" || name == "i8" || name == "<i8")
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
        index_dense_update_config_t config;
        config.force_vector_copy = force_copy;
        config.thread = thread_idx;
        key_t key = *reinterpret_cast<key_t const*>(keys_data + task_idx * keys_info.strides[0]);
        scalar_at const* vector = reinterpret_cast<scalar_at const*>(vectors_data + task_idx * vectors_info.strides[0]);
        dense_add_result_t result = index.add(key, vector, config);
        result.error.raise();
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

    Py_ssize_t keys_count = keys_info.shape[0];
    Py_ssize_t vectors_count = vectors_info.shape[0];
    Py_ssize_t vectors_dimensions = vectors_info.shape[1];
    if (vectors_dimensions != static_cast<Py_ssize_t>(index.scalar_words()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    if (keys_count != vectors_count)
        throw std::invalid_argument("Number of keys and vectors must match!");

    if (!threads)
        threads = std::thread::hardware_concurrency();
    if (!index.reserve(index_limits_t(ceil2(index.size() + vectors_count), threads)))
        throw std::invalid_argument("Out of memory!");

    // clang-format off
    switch (numpy_string_to_kind(vectors_info.format)) {
    case scalar_kind_t::b1x8_k: add_typed_to_index<b1x8_t>(index, keys_info, vectors_info, force_copy, threads); break;
    case scalar_kind_t::i8_k: add_typed_to_index<i8_bits_t>(index, keys_info, vectors_info, force_copy, threads); break;
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
    py::array_t<key_t>& keys_py, py::array_t<distance_t>& distances_py, py::array_t<Py_ssize_t>& counts_py,
    std::atomic<std::size_t>& stats_visited_members, std::atomic<std::size_t>& stats_computed_distances) {

    auto keys_py2d = keys_py.template mutable_unchecked<2>();
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
            static_cast<Py_ssize_t>(result.dump_to(&keys_py2d(task_idx, 0), &distances_py2d(task_idx, 0)));

        stats_visited_members += result.visited_members;
        stats_computed_distances += result.computed_distances;
        if (PyErr_CheckSignals() != 0)
            throw py::error_already_set();
    });
}

template <typename scalar_at>
static void search_typed(                                       //
    dense_indexes_py_t& indexes, py::buffer_info& vectors_info, //
    std::size_t wanted, bool exact, std::size_t threads,        //
    py::array_t<key_t>& keys_py, py::array_t<distance_t>& distances_py, py::array_t<Py_ssize_t>& counts_py,
    std::atomic<std::size_t>& stats_visited_members, std::atomic<std::size_t>& stats_computed_distances) {

    auto keys_py2d = keys_py.template mutable_unchecked<2>();
    auto distances_py2d = distances_py.template mutable_unchecked<2>();
    auto counts_py1d = counts_py.template mutable_unchecked<1>();

    Py_ssize_t vectors_count = vectors_info.shape[0];
    byte_t const* vectors_data = reinterpret_cast<byte_t const*>(vectors_info.ptr);
    for (std::size_t vector_idx = 0; vector_idx != static_cast<std::size_t>(vectors_count); ++vector_idx)
        counts_py1d(vector_idx) = 0;

    if (!threads)
        threads = std::thread::hardware_concurrency();

    visits_bitset_t query_mutexes(static_cast<std::size_t>(vectors_count));
    if (!query_mutexes)
        throw std::bad_alloc();

    executor_default_t{threads}.execute_bulk(indexes.shards_.size(), [&](std::size_t, std::size_t task_idx) {
        dense_index_py_t& index = *indexes.shards_[task_idx].get();

        index_limits_t limits;
        limits.members = index.size();
        limits.threads_add = 0;
        limits.threads_search = 1;
        if (!index.reserve(limits))
            throw std::bad_alloc();

        index_search_config_t config;
        config.thread = 0;
        config.exact = exact;

        for (std::size_t vector_idx = 0; vector_idx != static_cast<std::size_t>(vectors_count); ++vector_idx) {
            scalar_at const* vector = (scalar_at const*)(vectors_data + vector_idx * vectors_info.strides[0]);
            dense_search_result_t result = index.search(vector, wanted, config);
            result.error.raise();
            {
                auto lock = query_mutexes.lock(vector_idx);
                counts_py1d(vector_idx) = static_cast<Py_ssize_t>(result.merge_into( //
                    &keys_py2d(vector_idx, 0),                                       //
                    &distances_py2d(vector_idx, 0),                                  //
                    static_cast<std::size_t>(counts_py1d(vector_idx)),               //
                    wanted));
            }

            stats_visited_members += result.visited_members;
            stats_computed_distances += result.computed_distances;
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
        return py::tuple(5);

    if (index.limits().threads_search < threads)
        throw std::invalid_argument("Can't use that many threads!");

    py::buffer_info vectors_info = vectors.request();
    if (vectors_info.ndim != 2)
        throw std::invalid_argument("Expects a matrix of vectors to add!");

    Py_ssize_t vectors_count = vectors_info.shape[0];
    Py_ssize_t vectors_dimensions = vectors_info.shape[1];
    if (vectors_dimensions != static_cast<Py_ssize_t>(index.scalar_words()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    py::array_t<key_t> keys_py({vectors_count, static_cast<Py_ssize_t>(wanted)});
    py::array_t<distance_t> distances_py({vectors_count, static_cast<Py_ssize_t>(wanted)});
    py::array_t<Py_ssize_t> counts_py(vectors_count);
    std::atomic<std::size_t> stats_visited_members(0);
    std::atomic<std::size_t> stats_computed_distances(0);

    // clang-format off
    switch (numpy_string_to_kind(vectors_info.format)) {
    case scalar_kind_t::b1x8_k: search_typed<b1x8_t>(index, vectors_info, wanted, exact, threads, keys_py, distances_py, counts_py, stats_visited_members, stats_computed_distances); break;
    case scalar_kind_t::i8_k: search_typed<i8_bits_t>(index, vectors_info, wanted, exact, threads, keys_py, distances_py, counts_py, stats_visited_members, stats_computed_distances); break;
    case scalar_kind_t::f16_k: search_typed<f16_t>(index, vectors_info, wanted, exact, threads, keys_py, distances_py, counts_py, stats_visited_members, stats_computed_distances); break;
    case scalar_kind_t::f32_k: search_typed<f32_t>(index, vectors_info, wanted, exact, threads, keys_py, distances_py, counts_py, stats_visited_members, stats_computed_distances); break;
    case scalar_kind_t::f64_k: search_typed<f64_t>(index, vectors_info, wanted, exact, threads, keys_py, distances_py, counts_py, stats_visited_members, stats_computed_distances); break;
    default: throw std::invalid_argument("Incompatible scalars in the query matrix: " + vectors_info.format);
    }
    // clang-format on

    py::tuple results(5);
    results[0] = keys_py;
    results[1] = distances_py;
    results[2] = counts_py;
    results[3] = stats_visited_members.load();
    results[4] = stats_computed_distances.load();
    return results;
}

template <typename scalar_at>
static void search_typed_brute_force(                                //
    py::buffer_info& dataset_info, py::buffer_info& queries_info,    //
    std::size_t wanted, std::size_t threads, metric_t const& metric, //
    py::array_t<key_t>& keys_py, py::array_t<distance_t>& distances_py, py::array_t<Py_ssize_t>& counts_py) {

    auto keys_py2d = keys_py.template mutable_unchecked<2>();
    auto distances_py2d = distances_py.template mutable_unchecked<2>();
    auto counts_py1d = counts_py.template mutable_unchecked<1>();

    std::size_t dataset_count = static_cast<std::size_t>(dataset_info.shape[0]);
    std::size_t queries_count = static_cast<std::size_t>(queries_info.shape[0]);
    std::size_t dimensions = static_cast<std::size_t>(dataset_info.shape[1]);

    byte_t const* dataset_data = reinterpret_cast<byte_t const*>(dataset_info.ptr);
    byte_t const* queries_data = reinterpret_cast<byte_t const*>(queries_info.ptr);
    for (std::size_t query_idx = 0; query_idx != queries_count; ++query_idx)
        counts_py1d(query_idx) = 0;

    if (!threads)
        threads = std::thread::hardware_concurrency();

    std::size_t tasks_count = static_cast<std::size_t>(dataset_count * queries_count);
    visits_bitset_t query_mutexes(static_cast<std::size_t>(queries_count));
    if (!query_mutexes)
        throw std::bad_alloc();

    executor_default_t{threads}.execute_bulk(tasks_count, [&](std::size_t, std::size_t task_idx) {
        //
        std::size_t dataset_idx = task_idx / queries_count;
        std::size_t query_idx = task_idx % queries_count;

        byte_t const* dataset = dataset_data + dataset_idx * dataset_info.strides[0];
        byte_t const* query = queries_data + query_idx * queries_info.strides[0];
        distance_t distance = metric(dataset, query);

        {
            auto lock = query_mutexes.lock(query_idx);
            key_t* keys = &keys_py2d(query_idx, 0);
            distance_t* distances = &distances_py2d(query_idx, 0);
            std::size_t& matches = reinterpret_cast<std::size_t&>(counts_py1d(query_idx));
            if (matches == wanted)
                if (distances[wanted - 1] <= distance)
                    return;

            std::size_t offset = std::lower_bound(distances, distances + matches, distance) - distances;

            std::size_t count_worse = matches - offset - (wanted == matches);
            std::memmove(keys + offset + 1, keys + offset, count_worse * sizeof(key_t));
            std::memmove(distances + offset + 1, distances + offset, count_worse * sizeof(distance_t));
            keys[offset] = static_cast<key_t>(dataset_idx);
            distances[offset] = distance;
            matches += matches != wanted;
        }

        if (PyErr_CheckSignals() != 0)
            throw py::error_already_set();
    });
}

static py::tuple search_many_brute_force(    //
    py::buffer dataset, py::buffer queries,  //
    std::size_t wanted, std::size_t threads, //
    metric_kind_t metric_kind,               //
    metric_signature_t metric_signature,     //
    std::uintptr_t metric_uintptr) {

    if (wanted == 0)
        return py::tuple(5);

    py::buffer_info dataset_info = dataset.request();
    py::buffer_info queries_info = queries.request();
    if (dataset_info.ndim != 2 || queries_info.ndim != 2)
        throw std::invalid_argument("Expects a matrix of dataset to add!");

    Py_ssize_t dataset_count = dataset_info.shape[0];
    Py_ssize_t dataset_dimensions = dataset_info.shape[1];
    Py_ssize_t queries_count = queries_info.shape[0];
    Py_ssize_t queries_dimensions = queries_info.shape[1];
    if (dataset_dimensions != queries_dimensions)
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    scalar_kind_t dataset_kind = numpy_string_to_kind(dataset_info.format);
    scalar_kind_t queries_kind = numpy_string_to_kind(queries_info.format);
    if (dataset_kind != queries_kind)
        throw std::invalid_argument("The types of vectors don't match!");

    py::array_t<key_t> keys_py({dataset_count, static_cast<Py_ssize_t>(wanted)});
    py::array_t<distance_t> distances_py({dataset_count, static_cast<Py_ssize_t>(wanted)});
    py::array_t<Py_ssize_t> counts_py(dataset_count);

    std::size_t dimensions = static_cast<std::size_t>(queries_dimensions);
    metric_t metric =  //
        metric_uintptr //
            ? udf(metric_kind, metric_signature, metric_uintptr, queries_kind, dimensions)
            : metric_t(dimensions, metric_kind, queries_kind);

    // clang-format off
    switch (dataset_kind) {
    case scalar_kind_t::b1x8_k: search_typed_brute_force<b1x8_t>(dataset_info, queries_info, wanted, threads, metric, keys_py, distances_py, counts_py); break;
    case scalar_kind_t::i8_k: search_typed_brute_force<i8_bits_t>(dataset_info, queries_info, wanted, threads, metric, keys_py, distances_py, counts_py); break;
    case scalar_kind_t::f16_k: search_typed_brute_force<f16_t>(dataset_info, queries_info, wanted, threads, metric, keys_py, distances_py, counts_py); break;
    case scalar_kind_t::f32_k: search_typed_brute_force<f32_t>(dataset_info, queries_info, wanted, threads, metric, keys_py, distances_py, counts_py); break;
    case scalar_kind_t::f64_k: search_typed_brute_force<f64_t>(dataset_info, queries_info, wanted, threads, metric, keys_py, distances_py, counts_py); break;
    default: throw std::invalid_argument("Incompatible vector types: " + dataset_info.format);
    }
    // clang-format on

    py::tuple results(5);
    results[0] = keys_py;
    results[1] = distances_py;
    results[2] = counts_py;
    results[3] = 0;
    results[4] = static_cast<std::size_t>(dataset_count * queries_count);
    return results;
}

static std::unordered_map<key_t, key_t> join_index(       //
    dense_index_py_t const& a, dense_index_py_t const& b, //
    std::size_t max_proposals, bool exact) {

    std::unordered_map<key_t, key_t> a_to_b;
    dummy_label_to_label_mapping_t b_to_a;
    a_to_b.reserve((std::min)(a.size(), b.size()));

    index_join_config_t config;
    config.max_proposals = max_proposals;
    config.exact = exact;
    config.expansion = (std::max)(a.expansion_search(), b.expansion_search());
    std::size_t threads = (std::min)(a.limits().threads(), b.limits().threads());
    executor_default_t executor{threads};
    join_result_t result = a.join(b, config, a_to_b, b_to_a, executor);
    result.error.raise();

    return a_to_b;
}

static dense_index_py_t copy_index(dense_index_py_t const& index) {

    using copy_result_t = typename dense_index_py_t::copy_result_t;
    index_copy_config_t config;
    copy_result_t result = index.copy(config);
    result.error.raise();
    return std::move(result.index);
}

static void compact_index(dense_index_py_t& index, std::size_t threads) {

    if (!threads)
        threads = std::thread::hardware_concurrency();
    if (!index.reserve(index_limits_t(index.size(), threads)))
        throw std::invalid_argument("Out of memory!");

    index.compact(executor_default_t{threads});
}

// clang-format off
template <typename index_at> void save_index(index_at const& index, std::string const& path) { index.save(path.c_str()).error.raise(); }
template <typename index_at> void load_index(index_at& index, std::string const& path) { index.load(path.c_str()).error.raise(); }
template <typename index_at> void view_index(index_at& index, std::string const& path) { index.view(path.c_str()).error.raise(); }
template <typename index_at> void reset_index(index_at& index) { index.reset(); }
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
    else if (scalar_kind == scalar_kind_t::i8_k)
        return get_typed_member<i8_bits_t, std::int8_t>(index, key);
    else if (scalar_kind == scalar_kind_t::b1x8_k)
        return get_typed_member<b1x8_t, std::uint8_t>(index, key);
    else
        throw std::invalid_argument("Incompatible scalars in the query matrix!");
}

template <typename index_at> py::array_t<key_t> get_keys(index_at const& index, std::size_t offset, std::size_t limit) {
    limit = std::min(index.size(), limit);
    py::array_t<key_t> result_py(static_cast<Py_ssize_t>(limit));
    auto result_py1d = result_py.template mutable_unchecked<1>();
    index.export_keys(&result_py1d(0), offset, limit);
    return result_py;
}

template <typename index_at> py::array_t<key_t> get_all_keys(index_at const& index) {
    return get_keys(index, 0, index.size());
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

        .value("IP", metric_kind_t::cos_k)
        .value("Cos", metric_kind_t::cos_k)
        .value("L2sq", metric_kind_t::l2sq_k)

        .value("Haversine", metric_kind_t::haversine_k)
        .value("Pearson", metric_kind_t::pearson_k)
        .value("Jaccard", metric_kind_t::jaccard_k)
        .value("Hamming", metric_kind_t::hamming_k)
        .value("Tanimoto", metric_kind_t::tanimoto_k)
        .value("Sorensen", metric_kind_t::sorensen_k)

        .value("Cosine", metric_kind_t::cos_k)
        .value("InnerProduct", metric_kind_t::cos_k);

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

    m.def("exact_search", &search_many_brute_force,                        //
          py::arg("dataset"),                                              //
          py::arg("queries"),                                              //
          py::arg("count") = 10,                                           //
          py::kw_only(),                                                   //
          py::arg("threads") = 0,                                          //
          py::arg("metric_kind") = metric_kind_t::cos_k,                   //
          py::arg("metric_signature") = metric_signature_t::array_array_k, //
          py::arg("metric_pointer") = 0                                    //
    );

    auto i = py::class_<dense_index_py_t, std::shared_ptr<dense_index_py_t>>(m, "Index");

    i.def(py::init(&make_index),                                           //
          py::kw_only(),                                                   //
          py::arg("ndim") = 0,                                             //
          py::arg("dtype") = scalar_kind_t::f32_k,                         //
          py::arg("connectivity") = default_connectivity(),                //
          py::arg("expansion_add") = default_expansion_add(),              //
          py::arg("expansion_search") = default_expansion_search(),        //
          py::arg("metric_kind") = metric_kind_t::cos_k,                   //
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

            index.isolate(executor_default_t{threads});
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

            index.isolate(executor_default_t{threads});
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

    i.def(
        "change_metric",
        [](dense_index_py_t& index, metric_kind_t metric_kind, metric_signature_t metric_signature,
           std::uintptr_t metric_uintptr) {
            scalar_kind_t scalar_kind = index.scalar_kind();
            std::size_t dimensions = index.dimensions();
            metric_t metric =  //
                metric_uintptr //
                    ? udf(metric_kind, metric_signature, metric_uintptr, scalar_kind, dimensions)
                    : metric_t(dimensions, metric_kind, scalar_kind);
            index.change_metric(std::move(metric));
        },
        py::arg("metric_kind") = metric_kind_t::cos_k,                   //
        py::arg("metric_signature") = metric_signature_t::array_array_k, //
        py::arg("metric_pointer") = 0                                    //
    );

    i.def_property_readonly("keys", &get_all_keys<dense_index_py_t>);
    i.def("get_keys", &get_keys<dense_index_py_t>, py::arg("offset") = 0,
          py::arg("limit") = std::numeric_limits<std::size_t>::max());
    i.def("__contains__", &dense_index_py_t::contains);
    i.def("__getitem__", &get_member<dense_index_py_t>, py::arg("key"), py::arg("dtype") = scalar_kind_t::f32_k);

    i.def("save", &save_index<dense_index_py_t>, py::arg("path"), py::call_guard<py::gil_scoped_release>());
    i.def("load", &load_index<dense_index_py_t>, py::arg("path"), py::call_guard<py::gil_scoped_release>());
    i.def("view", &view_index<dense_index_py_t>, py::arg("path"), py::call_guard<py::gil_scoped_release>());
    i.def("reset", &reset_index<dense_index_py_t>, py::call_guard<py::gil_scoped_release>());
    i.def("clear", &clear_index<dense_index_py_t>, py::call_guard<py::gil_scoped_release>());
    i.def("copy", &copy_index, py::call_guard<py::gil_scoped_release>());
    i.def("compact", &compact_index, py::call_guard<py::gil_scoped_release>());
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
    is.def("__len__", &dense_indexes_py_t::size);
    is.def("add", &dense_indexes_py_t::add);
    is.def("add_paths", &dense_indexes_py_t::add_paths, py::arg("paths"), py::arg("view") = true,
           py::arg("threads") = 0);
    is.def(                                                  //
        "search", &search_many_in_index<dense_indexes_py_t>, //
        py::arg("query"),                                    //
        py::arg("count") = 10,                               //
        py::arg("exact") = false,                            //
        py::arg("threads") = 0                               //
    );
}

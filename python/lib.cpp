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
#define PY_SSIZE_T_CLEAN
#define NOMINMAX // Some of our dependencies call `std::max(x, y)`, which crashes Windows builds
#include <thread>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

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

using label_t = std::uint32_t;
using distance_t = punned_distance_t;
using metric_t = index_punned_dense_metric_t;
using id_t = std::uint32_t;
using big_id_t = std::uint64_t;

using punned_index_t = index_punned_dense_gt<label_t, id_t>;
using punned_add_result_t = typename punned_index_t::add_result_t;
using punned_search_result_t = typename punned_index_t::search_result_t;

struct punned_index_py_t : public punned_index_t {
    using native_t = punned_index_t;
    using native_t::add;
    using native_t::capacity;
    using native_t::reserve;
    using native_t::search;
    using native_t::size;

    punned_index_py_t(native_t&& base) : native_t(std::move(base)) {}
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
metric_t udf(metric_kind_t kind, metric_signature_t signature, std::uintptr_t metric_uintptr) {
    //
    metric_t result;
    result.kind_ = kind;
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
    case scalar_kind_t::f8_k: return udf<f8_bits_t>(kind, signature, metric_uintptr);
    case scalar_kind_t::f16_k: return udf<f16_t>(kind, signature, metric_uintptr);
    case scalar_kind_t::f32_k: return udf<f32_t>(kind, signature, metric_uintptr);
    case scalar_kind_t::f64_k: return udf<f64_t>(kind, signature, metric_uintptr);
    default: return {};
    }
}

static punned_index_py_t make_index(     //
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
        config = punned_index_t::optimize(config);

    if (metric_uintptr)
        return punned_index_t::make( //
            dimensions, udf(metric_kind, metric_signature, metric_uintptr, scalar_kind), config, scalar_kind,
            expansion_add, expansion_search);
    else
        return punned_index_t::make(dimensions, metric_kind, config, scalar_kind, expansion_add, expansion_search);
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

static void add_one_to_index(punned_index_py_t& index, label_t label, py::buffer vector, bool copy, std::size_t) {

    py::buffer_info vector_info = vector.request();
    if (vector_info.ndim != 1)
        throw std::invalid_argument("Expects a vector, not a higher-rank tensor!");

    Py_ssize_t vector_dimensions = vector_info.shape[0];
    char const* vector_data = reinterpret_cast<char const*>(vector_info.ptr);
    if (vector_dimensions != static_cast<Py_ssize_t>(index.scalar_words()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    if (index.size() + 1 >= index.capacity())
        index.reserve(ceil2(index.size() + 1));

    add_config_t config;
    config.store_vector = copy;

    // https://docs.python.org/3/library/struct.html#format-characters
    if (vector_info.format == "B" || vector_info.format == "u1" || vector_info.format == "|u1")
        index.add(label, reinterpret_cast<b1x8_t const*>(vector_data), config).error.raise();
    else if (vector_info.format == "b" || vector_info.format == "i1" || vector_info.format == "|i1")
        index.add(label, reinterpret_cast<f8_bits_t const*>(vector_data), config).error.raise();
    else if (vector_info.format == "e" || vector_info.format == "f2" || vector_info.format == "<f2")
        index.add(label, reinterpret_cast<f16_t const*>(vector_data), config).error.raise();
    else if (vector_info.format == "f" || vector_info.format == "f4" || vector_info.format == "<f4")
        index.add(label, reinterpret_cast<float const*>(vector_data), config).error.raise();
    else if (vector_info.format == "d" || vector_info.format == "f8" || vector_info.format == "<f8")
        index.add(label, reinterpret_cast<double const*>(vector_data), config).error.raise();
    else
        throw std::invalid_argument("Incompatible scalars in the vector: " + vector_info.format);
}

static void add_many_to_index(                                       //
    punned_index_py_t& index, py::buffer labels, py::buffer vectors, //
    bool copy, std::size_t threads) {

    if (index.limits().threads_add < threads)
        throw std::invalid_argument("Can't use that many threads!");

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

    if (index.size() + vectors_count >= index.capacity())
        index.reserve(ceil2(index.size() + vectors_count));

    char const* vectors_data = reinterpret_cast<char const*>(vectors_info.ptr);
    char const* labels_data = reinterpret_cast<char const*>(labels_info.ptr);

    // https://docs.python.org/3/library/struct.html#format-characters
    if (vectors_info.format == "B" || vectors_info.format == "u1" || vectors_info.format == "|u1")
        executor_default_t{threads}.execute_bulk(vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            add_config_t config;
            config.store_vector = copy;
            config.thread = thread_idx;
            label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
            b1x8_t const* vector = reinterpret_cast<b1x8_t const*>(vectors_data + task_idx * vectors_info.strides[0]);
            index.add(label, vector, config).error.raise();
        });
    else if (vectors_info.format == "b" || vectors_info.format == "i1" || vectors_info.format == "|i1")
        executor_default_t{threads}.execute_bulk(vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            add_config_t config;
            config.store_vector = copy;
            config.thread = thread_idx;
            label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
            f8_bits_t const* vector =
                reinterpret_cast<f8_bits_t const*>(vectors_data + task_idx * vectors_info.strides[0]);
            index.add(label, vector, config).error.raise();
        });
    else if (vectors_info.format == "e" || vectors_info.format == "f2" || vectors_info.format == "<f2")
        executor_default_t{threads}.execute_bulk(vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            add_config_t config;
            config.store_vector = copy;
            config.thread = thread_idx;
            label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
            f16_t const* vector = reinterpret_cast<f16_t const*>(vectors_data + task_idx * vectors_info.strides[0]);
            index.add(label, vector, config).error.raise();
        });
    else if (vectors_info.format == "f" || vectors_info.format == "f4" || vectors_info.format == "<f4")
        executor_default_t{threads}.execute_bulk(vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            add_config_t config;
            config.store_vector = copy;
            config.thread = thread_idx;
            label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
            float const* vector = reinterpret_cast<float const*>(vectors_data + task_idx * vectors_info.strides[0]);
            index.add(label, vector, config).error.raise();
        });
    else if (vectors_info.format == "d" || vectors_info.format == "f8" || vectors_info.format == "<f8")
        executor_default_t{threads}.execute_bulk(vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            add_config_t config;
            config.store_vector = copy;
            config.thread = thread_idx;
            label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
            double const* vector = reinterpret_cast<double const*>(vectors_data + task_idx * vectors_info.strides[0]);
            index.add(label, vector, config).error.raise();
        });
    else
        throw std::invalid_argument("Incompatible scalars in the vectors matrix: " + vectors_info.format);
}

static py::tuple search_one_in_index(punned_index_py_t& index, py::buffer vector, std::size_t wanted, bool exact) {

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

    // https://docs.python.org/3/library/struct.html#format-characters
    if (vector_info.format == "B" || vector_info.format == "u1" || vector_info.format == "|u1") {
        punned_search_result_t result = index.search(reinterpret_cast<b1x8_t const*>(vector_data), wanted, config);
        result.error.raise();
        count = result.dump_to(&labels_py1d(0), &distances_py1d(0));
    } else if (vector_info.format == "b" || vector_info.format == "i1" || vector_info.format == "|i1") {
        punned_search_result_t result = index.search(reinterpret_cast<f8_bits_t const*>(vector_data), wanted, config);
        result.error.raise();
        count = result.dump_to(&labels_py1d(0), &distances_py1d(0));
    } else if (vector_info.format == "e" || vector_info.format == "f2" || vector_info.format == "<f2") {
        punned_search_result_t result = index.search(reinterpret_cast<f16_t const*>(vector_data), wanted, config);
        result.error.raise();
        count = result.dump_to(&labels_py1d(0), &distances_py1d(0));
    } else if (vector_info.format == "f" || vector_info.format == "f4" || vector_info.format == "<f4") {
        punned_search_result_t result = index.search(reinterpret_cast<float const*>(vector_data), wanted, config);
        result.error.raise();
        count = result.dump_to(&labels_py1d(0), &distances_py1d(0));
    } else if (vector_info.format == "d" || vector_info.format == "f8" || vector_info.format == "<f8") {
        punned_search_result_t result = index.search(reinterpret_cast<double const*>(vector_data), wanted, config);
        result.error.raise();
        count = result.dump_to(&labels_py1d(0), &distances_py1d(0));
    } else
        throw std::invalid_argument("Incompatible scalars in the query vector: " + vector_info.format);

    labels_py.resize(py_shape_t{static_cast<Py_ssize_t>(count)});
    distances_py.resize(py_shape_t{static_cast<Py_ssize_t>(count)});

    py::tuple results(3);
    results[0] = labels_py;
    results[1] = distances_py;
    results[2] = static_cast<Py_ssize_t>(count);
    return results;
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
    punned_index_py_t& index, py::buffer vectors, std::size_t wanted, bool exact, std::size_t threads) {

    if (wanted == 0)
        return py::tuple(3);

    if (index.limits().threads_add < threads)
        throw std::invalid_argument("Can't use that many threads!");

    py::buffer_info vectors_info = vectors.request();
    if (vectors_info.ndim == 1)
        return search_one_in_index(index, vectors, wanted, exact);
    if (vectors_info.ndim != 2)
        throw std::invalid_argument("Expects a matrix of vectors to add!");

    Py_ssize_t vectors_count = vectors_info.shape[0];
    Py_ssize_t vectors_dimensions = vectors_info.shape[1];
    char const* vectors_data = reinterpret_cast<char const*>(vectors_info.ptr);
    if (vectors_dimensions != static_cast<Py_ssize_t>(index.scalar_words()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    py::array_t<label_t> labels_py({vectors_count, static_cast<Py_ssize_t>(wanted)});
    py::array_t<distance_t> distances_py({vectors_count, static_cast<Py_ssize_t>(wanted)});
    py::array_t<Py_ssize_t> counts_py(vectors_count);
    auto labels_py2d = labels_py.template mutable_unchecked<2>();
    auto distances_py2d = distances_py.template mutable_unchecked<2>();
    auto counts_py1d = counts_py.template mutable_unchecked<1>();

    // https://docs.python.org/3/library/struct.html#format-characters
    if (vectors_info.format == "B" || vectors_info.format == "u1" || vectors_info.format == "|u1")
        executor_default_t{threads}.execute_bulk(vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            search_config_t config;
            config.thread = thread_idx;
            config.exact = exact;
            b1x8_t const* vector = (b1x8_t const*)(vectors_data + task_idx * vectors_info.strides[0]);
            punned_search_result_t result = index.search(vector, wanted, config);
            result.error.raise();
            counts_py1d(task_idx) =
                static_cast<Py_ssize_t>(result.dump_to(&labels_py2d(task_idx, 0), &distances_py2d(task_idx, 0)));
        });
    else if (vectors_info.format == "b" || vectors_info.format == "i1" || vectors_info.format == "|i1")
        executor_default_t{threads}.execute_bulk(vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            search_config_t config;
            config.thread = thread_idx;
            config.exact = exact;
            f8_bits_t const* vector = (f8_bits_t const*)(vectors_data + task_idx * vectors_info.strides[0]);
            punned_search_result_t result = index.search(vector, wanted, config);
            result.error.raise();
            counts_py1d(task_idx) =
                static_cast<Py_ssize_t>(result.dump_to(&labels_py2d(task_idx, 0), &distances_py2d(task_idx, 0)));
        });
    else if (vectors_info.format == "e" || vectors_info.format == "f2" || vectors_info.format == "<f2")
        executor_default_t{threads}.execute_bulk(vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            search_config_t config;
            config.thread = thread_idx;
            config.exact = exact;
            f16_t const* vector = (f16_t const*)(vectors_data + task_idx * vectors_info.strides[0]);
            punned_search_result_t result = index.search(vector, wanted, config);
            result.error.raise();
            counts_py1d(task_idx) =
                static_cast<Py_ssize_t>(result.dump_to(&labels_py2d(task_idx, 0), &distances_py2d(task_idx, 0)));
        });
    else if (vectors_info.format == "f" || vectors_info.format == "f4" || vectors_info.format == "<f4")
        executor_default_t{threads}.execute_bulk(vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            search_config_t config;
            config.thread = thread_idx;
            config.exact = exact;
            float const* vector = (float const*)(vectors_data + task_idx * vectors_info.strides[0]);
            punned_search_result_t result = index.search(vector, wanted, config);
            result.error.raise();
            counts_py1d(task_idx) =
                static_cast<Py_ssize_t>(result.dump_to(&labels_py2d(task_idx, 0), &distances_py2d(task_idx, 0)));
        });
    else if (vectors_info.format == "d" || vectors_info.format == "f8" || vectors_info.format == "<f8")
        executor_default_t{threads}.execute_bulk(vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            search_config_t config;
            config.thread = thread_idx;
            config.exact = exact;
            double const* vector = (double const*)(vectors_data + task_idx * vectors_info.strides[0]);
            punned_search_result_t result = index.search(vector, wanted, config);
            result.error.raise();
            counts_py1d(task_idx) =
                static_cast<Py_ssize_t>(result.dump_to(&labels_py2d(task_idx, 0), &distances_py2d(task_idx, 0)));
        });
    else
        throw std::invalid_argument("Incompatible scalars in the query matrix: " + vectors_info.format);

    py::tuple results(3);
    results[0] = labels_py;
    results[1] = distances_py;
    results[2] = counts_py;
    return results;
}

// clang-format off
template <typename index_at> void save_index(index_at const& index, std::string const& path) { index.save(path.c_str()).error.raise(); }
template <typename index_at> void load_index(index_at& index, std::string const& path) { index.load(path.c_str()).error.raise(); }
template <typename index_at> void view_index(index_at& index, std::string const& path) { index.view(path.c_str()).error.raise(); }
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
        return get_typed_member<f16_t>(index, label);
    else if (scalar_kind == scalar_kind_t::f8_k)
        return get_typed_member<f8_bits_t, std::int8_t>(index, label);
    else if (scalar_kind == scalar_kind_t::b1x8_k)
        return get_typed_member<b1x8_t, std::uint8_t>(index, label);
    else
        throw std::invalid_argument("Incompatible scalars in the query matrix!");
}

template <typename index_at> py::array_t<label_t> get_labels(index_at const& index) {
    std::size_t result_length = index.size();
    py::array_t<label_t> result_py(static_cast<Py_ssize_t>(result_length));
    auto result_py1d = result_py.template mutable_unchecked<1>();
    index.export_labels(&result_py1d(0), result_length);
    return result_py;
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

    auto i = py::class_<punned_index_py_t>(m, "Index");

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

    i.def("__len__", &punned_index_py_t::size);
    i.def_property_readonly("size", &punned_index_py_t::size);
    i.def_property_readonly("ndim", &punned_index_py_t::dimensions);
    i.def_property_readonly("connectivity", &punned_index_py_t::connectivity);
    i.def_property_readonly("capacity", &punned_index_py_t::capacity);
    i.def_property_readonly( //
        "dtype", [](punned_index_py_t const& index) -> std::string { return scalar_kind_name(index.scalar_kind()); });
    i.def_property_readonly( //
        "memory_usage", [](punned_index_py_t const& index) -> std::size_t { return index.memory_usage(); });

    i.def_property("expansion_add", &punned_index_py_t::expansion_add, &punned_index_py_t::change_expansion_add);
    i.def_property("expansion_search", &punned_index_py_t::expansion_search,
                   &punned_index_py_t::change_expansion_search);

    i.def_property_readonly("labels", &get_labels<punned_index_py_t>);
    i.def("__contains__", &punned_index_py_t::contains);
    i.def( //
        "__getitem__", &get_member<punned_index_py_t>, py::arg("label"), py::arg("dtype") = scalar_kind_t::f32_k);

    i.def("save", &save_index<punned_index_py_t>, py::arg("path"));
    i.def("load", &load_index<punned_index_py_t>, py::arg("path"));
    i.def("view", &view_index<punned_index_py_t>, py::arg("path"));
    i.def("clear", &punned_index_py_t::clear);

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

    si.def("save", &sparse_index_py_t::save, py::arg("path"));
    si.def("load", &sparse_index_py_t::load, py::arg("path"));
    si.def("view", &sparse_index_py_t::view, py::arg("path"));
    si.def("clear", &sparse_index_py_t::clear);
}

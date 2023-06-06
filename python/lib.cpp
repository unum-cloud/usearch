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
#include <thread>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "punned.hpp"

using namespace unum::usearch;
using namespace unum;

namespace py = pybind11;
using py_shape_t = py::array::ShapeContainer;

using label_t = Py_ssize_t;
using distance_t = punned_distance_t;
using id_t = std::uint32_t;

using punned_index_t = punned_gt<label_t, id_t>;
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
using sets_index_t = index_gt<jaccard_gt<set_member_t>, label_t, id_t, set_member_t>;

struct sets_index_py_t : public sets_index_t {
    using native_t = sets_index_t;
    using native_t::add;
    using native_t::capacity;
    using native_t::reserve;
    using native_t::search;
    using native_t::size;

    sets_index_py_t(native_t&& base) : native_t(std::move(base)) {}
};

using bits_numpy_word_t = std::uint8_t;
using bits_native_word_t = std::uint64_t;
using bits_distance_t = float;
using bits_metric_t =
    std::function<bits_distance_t(bits_native_word_t const*, bits_native_word_t const*, std::size_t, std::size_t)>;
using bits_index_t = index_gt<bits_metric_t, label_t, id_t, bits_native_word_t>;
using bits_add_result_t = typename bits_index_t::add_result_t;
using bits_search_result_t = typename bits_index_t::search_result_t;

static constexpr std::size_t bits_per_hash_numpy_word_k = sizeof(bits_numpy_word_t) * CHAR_BIT;
static constexpr std::size_t bits_per_hash_native_word_k = sizeof(bits_native_word_t) * CHAR_BIT;

struct bits_index_py_t : public bits_index_t {
    using native_t = bits_index_t;
    using native_t::add;
    using native_t::capacity;
    using native_t::limits;
    using native_t::reserve;
    using native_t::search;
    using native_t::size;

    std::size_t bits_;
    std::size_t numpy_words_;
    std::size_t native_words_;
    std::vector<bits_native_word_t> word_aligned_buffer_;

    bits_index_py_t(native_t&& base) : native_t(std::move(base)) {}

    bits_index_py_t(index_config_t config, std::size_t bits, bits_metric_t metric)
        : bits_index_t(config, metric), bits_(bits),                         //
          numpy_words_(divide_round_up<bits_per_hash_numpy_word_k>(bits)),   //
          native_words_(divide_round_up<bits_per_hash_native_word_k>(bits)), //
          word_aligned_buffer_(native_words_* index_limits_t{}.threads()) {}
};

bits_metric_t bits_metric(common_metric_kind_t kind) {
    switch (kind) {
    case common_metric_kind_t::bitwise_hamming_k:
        return bits_metric_t(bitwise_hamming_gt<bits_native_word_t, bits_distance_t>{});
    case common_metric_kind_t::bitwise_tanimoto_k:
        return bits_metric_t(bitwise_tanimoto_gt<bits_native_word_t, bits_distance_t>{});
    case common_metric_kind_t::bitwise_sorensen_k:
        return bits_metric_t(bitwise_sorensen_gt<bits_native_word_t, bits_distance_t>{});
    default: throw std::invalid_argument("Not a bitwise metric!"); return {};
    }
}

template <typename scalar_at> punned_stateful_metric_t udf(std::size_t metric_uintptr) {
    return [metric_uintptr](byte_t const* a_bytes, byte_t const* b_bytes, std::size_t n, std::size_t m) -> distance_t {
        using metric_t = punned_distance_t (*)(scalar_at const*, scalar_at const*, std::size_t, std::size_t);
        metric_t metric_ptr = reinterpret_cast<metric_t>(metric_uintptr);
        scalar_at const* a = reinterpret_cast<scalar_at const*>(a_bytes);
        scalar_at const* b = reinterpret_cast<scalar_at const*>(b_bytes);
        return metric_ptr(a, b, n / sizeof(scalar_at), m / sizeof(scalar_at));
    };
}

punned_stateful_metric_t udf(std::size_t metric_uintptr, accuracy_t accuracy) {
    switch (accuracy) {
    case accuracy_t::f8_k: return udf<f8_bits_t>(metric_uintptr);
    case accuracy_t::f16_k: return udf<f16_bits_t>(metric_uintptr);
    case accuracy_t::f32_k: return udf<f32_t>(metric_uintptr);
    case accuracy_t::f64_k: return udf<f64_t>(metric_uintptr);
    }
}

static punned_index_py_t make_index( //
    std::size_t dimensions,          //
    std::string const& scalar_type,  //
    common_metric_kind_t metric,     //
    std::size_t connectivity,        //
    std::size_t expansion_add,       //
    std::size_t expansion_search,    //
    std::size_t metric_uintptr,      //
    bool tune) {

    index_config_t config;
    config.expansion_add = expansion_add;
    config.expansion_search = expansion_search;
    config.connectivity = connectivity;

    if (tune)
        config = punned_index_t::optimize(config);

    accuracy_t accuracy = accuracy_from_name(scalar_type.c_str(), scalar_type.size());
    if (metric_uintptr)
        return punned_index_t::udf(dimensions, udf(metric_uintptr, accuracy), accuracy, config);
    else
        return punned_index_t(make_punned<punned_index_t>(metric, dimensions, accuracy, config));
}

static std::unique_ptr<sets_index_py_t> make_sets_index( //
    std::size_t connectivity,                            //
    std::size_t expansion_add,                           //
    std::size_t expansion_search                         //
) {
    index_config_t config;
    config.expansion_add = expansion_add;
    config.expansion_search = expansion_search;
    config.connectivity = connectivity;

    return std::unique_ptr<sets_index_py_t>(new sets_index_py_t(sets_index_t(config)));
}

static std::unique_ptr<bits_index_py_t> make_hash_index( //
    std::size_t bits,                                    //
    common_metric_kind_t metric,                         //
    std::size_t connectivity,                            //
    std::size_t expansion_add,                           //
    std::size_t expansion_search                         //
) {
    index_config_t config;
    config.expansion_add = expansion_add;
    config.expansion_search = expansion_search;
    config.connectivity = connectivity;
    return std::unique_ptr<bits_index_py_t>(new bits_index_py_t(config, bits, bits_metric(metric)));
}

static void add_one_to_index(punned_index_py_t& index, label_t label, py::buffer vector, bool copy, std::size_t) {

    py::buffer_info vector_info = vector.request();
    if (vector_info.ndim != 1)
        throw std::invalid_argument("Expects a vector, not a higher-rank tensor!");

    Py_ssize_t vector_dimensions = vector_info.shape[0];
    char const* vector_data = reinterpret_cast<char const*>(vector_info.ptr);
    if (vector_dimensions != static_cast<Py_ssize_t>(index.dimensions()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    if (index.size() + 1 >= index.capacity())
        index.reserve(ceil2(index.size() + 1));

    add_config_t config;
    config.store_vector = copy;

    // https://docs.python.org/3/library/struct.html#format-characters
    if (vector_info.format == "c" || vector_info.format == "b")
        index.add(label, reinterpret_cast<f8_bits_t const*>(vector_data), config).error.raise();
    else if (vector_info.format == "e")
        index.add(label, reinterpret_cast<f16_bits_t const*>(vector_data), config).error.raise();
    else if (vector_info.format == "f" || vector_info.format == "f4" || vector_info.format == "<f4")
        index.add(label, reinterpret_cast<float const*>(vector_data), config).error.raise();
    else if (vector_info.format == "d" || vector_info.format == "f8" || vector_info.format == "<f8")
        index.add(label, reinterpret_cast<double const*>(vector_data), config).error.raise();
    else
        throw std::invalid_argument("Incompatible scalars in the vector!");
}

static void add_many_to_index(                                       //
    punned_index_py_t& index, py::buffer labels, py::buffer vectors, //
    bool copy, std::size_t threads) {

    if (index.limits().threads_add < threads)
        throw std::invalid_argument("Can't use that many threads!");

    py::buffer_info labels_info = labels.request();
    py::buffer_info vectors_info = vectors.request();

    if (labels_info.format != py::format_descriptor<label_t>::format())
        throw std::invalid_argument("Incompatible label type!");

    if (labels_info.ndim != 1)
        throw std::invalid_argument("Labels must be placed in a single-dimensional array!");

    if (vectors_info.ndim != 2)
        throw std::invalid_argument("Expects a matrix of vectors to add!");

    Py_ssize_t labels_count = labels_info.shape[0];
    Py_ssize_t vectors_count = vectors_info.shape[0];
    Py_ssize_t vectors_dimensions = vectors_info.shape[1];
    if (vectors_dimensions != static_cast<Py_ssize_t>(index.dimensions()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    if (labels_count != vectors_count)
        throw std::invalid_argument("Number of labels and vectors must match!");

    if (index.size() + vectors_count >= index.capacity())
        index.reserve(ceil2(index.size() + vectors_count));

    char const* vectors_data = reinterpret_cast<char const*>(vectors_info.ptr);
    char const* labels_data = reinterpret_cast<char const*>(labels_info.ptr);

    // https://docs.python.org/3/library/struct.html#format-characters
    if (vectors_info.format == "c" || vectors_info.format == "b")
        multithreaded(threads, vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            add_config_t config;
            config.store_vector = copy;
            config.thread = thread_idx;
            label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
            f8_bits_t const* vector =
                reinterpret_cast<f8_bits_t const*>(vectors_data + task_idx * vectors_info.strides[0]);
            index.add(label, vector, config).error.raise();
        });
    else if (vectors_info.format == "e")
        multithreaded(threads, vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            add_config_t config;
            config.store_vector = copy;
            config.thread = thread_idx;
            label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
            f16_bits_t const* vector =
                reinterpret_cast<f16_bits_t const*>(vectors_data + task_idx * vectors_info.strides[0]);
            index.add(label, vector, config).error.raise();
        });
    else if (vectors_info.format == "f" || vectors_info.format == "f4" || vectors_info.format == "<f4")
        multithreaded(threads, vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            add_config_t config;
            config.store_vector = copy;
            config.thread = thread_idx;
            label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
            float const* vector = reinterpret_cast<float const*>(vectors_data + task_idx * vectors_info.strides[0]);
            index.add(label, vector, config).error.raise();
        });
    else if (vectors_info.format == "d" || vectors_info.format == "f8" || vectors_info.format == "<f8")
        multithreaded(threads, vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            add_config_t config;
            config.store_vector = copy;
            config.thread = thread_idx;
            label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
            double const* vector = reinterpret_cast<double const*>(vectors_data + task_idx * vectors_info.strides[0]);
            index.add(label, vector, config).error.raise();
        });
    else
        throw std::invalid_argument("Incompatible scalars in the vectors matrix!");
}

static void add_many_to_bits_index(                                //
    bits_index_py_t& index, py::buffer labels, py::buffer vectors, //
    bool copy, std::size_t threads) {

    if (index.limits().threads_add < threads)
        throw std::invalid_argument("Can't use that many threads!");

    py::buffer_info labels_info = labels.request();
    py::buffer_info vectors_info = vectors.request();

    if (labels_info.format != py::format_descriptor<label_t>::format())
        throw std::invalid_argument("Incompatible label type!");

    if (labels_info.ndim != 1)
        throw std::invalid_argument("Labels must be placed in a single-dimensional array!");

    if (vectors_info.ndim != 2)
        throw std::invalid_argument("Expects a matrix of vectors to add!");

    Py_ssize_t labels_count = labels_info.shape[0];
    Py_ssize_t vectors_count = vectors_info.shape[0];
    Py_ssize_t vectors_dimensions = vectors_info.shape[1];
    if (vectors_dimensions != static_cast<Py_ssize_t>(index.numpy_words_))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    if (labels_count != vectors_count)
        throw std::invalid_argument("Number of labels and vectors must match!");

    if (index.size() + vectors_count >= index.capacity())
        index.reserve(ceil2(index.size() + vectors_count));

    char const* vectors_data = reinterpret_cast<char const*>(vectors_info.ptr);
    char const* labels_data = reinterpret_cast<char const*>(labels_info.ptr);

    // https://docs.python.org/3/library/struct.html#format-characters
    if (vectors_info.format != "B")
        throw std::invalid_argument("Incompatible scalars in the vectors matrix!");

    multithreaded(threads, vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
        add_config_t config;
        config.store_vector = copy;
        config.thread = thread_idx;
        label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
        bits_numpy_word_t const* vector =
            reinterpret_cast<bits_numpy_word_t const*>(vectors_data + task_idx * vectors_info.strides[0]);
        bits_native_word_t* vector_words = index.word_aligned_buffer_.data() + index.native_words_ * thread_idx;
        std::memcpy(vector_words, vector, vectors_dimensions);
        index.add(label, {vector_words, index.native_words_}, config).error.raise();
    });
}

static py::tuple search_one_in_index(punned_index_py_t& index, py::buffer vector, std::size_t wanted, bool exact) {

    py::buffer_info vector_info = vector.request();
    Py_ssize_t vector_dimensions = vector_info.shape[0];
    char const* vector_data = reinterpret_cast<char const*>(vector_info.ptr);
    if (vector_dimensions != static_cast<Py_ssize_t>(index.dimensions()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    py::array_t<label_t> labels_py(static_cast<Py_ssize_t>(wanted));
    py::array_t<distance_t> distances_py(static_cast<Py_ssize_t>(wanted));
    std::size_t count{};
    auto labels_py1d = labels_py.mutable_unchecked<1>();
    auto distances_py1d = distances_py.mutable_unchecked<1>();

    search_config_t config;
    config.exact = exact;

    // https://docs.python.org/3/library/struct.html#format-characters
    if (vector_info.format == "c" || vector_info.format == "b") {
        punned_search_result_t result = index.search(reinterpret_cast<f8_bits_t const*>(vector_data), wanted, config);
        result.error.raise();
        count = result.dump_to(&labels_py1d(0), &distances_py1d(0));
    } else if (vector_info.format == "e") {
        punned_search_result_t result = index.search(reinterpret_cast<f16_bits_t const*>(vector_data), wanted, config);
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
        throw std::invalid_argument("Incompatible scalars in the query vector!");

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
    if (vectors_dimensions != static_cast<Py_ssize_t>(index.dimensions()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    py::array_t<label_t> labels_py({vectors_count, static_cast<Py_ssize_t>(wanted)});
    py::array_t<distance_t> distances_py({vectors_count, static_cast<Py_ssize_t>(wanted)});
    py::array_t<Py_ssize_t> counts_py(vectors_count);
    auto labels_py2d = labels_py.mutable_unchecked<2>();
    auto distances_py2d = distances_py.mutable_unchecked<2>();
    auto counts_py1d = counts_py.mutable_unchecked<1>();

    // https://docs.python.org/3/library/struct.html#format-characters
    if (vectors_info.format == "c" || vectors_info.format == "b")
        multithreaded(threads, vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            search_config_t config;
            config.thread = thread_idx;
            config.exact = exact;
            f8_bits_t const* vector = (f8_bits_t const*)(vectors_data + task_idx * vectors_info.strides[0]);
            punned_search_result_t result = index.search(vector, wanted, config);
            result.error.raise();
            counts_py1d(task_idx) =
                static_cast<Py_ssize_t>(result.dump_to(&labels_py2d(task_idx, 0), &distances_py2d(task_idx, 0)));
        });
    else if (vectors_info.format == "e")
        multithreaded(threads, vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            search_config_t config;
            config.thread = thread_idx;
            config.exact = exact;
            f16_bits_t const* vector = (f16_bits_t const*)(vectors_data + task_idx * vectors_info.strides[0]);
            punned_search_result_t result = index.search(vector, wanted, config);
            result.error.raise();
            counts_py1d(task_idx) =
                static_cast<Py_ssize_t>(result.dump_to(&labels_py2d(task_idx, 0), &distances_py2d(task_idx, 0)));
        });
    else if (vectors_info.format == "f" || vectors_info.format == "f4" || vectors_info.format == "<f4")
        multithreaded(threads, vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
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
        multithreaded(threads, vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
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
        throw std::invalid_argument("Incompatible scalars in the query matrix!");

    py::tuple results(3);
    results[0] = labels_py;
    results[1] = distances_py;
    results[2] = counts_py;
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
static py::tuple search_many_in_bits_index( //
    bits_index_py_t& index, py::buffer vectors, std::size_t wanted, bool exact, std::size_t threads) {

    if (wanted == 0)
        return py::tuple(3);

    if (index.limits().threads_add < threads)
        throw std::invalid_argument("Can't use that many threads!");

    py::buffer_info vectors_info = vectors.request();
    // if (vectors_info.ndim == 1)
    //     return search_one_in_index(index, vectors, wanted, exact);
    if (vectors_info.ndim != 2)
        throw std::invalid_argument("Expects a matrix of vectors to add!");

    Py_ssize_t vectors_count = vectors_info.shape[0];
    Py_ssize_t vectors_dimensions = vectors_info.shape[1];
    char const* vectors_data = reinterpret_cast<char const*>(vectors_info.ptr);
    if (vectors_dimensions != static_cast<Py_ssize_t>(index.numpy_words_))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    py::array_t<label_t> labels_py({vectors_count, static_cast<Py_ssize_t>(wanted)});
    py::array_t<distance_t> distances_py({vectors_count, static_cast<Py_ssize_t>(wanted)});
    py::array_t<Py_ssize_t> counts_py(vectors_count);
    auto labels_py2d = labels_py.mutable_unchecked<2>();
    auto distances_py2d = distances_py.mutable_unchecked<2>();
    auto counts_py1d = counts_py.mutable_unchecked<1>();

    // https://docs.python.org/3/library/struct.html#format-characters
    if (vectors_info.format != "B")
        throw std::invalid_argument("Incompatible scalars in the query matrix!");

    multithreaded(threads, vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
        search_config_t config;
        config.thread = thread_idx;
        config.exact = exact;

        bits_numpy_word_t const* vector =
            reinterpret_cast<bits_numpy_word_t const*>(vectors_data + task_idx * vectors_info.strides[0]);
        bits_native_word_t* vector_words = index.word_aligned_buffer_.data() + index.native_words_ * thread_idx;
        std::memcpy(vector_words, vector, vectors_dimensions);

        bits_search_result_t result = index.search({vector_words, index.native_words_}, wanted, config);
        result.error.raise();
        counts_py1d(task_idx) =
            static_cast<Py_ssize_t>(result.dump_to(&labels_py2d(task_idx, 0), &distances_py2d(task_idx, 0)));
    });

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
template <typename index_at> void clear_index(index_at& index) { index.clear(); }
template <typename index_at> std::size_t get_expansion_add(index_at const &index) { return index.config().expansion_add; }
template <typename index_at> std::size_t get_expansion_search(index_at const &index) { return index.config().expansion_search; }
// clang-format on

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

inline bits_native_word_t hash_ror64(bits_native_word_t v, int r) noexcept { return (v >> r) | (v << (64 - r)); }

inline bits_native_word_t hash(bits_native_word_t v) noexcept {
    v ^= hash_ror64(v, 25) ^ hash_ror64(v, 50);
    v *= 0xA24BAED4963EE407UL;
    v ^= hash_ror64(v, 24) ^ hash_ror64(v, 49);
    v *= 0x9FB21C651E98DF25UL;
    return v ^ v >> 28;
}

template <typename scalar_at>
void hash_typed_row(                                                             //
    byte_t const* scalars, std::size_t scalar_stride, std::size_t scalars_count, //
    bits_numpy_word_t* hashes, std::size_t bits) noexcept {

    std::size_t words = divide_round_up<8>(bits);
    for (std::size_t i = 0; i != scalars_count; ++i) {
        scalar_at scalar;
        std::memcpy(&scalar, scalars + i * scalar_stride, sizeof(scalar_at));
        bits_native_word_t scalar_hash = hash(scalar);
        hashes[scalar_hash & (words - 1)] |= std::uint8_t(1) << (scalar_hash % 8);
    }
}

#if 0
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
        auto hashes_proxy = hashes_py.mutable_unchecked<2>();

        for (Py_ssize_t i = 0; i != rows_count; ++i)
            hash_typed_row(                                                                                 //
                data + i * rows_stride, hashes_proxy.stride(1), static_cast<std::size_t>(elements_per_row), //
                &hashes_proxy(i, 0), bits);

        return hashes_py;
    } else {
        auto hashes_shape = py_shape_t{words_per_row};
        auto hashes_py = py::array_t<bits_numpy_word_t>(hashes_shape);
        auto hashes_proxy = hashes_py.mutable_unchecked<1>();

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

    py::enum_<common_metric_kind_t>(m, "MetricKind")
        .value("Unknown", common_metric_kind_t::unknown_k)
        .value("IP", common_metric_kind_t::ip_k)
        .value("Cos", common_metric_kind_t::cos_k)
        .value("L2sq", common_metric_kind_t::l2sq_k)
        .value("Haversine", common_metric_kind_t::haversine_k)
        .value("Pearson", common_metric_kind_t::pearson_k)
        .value("Jaccard", common_metric_kind_t::jaccard_k)
        .value("BitwiseHamming", common_metric_kind_t::bitwise_hamming_k)
        .value("BitwiseTanimoto", common_metric_kind_t::bitwise_tanimoto_k)
        .value("BitwiseSorensen", common_metric_kind_t::bitwise_sorensen_k);

    auto i = py::class_<punned_index_py_t>(m, "Index");

    i.def(py::init(&make_index),                                    //
          py::kw_only(),                                            //
          py::arg("ndim") = 0,                                      //
          py::arg("dtype") = std::string("f32"),                    //
          py::arg("metric") = common_metric_kind_t::ip_k,           //
          py::arg("connectivity") = default_connectivity(),         //
          py::arg("expansion_add") = default_expansion_add(),       //
          py::arg("expansion_search") = default_expansion_search(), //
          py::arg("metric_pointer") = 0,                            //
          py::arg("tune") = false                                   //
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
        "dtype", [](punned_index_py_t const& index) -> std::string { return accuracy_name(index.accuracy()); });
    i.def_property_readonly("memory_usage",
                            [](punned_index_py_t const& index) -> std::size_t { return index.memory_usage(); });

    i.def_property("expansion_add", &get_expansion_add<punned_index_py_t>, &punned_index_py_t::change_expansion_add);
    i.def_property("expansion_search", &get_expansion_search<punned_index_py_t>,
                   &punned_index_py_t::change_expansion_search);

    i.def("save", &save_index<punned_index_py_t>, py::arg("path"));
    i.def("load", &load_index<punned_index_py_t>, py::arg("path"));
    i.def("view", &view_index<punned_index_py_t>, py::arg("path"));
    i.def("clear", &clear_index<punned_index_py_t>);

    auto bi = py::class_<bits_index_py_t>(m, "BitsIndex");

    bi.def(                                                          //
        py::init(&make_hash_index),                                  //
        py::kw_only(),                                               //
        py::arg("bits"),                                             //
        py::arg("metric") = common_metric_kind_t::bitwise_hamming_k, //
        py::arg("connectivity") = default_connectivity(),            //
        py::arg("expansion_add") = default_expansion_add(),          //
        py::arg("expansion_search") = default_expansion_search()     //
    );

    bi.def( //
        "add",
        &add_many_to_bits_index, //
        py::arg("label"),        //
        py::arg("array"),        //
        py::kw_only(),           //
        py::arg("copy") = true,  //
        py::arg("threads") = 0   //
    );

    bi.def(                                   //
        "search", &search_many_in_bits_index, //
        py::arg("query"),                     //
        py::arg("count") = 10,                //
        py::arg("exact") = false,             //
        py::arg("threads") = 0                //
    );

    bi.def("__len__", &bits_index_py_t::size);
    bi.def_property_readonly("size", &bits_index_py_t::size);
    bi.def_property_readonly("connectivity", &bits_index_py_t::connectivity);
    bi.def_property_readonly("capacity", &bits_index_py_t::capacity);

    bi.def("save", &save_index<bits_index_py_t>, py::arg("path"));
    bi.def("load", &load_index<bits_index_py_t>, py::arg("path"));
    bi.def("view", &view_index<bits_index_py_t>, py::arg("path"));
    bi.def("clear", &clear_index<bits_index_py_t>);

    auto si = py::class_<sets_index_py_t>(m, "SetsIndex");

    si.def(                                                      //
        py::init(&make_sets_index),                              //
        py::kw_only(),                                           //
        py::arg("connectivity") = default_connectivity(),        //
        py::arg("expansion_add") = default_expansion_add(),      //
        py::arg("expansion_search") = default_expansion_search() //
    );

    si.def( //
        "add",
        [](sets_index_py_t& index, label_t label, py::array_t<set_member_t> set, bool copy) {
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
        [](sets_index_py_t& index, py::array_t<set_member_t> set, std::size_t count) -> py::array_t<label_t> {
            validate_set(set);
            auto proxy = set.unchecked<1>();
            auto view = set_view_t{&proxy(0), static_cast<std::size_t>(proxy.shape(0))};
            auto labels_py = py::array_t<label_t>(py_shape_t{static_cast<Py_ssize_t>(count)});
            auto labels_proxy = labels_py.mutable_unchecked<1>();
            auto result = index.search(view, count);
            result.error.raise();
            auto found = result.dump_to(&labels_proxy(0));
            labels_py.resize(py_shape_t{static_cast<Py_ssize_t>(found)});
            return labels_py;
        },
        py::arg("set"),       //
        py::arg("count") = 10 //
    );

    si.def("__len__", &sets_index_py_t::size);
    si.def_property_readonly("size", &sets_index_py_t::size);
    si.def_property_readonly("connectivity", &sets_index_py_t::connectivity);
    si.def_property_readonly("capacity", &sets_index_py_t::capacity);

    si.def("save", &save_index<sets_index_py_t>, py::arg("path"));
    si.def("load", &load_index<sets_index_py_t>, py::arg("path"));
    si.def("view", &view_index<sets_index_py_t>, py::arg("path"));
    si.def("clear", &clear_index<sets_index_py_t>);
}

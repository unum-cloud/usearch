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

#include "advanced.hpp"

using namespace unum::usearch;
using namespace unum;

namespace py = pybind11;

using label_t = Py_ssize_t;
using distance_t = punned_distance_t;
using native_index_t = auto_index_gt<label_t>;

using set_member_t = std::uint32_t;
using set_view_t = span_gt<set_member_t const>;
using sets_index_t = index_gt<jaccard_gt<set_member_t>, label_t, std::uint32_t, set_member_t>;

using hash_index_t = index_gt<bit_hamming_gt<std::uint32_t>, label_t, std::uint32_t, std::size_t>;

static native_index_t make_index(   //
    std::size_t dimensions,         //
    std::size_t capacity,           //
    std::string const& scalar_type, //
    std::string const& metric,      //
    std::size_t connectivity,       //
    std::size_t expansion_add,      //
    std::size_t expansion_search,   //
    std::size_t metric_uintptr      //
) {

    config_t config;
    config.expansion_add = expansion_add;
    config.expansion_search = expansion_search;
    config.connectivity = connectivity;
    config.max_elements = capacity;
    config.max_threads_add = std::thread::hardware_concurrency();
    config.max_threads_search = std::thread::hardware_concurrency();

    accuracy_t accuracy = accuracy_from_name(scalar_type.c_str(), scalar_type.size());
    punned_metric_t metric_ptr = reinterpret_cast<punned_metric_t>(metric_uintptr);
    if (metric_ptr)
        return native_index_t::udf(dimensions, metric_ptr, accuracy, config);
    else
        return index_from_name<native_index_t>(metric.c_str(), metric.size(), dimensions, accuracy, config);
}

static std::unique_ptr<sets_index_t> make_sets_index( //
    std::size_t capacity,                             //
    std::size_t connectivity,                         //
    std::size_t expansion_add,                        //
    std::size_t expansion_search                      //
) {
    config_t config;
    config.expansion_add = expansion_add;
    config.expansion_search = expansion_search;
    config.connectivity = connectivity;
    config.max_elements = capacity;
    config.max_threads_add = 1;
    config.max_threads_search = 1;

    return std::unique_ptr<sets_index_t>(new sets_index_t(config));
}

static void add_one_to_index(native_index_t& index, label_t label, py::buffer vector, bool copy) {

    py::buffer_info vector_info = vector.request();
    if (vector_info.ndim != 1)
        throw std::invalid_argument("Expects a vector, not a higher-rank tensor!");

    ssize_t vector_dimensions = vector_info.shape[0];
    char const* vector_data = reinterpret_cast<char const*>(vector_info.ptr);
    if (vector_dimensions != static_cast<ssize_t>(index.dimensions()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    if (index.size() + 1 >= index.capacity())
        index.reserve(ceil2(index.size() + 1));

    // https://docs.python.org/3/library/struct.html#format-characters
    if (vector_info.format == "e")
        index.add(label, reinterpret_cast<f16_converted_t const*>(vector_data), 0, copy);
    else if (vector_info.format == "f")
        index.add(label, reinterpret_cast<float const*>(vector_data), 0, copy);
    else if (vector_info.format == "d")
        index.add(label, reinterpret_cast<double const*>(vector_data), 0, copy);
    else
        throw std::invalid_argument("Incompatible scalars in the vector!");
}

static void add_many_to_index(native_index_t& index, py::buffer labels, py::buffer vectors, bool copy) {

    py::buffer_info labels_info = labels.request();
    py::buffer_info vectors_info = vectors.request();

    if (labels_info.format != py::format_descriptor<label_t>::format())
        throw std::invalid_argument("Incompatible label type!");

    if (labels_info.ndim != 1)
        throw std::invalid_argument("Labels must be placed in a single-dimensional array!");

    if (vectors_info.ndim != 2)
        throw std::invalid_argument("Expects a matrix of vectors to add!");

    ssize_t labels_count = labels_info.shape[0];
    ssize_t vectors_count = vectors_info.shape[0];
    ssize_t vectors_dimensions = vectors_info.shape[1];
    if (vectors_dimensions != static_cast<ssize_t>(index.dimensions()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    if (labels_count != vectors_count)
        throw std::invalid_argument("Number of labels and vectors must match!");

    if (index.size() + vectors_count >= index.capacity())
        index.reserve(ceil2(index.size() + vectors_count));

    char const* vectors_data = reinterpret_cast<char const*>(vectors_info.ptr);
    char const* labels_data = reinterpret_cast<char const*>(labels_info.ptr);

    // https://docs.python.org/3/library/struct.html#format-characters
    if (vectors_info.format == "e")
        multithreaded(index.concurrency(), vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
            f16_converted_t const* vector =
                reinterpret_cast<f16_converted_t const*>(vectors_data + task_idx * vectors_info.strides[0]);
            index.add(label, vector, thread_idx, copy);
        });
    else if (vectors_info.format == "f")
        multithreaded(index.concurrency(), vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
            float const* vector = reinterpret_cast<float const*>(vectors_data + task_idx * vectors_info.strides[0]);
            index.add(label, vector, thread_idx, copy);
        });
    else if (vectors_info.format == "d")
        multithreaded(index.concurrency(), vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
            double const* vector = reinterpret_cast<double const*>(vectors_data + task_idx * vectors_info.strides[0]);
            index.add(label, vector, thread_idx, copy);
        });
    else
        throw std::invalid_argument("Incompatible scalars in the vectors matrix!");
}

static py::tuple search_one_in_index(native_index_t& index, py::buffer vector, std::size_t wanted) {

    py::buffer_info vector_info = vector.request();
    ssize_t vector_dimensions = vector_info.shape[0];
    char const* vector_data = reinterpret_cast<char const*>(vector_info.ptr);
    if (vector_dimensions != static_cast<ssize_t>(index.dimensions()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    py::array_t<label_t> labels_py(static_cast<ssize_t>(wanted));
    py::array_t<distance_t> distances_py(static_cast<ssize_t>(wanted));
    std::size_t count{};
    auto labels_py1d = labels_py.mutable_unchecked<1>();
    auto distances_py1d = distances_py.mutable_unchecked<1>();

    // https://docs.python.org/3/library/struct.html#format-characters
    if (vector_info.format == "e")
        count = index.search( //
            reinterpret_cast<f16_converted_t const*>(vector_data), wanted, &labels_py1d(0), &distances_py1d(0), 0);
    else if (vector_info.format == "f")
        count = index.search( //
            reinterpret_cast<float const*>(vector_data), wanted, &labels_py1d(0), &distances_py1d(0), 0);
    else if (vector_info.format == "d")
        count = index.search( //
            reinterpret_cast<double const*>(vector_data), wanted, &labels_py1d(0), &distances_py1d(0), 0);
    else
        throw std::invalid_argument("Incompatible scalars in the query vector!");

    labels_py.resize({static_cast<ssize_t>(count)});
    distances_py.resize({static_cast<ssize_t>(count)});

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
static py::tuple search_many_in_index(native_index_t& index, py::buffer vectors, std::size_t wanted) {

    if (wanted == 0)
        return py::tuple(3);

    py::buffer_info vectors_info = vectors.request();
    if (vectors_info.ndim == 1)
        return search_one_in_index(index, vectors, wanted);
    if (vectors_info.ndim != 2)
        throw std::invalid_argument("Expects a matrix of vectors to add!");

    ssize_t vectors_count = vectors_info.shape[0];
    ssize_t vectors_dimensions = vectors_info.shape[1];
    char const* vectors_data = reinterpret_cast<char const*>(vectors_info.ptr);
    if (vectors_dimensions != static_cast<ssize_t>(index.dimensions()))
        throw std::invalid_argument("The number of vector dimensions doesn't match!");

    py::array_t<label_t> labels_py({vectors_count, static_cast<ssize_t>(wanted)});
    py::array_t<distance_t> distances_py({vectors_count, static_cast<ssize_t>(wanted)});
    py::array_t<Py_ssize_t> counts_py(vectors_count);
    auto labels_py2d = labels_py.mutable_unchecked<2>();
    auto distances_py2d = distances_py.mutable_unchecked<2>();
    auto counts_py1d = counts_py.mutable_unchecked<1>();

    // https://docs.python.org/3/library/struct.html#format-characters
    if (vectors_info.format == "e")
        multithreaded(index.concurrency(), vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            f16_converted_t const* vector = (f16_converted_t const*)(vectors_data + task_idx * vectors_info.strides[0]);
            counts_py1d(task_idx) = static_cast<Py_ssize_t>(
                index.search(vector, wanted, &labels_py2d(task_idx, 0), &distances_py2d(task_idx, 0), thread_idx));
        });
    else if (vectors_info.format == "f")
        multithreaded(index.concurrency(), vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            float const* vector = (float const*)(vectors_data + task_idx * vectors_info.strides[0]);
            counts_py1d(task_idx) = static_cast<Py_ssize_t>(
                index.search(vector, wanted, &labels_py2d(task_idx, 0), &distances_py2d(task_idx, 0), thread_idx));
        });
    else if (vectors_info.format == "d")
        multithreaded(index.concurrency(), vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            double const* vector = (double const*)(vectors_data + task_idx * vectors_info.strides[0]);
            counts_py1d(task_idx) = static_cast<Py_ssize_t>(
                index.search(vector, wanted, &labels_py2d(task_idx, 0), &distances_py2d(task_idx, 0), thread_idx));
        });
    else
        throw std::invalid_argument("Incompatible scalars in the query matrix!");

    py::tuple results(3);
    results[0] = labels_py;
    results[1] = distances_py;
    results[2] = counts_py;
    return results;
}

template <typename index_at = native_index_t> //
static void save_index(index_at const& index, std::string const& path) {
    index.save(path.c_str());
}

template <typename index_at = native_index_t> //
static void load_index(index_at& index, std::string const& path) {
    index.load(path.c_str());
}

template <typename index_at = native_index_t> //
static void view_index(index_at& index, std::string const& path) {
    index.view(path.c_str());
}

template <typename index_at = native_index_t> //
static void clear_index(index_at& index) {
    index.clear();
}

PYBIND11_MODULE(usearch, m) {
    m.doc() = "Unum USearch Python bindings";

    auto i = py::class_<native_index_t>(m, "Index");

    i.def(py::init(&make_index),                                              //
          py::kw_only(),                                                      //
          py::arg("ndim") = 0,                                                //
          py::arg("capacity") = 0,                                            //
          py::arg("dtype") = std::string("f32"),                              //
          py::arg("metric") = std::string("ip"),                              //
          py::arg("connectivity") = config_t::connectivity_default_k,         //
          py::arg("expansion_add") = config_t::expansion_add_default_k,       //
          py::arg("expansion_search") = config_t::expansion_search_default_k, //
          py::arg("metric_pointer") = 0                                       //
    );

    i.def(                         //
        "add", &add_many_to_index, //
        py::arg("labels"),         //
        py::arg("vectors"),        //
        py::kw_only(),             //
        py::arg("copy") = true     //
    );

    i.def(                        //
        "add", &add_one_to_index, //
        py::arg("label"),         //
        py::arg("vector"),        //
        py::kw_only(),            //
        py::arg("copy") = true    //
    );

    i.def(                               //
        "search", &search_many_in_index, //
        py::arg("query"),                //
        py::arg("count") = 10            //
    );

    i.def("__len__", &native_index_t::size);
    i.def_property_readonly("size", &native_index_t::size);
    i.def_property_readonly("ndim", &native_index_t::dimensions);
    i.def_property_readonly("connectivity", &native_index_t::connectivity);
    i.def_property_readonly("capacity", &native_index_t::capacity);

    i.def("save", &save_index<native_index_t>, py::arg("path"));
    i.def("load", &load_index<native_index_t>, py::arg("path"));
    i.def("view", &view_index<native_index_t>, py::arg("path"));
    i.def("clear", &clear_index<native_index_t>);

    auto si = py::class_<sets_index_t>(m, "SetsIndex");

    si.def(                                                                //
        py::init(&make_sets_index),                                        //
        py::kw_only(),                                                     //
        py::arg("capacity") = 0,                                           //
        py::arg("connectivity") = config_t::connectivity_default_k,        //
        py::arg("expansion_add") = config_t::expansion_add_default_k,      //
        py::arg("expansion_search") = config_t::expansion_search_default_k //
    );

    si.def( //
        "add",
        [](sets_index_t& index, label_t label, py::array_t<set_member_t> set, bool copy) {
            if (set.ndim() != 1)
                throw std::runtime_error("Set can't be multi-dimensional!");
            if (set.strides(0) != sizeof(set_member_t))
                throw std::runtime_error("Set can't be strided!");
            if (index.size() + 1 >= index.capacity())
                index.reserve(ceil2(index.size() + 1));
            auto proxy = set.unchecked<1>();
            auto view = set_view_t{proxy.data(0), static_cast<std::size_t>(proxy.shape(0))};
            index.add(label, view, 0, copy);
        },                     //
        py::arg("label"),      //
        py::arg("set"),        //
        py::kw_only(),         //
        py::arg("copy") = true //
    );

    si.def( //
        "search",
        [](sets_index_t& index, py::array_t<set_member_t> set, std::size_t count) -> py::array_t<label_t> {
            auto proxy = set.unchecked<1>();
            auto view = set_view_t{proxy.data(0), static_cast<std::size_t>(proxy.shape(0))};
            auto labels_py = py::array_t<label_t>({static_cast<ssize_t>(count)});
            auto labels_proxy = labels_py.mutable_unchecked<1>();
            auto found = index.search(view, count, &labels_proxy(0), nullptr, 0);
            labels_py.resize({static_cast<ssize_t>(found)});
            return labels_py;
        },
        py::arg("set"),       //
        py::arg("count") = 10 //
    );

    si.def("__len__", &sets_index_t::size);
    si.def_property_readonly("size", &sets_index_t::size);
    si.def_property_readonly("connectivity", &sets_index_t::connectivity);
    si.def_property_readonly("capacity", &sets_index_t::capacity);

    si.def("save", &save_index<sets_index_t>, py::arg("path"));
    si.def("load", &load_index<sets_index_t>, py::arg("path"));
    si.def("view", &view_index<sets_index_t>, py::arg("path"));
}

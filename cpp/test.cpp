/**
 *  @file       test.cpp
 *  @author     Ash Vardanian
 *  @brief      Unit-testing vector-search functionality.
 *  @date       June 10, 2023
 */
#include <algorithm>     // `std::shuffle`
#include <cassert>       // `assert`
#include <random>        // `std::default_random_engine`
#include <stdexcept>     // `std::terminate`
#include <unordered_map> // `std::unordered_map`
#include <vector>        // `std::vector`

#define SZ_USE_X86_AVX512 0            // Sanitizers hate AVX512
#include <stringzilla/stringzilla.hpp> // Levenshtein distance implementation

#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>
#include <usearch/index_plugins.hpp>

using namespace unum::usearch;
using namespace unum;

void expect(bool must_be_true) {
    if (!must_be_true)
        __usearch_raise_runtime_error("Failed!");
}

template <typename value_at> void expect_eq(value_at a, value_at b) { expect(a == b); }

/**
 *  @brief  Convinience wrapper combining combined allocation and construction of an index.
 */
template <typename index_at> struct aligned_wrapper_gt {
    using index_t = index_at;
    using alloc_t = aligned_allocator_gt<index_t, 64>;

    alloc_t alloc;
    index_t* index = nullptr;

    template <typename... args_at> aligned_wrapper_gt(args_at&&... args) {

        alloc_t index_alloc;
        index_t* index_typed = index_alloc.allocate(1);
        expect(index_typed != nullptr);
        expect(((unsigned long long)(index_typed) % 64ull) == 0ull);

        new (index_typed) index_t(std::forward<args_at>(args)...);
        index = index_typed;
    }

    ~aligned_wrapper_gt() {
        if (index != nullptr) {
            index->~index_t();
            alloc.deallocate(index, 1);
        }
    }
};

/**
 * Tests the functionality of the custom uint40_t type ensuring consistent 
 * behavior across various constructors from uint32_t, uint64_t, and size_t types.
 */
void test_uint40() {
    // Constants for tests
    std::uint64_t max_uint40_k = (1ULL << 40) - 1;

    for (std::uint64_t original_value : {
        42ull,                        // Typical small number
        4242ull,                      // Larger number still within uint40 range
        1ull << 40,                   // Exactly at the boundary of uint40
        (1ull << 40) + 1,             // Just beyond the boundary of uint40
        1ull << 63                    // Well beyond the uint40 boundary, tests masking
    }) {
        std::uint32_t v_32 = static_cast<std::uint32_t>(original_value);
        std::uint64_t v_64 = original_value;
        std::size_t v_size = static_cast<std::size_t>(original_value);

        // Create uint40_t instances from different types
        uint40_t n_40_from_32(v_32);
        uint40_t n_40_from_64(v_64);
        uint40_t n_40_from_size(v_size);

        // Expected value after masking
        std::uint64_t expected_value = original_value & max_uint40_k;

        // Check if all conversions are equal to the masked value
        expect(n_40_from_32 == expected_value);
        expect(n_40_from_64 == expected_value);
        expect(n_40_from_size == expected_value);
    }
}
/**
 * Tests the behavior of various move-constructors and move-assignment operators for the index.
 *
 * Constructs an index and performs tests with it before and after move operations to ensure that the index maintains
 * its integrity and functionality after being moved.
 *
 * @param config Configuration settings for the index.
 * @tparam index_at Type of the index being tested.
 */
template <typename index_at> void test_move_constructors(index_config_t const& config) {
    {
        index_at index{config};
        test_sets(index);
    }
    {
        index_at index{config};
        index.reserve(1);
        test_sets(index_at(std::move(index)));
    }
    {
        index_at index{config};
        index.reserve(1);
        index_at index_moved = std::move(index);
        test_sets(index_moved);
    }
}

/**
 * The goal of this test is to invoke as many different interfaces as possible, making sure that all code-paths compile.
 * For that it only uses a tiny set of 3 predefined vectors.
 *
 * @param index Reference to the index where vectors will be stored and searched.
 * @param vectors A collection of vectors to be tested.
 * @param args Additional arguments for configuring search or index operations.
 * @tparam punned_ak Template parameter that determines specific behaviors or checks in the test based on its value.
 * @tparam index_at Type of the index being tested.
 * @tparam scalar_at Data type of the elements in the vectors.
 * @tparam extra_args_at Variadic template parameter types for additional configuration.
 */
template <bool punned_ak, typename index_at, typename scalar_at, typename... extra_args_at>
void test_minimal_three_vectors(index_at& index, //
                                typename index_at::vector_key_t key_first, std::vector<scalar_at> const& vector_first,
                                typename index_at::vector_key_t key_second, std::vector<scalar_at> const& vector_second,
                                typename index_at::vector_key_t key_third, std::vector<scalar_at> const& vector_third,
                                extra_args_at&&... args) {

    using scalar_t = scalar_at;
    using index_t = index_at;
    using vector_key_t = typename index_t::vector_key_t;
    using distance_t = typename index_t::distance_t;

    // Try checking the empty state
    if constexpr (punned_ak) {
        expect(!index.contains(key_first));
        expect(!index.get(key_first, (f32_t*)nullptr, 1));
    }

    // Add data
    index.reserve(10);
    index.add(key_first, vector_first.data(), args...);

    // Default approximate search
    vector_key_t matched_keys[10] = {0};
    distance_t matched_distances[10] = {0};
    std::size_t matched_count = index.search(vector_first.data(), 5, args...).dump_to(matched_keys, matched_distances);

    expect(matched_count == 1);
    expect(matched_keys[0] == key_first);
    expect(std::abs(matched_distances[0]) < 0.01);

    // Add more entries
    index.add(key_second, vector_second.data(), args...);
    index.add(key_third, vector_third.data(), args...);
    expect(index.size() == 3);

    // Perform exact search
    matched_count = index.search(vector_first.data(), 5, args...).dump_to(matched_keys, matched_distances);
    expect(matched_count != 0);

    // Perform filtered exact search, keeping only odd values
    if constexpr (punned_ak) {
        auto is_odd = [](vector_key_t key) -> bool { return (key & 1) != 0; };
        matched_count =
            index.filtered_search(vector_first.data(), 5, is_odd, args...).dump_to(matched_keys, matched_distances);
        expect(matched_count != 0);
        for (std::size_t i = 0; i < matched_count; i++)
            expect(is_odd(matched_keys[i]));
    }

    // Validate scans
    std::size_t count = 0;
    for (auto member : index) {
        vector_key_t id = member.key;
        expect(id >= key_first && id <= key_third);
        count++;
    }
    expect((count == 3));
    expect((index.stats(0).nodes == 3));

    // Check if clustering endpoint compiles
    index.cluster(vector_first.data(), 0, args...);

    // Try removals and replacements
    if constexpr (punned_ak) {
        using labeling_result_t = typename index_t::labeling_result_t;
        labeling_result_t result = index.remove(key_third);
        expect(bool(result));
        expect(index.size() == 2);
        index.add(key_third, vector_third.data(), args...);
        expect(index.size() == 3);
    }

    index.save("tmp.usearch");

    // Check if metadata is retrieved correctly
    if constexpr (punned_ak) {
        auto head = index_dense_metadata_from_path("tmp.usearch");
        expect_eq<std::size_t>(head.head.count_present, 3);
    }

    // Search again over reconstructed index
    index.load("tmp.usearch");
    matched_count = index.search(vector_first.data(), 5, args...).dump_to(matched_keys, matched_distances);
    expect(matched_count == 3);
    expect(matched_keys[0] == key_first);
    expect(std::abs(matched_distances[0]) < 0.01);

    if constexpr (punned_ak) {
        std::size_t dimensions = vector_first.size();
        std::vector<scalar_t> vector_reloaded(dimensions);
        index.get(key_second, vector_reloaded.data());
        expect(std::equal(vector_second.data(), vector_second.data() + dimensions, vector_reloaded.data()));
    }
}

/**
 * Tests the normal operational mode of the library, dealing with a variable length collection
 * of `vectors` with monotonically increasing keys starting from `start_key`.
 *
 * @param index Reference to the index where vectors will be stored and searched.
 * @param start_key The key for the first `vector`, others are generated with increments.
 * @param vectors A collection of vectors to be tested.
 * @param args Additional arguments for configuring search or index operations.
 * @tparam punned_ak Template parameter that determines specific behaviors or checks in the test based on its value.
 * @tparam index_at Type of the index being tested.
 * @tparam scalar_at Data type of the elements in the vectors.
 * @tparam extra_args_at Variadic template parameter types for additional configuration.
 */
template <bool punned_ak, typename index_at, typename scalar_at, typename... extra_args_at>
void test_collection(index_at& index, typename index_at::vector_key_t const start_key,
                     std::vector<std::vector<scalar_at>> const& vectors, extra_args_at&&... args) {

    using scalar_t = scalar_at;
    using index_t = index_at;
    using vector_key_t = typename index_t::vector_key_t;
    using distance_t = typename index_t::distance_t;

    // Generate some keys starting from end,
    // for three vectors from the dataset
    vector_key_t const key_first = start_key;
    std::vector<scalar_at> const& vector_first = vectors[0];
    std::size_t dimensions = vector_first.size();

    // Try batch requests, heavily obersubscribing the CPU cores
    std::size_t executor_threads = std::thread::hardware_concurrency();
    executor_default_t executor(executor_threads);
    index.reserve({vectors.size(), executor.size()});
    executor.fixed(vectors.size(), [&](std::size_t thread, std::size_t task) {
        if constexpr (punned_ak) {
            index.add(start_key + task, vectors[task].data(), args...);
        } else {
            index_update_config_t config;
            config.thread = thread;
            index.add(start_key + task, vectors[task].data(), args..., config);
        }
    });

    // Check for duplicates
    if constexpr (punned_ak) {
        index.reserve({vectors.size() + 1u, executor.size()});
        auto result = index.add(key_first, vector_first.data(), args...);
        expect_eq<std::size_t>(!!result, index.multi());
        result.error.release();

        std::size_t first_key_count = index.count(key_first);
        expect_eq<std::size_t>(first_key_count, 1ul + index.multi());
    }

    // Search again over mapped index
    index.view("tmp.usearch");
    vector_key_t matched_keys[10] = {0};
    distance_t matched_distances[10] = {0};
    std::size_t matched_count = index.search(vector_first.data(), 5, args...).dump_to(matched_keys, matched_distances);
    expect(matched_count == 3);
    expect(matched_keys[0] == key_first);
    expect(std::abs(matched_distances[0]) < 0.01);

    // Check over-sampling beyond the size of the collection
    {
        std::size_t max_possible_matches = vectors.size();
        std::size_t count_requested = max_possible_matches * 4;
        std::vector<vector_key_t> matched_keys(count_requested);
        std::vector<distance_t> matched_distances(count_requested);

        matched_count = index                                                      //
                            .search(vector_first.data(), count_requested, args...) //
                            .dump_to(matched_keys.data(), matched_distances.data());
        expect(matched_count <= max_possible_matches);
        expect(matched_keys[0] == key_first);
        expect(std::abs(matched_distances[0]) < 0.01);

        // Check that all the distance are monotonically rising
        for (std::size_t i = 1; i < matched_count; i++)
            expect(matched_distances[i - 1] <= matched_distances[i]);
    }

    if constexpr (punned_ak) {
        std::vector<scalar_t> vector_reloaded(dimensions);
        index.get(key_first, vector_reloaded.data());
        expect(std::equal(vector_first.data(), vector_first.data() + dimensions, vector_reloaded.data()));

        auto compaction_result = index.compact();
        expect(bool(compaction_result));
    }

    expect(index.memory_usage() > 0);
    expect(index.stats().max_edges > 0);

    // Check metadata
    if constexpr (punned_ak) {
        index_dense_metadata_result_t meta = index_dense_metadata_from_path("tmp.usearch");
        expect(bool(meta));
    }
}

/**
 * Stress-tests the behavior of the type-punned higher-level index under heavy concurrent insertions,
 * removals and updates.
 *
 * @param index Reference to the index where vectors will be stored and searched.
 * @param start_key The key for the first `vector`, others are generated with increments.
 * @param vectors A collection of vectors to be tested.
 * @param executor_threads Number of threads to be used for concurrent operations.
 * @tparam punned_ak Template parameter that determines specific behaviors or checks in the test based on its value.
 * @tparam index_at Type of the index being tested.
 * @tparam scalar_at Data type of the elements in the vectors.
 * @tparam extra_args_at Variadic template parameter types for additional configuration.
 */
template <typename index_at, typename scalar_at, typename... extra_args_at>
void test_punned_concurrent_updates(index_at& index, typename index_at::vector_key_t const start_key,
                                    std::vector<std::vector<scalar_at>> const& vectors, std::size_t executor_threads) {

    using scalar_t = scalar_at;
    using index_t = index_at;
    using vector_key_t = typename index_t::vector_key_t;
    using distance_t = typename index_t::distance_t;

    // Generate some keys starting from end,
    // for three vectors from the dataset
    std::size_t dimensions = vectors[0].size();

    // Try batch requests, heavily obersubscribing the CPU cores
    executor_default_t executor(executor_threads);
    index.reserve({vectors.size(), executor.size()});
    executor.fixed(vectors.size(), [&](std::size_t thread, std::size_t task) {
        using add_result_t = typename index_t::add_result_t;
        add_result_t result = index.add(start_key + task, vectors[task].data());
        expect(bool(result));
    });
    expect_eq<std::size_t>(index.size(), vectors.size());

    // Remove all the keys
    executor.fixed(vectors.size(), [&](std::size_t thread, std::size_t task) {
        using labeling_result_t = typename index_t::labeling_result_t;
        labeling_result_t result = index.remove(start_key + task);
        expect(bool(result));
    });
    expect_eq<std::size_t>(index.size(), 0);

    // Add them back, which under the hood will trigger the `update`
    executor.fixed(vectors.size(), [&](std::size_t thread, std::size_t task) {
        using add_result_t = typename index_t::add_result_t;
        add_result_t result = index.add(start_key + task, vectors[task].data());
        expect(bool(result));
    });
    expect_eq<std::size_t>(index.size(), vectors.size());
}

/**
 * Overloaded function to test cosine similarity index functionality using specific scalar, key, and slot types.
 *
 * This function initializes vectors and an index instance to test cosine similarity calculations and index operations.
 * It involves creating vectors with random values, constructing an index, and verifying that the index operations
 * like search work correctly with respect to the cosine similarity metric.
 *
 * @param collection_size Number of vectors to be included in the test.
 * @param dimensions Number of dimensions each vector should have.
 * @tparam scalar_at Data type of the elements in the vectors.
 * @tparam key_at Data type used for the keys in the index.
 * @tparam slot_at Data type used for slots in the index.
 */
template <typename scalar_at, typename key_at, typename slot_at> //
void test_cosine(std::size_t collection_size, std::size_t dimensions) {

    using scalar_t = scalar_at;
    using vector_key_t = key_at;
    using slot_t = slot_at;

    using index_typed_t = index_gt<float, vector_key_t, slot_t>;
    using member_cref_t = typename index_typed_t::member_cref_t;
    using member_citerator_t = typename index_typed_t::member_citerator_t;

    using vector_of_vectors_t = std::vector<std::vector<scalar_at>>;
    vector_of_vectors_t vector_of_vectors(collection_size);
    for (auto& vector : vector_of_vectors) {
        vector.resize(dimensions);
        std::generate(vector.begin(), vector.end(), [=] { return float(std::rand()) / float(INT_MAX); });
    }

    struct metric_t {
        vector_of_vectors_t const* vector_of_vectors_ptr = nullptr;
        std::size_t dimensions = 0;

        scalar_t const* row(std::size_t i) const noexcept { return (*vector_of_vectors_ptr)[i].data(); }

        float operator()(member_cref_t const& a, member_cref_t const& b) const {
            return metric_cos_gt<scalar_t>{}(row(get_slot(b)), row(get_slot(a)), dimensions);
        }
        float operator()(scalar_t const* some_vector, member_cref_t const& member) const {
            return metric_cos_gt<scalar_t>{}(some_vector, row(get_slot(member)), dimensions);
        }
        float operator()(member_citerator_t const& a, member_citerator_t const& b) const {
            return metric_cos_gt<scalar_t>{}(row(get_slot(b)), row(get_slot(a)), dimensions);
        }
        float operator()(scalar_t const* some_vector, member_citerator_t const& member) const {
            return metric_cos_gt<scalar_t>{}(some_vector, row(get_slot(member)), dimensions);
        }
    };

    // Template:
    for (std::size_t connectivity : {3, 13, 50}) {
        std::printf("- templates with connectivity %zu \n", connectivity);
        metric_t metric{&vector_of_vectors, dimensions};
        index_config_t config(connectivity);

        // Toy example
        if (vector_of_vectors.size() >= 3) {
            aligned_wrapper_gt<index_typed_t> aligned_index(config);
            test_minimal_three_vectors<false, index_typed_t>(*aligned_index.index,     //
                                                             42, vector_of_vectors[0], //
                                                             43, vector_of_vectors[1], //
                                                             44, vector_of_vectors[2], metric);
        }
        // Larger collection
        {
            aligned_wrapper_gt<index_typed_t> aligned_index(config);
            test_collection<false, index_typed_t>(*aligned_index.index, 42, vector_of_vectors, metric);
        }
    }

    // Type-punned:
    for (bool multi : {false, true}) {
        for (std::size_t connectivity : {3, 13, 50}) {
            std::printf("- punned with connectivity %zu \n", connectivity);
            using index_t = index_dense_gt<vector_key_t, slot_t>;
            using index_result_t = typename index_t::state_result_t;
            metric_punned_t metric(dimensions, metric_kind_t::cos_k, scalar_kind<scalar_at>());
            index_dense_config_t config(connectivity);
            config.multi = multi;

            // Toy example
            if (vector_of_vectors.size() >= 3) {
                index_result_t index_result = index_t::make(metric, config);
                index_t& index = index_result.index;
                test_minimal_three_vectors<true>(index,                    //
                                                 42, vector_of_vectors[0], //
                                                 43, vector_of_vectors[1], //
                                                 44, vector_of_vectors[2]);
            }
            // Larger collection
            {
                index_result_t index_result = index_t::make(metric, config);
                index_t& index = index_result.index;
                test_collection<true>(index, 42, vector_of_vectors);
            }
            // Try running benchmarks with a different number of threads
            for (std::size_t threads : {
                     static_cast<std::size_t>(1),
                     static_cast<std::size_t>(2),
                     static_cast<std::size_t>(std::thread::hardware_concurrency()),
                     static_cast<std::size_t>(std::thread::hardware_concurrency() * 4),
                     static_cast<std::size_t>(vector_of_vectors.size()),
                 }) {
                index_result_t index_result = index_t::make(metric, config);
                index_t& index = index_result.index;
                test_punned_concurrent_updates(index, 42, vector_of_vectors, threads);
            }
        }
    }
}

/**
 * Tests the functionality of the Tanimoto coefficient calculation and indexing.
 *
 * Initializes a dense index configured for Tanimoto similarity and fills it with randomly generated binary vectors.
 * It performs concurrent additions of these vectors to the index to ensure thread safety and correctness of concurrent
 * operations.
 *
 * @param dimensions Number of dimensions for the binary vectors.
 * @param connectivity The degree of connectivity for the index configuration.
 * @tparam key_at Data type used for the keys in the index.
 * @tparam slot_at Data type used for slots in the index.
 */
template <typename key_at, typename slot_at> void test_tanimoto(std::size_t dimensions, std::size_t connectivity) {

    using vector_key_t = key_at;
    using slot_t = slot_at;

    using index_punned_t = index_dense_gt<vector_key_t, slot_t>;
    std::size_t words = divide_round_up<CHAR_BIT>(dimensions);
    metric_punned_t metric(words, metric_kind_t::tanimoto_k, scalar_kind_t::b1x8_k);
    index_config_t config(connectivity);
    auto index_result = index_punned_t::make(metric, config);
    expect(bool(index_result));
    index_punned_t& index = index_result.index;

    executor_default_t executor;
    std::size_t batch_size = 1000;
    std::vector<b1x8_t> scalars(batch_size * index.scalar_words());
    std::generate(scalars.begin(), scalars.end(), [] { return static_cast<b1x8_t>(std::rand()); });

    index.reserve({batch_size + index.size(), executor.size()});
    executor.fixed(batch_size, [&](std::size_t thread, std::size_t task) {
        index.add(task + 25000, scalars.data() + index.scalar_words() * task, thread);
    });
}

/**
 * Performs a unit test on the index with a ridiculous variety of configurations and parameters.
 *
 * This test aims to evaluate the index under extreme conditions, including small and potentially invalid parameters for
 * connectivity, dimensions, and other configurations. It tests both the addition of vectors and their retrieval in
 * these edge cases to ensure stability and error handling.
 *
 * @param dimensions Number of dimensions for the vectors.
 * @param connectivity Index connectivity configuration.
 * @param expansion_add Expansion factor during addition operations.
 * @param expansion_search Expansion factor during search operations.
 * @param count_vectors Number of vectors to add to the index.
 * @param count_wanted Number of results wanted from search operations.
 * @tparam key_at Data type used for the keys in the index.
 * @tparam slot_at Data type used for slots in the index.
 */
template <typename key_at, typename slot_at>
void test_absurd(std::size_t dimensions, std::size_t connectivity, std::size_t expansion_add,
                 std::size_t expansion_search, std::size_t count_vectors, std::size_t count_wanted) {

    using vector_key_t = key_at;
    using slot_t = slot_at;

    using index_punned_t = index_dense_gt<vector_key_t, slot_t>;
    metric_punned_t metric(dimensions, metric_kind_t::cos_k, scalar_kind_t::f32_k);
    index_dense_config_t config(connectivity, expansion_add, expansion_search);
    auto index_result = index_punned_t::make(metric, config);
    expect(bool(index_result));
    index_punned_t& index = index_result.index;

    std::size_t count_max = (std::max)(count_vectors, count_wanted);
    std::size_t needed_scalars = count_max * dimensions;
    std::vector<f32_t> scalars(needed_scalars);
    std::generate(scalars.begin(), scalars.end(), [] { return static_cast<f32_t>(std::rand()); });

    expect(index.try_reserve({count_vectors, count_max}));
    index.change_expansion_add(expansion_add);
    index.change_expansion_search(expansion_search);

    // Parallel construction
    {
        executor_default_t executor(count_vectors);
        executor.fixed(count_vectors, [&](std::size_t thread, std::size_t task) {
            expect((bool)index.add(task + 25000, scalars.data() + index.scalar_words() * task, thread));
        });
    }

    // Parallel search
    {
        executor_default_t executor(count_max);
        executor.fixed(count_max, [&](std::size_t thread, std::size_t task) {
            std::vector<vector_key_t> keys(count_wanted);
            std::vector<f32_t> distances(count_wanted);
            auto results = index.search(scalars.data() + index.scalar_words() * task, count_wanted, thread);
            expect((bool)results);
            auto count_found = results.dump_to(keys.data(), distances.data());
            expect(count_found <= count_wanted);
            if (count_vectors && count_wanted)
                expect(count_found > 0);
        });
    }
}

/**
 * Tests the exact search functionality over a dataset of vectors, @b wigthout constructing the index.
 *
 * Generates a dataset of vectors and performs exact search queries to verify that the search results are correct.
 * This function mainly validates the basic functionality of exact searches using a given similarity metric.
 *
 * @param dataset_count Number of vectors in the dataset.
 * @param queries_count Number of query vectors.
 * @param wanted_count Number of top matches required from each query.
 */
void test_exact_search(std::size_t dataset_count, std::size_t queries_count, std::size_t wanted_count) {
    std::size_t dimensions = 10;
    metric_punned_t metric(dimensions, metric_kind_t::cos_k);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::vector<float> dataset(dataset_count * dimensions);
    std::generate(dataset.begin(), dataset.end(), [&] { return dis(gen); });

    exact_search_t search;
    auto results = search(                                                        //
        (byte_t const*)dataset.data(), dataset_count, dimensions * sizeof(float), //
        (byte_t const*)dataset.data(), queries_count, dimensions * sizeof(float), //
        wanted_count, metric);

    for (std::size_t i = 0; i < results.size(); ++i)
        assert(results.at(i)[0].offset == i); // Validate the top match
}

/**
 * Tests handling of variable length sets (group of sorted unique integers), as opposed to @b equi-dimensional vectors.
 *
 * Adds a predefined number of vectors to an index and checks if the size of the index is updated correctly.
 * It serves as a simple verification and showcase of how the same index can be used to handle strings and other types.
 *
 * @param index A reference to the index instance to be tested.
 * @tparam index_at Type of the index being tested.
 */
template <typename key_at, typename slot_at>
void test_sets(std::size_t collection_size, std::size_t min_set_length, std::size_t max_set_length) {

    /// Type of set elements, should support strong ordering
    using set_member_t = std::uint32_t;
    /// Jaccard is a fraction, so let's use a some float
    using set_distance_t = double;

    // Aliasis for the index overload
    using vector_key_t = key_at;
    using slot_t = slot_at;
    using index_t = index_gt<set_distance_t, vector_key_t, slot_t>;

    // Let's allocate some data for indexing
    using set_view_t = span_gt<set_member_t const>;
    using sets_t = std::vector<std::vector<set_member_t>>;
    sets_t sets(collection_size);
    for (auto& set : sets) {
        std::size_t set_size = min_set_length + std::rand() % (max_set_length - min_set_length);
        set.resize(set_size);
        std::size_t upper_bound = (max_set_length - min_set_length) * 3;
        std::generate(set.begin(), set.end(), [=] { return static_cast<set_member_t>(std::rand() % upper_bound); });
        std::sort(set.begin(), set.end());
    }

    // Wrap the data into a proxy object
    struct metric_t {
        using member_cref_t = typename index_t::member_cref_t;
        using member_citerator_t = typename index_t::member_citerator_t;

        sets_t const* sets_ptr = nullptr;

        set_view_t set_at(std::size_t i) const noexcept { return {(*sets_ptr)[i].data(), (*sets_ptr)[i].size()}; }
        set_distance_t between(set_view_t a, set_view_t b) const {
            return metric_jaccard_gt<set_member_t, set_distance_t>{}(a.data(), b.data(), a.size(), b.size());
        }

        set_distance_t operator()(member_cref_t const& a, member_cref_t const& b) const {
            return between(set_at(get_slot(b)), set_at(get_slot(a)));
        }
        set_distance_t operator()(set_view_t some_vector, member_cref_t const& member) const {
            return between(some_vector, set_at(get_slot(member)));
        }
        set_distance_t operator()(member_citerator_t const& a, member_citerator_t const& b) const {
            return between(set_at(get_slot(b)), set_at(get_slot(a)));
        }
        set_distance_t operator()(set_view_t some_vector, member_citerator_t const& member) const {
            return between(some_vector, set_at(get_slot(member)));
        }
    };

    // Perform indexing
    aligned_wrapper_gt<index_t> aligned_index;
    aligned_index.index->reserve(sets.size());
    for (std::size_t i = 0; i < sets.size(); i++)
        aligned_index.index->add(i, set_view_t{sets[i].data(), sets[i].size()}, metric_t{&sets});
    expect(aligned_index.index->size() == sets.size());

    // Perform the search queries
    for (std::size_t i = 0; i < sets.size(); i++) {
        auto results = aligned_index.index->search(set_view_t{sets[i].data(), sets[i].size()}, 5, metric_t{&sets});
        expect(results.size() > 0);
    }
}

/**
 * Tests similarity search over strings using Levenshtein distances
 * implementation from StringZilla.
 *
 * Adds a predefined number of long strings, comparing them.
 *
 * @param index A reference to the index instance to be tested.
 * @tparam index_at Type of the index being tested.
 */
template <typename key_at, typename slot_at> void test_strings() {

    namespace sz = ashvardanian::stringzilla;

    /// Levenshtein distance is an integer
    using levenshtein_distance_t = std::uint64_t;

    // Aliasis for the index overload
    using vector_key_t = key_at;
    using slot_t = slot_at;
    using index_t = index_gt<levenshtein_distance_t, vector_key_t, slot_t>;

    std::string_view str0 = "ACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    std::string_view str1 = "ACG_ACTC_TAC-TACGTA_GTACACG_ACGT";
    std::string_view str2 = "A_GTACTACGTA-GTAC_TACGTACGTA-GTAGT";
    std::string_view str3 = "GTACGTAGT-ACGTACGACGTACGTACG-TACGTAC";
    std::vector<std::string_view> strings({str0, str1, str2, str3});

    // Wrap the data into a proxy object
    struct metric_t {
        using member_cref_t = typename index_t::member_cref_t;
        using member_citerator_t = typename index_t::member_citerator_t;

        std::vector<std::string_view> const* strings_ptr = nullptr;

        std::string_view str_at(std::size_t i) const noexcept { return (*strings_ptr)[i]; }
        levenshtein_distance_t between(std::string_view a, std::string_view b) const {
            sz::string_view asz{a.data(), a.size()};
            sz::string_view bsz{b.data(), b.size()};
            return sz::edit_distance<char const>(asz, bsz);
        }

        levenshtein_distance_t operator()(member_cref_t const& a, member_cref_t const& b) const {
            return between(str_at(get_slot(b)), str_at(get_slot(a)));
        }
        levenshtein_distance_t operator()(std::string_view some_vector, member_cref_t const& member) const {
            return between(some_vector, str_at(get_slot(member)));
        }
        levenshtein_distance_t operator()(member_citerator_t const& a, member_citerator_t const& b) const {
            return between(str_at(get_slot(b)), str_at(get_slot(a)));
        }
        levenshtein_distance_t operator()(std::string_view some_vector, member_citerator_t const& member) const {
            return between(some_vector, str_at(get_slot(member)));
        }
    };

    // Perform indexing
    aligned_wrapper_gt<index_t> aligned_index;
    aligned_index.index->reserve(strings.size());
    for (std::size_t i = 0; i < strings.size(); i++)
        aligned_index.index->add(i, strings[i], metric_t{&strings});
    expect(aligned_index.index->size() == strings.size());

    // Perform the search queries
    for (std::size_t i = 0; i < strings.size(); i++) {
        auto results = aligned_index.index->search(strings[i], 5, metric_t{&strings});
        expect(results.size() > 0);
    }
}

/**
 * @brief Tests replacing and updating entries in index_dense_gt to ensure consistency after modifications.
 */
template <typename key_at, typename slot_at> void test_replacing_update() {

    using vector_key_t = key_at;
    using slot_t = slot_at;

    using index_punned_t = index_dense_gt<vector_key_t, slot_t>;
    metric_punned_t metric(1, metric_kind_t::l2sq_k, scalar_kind_t::f32_k);
    auto index_result = index_punned_t::make(metric);
    expect(bool(index_result));
    index_punned_t& index = index_result.index;

    // Reserve space for 3 entries
    index.reserve(3);
    auto as_ptr = [](float v) {
        static float value;
        value = v;
        return &value;
    };

    // Add 3 entries
    index.add(42, as_ptr(1.1f));
    index.add(43, as_ptr(2.1f));
    index.add(44, as_ptr(3.1f));
    expect_eq<std::size_t>(index.size(), 3);

    // Assert initial state
    auto initial_search = index.search(as_ptr(1.0f), 3);
    expect_eq<std::size_t>(initial_search.size(), 3);
    expect_eq<vector_key_t>(initial_search[0].member.key, 42);
    expect_eq<vector_key_t>(initial_search[1].member.key, 43);
    expect_eq<vector_key_t>(initial_search[2].member.key, 44);

    // Replace the second entry
    index.remove(43);
    index.add(43, as_ptr(2.2f));
    expect_eq<std::size_t>(index.size(), 3);

    // Assert state after replacing second entry
    auto post_second_replacement = index.search(as_ptr(1.0f), 3);
    expect_eq<std::size_t>(post_second_replacement.size(), 3);
    expect_eq<vector_key_t>(post_second_replacement[0].member.key, 42);
    expect_eq<vector_key_t>(post_second_replacement[1].member.key, 43);
    expect_eq<vector_key_t>(post_second_replacement[2].member.key, 44);

    // Replace the first entry
    index.remove(42);
    index.add(42, as_ptr(1.2f));
    expect_eq<std::size_t>(index.size(), 3);

    // Assert state after replacing first entry
    auto final_search = index.search(as_ptr(1.0f), 3, 0);
    expect_eq<std::size_t>(final_search.size(), 3);
    expect_eq<vector_key_t>(final_search[0].member.key, 42);
    expect_eq<vector_key_t>(final_search[1].member.key, 43);
    expect_eq<vector_key_t>(final_search[2].member.key, 44);
}

int main(int, char**) {
    test_uint40();

    // Weird corner cases
    // test_replacing_update<std::int64_t, std::uint32_t>();

    // Exact search without constructing indexes.
    // Great for validating the distance functions.
    std::printf("Testing exact search\n");
    for (std::size_t dataset_count : {10, 100})
        for (std::size_t queries_count : {1, 10})
            for (std::size_t wanted_count : {1, 5})
                test_exact_search(dataset_count, queries_count, wanted_count);

    // Make sure the initializers and the algorithms can work with inadequately small values.
    // Be warned - this combinatorial explosion of tests produces close to __500'000__ tests!
    std::printf("Testing absurd index configs\n");
    for (std::size_t connectivity : {2, 3})      // ! Zero maps to default, one degenerates
        for (std::size_t dimensions : {1, 2, 3}) // ! Zero will raise
            for (std::size_t expansion_add : {0, 1, 2, 3})
                for (std::size_t expansion_search : {0, 1, 2, 3})
                    for (std::size_t count_vectors : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
                        for (std::size_t count_wanted : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
                            // test_absurd<std::int64_t, std::uint32_t>(dimensions, connectivity, expansion_add,
                            //                                          expansion_search, count_vectors, count_wanted);
                            continue;

    // Use just one
    for (std::size_t collection_size : {10, 500})
        for (std::size_t dimensions : {97, 256}) {
            std::printf("Indexing %zu vectors with cos: <float, std::int64_t, std::uint32_t> \n", collection_size);
            test_cosine<float, std::int64_t, std::uint32_t>(collection_size, dimensions);
            std::printf("Indexing %zu vectors with cos: <float, std::int64_t, uint40_t> \n", collection_size);
            test_cosine<float, std::int64_t, uint40_t>(collection_size, dimensions);
        }

    // Test with binaty vectors
    std::printf("Testing binary vectors\n");
    for (std::size_t connectivity : {3, 13, 50})
        for (std::size_t dimensions : {97, 256})
            test_tanimoto<std::int64_t, std::uint32_t>(dimensions, connectivity);

    // Beyond dense equi-dimensional vectors - integer sets
    std::printf("Testing sparse vectors, strings, and sets\n");
    for (std::size_t set_size : {1, 100, 1000})
        test_sets<std::int64_t, std::uint32_t>(set_size, 20, 30);
    test_strings<std::int64_t, std::uint32_t>();

    return 0;
}

/**
 * @brief A trivial test.
 */
#include <algorithm> // `std::shuffle`
#include <cassert>   // `assert`
#include <random>    // `std::default_random_engine`
#include <stdexcept>
#include <vector> // for std::vector

#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>
#include <usearch/index_plugins.hpp>

using namespace unum::usearch;
using namespace unum;

void expect(bool must_be_true) {
    if (!must_be_true)
        throw std::runtime_error("Failed!");
}

template <bool punned_ak, typename index_at, typename scalar_at, typename... extra_args_at>
void test_cosine(index_at& index, std::vector<std::vector<scalar_at>> const& vectors, extra_args_at&&... args) {

    using scalar_t = scalar_at;
    using index_t = index_at;
    using vector_key_t = typename index_t::vector_key_t;
    using distance_t = typename index_t::distance_t;

    // Generate some keys starting from end,
    // for three vectors from the dataset
    vector_key_t const key_max = default_free_value<vector_key_t>() - 1;
    vector_key_t const key_first = key_max - 0;
    vector_key_t const key_second = key_max - 1;
    vector_key_t const key_third = key_max - 2;
    scalar_t const* vector_first = vectors[0].data();
    scalar_t const* vector_second = vectors[1].data();
    scalar_t const* vector_third = vectors[2].data();
    std::size_t dimensions = vectors[0].size();

    // Try checking the empty state
    if constexpr (punned_ak) {
        expect(!index.contains(key_first));
        expect(!index.get(key_first, (f32_t*)nullptr, 1));
    }

    // Add data
    index.reserve(10);
    index.add(key_first, vector_first, args...);

    // Default approximate search
    vector_key_t matched_keys[10] = {0};
    distance_t matched_distances[10] = {0};
    std::size_t matched_count = index.search(vector_first, 5, args...).dump_to(matched_keys, matched_distances);

    expect(matched_count == 1);
    expect(matched_keys[0] == key_first);
    expect(std::abs(matched_distances[0]) < 0.01);

    // Add more entries
    index.add(key_second, vector_second, args...);
    index.add(key_third, vector_third, args...);
    expect(index.size() == 3);

    // Perform exact search
    matched_count = index.search(vector_first, 5, args...).dump_to(matched_keys, matched_distances);

    // Validate scans
    std::size_t count = 0;
    for (auto member : index) {
        vector_key_t id = member.key;
        expect(id <= key_first && id >= key_third);
        count++;
    }
    expect((count == 3));
    expect((index.stats(0).nodes == 3));

    // Check if clustering endpoint compiles
    // index.cluster(vector_first, 0, args...);

    // Try removals and replacements
    if constexpr (punned_ak) {
        using labeling_result_t = typename index_t::labeling_result_t;
        labeling_result_t result = index.remove(key_third);
        expect(bool(result));
        expect(index.size() == 2);
        index.add(key_third, vector_third, args...);
        expect(index.size() == 3);
    }

    // Search again over reconstructed index
    index.save("tmp.usearch");
    index.load("tmp.usearch");
    matched_count = index.search(vector_first, 5, args...).dump_to(matched_keys, matched_distances);
    expect(matched_count == 3);
    expect(matched_keys[0] == key_first);
    expect(std::abs(matched_distances[0]) < 0.01);

    if constexpr (punned_ak) {
        std::vector<scalar_t> vec_recovered_from_load(dimensions);
        index.get(key_second, vec_recovered_from_load.data());
        expect(std::equal(vector_second, vector_second + dimensions, vec_recovered_from_load.data()));
    }

    // Try batch requests
    executor_default_t executor;
    index.reserve({vectors.size(), executor.size()});
    executor.fixed(vectors.size() - 3, [&](std::size_t thread, std::size_t task) {
        if constexpr (punned_ak) {
            index.add(key_max - task - 3, vectors[task + 3].data(), args...);
        } else {
            index_update_config_t config;
            config.thread = thread;
            index.add(key_max - task - 3, vectors[task + 3].data(), args..., config);
        }
    });

    // Check for duplicates
    if constexpr (punned_ak) {
        index.reserve({vectors.size() + 1u, executor.size()});
        auto result = index.add(key_first, vector_first, args...);
        expect(!!result == index.multi());
        result.error.release();

        std::size_t first_key_count = index.count(key_first);
        expect(first_key_count == (1ul + index.multi()));
    }

    // Search again over mapped index
    // file_head_result_t head = index_dense_metadata_from_path("tmp.usearch");
    // expect(head.size == 3);
    index.view("tmp.usearch");
    matched_count = index.search(vector_first, 5, args...).dump_to(matched_keys, matched_distances);
    expect(matched_count == 3);
    expect(matched_keys[0] == key_first);
    expect(std::abs(matched_distances[0]) < 0.01);

    if constexpr (punned_ak) {
        std::vector<scalar_t> vec_recovered_from_view(dimensions);
        index.get(key_second, vec_recovered_from_view.data());
        expect(std::equal(vector_second, vector_second + dimensions, vec_recovered_from_view.data()));

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

template <typename scalar_at, typename key_at, typename slot_at> //
void test_cosine(std::size_t collection_size, std::size_t dimensions) {

    using scalar_t = scalar_at;
    using vector_key_t = key_at;
    using slot_t = slot_at;

    // using index_storage_t = storage_proxy_t<vector_key_t, slot_t>;
    using index_storage_t = dummy_storage_single_threaded<vector_key_t, slot_t>;
    using index_typed_t = index_gt<index_storage_t, float, vector_key_t, slot_t>;
    using member_cref_t = typename index_typed_t::member_cref_t;
    using member_citerator_t = typename index_typed_t::member_citerator_t;

    std::vector<std::vector<scalar_at>> matrix(collection_size);
    for (std::vector<scalar_at>& vector : matrix) {
        vector.resize(dimensions);
        std::generate(vector.begin(), vector.end(), [=] { return float(std::rand()) / float(INT_MAX); });
    }

    struct metric_t {
        std::vector<std::vector<scalar_at>> const* matrix_ptr;
        std::size_t dimensions;

        scalar_t const* row(std::size_t i) const noexcept { return (*matrix_ptr)[i].data(); }

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
        metric_t metric{&matrix, dimensions};
        index_config_t config(connectivity);
        std::vector<node_at<vector_key_t, slot_t>> nodes;
        bitset_gt nodes_mutexes;
        // index_storage_t storage{&nodes, &nodes_mutexes, config};
        index_storage_t storage{config};
        index_typed_t index_typed(storage, config);
        test_cosine<false>(index_typed, matrix, metric);
    }

    // Type-punned:
    for (bool multi : {false, true}) {
        for (std::size_t connectivity : {3, 13, 50}) {
            std::printf("- punned with connectivity %zu \n", connectivity);
            using index_t = index_dense_gt<vector_key_t, slot_t>;
            metric_punned_t metric(dimensions, metric_kind_t::cos_k, scalar_kind<scalar_at>());
            index_dense_config_t config(connectivity);
            config.multi = multi;
            index_t index;
            {
                index_t index_tmp1 = index_t::make(metric, config);
                // move construction
                index_t index_tmp2 = std::move(index_tmp1);
                // move assignment
                index = std::move(index_tmp2);
            }
            test_cosine<true>(index, matrix);
        }
    }
}

template <typename key_at, typename slot_at> void test_tanimoto(std::size_t dimensions, std::size_t connectivity) {

    using vector_key_t = key_at;
    using slot_t = slot_at;

    using index_punned_t = index_dense_gt<vector_key_t, slot_t>;
    std::size_t words = divide_round_up<CHAR_BIT>(dimensions);
    metric_punned_t metric(words, metric_kind_t::tanimoto_k, scalar_kind_t::b1x8_k);
    index_config_t config(connectivity);
    index_punned_t index = index_punned_t::make(metric, config);

    executor_default_t executor;
    std::size_t batch_size = 1000;
    std::vector<b1x8_t> scalars(batch_size * index.scalar_words());
    std::generate(scalars.begin(), scalars.end(), [] { return static_cast<b1x8_t>(std::rand()); });

    index.reserve({batch_size + index.size(), executor.size()});
    executor.fixed(batch_size, [&](std::size_t thread, std::size_t task) {
        index.add(task + 25000, scalars.data() + index.scalar_words() * task, thread);
    });
}

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

template <typename index_at> void test_sets(index_at&& index) {

    using index_t = typename std::remove_reference<index_at>::type;
    using scalar_t = typename index_t::scalar_t;
    using view_t = span_gt<scalar_t const>;

    scalar_t vec0[] = {10, 20};
    scalar_t vec1[] = {10, 15, 20};
    scalar_t vec2[] = {10, 20, 30, 35};

    index.reserve(10);
    index.add(42, view_t{&vec0[0], 2ul});
    index.add(43, view_t{&vec1[0], 3ul});
    index.add(44, view_t{&vec2[0], 4ul});

    expect(index.size() == 3);
}

template <typename index_at> void test_sets_moved(index_config_t const& config) {
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

int main(int, char**) {

    for (std::size_t dataset_count : {10, 100})
        for (std::size_t queries_count : {1, 10})
            for (std::size_t wanted_count : {1, 5})
                test_exact_search(dataset_count, queries_count, wanted_count);

    for (std::size_t collection_size : {10, 500})
        for (std::size_t dimensions : {97, 256}) {
            std::printf("Indexing %zu vectors with cos: <float, std::int64_t, std::uint32_t> \n", collection_size);
            test_cosine<float, std::int64_t, std::uint32_t>(collection_size, dimensions);
            std::printf("Indexing %zu vectors with cos: <float, std::int64_t, uint40_t> \n", collection_size);
            test_cosine<float, std::int64_t, uint40_t>(collection_size, dimensions);
        }

    for (std::size_t connectivity : {3, 13, 50})
        for (std::size_t dimensions : {97, 256})
            test_tanimoto<std::int64_t, std::uint32_t>(dimensions, connectivity);

    return 0;
}

/**
 * @brief A trivial test.
 */
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

#include <usearch/index.hpp>
#include <usearch/index_punned_helpers.hpp>

using namespace unum::usearch;
using namespace unum;

void expect(bool must_be_true) {
    if (!must_be_true)
        throw std::runtime_error("Failed!");
}

template <typename scalar_at, typename label_at, typename id_at> void test3d() {

    using scalar_t = scalar_at;
    using label_t = label_at;
    using id_t = id_at;

    scalar_t vec[3] = {10, 20, 15};
    scalar_t vec_a[3] = {15, 16, 17};
    scalar_t vec_b[3] = {16, 17, 18};

    std::unordered_map<label_t, scalar_t const*> vecs;
    vecs[42] = &vec[0];
    vecs[43] = &vec_a[0];
    vecs[44] = &vec_b[0];

    struct metric_t {
        std::unordered_map<label_t, scalar_t const*> const* vecs = nullptr;

        float operator()(label_t a, label_t b) const { return cos_gt<scalar_t>{}(vecs->at(b), vecs->at(a), 3); }
        float operator()(scalar_t const* some_vector, label_t stored_label) const {
            return cos_gt<scalar_t>{}(some_vector, vecs->at(stored_label), 3);
        }
    };

    using index_t = index_gt<metric_t, label_t, id_t>;
    using distance_t = typename index_t::distance_t;

    metric_t metric{{&vecs}};
    index_t index(index_config_t{}, metric);
    index.reserve(10);
    index.add(42, &vec[0]);

    // Default approximate search
    label_t matched_labels[10] = {0};
    distance_t matched_distances[10] = {0};
    std::size_t matched_count = index.search(&vec[0], 5).dump_to(matched_labels, matched_distances);

    expect(matched_count == 1);
    expect(matched_labels[0] == 42);
    expect(std::abs(matched_distances[0]) < 0.01);

    // Add more entries
    search_config_t search_config;
    search_config.exact = true;
    index.add(43, &vec_a[0]);
    index.add(44, &vec_b[0]);
    expect(index.size() == 3);

    // Perform exact search
    matched_count = index.search(&vec[0], 5, search_config).dump_to(matched_labels, matched_distances);

    // Validate scans
    std::size_t count = 0;
    for (auto member : index) {
        label_t id = member.label;
        expect(id >= 42 && id <= 44);
        count++;
    }
    expect((count == 3));
    expect((index.stats(0).nodes == 3));

    // Search again over reconstructed index
    index.save("tmp.usearch");
    index.load("tmp.usearch");
    matched_count = index.search(&vec[0], 5).dump_to(matched_labels, matched_distances);
    expect(matched_count == 3);
    expect(matched_labels[0] == 42);
    expect(std::abs(matched_distances[0]) < 0.01);

    // Search again over mapped index
    // file_head_result_t head = index_metadata("tmp.usearch");
    // expect(head.size == 3);
    index.view("tmp.usearch");
    matched_count = index.search(&vec[0], 5).dump_to(matched_labels, matched_distances);
    expect(matched_count == 3);
    expect(matched_labels[0] == 42);
    expect(std::abs(matched_distances[0]) < 0.01);

    expect(index.memory_usage() > 0);
    expect(index.stats().max_edges > 0);
}

template <typename scalar_at, typename index_at> void test3d_punned(index_at&& index) {

    using scalar_t = scalar_at;
    using view_t = span_gt<scalar_t const>;
    using span_t = span_gt<scalar_at>;

    scalar_t vec42[3] = {10, 20, 15};
    scalar_t vec43[3] = {19, 22, 11};

    index.reserve(10);
    index.add(42, view_t{&vec42[0], 3ul});

    // Reconstruct
    scalar_t vec42_reconstructed[3] = {0, 0, 0};
    index.get(42, span_t{&vec42_reconstructed[0], 3ul});
    expect(vec42_reconstructed[0] == vec42[0]);
    expect(vec42_reconstructed[1] == vec42[1]);
    expect(vec42_reconstructed[2] == vec42[2]);

    index.add(43, view_t{&vec43[0], 3ul});
    expect(index.size() == 2);
    index.remove(43);
    expect(index.size() == 1);
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

    // static_assert(!std::is_same<index_gt<hamming_gt<>>::value_type, std::true_type>());
    // static_assert(!std::is_same<index_gt<tanimoto_gt<>>::value_type, std::true_type>());
    // static_assert(!std::is_same<index_gt<sorensen_gt<>>::value_type, std::true_type>());
    // static_assert(!std::is_same<index_gt<jaccard_gt<>>::value_type, std::true_type>());
    // static_assert(!std::is_same<index_gt<pearson_correlation_gt<>>::value_type, std::true_type>());
    // static_assert(!std::is_same<index_gt<haversine_gt<>>::value_type, std::true_type>());

    using big_point_id_t = std::int64_t;

    test3d<float, big_point_id_t, std::uint32_t>();

    // test3d<double>(index_gt<cos_gt<double>, big_point_id_t, std::uint32_t>{config2});
    // test3d<double>(index_gt<l2sq_gt<double>, big_point_id_t, std::uint32_t>{config2});

    // test3d<float>(punned_small_t::make(3, metric_kind_t::cos_k, config1));
    // test3d<float>(punned_small_t::make(3, metric_kind_t::l2sq_k, config1));

    // test3d<double>(punned_small_t::make(3, metric_kind_t::cos_k, config1));
    // test3d<double>(punned_small_t::make(3, metric_kind_t::l2sq_k, config1));

    // test3d_punned<float>(punned_small_t::make(3, metric_kind_t::cos_k, config1));
    // test3d_punned<float>(punned_small_t::make(3, metric_kind_t::l2sq_k, config1));

    // test_sets(index_gt<jaccard_gt<std::int32_t, float>, big_point_id_t, std::uint32_t>{config1});
    // test_sets(index_gt<jaccard_gt<std::int64_t, float>, big_point_id_t, std::uint32_t>{config1});

    // test_sets_moved<index_gt<jaccard_gt<std::int32_t, float>, big_point_id_t, std::uint32_t>>(config1);
    // test_sets_moved<index_gt<jaccard_gt<std::int64_t, float>, big_point_id_t, std::uint32_t>>(config1);

    return 0;
}
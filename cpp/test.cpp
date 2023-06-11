/**
 * @brief A trivial test.
 */
#include <algorithm>
#include <stdexcept>

#include <usearch/index_punned_dense.hpp>

using namespace unum::usearch;
using namespace unum;

using point_id_t = std::int64_t;

void expect(bool must_be_true) {
    if (!must_be_true)
        throw std::runtime_error("Failed!");
}

template <typename scalar_at, typename index_at> void test3d(index_at&& index) {

    using scalar_t = scalar_at;
    using view_t = span_gt<scalar_t const>;
    using distance_t = typename index_at::distance_t;

    scalar_t vec[3] = {10, 20, 15};
    scalar_t vec_a[3] = {15, 16, 17};
    scalar_t vec_b[3] = {16, 17, 18};

    index.reserve(10);
    index.add(42, view_t{&vec[0], 3ul});

    // Default approximate search
    point_id_t matched_labels[10] = {0};
    distance_t matched_distances[10] = {0};
    std::size_t matched_count = index.search(view_t{&vec[0], 3ul}, 5).dump_to(matched_labels, matched_distances);

    expect(matched_count == 1);
    expect(matched_labels[0] == 42);
    expect(std::abs(matched_distances[0]) < 0.01);

    // Add more entries
    search_config_t search_config;
    search_config.exact = true;
    index.add(43, view_t{&vec_a[0], 3ul});
    index.add(44, view_t{&vec_b[0], 3ul});
    expect(index.size() == 3);

    // Perform exact search
    matched_count = index.search(view_t{&vec[0], 3ul}, 5, search_config).dump_to(matched_labels, matched_distances);

    // Validate scans
    std::size_t count = 0;
    for (auto member : index) {
        point_id_t id = member.label;
        expect(id >= 42 && id <= 44);
        count++;
    }
    expect((count == 3));
    expect((index.stats(0).nodes == 3));

    // Search again over reconstructed index
    index.save("tmp.usearch");
    index.load("tmp.usearch");
    matched_count = index.search(view_t{&vec[0], 3ul}, 5).dump_to(matched_labels, matched_distances);
    expect(matched_count == 3);
    expect(matched_labels[0] == 42);
    expect(std::abs(matched_distances[0]) < 0.01);

    // Search again over mapped index
    index.view("tmp.usearch");
    matched_count = index.search(view_t{&vec[0], 3ul}, 5).dump_to(matched_labels, matched_distances);
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

    scalar_t vec[3] = {10, 20, 15};

    index.reserve(10);
    index.add(42, view_t{&vec[0], 3ul});

    // Reconstruct
    scalar_t vec_reconstructed[3] = {0, 0, 0};
    index.reconstruct(42, span_t{&vec_reconstructed[0], 3ul});
    expect(vec_reconstructed[0] == vec[0]);
    expect(vec_reconstructed[1] == vec[1]);
    expect(vec_reconstructed[2] == vec[2]);
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

template <typename index_at> void test_sets_moved() {
    {
        index_at index;
        test_sets(index);
    }
    {
        index_at index;
        index.reserve(1);
        test_sets(index_at(std::move(index)));
    }
    {
        index_at index;
        index.reserve(1);
        index_at index_moved = std::move(index);
        test_sets(index_moved);
    }
}

int main(int, char**) {

    test3d<float>(index_gt<cos_gt<float>, point_id_t, std::uint32_t, float>{});
    test3d<float>(index_gt<l2sq_gt<float>, point_id_t, std::uint32_t, float>{});

    test3d<double>(index_gt<cos_gt<double>, point_id_t, std::uint32_t, double>{});
    test3d<double>(index_gt<l2sq_gt<double>, point_id_t, std::uint32_t, double>{});

    test3d<float>(punned_small_t::make(3, metric_kind_t::cos_k));
    test3d<float>(punned_small_t::make(3, metric_kind_t::l2sq_k));

    test3d<double>(punned_small_t::make(3, metric_kind_t::cos_k));
    test3d<double>(punned_small_t::make(3, metric_kind_t::l2sq_k));

    test3d_punned<float>(punned_small_t::make(3, metric_kind_t::cos_k));
    test3d_punned<float>(punned_small_t::make(3, metric_kind_t::l2sq_k));

    test_sets(index_gt<jaccard_gt<std::int32_t, float>, point_id_t, std::uint32_t, std::int32_t>{});
    test_sets(index_gt<jaccard_gt<std::int64_t, float>, point_id_t, std::uint32_t, std::int64_t>{});

    test_sets_moved<index_gt<jaccard_gt<std::int32_t, float>, point_id_t, std::uint32_t, std::int32_t>>();
    test_sets_moved<index_gt<jaccard_gt<std::int64_t, float>, point_id_t, std::uint32_t, std::int64_t>>();

    return 0;
}
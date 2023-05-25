/**
 * @brief A trivial test.
 */
#include <algorithm>
#include <cassert>

#include "punned.hpp"

using namespace unum::usearch;
using namespace unum;

using point_id_t = std::int64_t;

template <typename scalar_at, typename index_at> void test3d(index_at&& index) {
    using span_t = span_gt<scalar_at const>;
    using distance_t = typename index_at::distance_t;

    scalar_at vec[3] = {10, 20, 15};
    scalar_at vec_a[3] = {15, 16, 17};
    scalar_at vec_b[3] = {16, 17, 18};

    index.reserve(10);
    index.add(42, span_t{&vec[0], 3ul});

    // Default approximate search
    point_id_t matched_labels[10] = {0};
    distance_t matched_distances[10] = {0};
    std::size_t matched_count = index.search(span_t{&vec[0], 3ul}, 5).dump_to(matched_labels, matched_distances);

    assert(matched_count == 1);
    assert(matched_labels[0] == 42);
    assert(std::abs(matched_distances[0]) < 0.01);

    // Add more entries
    search_config_t search_config;
    search_config.exact = true;
    index.add(43, span_t{&vec_a[0], 3ul});
    index.add(44, span_t{&vec_b[0], 3ul});
    assert(index.size() == 3);

    // Perform exact search
    matched_count = index.search(span_t{&vec[0], 3ul}, 5, search_config).dump_to(matched_labels, matched_distances);

    // Validate scans
    std::size_t count = 0;
    for (auto member : index) {
        point_id_t id = member.label;
        assert(id >= 42 && id <= 44);
        count++;
    }
    assert((count == 3));
    assert((index.stats(0).nodes == 3));

    // Search again over reconstructed index
    index.save("tmp.usearch");
    index.load("tmp.usearch");
    matched_count = index.search(span_t{&vec[0], 3ul}, 5).dump_to(matched_labels, matched_distances);
    assert(matched_count == 3);
    assert(matched_labels[0] == 42);
    assert(std::abs(matched_distances[0]) < 0.01);

    // Search again over mapped index
    index.view("tmp.usearch");
    matched_count = index.search(span_t{&vec[0], 3ul}, 5).dump_to(matched_labels, matched_distances);
    assert(matched_count == 3);
    assert(matched_labels[0] == 42);
    assert(std::abs(matched_distances[0]) < 0.01);
}

int main(int, char**) {

    test3d<float>(index_gt<cos_gt<float>, point_id_t, std::uint32_t, float>{});
    test3d<float>(index_gt<l2sq_gt<float>, point_id_t, std::uint32_t, float>{});

    test3d<double>(index_gt<cos_gt<double>, point_id_t, std::uint32_t, double>{});
    test3d<double>(index_gt<l2sq_gt<double>, point_id_t, std::uint32_t, double>{});

    test3d<float>(punned_small_t::cos(3));
    test3d<float>(punned_small_t::l2sq(3));

    test3d<double>(punned_small_t::cos(3));
    test3d<double>(punned_small_t::l2sq(3));

    return 0;
}
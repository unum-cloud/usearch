/**
 * @brief A trivial test.
 */
#include <cassert>

#include "punned.hpp"

using namespace unum::usearch;
using namespace unum;

using point_id_t = std::int64_t;

template <typename scalar_at, typename index_at> void test3d(index_at&& index) {
    using span_t = span_gt<scalar_at const>;
    using distance_t = typename index_at::distance_t;

    scalar_at vec[3] = {10, 20, 15};

    index.reserve(10);
    index.add(42, span_t{&vec[0], 3ul});

    point_id_t matched_labels[10] = {0};
    distance_t matched_distances[10] = {0};
    std::size_t matched_count = index.search(span_t{&vec[0], 3ul}, 5).dump_to(matched_labels, matched_distances);

    assert(matched_count == 1);
    assert(matched_labels[0] == 42);
    assert(std::abs(matched_distances[0]) < 0.01);

    index.save("tmp.usearch");
    index.load("tmp.usearch");
    index.view("tmp.usearch");
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
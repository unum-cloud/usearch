/**
 * @brief A trivial test.
 */
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

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
void test3d(index_at& index, scalar_at* vectors, extra_args_at&&... args) {

    using scalar_t = scalar_at;
    using index_t = index_at;
    using key_t = typename index_t::key_t;
    using distance_t = typename index_t::distance_t;

    scalar_t* vec = &vectors[0];
    scalar_t* vec_a = &vectors[3];
    scalar_t* vec_b = &vectors[6];

    index.reserve(10);
    index.add(42, &vec[0], args...);

    // Default approximate search
    key_t matched_labels[10] = {0};
    distance_t matched_distances[10] = {0};
    std::size_t matched_count = index.search(&vec[0], 5, args...).dump_to(matched_labels, matched_distances);

    expect(matched_count == 1);
    expect(matched_labels[0] == 42);
    expect(std::abs(matched_distances[0]) < 0.01);

    // Add more entries
    index_search_config_t search_config;
    search_config.exact = true;
    index.add(43, &vec_a[0], args...);
    index.add(44, &vec_b[0], args...);
    expect(index.size() == 3);

    // Perform exact search
    matched_count = index.search(&vec[0], 5, args..., search_config).dump_to(matched_labels, matched_distances);

    // Validate scans
    std::size_t count = 0;
    for (auto member : index) {
        key_t id = member.key;
        expect(id >= 42 && id <= 44);
        count++;
    }
    expect((count == 3));
    expect((index.stats(0).nodes == 3));

    // Search again over reconstructed index
    index.save("tmp.usearch");
    index.load("tmp.usearch");
    matched_count = index.search(&vec[0], 5, args...).dump_to(matched_labels, matched_distances);
    expect(matched_count == 3);
    expect(matched_labels[0] == 42);
    expect(std::abs(matched_distances[0]) < 0.01);

    if constexpr (punned_ak) {
        scalar_t vec_recovered_from_load[3];
        index.get(42, &vec_recovered_from_load[0]);
        expect(std::equal(&vec[0], &vec[3], &vec_recovered_from_load[0]));
    }

    // Try batch requests
    if constexpr (punned_ak) {
        executor_default_t executor;
        std::size_t batch_size = 1000;
        std::vector<scalar_at> scalars(batch_size * index.dimensions_upper_bound());
        index.reserve({batch_size + index.size(), executor.size()});
        executor.execute_bulk(batch_size, [&](std::size_t thread, std::size_t task) {
            index_add_config_t config;
            config.thread = thread;
            index.add(task + 25000, scalars.data() + index.dimensions_upper_bound() * task, config);
        });
    }

    // Search again over mapped index
    // file_head_result_t head = index_metadata("tmp.usearch");
    // expect(head.size == 3);
    index.view("tmp.usearch");
    matched_count = index.search(&vec[0], 5, args...).dump_to(matched_labels, matched_distances);
    expect(matched_count == 3);
    expect(matched_labels[0] == 42);
    expect(std::abs(matched_distances[0]) < 0.01);

    if constexpr (punned_ak) {
        scalar_t vec_recovered_from_view[3];
        index.get(42, &vec_recovered_from_view[0]);
        expect(std::equal(&vec[0], &vec[3], &vec_recovered_from_view[0]));
    }

    expect(index.memory_usage() > 0);
    expect(index.stats().max_edges > 0);

    // Check metadata
    if constexpr (punned_ak) {
        index_dense_metadata_result_t meta = index_metadata("tmp.usearch");
        expect(bool(meta));
    }
}

template <typename scalar_at, typename key_at, typename id_at> void test3d() {

    using scalar_t = scalar_at;
    using key_t = key_at;
    using slot_t = id_at;

    using index_typed_t = index_gt<float, key_t, slot_t>;
    using member_cref_t = typename index_typed_t::member_cref_t;
    using member_citerator_t = typename index_typed_t::member_citerator_t;

    using vecs_table_t = scalar_t[3][3];
    vecs_table_t vecs_table = {
        {10, 11, 12},
        {13, 14, 15},
        {16, 17, 18},
    };

    struct metric_t {
        vecs_table_t const* vecs = nullptr;

        scalar_t const* row(std::size_t i) const noexcept { return (*vecs)[i]; }

        float operator()(member_cref_t const& a, member_cref_t const& b) const {
            return metric_cos_gt<scalar_t>{}(row(get_slot(b)), row(get_slot(a)), 3ul);
        }
        float operator()(scalar_t const* some_vector, member_cref_t const& member) const {
            return metric_cos_gt<scalar_t>{}(some_vector, row(get_slot(member)), 3ul);
        }
        float operator()(member_citerator_t const& a, member_citerator_t const& b) const {
            return metric_cos_gt<scalar_t>{}(row(get_slot(b)), row(get_slot(a)), 3ul);
        }
        float operator()(scalar_t const* some_vector, member_citerator_t const& member) const {
            return metric_cos_gt<scalar_t>{}(some_vector, row(get_slot(member)), 3ul);
        }
    };

    metric_t metric{{&vecs_table}};
    index_typed_t index_typed(index_config_t{});
    test3d<false>(index_typed, (scalar_at*)vecs_table, metric);

    using index_punned_t = index_dense_gt<key_t, slot_t>;
    metric_punned_t metric_punned(3 * sizeof(scalar_at), metric_kind_t::cos_k, scalar_kind<scalar_at>());
    index_punned_t index_punned = index_punned_t::make(metric_punned, index_config_t{});
    test3d<true>(index_punned, (scalar_at*)vecs_table);
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
    test3d<float, std::int64_t, std::uint32_t>();
    return 0;
}
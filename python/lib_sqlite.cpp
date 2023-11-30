/**
 *  @brief      SQLite3 bindings for USearch.
 *  @file       lib_sqlite.cpp
 *  @author     Ash Vardanian
 *  @date       November 28, 2023
 *  @copyright  Copyright (c) 2023
 */
#include <stringzilla/stringzilla.h>

#include <charconv>

#include <sqlite3ext.h>
SQLITE_EXTENSION_INIT1

template <scalar_kind_t scalar_kind_ak> struct parsed_scalar_kind_gt {
    using type = f32_t;
    static constexpr scalar_kind_t kind = scalar_kind_t::f32_k;
};

template <> struct parsed_scalar_kind_gt<scalar_kind_t::f64_k> {
    using type = f64_t;
    static constexpr scalar_kind_t kind = scalar_kind_t::f64_k;
};

template <scalar_kind_t scalar_kind_ak, metric_kind_t metric_kind_ak>
static void sqlite_dense(sqlite3_context* context, int argc, sqlite3_value** argv) {
    if (argc != 2) {
        sqlite3_result_error(context, "Wrong number of arguments", -1);
        return;
    }

    int type1 = sqlite3_value_type(argv[0]);
    int type2 = sqlite3_value_type(argv[1]);
    if (type1 != type2) {
        sqlite3_result_error(context, "Arguments types don't match", -1);
        return;
    }

    if (type1 == SQLITE_NULL)
        sqlite3_result_error(context, "Only BLOB types are supported, received a NULL", -1);
    else if (type1 == SQLITE_INTEGER)
        sqlite3_result_error(context, "Only BLOB types are supported, received an INTEGER", -1);
    else if (type1 == SQLITE_FLOAT)
        sqlite3_result_error(context, "Only BLOB types are supported, received a FLOAT", -1);
    // Textual JSON objects
    else if (type1 == SQLITE_TEXT) {
        char const* vec1 = (char const*)sqlite3_value_text(argv[0]);
        char const* vec2 = (char const*)sqlite3_value_text(argv[1]);
        size_t bytes1 = (size_t)sqlite3_value_bytes(argv[0]);
        size_t bytes2 = (size_t)sqlite3_value_bytes(argv[1]);
        size_t commas1 = sz_count_char_swar(vec1, bytes1, ",");
        size_t commas2 = sz_count_char_swar(vec2, bytes2, ",");
        if (commas1 != commas2) {
            sqlite3_result_error(context, "Vectors have different number of dimensions", -1);
            return;
        }

        // Valid JSON array of numbers would be packed into [] square brackets
        if (bytes1 && vec1[0] == '[')
            ++vec1, --bytes1;
        if (bytes2 && vec2[0] == '[')
            ++vec2, --bytes2;
        // We don't have to trim the end
        // if (bytes1 && vec1[bytes1 - 1] == ']')
        //     --bytes1;
        // if (bytes2 && vec2[bytes2 - 1] == ']')
        //     --bytes2;

        // Allocate vectors on stack and parse strings into them
        using scalar_t = typename parsed_scalar_kind_gt<scalar_kind_ak>::type;
        size_t dimensions = commas1 + 1;
        scalar_t parsed1[dimensions], parsed2[dimensions];
        for (size_t i = 0; i != dimensions; ++i) {
            // Skip whitespace
            while (bytes1 && vec1[0] == ' ')
                ++vec1, --bytes1;
            while (bytes2 && vec2[0] == ' ')
                ++vec2, --bytes2;

            // Parse the floating-point numbers
            std::from_chars_result result1 = std::from_chars(vec1, vec1 + bytes1, parsed1[i]);
            std::from_chars_result result2 = std::from_chars(vec2, vec2 + bytes2, parsed1[i]);
            if (result1.ec != std::errc() || result2.ec != std::errc()) {
                sqlite3_result_error(context, "Number can't be parsed", -1);
                return;
            }

            // Skip the number, whitespaces, and commas
            bytes1 -= result1.ptr - vec1;
            bytes2 -= result2.ptr - vec2;
            vec1 = result1.ptr;
            vec2 = result2.ptr;
            while (bytes1 && (vec1[0] == ' ' || vec1[0] == ','))
                ++vec1, --bytes1;
            while (bytes2 && (vec2[0] == ' ' || vec2[0] == ','))
                ++vec2, --bytes2;
        }

        // Compute the distance itself
        metric_t metric = metric_t(dimensions, metric_kind_ak, parsed_scalar_kind_gt<scalar_kind_ak>::kind);
        distance_punned_t distance =
            metric(reinterpret_cast<byte_t const*>(parsed1), reinterpret_cast<byte_t const*>(parsed2));
        sqlite3_result_double(context, distance);
    }
    // Binary objects
    else if (type1 == SQLITE_BLOB) {
        void const* vec1 = sqlite3_value_blob(argv[0]);
        void const* vec2 = sqlite3_value_blob(argv[1]);
        int bytes1 = sqlite3_value_bytes(argv[0]);
        int bytes2 = sqlite3_value_bytes(argv[1]);
        if (bytes1 != bytes2) {
            sqlite3_result_error(context, "Vectors have different number of dimensions", -1);
            return;
        }

        std::size_t dimensions = (size_t)(bytes1)*CHAR_BIT / bits_per_scalar(scalar_kind_ak);
        metric_t metric = metric_t(dimensions, metric_kind_ak, scalar_kind_ak);
        distance_punned_t distance =
            metric(reinterpret_cast<byte_t const*>(vec1), reinterpret_cast<byte_t const*>(vec2));
        sqlite3_result_double(context, distance);
    } else
        sqlite3_result_error(context, "Unknown argument types", -1);
}

extern "C" PYBIND11_MAYBE_UNUSED PYBIND11_EXPORT int sqlite3_compiled_init( //
    sqlite3* db,                                                            //
    char** error_message,                                                   //
    sqlite3_api_routines const* api) {
    SQLITE_EXTENSION_INIT2(api)

    int flags = SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_INNOCUOUS;

    sqlite3_create_function(db, "distance_hamming_binary", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::b1x8_k, metric_kind_t::hamming_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_jaccard_binary", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::b1x8_k, metric_kind_t::jaccard_k>, NULL, NULL);

    sqlite3_create_function(db, "distance_sqeuclidean_f64", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::f64_k, metric_kind_t::l2sq_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_cosine_f64", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::f64_k, metric_kind_t::cos_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_inner_f64", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::f64_k, metric_kind_t::ip_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_divergence_f64", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::f64_k, metric_kind_t::divergence_k>, NULL, NULL);

    sqlite3_create_function(db, "distance_sqeuclidean_f32", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::f32_k, metric_kind_t::l2sq_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_cosine_f32", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::f32_k, metric_kind_t::cos_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_inner_f32", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::f32_k, metric_kind_t::ip_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_divergence_f32", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::f32_k, metric_kind_t::divergence_k>, NULL, NULL);

    sqlite3_create_function(db, "distance_sqeuclidean_f16", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::f16_k, metric_kind_t::l2sq_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_cosine_f16", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::f16_k, metric_kind_t::cos_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_inner_f16", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::f16_k, metric_kind_t::ip_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_divergence_f16", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::f16_k, metric_kind_t::divergence_k>, NULL, NULL);

    sqlite3_create_function(db, "distance_sqeuclidean_i8", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::i8_k, metric_kind_t::l2sq_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_cosine_i8", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::i8_k, metric_kind_t::cos_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_inner_i8", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::i8_k, metric_kind_t::ip_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_divergence_i8", 2, flags, NULL,
                            sqlite_dense<scalar_kind_t::i8_k, metric_kind_t::divergence_k>, NULL, NULL);

    return SQLITE_OK;
}

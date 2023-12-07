/**
 *  @brief      SQLite3 bindings for USearch.
 *  @file       lib_sqlite.cpp
 *  @author     Ash Vardanian
 *  @date       November 28, 2023
 *  @copyright  Copyright (c) 2023
 */
#include <stringzilla.h>

#include <charconv> // `std::from_chars`
#include <cstdlib>  // `std::strtod`

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

    if (argc < 2) {
        sqlite3_result_error(context, "Distance function expects at least two arguments", -1);
        return;
    }

    int type1 = sqlite3_value_type(argv[0]);
    int type2 = sqlite3_value_type(argv[1]);

    // Our primary case is having two BLOBs containing dense vector representations.
    if (argc == 2 && type1 == SQLITE_BLOB && type2 == SQLITE_BLOB) {
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
    }

    // Worst case is to have JSON arrays or comma-separated values
    else if (argc == 2 && type1 == SQLITE_TEXT && type2 == SQLITE_TEXT) {
        char* vec1 = (char*)sqlite3_value_text(argv[0]);
        char* vec2 = (char*)sqlite3_value_text(argv[1]);
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
                // Sadly, most modern compilers don't support the `std::from_chars` yet
#if __cpp_lib_to_chars
            std::from_chars_result result1 = std::from_chars(vec1, vec1 + bytes1, parsed1[i]);
            std::from_chars_result result2 = std::from_chars(vec2, vec2 + bytes2, parsed2[i]);
            if (result1.ec != std::errc() || result2.ec != std::errc()) {
                sqlite3_result_error(context, "Number can't be parsed", -1);
                return;
            }
            bytes1 -= result1.ptr - vec1;
            bytes2 -= result2.ptr - vec2;
            vec1 = (char*)result1.ptr;
            vec2 = (char*)result2.ptr;
#else
            char* parsed1_end = vec1 + bytes1;
            parsed1[i] = std::strtod(vec1, &parsed1_end);
            char* parsed2_end = vec2 + bytes2;
            parsed2[i] = std::strtod(vec2, &parsed2_end);
            if (vec1 == parsed1_end || vec2 == parsed2_end) {
                sqlite3_result_error(context, "Number can't be parsed", -1);
                return;
            }
            bytes1 -= parsed1_end - vec1;
            bytes2 -= parsed2_end - vec2;
            vec1 = parsed1_end;
            vec2 = parsed2_end;
#endif

            // Skip the whitespaces and commas
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

    // Less efficient, yet still common case is to have many scalar columns
    else if (argc % 2 == 0) {

        // Allocate vectors on stack and parse floating-point values into them
        using scalar_t = typename parsed_scalar_kind_gt<scalar_kind_ak>::type;
        size_t dimensions = argc / 2;
        scalar_t parsed1[dimensions], parsed2[dimensions];
        for (size_t i = 0; i != dimensions; ++i) {
            switch (sqlite3_value_type(argv[i])) {
            case SQLITE_FLOAT: parsed1[i] = sqlite3_value_double(argv[i]); break;
            case SQLITE_INTEGER: parsed1[i] = sqlite3_value_int(argv[i]); break;
            case SQLITE_NULL: parsed1[i] = 0; break;
            default:
                sqlite3_result_error(context, "Scalar columns may only contain 32-bit integers, floats, or NULLs.", -1);
                return;
            }
            switch (sqlite3_value_type(argv[dimensions + i])) {
            case SQLITE_FLOAT: parsed2[i] = sqlite3_value_double(argv[dimensions + i]); break;
            case SQLITE_INTEGER: parsed2[i] = sqlite3_value_int(argv[dimensions + i]); break;
            case SQLITE_NULL: parsed2[i] = 0; break;
            default:
                sqlite3_result_error(context, "Scalar columns may only contain 32-bit integers, floats, or NULLs.", -1);
                return;
            }
        }

        // Compute the distance itself
        metric_t metric = metric_t(dimensions, metric_kind_ak, parsed_scalar_kind_gt<scalar_kind_ak>::kind);
        distance_punned_t distance =
            metric(reinterpret_cast<byte_t const*>(parsed1), reinterpret_cast<byte_t const*>(parsed2));
        sqlite3_result_double(context, distance);
    }
    // Unsupported arguments combination
    else {
        sqlite3_result_error(context, "Number of columns in two vectors must be divisible by two", -1);
    }
}

extern "C" PYBIND11_MAYBE_UNUSED PYBIND11_EXPORT int sqlite3_compiled_init( //
    sqlite3* db,                                                            //
    char** error_message,                                                   //
    sqlite3_api_routines const* api) {
    SQLITE_EXTENSION_INIT2(api)

    int flags = SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_INNOCUOUS;
    int num_params = -1; // Any number will be accepted

    sqlite3_create_function(db, "distance_hamming_binary", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::b1x8_k, metric_kind_t::hamming_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_jaccard_binary", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::b1x8_k, metric_kind_t::jaccard_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_haversine_meters", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::f64_k, metric_kind_t::haversine_k>, NULL, NULL);
    // sqlite3_create_function(db, "distance_levenshtein", num_params, flags, NULL,
    //                         sqlite_dense<scalar_kind_t::u8_k, metric_kind_t::haversine_k>, NULL, NULL);

    sqlite3_create_function(db, "distance_sqeuclidean_f64", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::f64_k, metric_kind_t::l2sq_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_cosine_f64", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::f64_k, metric_kind_t::cos_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_inner_f64", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::f64_k, metric_kind_t::ip_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_divergence_f64", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::f64_k, metric_kind_t::divergence_k>, NULL, NULL);

    sqlite3_create_function(db, "distance_sqeuclidean_f32", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::f32_k, metric_kind_t::l2sq_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_cosine_f32", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::f32_k, metric_kind_t::cos_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_inner_f32", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::f32_k, metric_kind_t::ip_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_divergence_f32", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::f32_k, metric_kind_t::divergence_k>, NULL, NULL);

    sqlite3_create_function(db, "distance_sqeuclidean_f16", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::f16_k, metric_kind_t::l2sq_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_cosine_f16", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::f16_k, metric_kind_t::cos_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_inner_f16", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::f16_k, metric_kind_t::ip_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_divergence_f16", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::f16_k, metric_kind_t::divergence_k>, NULL, NULL);

    sqlite3_create_function(db, "distance_sqeuclidean_i8", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::i8_k, metric_kind_t::l2sq_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_cosine_i8", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::i8_k, metric_kind_t::cos_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_inner_i8", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::i8_k, metric_kind_t::ip_k>, NULL, NULL);
    sqlite3_create_function(db, "distance_divergence_i8", num_params, flags, NULL,
                            sqlite_dense<scalar_kind_t::i8_k, metric_kind_t::divergence_k>, NULL, NULL);

    return SQLITE_OK;
}

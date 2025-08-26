#include <jni.h>

#include "cloud_unum_usearch_Index.h"

#include <thread>

#include <usearch/index_dense.hpp>

using namespace unum::usearch;
using namespace unum;

using f32_span_t = unum::usearch::span_gt<float>;
using f64_span_t = unum::usearch::span_gt<double>;
using i8_span_t = unum::usearch::span_gt<std::int8_t>;
static_assert(sizeof(jlong) == sizeof(index_dense_t::vector_key_t));

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1create( //
    JNIEnv* env, jclass,                                         //
    jstring metric, jstring quantization,                        //
    jlong dimensions, jlong capacity, jlong connectivity,        //
    jlong expansion_add, jlong expansion_search) {

    jlong result{};
    char const* metric_cstr{};
    char const* quantization_cstr{};
    try {

        metric_cstr = (*env).GetStringUTFChars(metric, 0);
        std::size_t metric_length = (*env).GetStringUTFLength(metric);
        quantization_cstr = (*env).GetStringUTFChars(quantization, 0);
        std::size_t quantization_length = (*env).GetStringUTFLength(quantization);
        metric_kind_t metric_kind = metric_from_name(metric_cstr, metric_length);
        scalar_kind_t quantization = scalar_kind_from_name(quantization_cstr, quantization_length);
        index_dense_config_t config(static_cast<std::size_t>(connectivity), static_cast<std::size_t>(expansion_add),
                                    static_cast<std::size_t>(expansion_search));
        metric_punned_t metric(static_cast<std::size_t>(dimensions), metric_kind, quantization);
        if (metric.missing()) {
            jclass jc = (*env).FindClass("java/lang/Error");
            if (jc)
                (*env).ThrowNew(jc, "Failed to initialize the metric!");
            goto cleanup;
        }

        index_dense_t index = index_dense_t::make(metric, config);
        if (!index.try_reserve(static_cast<std::size_t>(capacity))) {
            jclass jc = (*env).FindClass("java/lang/Error");
            if (jc)
                (*env).ThrowNew(jc, "Failed to reserve desired capacity!");
        } else {
            index_dense_t* result_ptr = new index_dense_t(std::move(index));
            std::memcpy(&result, &result_ptr, sizeof(jlong));
        }

    } catch (...) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, "Failed to initialize the vector index!");
    }

cleanup:
    (*env).ReleaseStringUTFChars(metric, metric_cstr);
    (*env).ReleaseStringUTFChars(quantization, quantization_cstr);
    return result;
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1createFromFile(JNIEnv* env, jclass, jstring path,
                                                                        jboolean view) {
    char const* path_cstr = env->GetStringUTFChars(path, 0);
    index_dense_t::state_result_t make_result = index_dense_t::make(path_cstr, view);
    env->ReleaseStringUTFChars(path, path_cstr);
    if (!make_result) {
        jclass jc = env->FindClass("java/lang/Error");
        if (jc) {
            env->ThrowNew(jc, make_result.error.release());
        }
    }
    index_dense_t* result_ptr = new index_dense_t(std::move(make_result.index));
    jlong result;
    std::memcpy(&result, &result_ptr, sizeof(jlong));
    return result;
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1save(JNIEnv* env, jclass, jlong c_ptr, jstring path) {
    char const* path_cstr = (*env).GetStringUTFChars(path, 0);
    serialization_result_t result = reinterpret_cast<index_dense_t*>(c_ptr)->save(path_cstr);
    if (!result) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, result.error.release());
    }
    (*env).ReleaseStringUTFChars(path, path_cstr);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1load(JNIEnv* env, jclass, jlong c_ptr, jstring path) {
    char const* path_cstr = (*env).GetStringUTFChars(path, 0);
    serialization_result_t result = reinterpret_cast<index_dense_t*>(c_ptr)->load(path_cstr);
    if (!result) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, result.error.release());
    }
    (*env).ReleaseStringUTFChars(path, path_cstr);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1view(JNIEnv* env, jclass, jlong c_ptr, jstring path) {
    char const* path_cstr = (*env).GetStringUTFChars(path, 0);
    serialization_result_t result = reinterpret_cast<index_dense_t*>(c_ptr)->view(path_cstr);
    if (!result) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, result.error.release());
    }
    (*env).ReleaseStringUTFChars(path, path_cstr);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1destroy(JNIEnv*, jclass, jlong c_ptr) {
    delete reinterpret_cast<index_dense_t*>(c_ptr);
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1size(JNIEnv*, jclass, jlong c_ptr) {
    return reinterpret_cast<index_dense_t*>(c_ptr)->size();
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1connectivity(JNIEnv*, jclass, jlong c_ptr) {
    return reinterpret_cast<index_dense_t*>(c_ptr)->connectivity();
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1dimensions(JNIEnv*, jclass, jlong c_ptr) {
    return reinterpret_cast<index_dense_t*>(c_ptr)->dimensions();
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1capacity(JNIEnv*, jclass, jlong c_ptr) {
    return reinterpret_cast<index_dense_t*>(c_ptr)->capacity();
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1reserve(JNIEnv* env, jclass, jlong c_ptr, jlong capacity) {
    if (!reinterpret_cast<index_dense_t*>(c_ptr)->try_reserve(static_cast<std::size_t>(capacity))) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, "Failed to grow vector index!");
    }
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1add_1f32( //
    JNIEnv* env, jclass, jlong c_ptr, jlong key, jfloatArray vector) {

    jfloat* vector_data = (*env).GetFloatArrayElements(vector, 0);
    jsize vector_length = (*env).GetArrayLength(vector);

    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    size_t dimensions = index->dimensions();

    using vector_key_t = typename index_dense_t::vector_key_t;
    using add_result_t = typename index_dense_t::add_result_t;

    // Handle both single and batch processing uniformly
    if (vector_length % dimensions != 0) {
        (*env).ReleaseFloatArrayElements(vector, vector_data, 0);
        jclass jc = (*env).FindClass("java/lang/IllegalArgumentException");
        if (jc)
            (*env).ThrowNew(jc, "Vector length must be a multiple of dimensions");
        return;
    }

    size_t num_vectors = vector_length / dimensions;
    for (size_t i = 0; i < num_vectors; i++) {
        f32_span_t vector_span = f32_span_t{vector_data + i * dimensions, dimensions};
        add_result_t result = index->add(static_cast<vector_key_t>(key + i), vector_span);
        if (!result) {
            (*env).ReleaseFloatArrayElements(vector, vector_data, 0);
            jclass jc = (*env).FindClass("java/lang/Error");
            if (jc)
                (*env).ThrowNew(jc, result.error.release());
            return;
        }
    }
    (*env).ReleaseFloatArrayElements(vector, vector_data, 0);
}

JNIEXPORT jfloatArray JNICALL Java_cloud_unum_usearch_Index_c_1get(JNIEnv* env, jclass, jlong c_ptr, jlong key) {
    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    size_t dim = index->dimensions();
    std::unique_ptr<jfloat[]> vector(new jfloat[dim]);
    if (index->get(key, vector.get()) == 0) {
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "key not found");
        }
    }
    jfloatArray jvector = env->NewFloatArray(dim);
    if (jvector == nullptr) { // out of memory
        return nullptr;
    }
    env->SetFloatArrayRegion(jvector, 0, dim, vector.get());
    return jvector;
}

JNIEXPORT jlongArray JNICALL Java_cloud_unum_usearch_Index_c_1search_1f32( //
    JNIEnv* env, jclass, jlong c_ptr, jfloatArray vector, jlong wanted) {

    jfloat* vector_data = (*env).GetFloatArrayElements(vector, 0);
    jsize vector_dims = (*env).GetArrayLength(vector);
    f32_span_t vector_span = f32_span_t{vector_data, static_cast<std::size_t>(vector_dims)};

    using vector_key_t = typename index_dense_t::vector_key_t;
    using search_result_t = typename index_dense_t::search_result_t;

    search_result_t result = 
        reinterpret_cast<index_dense_t*>(c_ptr)->search(vector_span, static_cast<std::size_t>(wanted));
    (*env).ReleaseFloatArrayElements(vector, vector_data, 0);

    if (result) {
        std::size_t found = result.count;
        jlongArray matches = (*env).NewLongArray(found);
        if (matches == NULL)
            return NULL;

        jlong* matches_data = (*env).GetLongArrayElements(matches, 0);
        result.dump_to(reinterpret_cast<vector_key_t*>(matches_data));
        (*env).ReleaseLongArrayElements(matches, matches_data, JNI_COMMIT);

        return matches;
    } else {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, result.error.release());
        return NULL;
    }
}

JNIEXPORT jboolean JNICALL Java_cloud_unum_usearch_Index_c_1remove(JNIEnv* env, jclass, jlong c_ptr, jlong key) {
    using vector_key_t = typename index_dense_t::vector_key_t;
    using labeling_result_t = typename index_dense_t::labeling_result_t;
    labeling_result_t result = reinterpret_cast<index_dense_t*>(c_ptr)->remove(static_cast<vector_key_t>(key));
    if (!result) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, result.error.release());
    }
    return result.completed;
}

JNIEXPORT jboolean JNICALL Java_cloud_unum_usearch_Index_c_1rename(JNIEnv* env, jclass, jlong c_ptr, jlong from,
                                                                   jlong to) {
    using vector_key_t = typename index_dense_t::vector_key_t;
    using labeling_result_t = typename index_dense_t::labeling_result_t;
    labeling_result_t result =
        reinterpret_cast<index_dense_t*>(c_ptr)->rename(static_cast<vector_key_t>(from), static_cast<vector_key_t>(to));
    if (!result) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, result.error.release());
    }
    return result.completed;
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1add_1f64( //
    JNIEnv* env, jclass, jlong c_ptr, jlong key, jdoubleArray vector) {

    jdouble* vector_data = (*env).GetDoubleArrayElements(vector, 0);
    jsize vector_length = (*env).GetArrayLength(vector);

    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    size_t dimensions = index->dimensions();

    using vector_key_t = typename index_dense_t::vector_key_t;
    using add_result_t = typename index_dense_t::add_result_t;

    // Handle both single and batch processing uniformly
    if (vector_length % dimensions != 0) {
        (*env).ReleaseDoubleArrayElements(vector, vector_data, 0);
        jclass jc = (*env).FindClass("java/lang/IllegalArgumentException");
        if (jc)
            (*env).ThrowNew(jc, "Vector length must be a multiple of dimensions");
        return;
    }
    
    size_t num_vectors = vector_length / dimensions;
    for (size_t i = 0; i < num_vectors; i++) {
        f64_span_t vector_span = f64_span_t{vector_data + i * dimensions, dimensions};
        add_result_t result = index->add(static_cast<vector_key_t>(key + i), vector_span);
        if (!result) {
            (*env).ReleaseDoubleArrayElements(vector, vector_data, 0);
            jclass jc = (*env).FindClass("java/lang/Error");
            if (jc)
                (*env).ThrowNew(jc, result.error.release());
            return;
        }
    }
    (*env).ReleaseDoubleArrayElements(vector, vector_data, 0);
}

JNIEXPORT jlongArray JNICALL Java_cloud_unum_usearch_Index_c_1search_1f64( //
    JNIEnv* env, jclass, jlong c_ptr, jdoubleArray vector, jlong wanted) {

    jdouble* vector_data = (*env).GetDoubleArrayElements(vector, 0);
    jsize vector_dims = (*env).GetArrayLength(vector);
    f64_span_t vector_span = f64_span_t{vector_data, static_cast<std::size_t>(vector_dims)};

    using vector_key_t = typename index_dense_t::vector_key_t;
    using search_result_t = typename index_dense_t::search_result_t;

    search_result_t result = 
        reinterpret_cast<index_dense_t*>(c_ptr)->search(vector_span, static_cast<std::size_t>(wanted));
    (*env).ReleaseDoubleArrayElements(vector, vector_data, 0);

    if (result) {
        std::size_t found = result.count;
        jlongArray matches = (*env).NewLongArray(found);
        if (matches == NULL)
            return NULL;

        jlong* matches_data = (*env).GetLongArrayElements(matches, 0);
        result.dump_to(reinterpret_cast<vector_key_t*>(matches_data));
        (*env).ReleaseLongArrayElements(matches, matches_data, JNI_COMMIT);

        return matches;
    } else {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, result.error.release());
        return NULL;
    }
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1get_1into_1f64(JNIEnv* env, jclass, jlong c_ptr, jlong key,
                                                                       jdoubleArray buffer) {
    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    jdouble* buffer_data = (*env).GetDoubleArrayElements(buffer, 0);

    if (index->get(key, buffer_data) == 0) {
        (*env).ReleaseDoubleArrayElements(buffer, buffer_data, JNI_ABORT);
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "key not found");
        }
        return;
    }
    (*env).ReleaseDoubleArrayElements(buffer, buffer_data, JNI_COMMIT);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1add_1i8( //
    JNIEnv* env, jclass, jlong c_ptr, jlong key, jbyteArray vector) {

    jbyte* vector_data = (*env).GetByteArrayElements(vector, 0);
    jsize vector_length = (*env).GetArrayLength(vector);
    
    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    size_t dimensions = index->dimensions();
    
    using vector_key_t = typename index_dense_t::vector_key_t;
    using add_result_t = typename index_dense_t::add_result_t;

    // Handle both single and batch processing uniformly
    if (vector_length % dimensions != 0) {
        (*env).ReleaseByteArrayElements(vector, vector_data, 0);
        jclass jc = (*env).FindClass("java/lang/IllegalArgumentException");
        if (jc)
            (*env).ThrowNew(jc, "Vector length must be a multiple of dimensions");
        return;
    }
    
    size_t num_vectors = vector_length / dimensions;
    for (size_t i = 0; i < num_vectors; i++) {
        i8_span_t vector_span = i8_span_t{reinterpret_cast<std::int8_t*>(vector_data + i * dimensions), dimensions};
        add_result_t result = index->add(static_cast<vector_key_t>(key + i), vector_span);
        if (!result) {
            (*env).ReleaseByteArrayElements(vector, vector_data, 0);
            jclass jc = (*env).FindClass("java/lang/Error");
            if (jc)
                (*env).ThrowNew(jc, result.error.release());
            return;
        }
    }
    (*env).ReleaseByteArrayElements(vector, vector_data, 0);
}

JNIEXPORT jlongArray JNICALL Java_cloud_unum_usearch_Index_c_1search_1i8( //
    JNIEnv* env, jclass, jlong c_ptr, jbyteArray vector, jlong wanted) {

    jbyte* vector_data = (*env).GetByteArrayElements(vector, 0);
    jsize vector_dims = (*env).GetArrayLength(vector);
    i8_span_t vector_span = i8_span_t{reinterpret_cast<std::int8_t*>(vector_data), static_cast<std::size_t>(vector_dims)};

    using vector_key_t = typename index_dense_t::vector_key_t;
    using search_result_t = typename index_dense_t::search_result_t;

    search_result_t result = 
        reinterpret_cast<index_dense_t*>(c_ptr)->search(vector_span, static_cast<std::size_t>(wanted));
    (*env).ReleaseByteArrayElements(vector, vector_data, 0);

    if (result) {
        std::size_t found = result.count;
        jlongArray matches = (*env).NewLongArray(found);
        if (matches == NULL)
            return NULL;

        jlong* matches_data = (*env).GetLongArrayElements(matches, 0);
        result.dump_to(reinterpret_cast<vector_key_t*>(matches_data));
        (*env).ReleaseLongArrayElements(matches, matches_data, JNI_COMMIT);

        return matches;
    } else {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, result.error.release());
        return NULL;
    }
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1get_1into_1i8(JNIEnv* env, jclass, jlong c_ptr, jlong key,
                                                                      jbyteArray buffer) {
    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    jbyte* buffer_data = (*env).GetByteArrayElements(buffer, 0);

    if (index->get(key, reinterpret_cast<std::int8_t*>(buffer_data)) == 0) {
        (*env).ReleaseByteArrayElements(buffer, buffer_data, JNI_ABORT);
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "key not found");
        }
        return;
    }
    (*env).ReleaseByteArrayElements(buffer, buffer_data, JNI_COMMIT);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1get_1into_1f32(JNIEnv* env, jclass, jlong c_ptr, jlong key,
                                                                       jfloatArray buffer) {
    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    jfloat* buffer_data = (*env).GetFloatArrayElements(buffer, 0);

    if (index->get(key, buffer_data) == 0) {
        (*env).ReleaseFloatArrayElements(buffer, buffer_data, JNI_ABORT);
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "key not found");
        }
        return;
    }
    (*env).ReleaseFloatArrayElements(buffer, buffer_data, JNI_COMMIT);
}

#include <jni.h>

#include "cloud_unum_usearch_Index.h"

#include <thread>

#include <usearch/index_dense.hpp>

using namespace unum::usearch;
using namespace unum;

using float_span_t = unum::usearch::span_gt<float>;
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

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1add( //
    JNIEnv* env, jclass, jlong c_ptr, jlong key, jfloatArray vector) {

    jfloat* vector_data = (*env).GetFloatArrayElements(vector, 0);
    jsize vector_dims = (*env).GetArrayLength(vector);
    float_span_t vector_span = float_span_t{vector_data, static_cast<std::size_t>(vector_dims)};

    using vector_key_t = typename index_dense_t::vector_key_t;
    using add_result_t = typename index_dense_t::add_result_t;

    add_result_t result = reinterpret_cast<index_dense_t*>(c_ptr)->add(static_cast<vector_key_t>(key), vector_span);
    if (!result) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, result.error.release());
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

JNIEXPORT jlongArray JNICALL Java_cloud_unum_usearch_Index_c_1search( //
    JNIEnv* env, jclass, jlong c_ptr, jfloatArray vector, jlong wanted) {

    jfloat* vector_data = (*env).GetFloatArrayElements(vector, 0);
    jsize vector_dims = (*env).GetArrayLength(vector);
    float_span_t vector_span = float_span_t{vector_data, static_cast<std::size_t>(vector_dims)};

    using vector_key_t = typename index_dense_t::vector_key_t;
    using search_result_t = typename index_dense_t::search_result_t;

    search_result_t result =
        reinterpret_cast<index_dense_t*>(c_ptr)->search(vector_span, static_cast<std::size_t>(wanted));
    (*env).ReleaseFloatArrayElements(vector, vector_data, 0);

    if (result) {
        std::size_t found = result.count;
        jlongArray matches = (*env).NewLongArray(found);
        if (matches == NULL)
            // The exception is already set by JNI.
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

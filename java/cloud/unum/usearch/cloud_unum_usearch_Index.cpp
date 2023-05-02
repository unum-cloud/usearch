#include <jni.h>

#include "cloud_unum_usearch_Index.h"

#include <thread>

#include "advanced.hpp"

using namespace unum::usearch;
using namespace unum;

using label_t = jint;
using distance_t = punned_distance_t;
using native_index_t = auto_index_gt<label_t>;
using span_t = span_gt<float>;

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1create( //
    JNIEnv* env, jclass,                                         //
    jstring metric, jstring accuracy,                            //
    jlong dimensions, jlong capacity, jlong connectivity,        //
    jlong expansion_add, jlong expansion_search) {

    jlong result{};
    char const* metric_cstr{};
    char const* accuracy_cstr{};
    try {

        config_t config;
        config.expansion_add = static_cast<std::size_t>(expansion_add);
        config.expansion_search = static_cast<std::size_t>(expansion_search);
        config.connectivity = static_cast<std::size_t>(connectivity);
        config.max_elements = static_cast<std::size_t>(capacity);
        config.max_threads_add = std::thread::hardware_concurrency();
        config.max_threads_search = std::thread::hardware_concurrency();

        metric_cstr = (*env).GetStringUTFChars(metric, 0);
        std::size_t metric_length = (*env).GetStringUTFLength(metric);
        accuracy_cstr = (*env).GetStringUTFChars(accuracy, 0);
        std::size_t accuracy_length = (*env).GetStringUTFLength(accuracy);

        accuracy_t accuracy = accuracy_from_name(accuracy_cstr, accuracy_length);
        native_index_t index = index_from_name<native_index_t>( //
            metric_cstr, metric_length, static_cast<std::size_t>(dimensions), accuracy, config);

        native_index_t* result_ptr = new native_index_t(std::move(index));
        std::memcpy(&result, &result_ptr, sizeof(jlong));
    } catch (...) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, "Failed to initialize the vector index!");
    }

    (*env).ReleaseStringUTFChars(metric, metric_cstr);
    (*env).ReleaseStringUTFChars(accuracy, accuracy_cstr);
    return result;
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1save(JNIEnv* env, jclass, jlong c_ptr, jstring path) {
    char const* path_cstr{};
    try {
        path_cstr = (*env).GetStringUTFChars(path, 0);
        reinterpret_cast<native_index_t*>(c_ptr)->save(path_cstr);
    } catch (...) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, "Failed to dump vector index to path!");
    }
    (*env).ReleaseStringUTFChars(path, path_cstr);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1load(JNIEnv* env, jclass, jlong c_ptr, jstring path) {
    char const* path_cstr{};
    try {
        path_cstr = (*env).GetStringUTFChars(path, 0);
        reinterpret_cast<native_index_t*>(c_ptr)->load(path_cstr);
    } catch (...) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, "Failed to load vector index from path!");
    }
    (*env).ReleaseStringUTFChars(path, path_cstr);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1view(JNIEnv* env, jclass, jlong c_ptr, jstring path) {
    char const* path_cstr{};
    try {
        path_cstr = (*env).GetStringUTFChars(path, 0);
        reinterpret_cast<native_index_t*>(c_ptr)->view(path_cstr);
    } catch (...) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, "Failed to view vector index from path!");
    }
    (*env).ReleaseStringUTFChars(path, path_cstr);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1destroy(JNIEnv*, jclass, jlong c_ptr) {
    delete reinterpret_cast<native_index_t*>(c_ptr);
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1size(JNIEnv*, jclass, jlong c_ptr) {
    return reinterpret_cast<native_index_t*>(c_ptr)->size();
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1connectivity(JNIEnv*, jclass, jlong c_ptr) {
    return reinterpret_cast<native_index_t*>(c_ptr)->connectivity();
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1dimensions(JNIEnv*, jclass, jlong c_ptr) {
    return reinterpret_cast<native_index_t*>(c_ptr)->dimensions();
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1capacity(JNIEnv*, jclass, jlong c_ptr) {
    return reinterpret_cast<native_index_t*>(c_ptr)->capacity();
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1reserve(JNIEnv* env, jclass, jlong c_ptr, jlong capacity) {
    try {
        return reinterpret_cast<native_index_t*>(c_ptr)->reserve(static_cast<std::size_t>(capacity));
    } catch (...) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, "Failed to grow vector index!");
    }
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1add( //
    JNIEnv* env, jclass, jlong c_ptr, jint label, jfloatArray vector) {

    jfloat* vector_data{};
    try {
        vector_data = (*env).GetFloatArrayElements(vector, 0);
        jsize vector_dims = (*env).GetArrayLength(vector);
        span_t vector_span = span_t{vector_data, static_cast<std::size_t>(vector_dims)};
        reinterpret_cast<native_index_t*>(c_ptr)->add(label, vector_span);
    } catch (...) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, "Failed to insert a new point in vector index!");
    }
    (*env).ReleaseFloatArrayElements(vector, vector_data, 0);
}

JNIEXPORT jintArray JNICALL Java_cloud_unum_usearch_Index_c_1search( //
    JNIEnv* env, jclass, jlong c_ptr, jfloatArray vector, jlong wanted) {

    jintArray matches;
    matches = (*env).NewIntArray(wanted);
    if (matches == NULL)
        return NULL;

    jint* matches_data = (jint*)std::malloc(sizeof(jint) * wanted);
    if (matches_data == NULL)
        return NULL;

    jfloat* vector_data{};
    std::size_t found{};
    try {
        vector_data = (*env).GetFloatArrayElements(vector, 0);
        jsize vector_dims = (*env).GetArrayLength(vector);
        span_t vector_span = span_t{vector_data, static_cast<std::size_t>(vector_dims)};
        found = reinterpret_cast<native_index_t*>(c_ptr)->search( //
            vector_span, static_cast<std::size_t>(wanted), matches_data, NULL);
        (*env).SetIntArrayRegion(matches, 0, found, matches_data);
    } catch (...) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, "Failed to find in vector index!");
    }
    (*env).ReleaseFloatArrayElements(vector, vector_data, 0);
    std::free(matches_data);
    return matches;
}

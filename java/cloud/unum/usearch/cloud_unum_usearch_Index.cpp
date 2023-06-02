#include <jni.h>

#include "cloud_unum_usearch_Index.h"

#include <thread>

#include "punned.hpp"

using namespace unum::usearch;
using namespace unum;

using label_t = jint;
using distance_t = punned_distance_t;
using punned_t = punned_gt<label_t>;
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

        index_config_t config;
        config.expansion_add = static_cast<std::size_t>(expansion_add);
        config.expansion_search = static_cast<std::size_t>(expansion_search);
        config.connectivity = static_cast<std::size_t>(connectivity);

        metric_cstr = (*env).GetStringUTFChars(metric, 0);
        std::size_t metric_length = (*env).GetStringUTFLength(metric);
        accuracy_cstr = (*env).GetStringUTFChars(accuracy, 0);
        std::size_t accuracy_length = (*env).GetStringUTFLength(accuracy);

        accuracy_t accuracy = accuracy_from_name(accuracy_cstr, accuracy_length);
        common_metric_kind_t metric_kind = common_metric_from_name(metric_cstr, metric_length);
        punned_t index = make_punned<punned_t>(metric_kind, static_cast<std::size_t>(dimensions), accuracy, config);
        if (!index.reserve(static_cast<std::size_t>(capacity))) {
            jclass jc = (*env).FindClass("java/lang/Error");
            if (jc)
                (*env).ThrowNew(jc, "Failed to reserve desired capacity!");
        } else {
            punned_t* result_ptr = new punned_t(std::move(index));
            std::memcpy(&result, &result_ptr, sizeof(jlong));
        }

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
    char const* path_cstr = (*env).GetStringUTFChars(path, 0);
    if (!reinterpret_cast<punned_t*>(c_ptr)->save(path_cstr)) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, "Failed to dump vector index to path!");
    }
    (*env).ReleaseStringUTFChars(path, path_cstr);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1load(JNIEnv* env, jclass, jlong c_ptr, jstring path) {
    char const* path_cstr = (*env).GetStringUTFChars(path, 0);
    if (!reinterpret_cast<punned_t*>(c_ptr)->load(path_cstr)) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, "Failed to load vector index from path!");
    }
    (*env).ReleaseStringUTFChars(path, path_cstr);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1view(JNIEnv* env, jclass, jlong c_ptr, jstring path) {
    char const* path_cstr = (*env).GetStringUTFChars(path, 0);
    if (!reinterpret_cast<punned_t*>(c_ptr)->view(path_cstr)) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, "Failed to view vector index from path!");
    }
    (*env).ReleaseStringUTFChars(path, path_cstr);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1destroy(JNIEnv*, jclass, jlong c_ptr) {
    delete reinterpret_cast<punned_t*>(c_ptr);
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1size(JNIEnv*, jclass, jlong c_ptr) {
    return reinterpret_cast<punned_t*>(c_ptr)->size();
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1connectivity(JNIEnv*, jclass, jlong c_ptr) {
    return reinterpret_cast<punned_t*>(c_ptr)->connectivity();
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1dimensions(JNIEnv*, jclass, jlong c_ptr) {
    return reinterpret_cast<punned_t*>(c_ptr)->dimensions();
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1capacity(JNIEnv*, jclass, jlong c_ptr) {
    return reinterpret_cast<punned_t*>(c_ptr)->capacity();
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1reserve(JNIEnv* env, jclass, jlong c_ptr, jlong capacity) {
    if (!reinterpret_cast<punned_t*>(c_ptr)->reserve(static_cast<std::size_t>(capacity))) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, "Failed to grow vector index!");
    }
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1add( //
    JNIEnv* env, jclass, jlong c_ptr, jint label, jfloatArray vector) {

    jfloat* vector_data = (*env).GetFloatArrayElements(vector, 0);
    jsize vector_dims = (*env).GetArrayLength(vector);
    span_t vector_span = span_t{vector_data, static_cast<std::size_t>(vector_dims)};

    if (!reinterpret_cast<punned_t*>(c_ptr)->add(label, vector_span)) {
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

    jfloat* vector_data = (*env).GetFloatArrayElements(vector, 0);
    jsize vector_dims = (*env).GetArrayLength(vector);
    span_t vector_span = span_t{vector_data, static_cast<std::size_t>(vector_dims)};
    search_result_t result = reinterpret_cast<punned_t*>(c_ptr)->search(vector_span, static_cast<std::size_t>(wanted));
    if (result) {
        std::size_t found = result.dump_to(matches_data, NULL);
        (*env).SetIntArrayRegion(matches, 0, found, matches_data);
    } else {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, "Failed to find in vector index!");
    }

    (*env).ReleaseFloatArrayElements(vector, vector_data, 0);
    std::free(matches_data);
    return matches;
}

#include <jni.h>

#include "cloud_unum_usearch_Index.h"

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1create(JNIEnv* env, jclass, jstring, jstring, jlong, jlong,
                                                                jlong, jlong, jlong) {
    return 0;
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1destroy(JNIEnv*, jclass, jlong) {}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1size(JNIEnv*, jclass, jlong) { return 0; }

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1connectivity(JNIEnv*, jclass, jlong) { return 0; }

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1dimensions(JNIEnv*, jclass, jlong) { return 0; }

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1capacity(JNIEnv*, jclass, jlong) { return 0; }

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1reserve(JNIEnv*, jclass, jlong, jlong) {}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1add(JNIEnv*, jclass, jlong, jint, jfloatArray) {}

JNIEXPORT jintArray JNICALL Java_cloud_unum_usearch_Index_c_1search(JNIEnv*, jclass, jlong, jfloatArray, jlong) {
    return {};
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1save(JNIEnv*, jclass, jlong, jstring) {}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1load(JNIEnv*, jclass, jlong, jstring) {}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1view(JNIEnv*, jclass, jlong, jstring) {}

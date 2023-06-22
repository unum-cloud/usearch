#include "WolframLibrary.h"
#include <usearch/index_punned_dense.hpp>

using namespace unum::usearch;
using distance_t = punned_distance_t;
using punned_t = index_punned_dense_gt<int>;
using punned_search_result_t = typename punned_t::search_result_t;
using span_t = span_gt<float>;

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) { return LIBRARY_NO_ERROR; }
EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData) { return; }

EXTERN_C DLLEXPORT int IndexCreate(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    index_config_t config;
    char* accuracy_cstr = nullptr;
    char* metric_cstr = nullptr;
    try {
        accuracy_cstr = MArgument_getUTF8String(Args[1]);
        metric_cstr = MArgument_getUTF8String(Args[0]);
        std::size_t dimensions = static_cast<std::size_t>(MArgument_getInteger(Args[2]));
        std::size_t capacity = static_cast<std::size_t>(MArgument_getInteger(Args[3]));
        config.connectivity = static_cast<std::size_t>(MArgument_getInteger(Args[4]));
        std::size_t expansion_add = static_cast<std::size_t>(MArgument_getInteger(Args[5]));
        std::size_t expansion_search = static_cast<std::size_t>(MArgument_getInteger(Args[6]));

        scalar_kind_t accuracy = scalar_kind_from_name(accuracy_cstr, std::strlen(accuracy_cstr));
        metric_kind_t metric_kind = metric_from_name(metric_cstr, std::strlen(metric_cstr));
        punned_t index = punned_t::make(dimensions, metric_kind, config, accuracy, expansion_add, expansion_search);
        index.reserve(capacity);

        punned_t* result_ptr = new punned_t(std::move(index));
        
        MArgument_setInteger(Res, (long)result_ptr);

    } catch (...) {
        return LIBRARY_FUNCTION_ERROR;
    }

    libData->UTF8String_disown(accuracy_cstr);
    libData->UTF8String_disown(metric_cstr);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexSave(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    char* path_cstr = nullptr;
    punned_t* c_ptr = (punned_t*)MArgument_getUTF8String(Args[0]);
    try {
        path_cstr = MArgument_getUTF8String(Args[1]);
        c_ptr->save(path_cstr);
    } catch (...) {
        return LIBRARY_FUNCTION_ERROR;
    }
    libData->UTF8String_disown(path_cstr);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexLoad(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    char* path_cstr = nullptr;
    punned_t* c_ptr = (punned_t*)MArgument_getUTF8String(Args[0]);
    try {
        path_cstr = MArgument_getUTF8String(Args[1]);
        c_ptr->load(path_cstr);
    } catch (...) {
        return LIBRARY_FUNCTION_ERROR;
    }
    libData->UTF8String_disown(path_cstr);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexView(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    char* path_cstr = nullptr;
    punned_t* c_ptr = (punned_t*)MArgument_getUTF8String(Args[0]);
    try {
        path_cstr = MArgument_getUTF8String(Args[1]);
        c_ptr->view(path_cstr);
    } catch (...) {
        return LIBRARY_FUNCTION_ERROR;
    }
    libData->UTF8String_disown(path_cstr);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexDestroy(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    delete (punned_t*)MArgument_getUTF8String(Args[0]);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexSize(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    size_t res = ((punned_t*)MArgument_getUTF8String(Args[0]))->size();
    MArgument_setInteger(Res, res);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexConnectivity(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    size_t res = ((punned_t*)MArgument_getUTF8String(Args[0]))->connectivity();
    MArgument_setInteger(Res, res);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexDimensions(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    size_t res = ((punned_t*)MArgument_getUTF8String(Args[0]))->dimensions();
    MArgument_setInteger(Res, res);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexCapacity(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    size_t res = ((punned_t*)MArgument_getUTF8String(Args[0]))->capacity();
    MArgument_setInteger(Res, res);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexAdd(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    char* path_cstr = nullptr;
    punned_t* c_ptr = (punned_t*)MArgument_getUTF8String(Args[0]);
    float* vector_data = nullptr;
    try {
        int label = MArgument_getInteger(Args[1]);
        MTensor tens = MArgument_getMTensor(Args[2]);
        size_t len = libData->MTensor_getFlattenedLength(tens);
        vector_data = (float*)libData->MTensor_getRealData(tens);
        span_t vector_span = span_t{vector_data, len};
        c_ptr->add(label, vector_span);
    } catch (...) {
        return LIBRARY_FUNCTION_ERROR;
    }
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexSearch(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    punned_t* c_ptr = (punned_t*)MArgument_getUTF8String(Args[0]);
    MTensor matches;
    mint dims[] = {1};
    int wanted = MArgument_getInteger(Args[2]);
    float* vector_data = nullptr;
    int* matches_data = nullptr;

    try {
        libData->MTensor_new(MType_Integer, 1, dims, &matches);

        MTensor tens = MArgument_getMTensor(Args[1]);
        size_t len = libData->MTensor_getFlattenedLength(tens);
        vector_data = (float*)libData->MTensor_getRealData(tens);
        span_t vector_span = span_t{vector_data, len};
        search_config_t config;

        punned_search_result_t result = c_ptr->search(vector_span, (size_t)wanted, config);
        for (mint i = 0; i < result.size(); i++)
            libData->MTensor_setInteger(matches, &i, result[i].element.label);

        MArgument_setMTensor(Res, matches);
    } catch (...) {
        return LIBRARY_FUNCTION_ERROR;
    }
    return LIBRARY_NO_ERROR;
}
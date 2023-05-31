#include "WolframLibrary.h"
#include "punned.hpp"

using namespace unum::usearch;
using distance_t = punned_distance_t;
using punned_t = punned_gt<int>;
using span_t = span_gt<float>;

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) { return LIBRARY_NO_ERROR; }
EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData) { return; }

EXTERN_C DLLEXPORT int IndexCreate(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    config_t config;
    char* accuracy_cstr = nullptr;
    char* metric_cstr = nullptr;
    try {
        accuracy_cstr = MArgument_getUTF8String(Args[1]);
        metric_cstr = MArgument_getUTF8String(Args[0]);
        long dimensions = static_cast<std::size_t>(MArgument_getInteger(Args[2]));
        config.max_elements = static_cast<std::size_t>(MArgument_getInteger(Args[3]));
        config.connectivity = static_cast<std::size_t>(MArgument_getInteger(Args[4]));
        config.expansion_add = static_cast<std::size_t>(MArgument_getInteger(Args[5]));
        config.expansion_search = static_cast<std::size_t>(MArgument_getInteger(Args[6]));
        config.max_threads_add = 4;
        config.max_threads_search = 4;

        accuracy_t accuracy = accuracy_from_name(accuracy_cstr, strlen(accuracy_cstr));
        punned_t index = make_punned<punned_t>( //
            metric_cstr, strlen(metric_cstr), dimensions, accuracy, config);

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
    std::size_t found = 0;

    try {
        libData->MTensor_new(MType_Integer, 1, dims, &matches);

        MTensor tens = MArgument_getMTensor(Args[1]);
        size_t len = libData->MTensor_getFlattenedLength(tens);
        vector_data = (float*)libData->MTensor_getRealData(tens);
        span_t vector_span = span_t{vector_data, len};

        matches_data = (int*)std::malloc(sizeof(int) * wanted);
        found = c_ptr->search(vector_span, (size_t)wanted, matches_data, nullptr);
        for (mint i = 0; i < found; i++)
            libData->MTensor_setInteger(matches, &i, matches_data[i]);

        MArgument_setMTensor(Res, matches);
    } catch (...) {
        return LIBRARY_FUNCTION_ERROR;
    }
    std::free(matches_data);
    return LIBRARY_NO_ERROR;
}

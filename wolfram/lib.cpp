#include "WolframLibrary.h"
#include <usearch/index_dense.hpp>

using namespace unum::usearch;

using distance_t = distance_punned_t;
using index_t = index_dense_t;
using vector_view_t = span_gt<float>;

using add_result_t = typename index_t::add_result_t;
using search_result_t = typename index_t::search_result_t;
using vector_key_t = typename index_t::vector_key_t;

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) { return LIBRARY_NO_ERROR; }
EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData) { return; }

EXTERN_C DLLEXPORT int IndexCreate(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    index_config_t config;
    char* quantization_cstr = nullptr;
    char* metric_cstr = nullptr;
    try {
        quantization_cstr = MArgument_getUTF8String(Args[1]);
        metric_cstr = MArgument_getUTF8String(Args[0]);
        std::size_t dimensions = static_cast<std::size_t>(MArgument_getInteger(Args[2]));
        std::size_t capacity = static_cast<std::size_t>(MArgument_getInteger(Args[3]));
        config.connectivity = static_cast<std::size_t>(MArgument_getInteger(Args[4]));
        config.expansion_add = static_cast<std::size_t>(MArgument_getInteger(Args[5]));
        config.expansion_search = static_cast<std::size_t>(MArgument_getInteger(Args[6]));

        scalar_kind_t quantization = scalar_kind_from_name(quantization_cstr, std::strlen(quantization_cstr));
        metric_kind_t metric_kind = metric_from_name(metric_cstr, std::strlen(metric_cstr));
        index_t index = make_punned<index_t>(metric_kind, dimensions, quantization, config);
        index.reserve(capacity);

        index_t* result_ptr = new index_t(std::move(index));
        MArgument_setInteger(Res, (long)result_ptr);

    } catch (...) {
        return LIBRARY_FUNCTION_ERROR;
    }

    libData->UTF8String_disown(quantization_cstr);
    libData->UTF8String_disown(metric_cstr);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexSave(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    char* path_cstr = nullptr;
    index_t* c_ptr = (index_t*)MArgument_getUTF8String(Args[0]);
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
    index_t* c_ptr = (index_t*)MArgument_getUTF8String(Args[0]);
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
    index_t* c_ptr = (index_t*)MArgument_getUTF8String(Args[0]);
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
    delete (index_t*)MArgument_getUTF8String(Args[0]);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexSize(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    std::size_t res = ((index_t*)MArgument_getUTF8String(Args[0]))->size();
    MArgument_setInteger(Res, res);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexConnectivity(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    std::size_t res = ((index_t*)MArgument_getUTF8String(Args[0]))->connectivity();
    MArgument_setInteger(Res, res);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexDimensions(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    std::size_t res = ((index_t*)MArgument_getUTF8String(Args[0]))->dimensions();
    MArgument_setInteger(Res, res);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexCapacity(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    std::size_t res = ((index_t*)MArgument_getUTF8String(Args[0]))->capacity();
    MArgument_setInteger(Res, res);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexAdd(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    char* path_cstr = nullptr;
    index_t* c_ptr = (index_t*)MArgument_getUTF8String(Args[0]);
    float* vector_data = nullptr;
    try {
        int key = MArgument_getInteger(Args[1]);
        MTensor tens = MArgument_getMTensor(Args[2]);
        std::size_t len = libData->MTensor_getFlattenedLength(tens);
        vector_data = (float*)libData->MTensor_getRealData(tens);
        vector_view_t vector_span = vector_view_t{vector_data, len};
        c_ptr->add(key, vector_span);
    } catch (...) {
        return LIBRARY_FUNCTION_ERROR;
    }
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexSearch(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    index_t* c_ptr = (index_t*)MArgument_getUTF8String(Args[0]);
    MTensor matches;
    mint dims[] = {1};
    int wanted = MArgument_getInteger(Args[2]);
    float* vector_data = nullptr;
    int* matches_data = nullptr;
    std::size_t found = 0;

    try {
        libData->MTensor_new(MType_Integer, 1, dims, &matches);

        MTensor tens = MArgument_getMTensor(Args[1]);
        std::size_t len = libData->MTensor_getFlattenedLength(tens);
        vector_data = (float*)libData->MTensor_getRealData(tens);
        vector_view_t vector_span = vector_view_t{vector_data, len};

        matches_data = (int*)std::malloc(sizeof(int) * wanted);
        found = c_ptr->search(vector_span, static_cast<std::size_t>(wanted), matches_data, nullptr);
        for (mint i = 0; i < found; i++)
            libData->MTensor_setInteger(matches, &i, matches_data[i]);

        MArgument_setMTensor(Res, matches);
    } catch (...) {
        return LIBRARY_FUNCTION_ERROR;
    }
    std::free(matches_data);
    return LIBRARY_NO_ERROR;
}

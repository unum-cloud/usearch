#include "WolframLibrary.h"

EXTERN_C DLLEXPORT mint WolframLibrary_getVersion() { return WolframLibraryVersion; }
EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) { return 0; }
EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData) { return; }

EXTERN_C DLLEXPORT int IndexInit(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexReserve(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexAdd(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int IndexSearch(WolframLibraryData libData, mint Argc, MArgument* Args, MArgument Res) {
    return LIBRARY_NO_ERROR;
}

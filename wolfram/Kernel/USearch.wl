BeginPackage[ "UnumCloud`USearch`" ];

IndexCreate;
IndexSave;
IndexLoad;
IndexView;
IndexDestroy;
IndexSize;
IndexConnectivity;
IndexDimensions;
IndexCapacity;
IndexAdd;
IndexSearch;

Begin[ "`Private`" ];

IndexCreate[ a1_String, a2_String, a3_Integer, a4_Integer, a5_Integer, a6_Integer, a7_Integer] := indexCreate[a1, a2, a3, a4, a5, a6, a7];
IndexSave[ a1_Integer, a2_String] := indexSave[a1, a2];
IndexLoad[ a1_Integer, a2_String] := indexLoad[a1, a2];
IndexView[ a1_Integer, a2_String] := indexView[a1, a2];
IndexDestroy[ a1_Integer] := indexDestroy[a1];
IndexSize[ a1_Integer] := indexSize[a1];
IndexConnectivity[ a1_Integer] := indexConnectivity[a1];
IndexDimensions[ a1_Integer] := indexDimensions[a1];
IndexCapacity[ a1_Integer] := indexCapacity[a1];
IndexAdd[ a1_Integer, a2_Integer, a3_Real] := indexAdd[a1, a2, a3];
IndexSearch[ a1_Integer, a2_Real, a3_Integer] := indexSearch[a1, a2, a3];


indexCreate := indexCreate = LibraryFunctionLoad[$lib, "IndexCreate", {String, String, Integer, Integer, Integer, Integer, Integer}, Integer];
indexSave := indexSave = LibraryFunctionLoad[$lib, "IndexSave", {Integer, "UTF8String"}, "Void"]
indexLoad := indexLoad = LibraryFunctionLoad[$lib, "IndexLoad", {Integer, "UTF8String"}, "Void"]
indexView := indexView = LibraryFunctionLoad[$lib, "IndexView", {Integer, "UTF8String"}, "Void"]
indexDestroy := indexDestroy = LibraryFunctionLoad[$lib, "IndexDestroy", {Integer}, "Void"]
indexSize := indexSize = LibraryFunctionLoad[$lib, "IndexSize", {Integer}, Integer]
indexConnectivity := indexConnectivity = LibraryFunctionLoad[$lib, "IndexConnectivity", {Integer}, Integer]
indexDimensions := indexDimensions = LibraryFunctionLoad[$lib, "IndexDimensions", {Integer}, Integer]
indexCapacity := indexCapacity = LibraryFunctionLoad[$lib, "IndexCapacity", {Integer}, Integer]
indexAdd := indexAdd = LibraryFunctionLoad[$lib, "IndexAdd", {Integer, Integer, {Real, 1}}, "Void"]
indexSearch := indexSearch = LibraryFunctionLoad[$lib, "IndexSearch", {Integer, {Real, 1}, Integer}, {Integer, 1}]


$lib = FileNameJoin @ {
    DirectoryName[ $InputFileName, 2 ],
    "LibraryResources",
    $SystemID,
    "usearchWFM." <> Internal`DynamicLibraryExtension[ ]
};


End[ ];

EndPackage[ ];
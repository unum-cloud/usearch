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

IndexCreate[ a1_, a2_, a3_, a4_, a5_, a6_, a7_] := indexCreate[a1, a2, a3, a4, a5, a6, a7];
IndexSave[ a1_, a2_] := indexSave[a1, a2];
IndexLoad[ a1_, a2_] := indexLoad[a1, a2];
IndexView[ a1_, a2_] := indexView[a1, a2];
IndexDestroy[ a1_] := indexDestroy[a1];
IndexSize[ a1_] := indexSize[a1];
IndexConnectivity[ a1_] := indexConnectivity[a1];
IndexDimensions[ a1_] := indexDimensions[a1];
IndexCapacity[ a1_] := indexCapacity[a1];
IndexAdd[ a1_, a2_, a3_] := indexAdd[a1, a2, a3];
IndexSearch[ a1_, a2_, a3_] := indexSearch[a1, a2, a3];


indexCreate := indexCreate = LibraryFunctionLoad[$lib, "IndexCreate", {"UTF8String", "UTF8String", Integer, Integer, Integer, Integer, Integer}, Integer];
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
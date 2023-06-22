#!/usr/bin/env wolframscript

Needs["UnumCloud`USearch`"]
Print["Functions Loaded"]

$index = IndexCreate["cos", "f16", 10, 10, 10, 10, 10]
Print["New Index Address: ", $index]
Print["Index Size: ", IndexSize[$index]]
Print["Index Capacity: ", IndexCapacity[$index]]
Print["Index Connectivity: ", IndexConnectivity[$index]]
$arr = {1.0, 2.0, 3.0, 4.0, 5.0};
IndexAdd[$index, 42, $arr]
Print["Added Array: ", $arr]
$res = IndexSearch[$index, $arr, 42]
Print["Searched: ", $res]
IndexDestroy[$index]
Print["Index Destroyed"]
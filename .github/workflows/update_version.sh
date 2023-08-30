#!/bin/sh

echo $1 > VERSION && 
    sed -i "s/\"version\": \".*\"/\"version\": \"$1\"/" package.json &&
    sed -i "s/^version: .*/version: $1/" CITATION.cff &&
    sed -i "s/version = \".*\"/version = \"$1\"/" Cargo.toml &&
    sed -i "s/version = '.*'/version = '$1'/" conanfile.py &&
    sed -i "s/^\(#define USEARCH_VERSION_MAJOR \).*/\1$(echo "$1" | cut -d. -f1)/" ./include/usearch/index.hpp &&
    sed -i "s/^\(#define USEARCH_VERSION_MINOR \).*/\1$(echo "$1" | cut -d. -f2)/" ./include/usearch/index.hpp &&
    sed -i "s/^\(#define USEARCH_VERSION_PATCH \).*/\1$(echo "$1" | cut -d. -f3)/" ./include/usearch/index.hpp &&
    sed -i "s/<version>[0-9]\+\.[0-9]\+\.[0-9]\+/<version>$1/" README.md &&
    sed -i "s/version = {[0-9]\+\.[0-9]\+\.[0-9]\+}/version = {$1}/" README.md &&
    sed -i "s/>[0-9]\+\.[0-9]\+\.[0-9]\+<\/Version>/>$1<\/Version>/" ./csharp/nuget/nuget-package.props &&
    sed -i "s/version=\".*\"/version=\"$1\"/" wasmer.toml

#!/bin/sh

echo $1 > VERSION && 
    sed -i "s/\"version\": \".*\"/\"version\": \"$1\"/" package.json &&
    sed -i "s/^version: .*/version: $1/" CITATION.cff &&
    sed -i "s/version = \".*\"/version = \"$1\"/" Cargo.toml &&
    sed -i "s/version = '.*'/version = '$1'/" conanfile.py

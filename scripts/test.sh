#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 [--cpu | --gpu]"
    exit 1
fi

BUILD_DIR="../build"

if [ -f "$BUILD_DIR/CMakeCache.txt" ] && [ -f "$BUILD_DIR/Makefile" ]; then
    echo "Build already exists. Skipping build step."
    cd ../build
else
    if [ "$1" == "--cpu" ]; then
        echo "=== Building for CPU... ==="
        cd ..
        mkdir -p build
        cd build
        cmake -DTEST=ON -DCUDA=OFF ..
        if [ $? -ne 0 ]; then
            echo "CMake configuration failed"
            exit 1
        fi
        make
        if [ $? -ne 0 ]; then
            echo "Make failed"
            exit 1
        fi
    elif [ "$1" == "--gpu" ]; then
        echo "=== Building for GPU... ==="
        cd ..
        mkdir -p build
        cd build
        cmake -DTEST=ON -DCUDA=ON ..
        if [ $? -ne 0 ]; then
            echo "CMake configuration failed"
            exit 1
        fi
        make
        if [ $? -ne 0 ]; then
            echo "Make failed"
            exit 1
        fi
    else
        echo "Invalid option: $1"
        echo "Usage: $0 [--cpu | --gpu]"
        exit 1
    fi
fi

echo "=== Running tests... ==="
ctest --verbose --output-on-failure -C Debug -T test
cd ../scripts
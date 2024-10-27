#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 [--cpu | --gpu]"
    exit 1
fi

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
    cd ../scripts
elif [ "$1" == "--gpu" ]; then
    echo "=== Building for GPU... ==="
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
    cd ../scripts
else
    echo "Invalid option: $1"
    echo "Usage: $0 [--cpu | --gpu]"
    exit 1
fi
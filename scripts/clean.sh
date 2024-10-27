#!/bin/bash

cd ..

if [ -d "build" ]; then
    echo "Removing build directory..."
    rm -r build
    echo "Build directory removed."
else
    echo "Build directory does not exist. Skipping removal."
fi

cd scripts
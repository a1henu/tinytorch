# tinytorch

This is a project for the course "[Programming in AI](https://pkuprogramminginai.github.io/Labs-Documentation/#/)" in Peking University.

We purpose to implement a tiny deep learning framework, and we will use it to train a simple neural network.

## Installation

**TODO**: Describe the installation process

## Build C++/CUDA Part

You can run the script `build.sh` to build the C++/CUDA part of the project. The script will create a folder `build` and compile the source code in it.

```bash
bash build.sh --cpu # build the CPU Version
bash build.sh --gpu # build the GPU Version
```

If you want to clean the build folder, you can run the following script:

```bash
bash clean.sh
```

You can also run the following commands to do the same thing:

```bash
mkdir build
cd build
cmake -DCUDA=OFF -DTEST=ON .. # build the CPU Version
cmake -DCUDA=ON -DTEST=ON .. # build the GPU Version
make
```


## Usage

**TODO**: Write usage instructions

## Unit Test

`tinytorch` is a C++ project binded with [pybind11](https://github.com/pybind/pybind11). For C++ source code, tinytorch uses [GoogleTest](https://github.com/google/googletest) to implement unit tests. For Python source code, tinytorch uses ...

### C++/CUDA Part

To test the C++/CUDA code, you need to install [GoogleTest](https://github.com/google/googletest) first. This module is included in `third_party` folder. You can install it by running the following commands:

```bash
git submodule update --init --recursive
```

**Testing C++ Code**

You can run the script `test.sh` to test the C++ code. The script will create a folder `build` and compile the source code in it.

```bash
bash test.sh --cpu # test the CPU Version
bash test.sh --gpu # test the GPU Version
```

You can also run the following commands to test the C++ code:

```bash
mkdir build
cd build
cmake -DTEST=ON -DCUDA=OFF ..
make
ctest --verbose --output-on-failure -C Debug -T test
```

**Testing CUDA Code**

You can run the following commands to test the CUDA code:

```bash
mkdir build
cd build
cmake -DTEST=ON -DCUDA=ON ..
make
ctest --verbose --output-on-failure -C Debug -T test
```
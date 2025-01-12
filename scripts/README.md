# Scripts

`scripts` is a collection of scripts that I use to automate tasks in this project.

## `build.sh`

This script is used to build the project. It will compile the source code and generate the executable file.

Usage:

```bash
./build.sh [--cpu | --gpu]
```

Options:
- `--cpu`: Build the project for CPU.
- `--gpu`: Build the project for GPU.

## `test.sh`

This script is used to test the project. It will run the executable file with the test data.

Usage:

```bash
./test.sh [--cpu | --gpu]
```

Options:
- `--cpu`: Test the project with CPU.
- `--gpu`: Test the project with GPU.

## `clean.sh`

This script is used to clean the project. It will remove the executable file.

Usage:

```bash
./clean.sh
```

## `all_test.sh`

This script is used to test the project with all available devices.

Usage:

```bash
./all_test.sh
```
# SRT

SRT(Super Rabbit Tracer) is a powerful rendering framework based on the OptiX ray tracing engine. Its goal is to provide a simple and flexible API for building ray tracing applications, making it easier for researchers to quickly prototype new ideas and discover innovative methods.

**Work in progress.**

## Features

* [x] OptiX Ray Tracing Engine
* [x] Disney BRDF
* [x] Skeletal animation

## Requirements

* [OptiX SDK](https://developer.nvidia.com/optix) (version 7.3 or higher)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (version 10.0 or higher)
* [CMake](https://cmake.org/) (version 3.18 or higher)
* [Assimp Requirements](https://github.com/assimp/assimp)  SRT uses submodules to manage the Assimp library, you don't need to install it manually, but you need to install the dependencies of it.

## Build and Run

Download the SRT repository. Note that SRT uses submodules, so you need to clone the repository recursively:

```shell
git clone --recursive https://github.com/woAIxuexiSR/SRT.git
```

If you have already cloned the repository without the `--recursive` flag, you can initialize and update the submodules manually:

```shell
git submodule init
git submodule update
```

Use CMake to generate and build the project:

```shell
cmake -B build
cmake --build build
```

## Usage

SRT takes the json configuration file as input. For more information about the configuration file, please refer to the [SRT Scene Configuration File](example/readme.md).

```
Usage: main [options]
    --help, -h: Print this usage message and exit.
    --config, -c: Path to config file.
    --render, -r: Override the render config.
    --passes, -p: Override the passes config.
    --scene, -s: Override the scene config.
```

## Algorithms

* [x] Path tracing
* [ ] Bidirectional path tracing
* [ ] ReSTIR

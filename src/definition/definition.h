#pragma once

#include "stb_image.h"
#include "stb_image_write.h"

#define TINYEXR_USE_MINIZ 0
#include "zlib.h"
#include "tinyexr.h"

#include "tiny_obj_loader.h"

#include "tiny-cuda-nn/gpu_memory.h"
#include "tiny-cuda-nn/gpu_matrix.h"
using tcnn::GPUMemory;
using tcnn::GPUMatrix;

#include <chrono>
#include <iostream>
#define TICK(x) auto x = std::chrono::system_clock::now()
#define TOCK(x) std::cout << #x << " : " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - x).count() << "ms" << std::endl
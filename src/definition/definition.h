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
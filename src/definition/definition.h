#pragma once

#include "stb_image.h"
#include "stb_image_write.h"

#define TINYEXR_USE_MINIZ 0
#include "zlib.h"
#include "tinyexr.h"
#include "tinyply.h"

#include "json.hpp"
using json = nlohmann::json;

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"
#include "implot.h"
#include "implot_internal.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "tiny-cuda-nn/common.h"
#include "tiny-cuda-nn/loss.h"
#include "tiny-cuda-nn/optimizer.h"
#include "tiny-cuda-nn/encoding.h"
#include "tiny-cuda-nn/network.h"
#include "tiny-cuda-nn/network_with_input_encoding.h"
#include "tiny-cuda-nn/gpu_memory.h"
#include "tiny-cuda-nn/gpu_matrix.h"
using tcnn::GPUMemory, tcnn::GPUMatrix;
using tcnn::Loss, tcnn::Optimizer, tcnn::Encoding;
using tcnn::Network, tcnn::NetworkWithInputEncoding;
using precision_t = tcnn::network_precision_t;

#include <chrono>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
using std::vector, std::string, std::unordered_map;
using std::shared_ptr, std::make_shared;
using std::cout, std::endl;


// parse a string to a value of T
template<class T>
inline T parse_value(const string& value)
{
    std::stringstream ss(value);
    T result;
    ss >> result;
    return result;
}

// parse a string to a vector of T (split by space)
template<class T>
inline vector<T> parse_to_vector(const string& str)
{
    std::stringstream ss(str);
    vector<T> vec;
    T value;
    while (ss >> value)
        vec.push_back(value);
    return vec;
}

// dequote a string
inline string dequote(const string& s)
{
    auto start = s.find_first_not_of(' ');
    auto end = s.find_last_not_of(' ');
    string ans = s.substr(start, end - start + 1);
    return ans.substr(1, ans.size() - 2);
}

// convert a value of T to a string
template<class T>
inline string to_str(const T& value)
{
    std::stringstream ss;
    ss << value;
    return ss.str();
}
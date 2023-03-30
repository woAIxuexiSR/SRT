#pragma once

#include "stb_image.h"
#include "stb_image_write.h"

#define TINYEXR_USE_MINIZ 0
#include "zlib.h"
#include "tinyexr.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "tiny-cuda-nn/gpu_memory.h"
#include "tiny-cuda-nn/gpu_matrix.h"
using tcnn::GPUMemory, tcnn::GPUMatrix;

#include <chrono>
#include <iostream>
#define TICK(x) auto x = std::chrono::system_clock::now()
#define TOCK(x) std::cout << #x << " : " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - x).count() << "ms" << std::endl

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
using std::vector, std::string, std::unordered_map;
using std::shared_ptr, std::make_shared;
using std::cout, std::endl;



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

inline string dequote(const string& s)
{
    auto start = s.find_first_not_of(' ');
    auto end = s.find_last_not_of(' ');
    string ans = s.substr(start, end - start + 1);
    return ans.substr(1, ans.size() - 2);
}
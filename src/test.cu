#include <iostream>
#include "renderer.h"
#include "profiler.h"

__global__ void kernel(int num, Ray* ray, int width, int height, Camera camera)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num) return;
    float xx = (idx % width) / (float)width;
    float yy = (idx / width) / (float)height;
    // for (int i = 0; i < 128; i++)
    //     ray[idx] = camera.get_ray(xx, yy);
}

int main()
{
    int width = 1920, height = 1080;
    int num = width * height;
    Ray* ray;
    checkCudaErrors(cudaMallocManaged(&ray, sizeof(Ray) * num));
    Camera camera;
    camera.set_aspect_fov((float)width / (float)height, 90.0f);

    TICK(test);

    int block = 256;
    int grid = (num + block - 1) / block;
    kernel << <grid, block >> > (num, ray, width, height, camera);
    checkCudaErrors(cudaDeviceSynchronize());

    TOCK(test);


    return 0;
}
#include "scene/material.h"
#include "definition.h"
#include "renderer.h"

class A
{
public:
    float4* ptr;

    void func(shared_ptr<Film> _film)
    {
        int width = _film->get_width();
        int height = _film->get_height();
        ptr = _film->get_fptr();

        tcnn::parallel_for_gpu(width * height, [=, *this] __device__(int index) {
            float x = (index % width) / (float)width;
            float y = (index / width) / (float)height;
            ptr[index] = make_float4(x, y, 0, 1);
        });

        checkCudaErrors(cudaDeviceSynchronize());
    }
};

__global__ void test(int* a)
{
    atomicAdd(a, 1);
}

int main()
{
    int width = 10, height = 12;
    shared_ptr<Film> film = make_shared<Film>(width, height);

    A a;
    a.func(film);

    film->f_to_uchar();
    film->save_png("hhh.png");

    return 0;
}
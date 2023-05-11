#include <iostream>
#include "renderer.h"
#include "profiler.h"

void func()
{
    PROFILE("func");
    for(int i = 0; i < 20; i++)
        cout << i << " ";
    cout << endl;
}

__global__ void kernel()
{
    printf("hhh\n");
}

void test()
{
    PROFILE("test");
    kernel<<<3, 5>>>();
    checkCudaErrors(cudaDeviceSynchronize());
}

int main()
{
    Profiler::reset();

    PROFILE("main");

    func();
    test();

    Profiler::stop();
    Profiler::print();

    Profiler::reset();
    return 0;
}
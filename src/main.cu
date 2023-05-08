#include <iostream>
#include "renderer.h"

__global__ void kernel(int num, int width, int height, Camera* camera, float4* pixels, cudaTextureObject_t tex_obj)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;

    int x = idx % width, y = idx / width;
    float xx = (float)x / width, yy = (float)y / height;
    Ray ray = camera->get_ray(xx, yy);
    float4 color = texCubemap<float4>(tex_obj, ray.dir.x, ray.dir.y, ray.dir.z);
    pixels[idx] = color;
}

void test()
{
    const int width = 2048, height = 2048;

    shared_ptr<Film> film = make_shared<Film>(width, height);

    Transform transform = LookAt({ 0.0f, 1.0f, 5.5f }, { 0.0f,2.0f,0.5f }, { 0.0f,1.0f,0.0f });
    float aspect = (float)width / (float)height;
    float fov = 60.0f;
    shared_ptr<Camera> camera = make_shared<Camera>(transform.get_matrix(), aspect, fov);

    GPUMemory<Camera> gpu_camera;
    gpu_camera.resize_and_copy_from_host(camera.get(), 1);

    vector<uchar4> tex_data(width * height * 6);

    std::filesystem::path current_path(__FILE__);
    std::filesystem::path skybox_path;
    skybox_path = current_path.parent_path().parent_path() / "data" / "skybox";
    vector<string> faces = {
        skybox_path / "right.jpg",
        skybox_path / "left.jpg",
        skybox_path / "top.jpg",
        skybox_path / "bottom.jpg",
        skybox_path / "front.jpg",
        skybox_path / "back.jpg"
    };

    for(int i = 0; i < 6; i++)
    {
        shared_ptr<Texture> tex = make_shared<Texture>();
        tex->load_from_file(faces[i]);
        memcpy(tex_data.data() + i * width * height, tex->pixels.data(), width * height * sizeof(uchar4)); 
    }


    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
    cudaExtent extent = make_cudaExtent(width, height, 6);
    cudaArray_t cu_array;
    checkCudaErrors(cudaMalloc3DArray(&cu_array, &channel_desc, extent, cudaArrayCubemap));

    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr(tex_data.data(), width * sizeof(uchar4), width, height);
    copyParams.dstArray = cu_array;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cu_array;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.addressMode[2] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;

    cudaTextureObject_t tex_obj = 0;
    checkCudaErrors(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

    tcnn::linear_kernel(kernel, 0, 0, width * height, width, height, gpu_camera.data(), film->get_pixels(), tex_obj);
    checkCudaErrors(cudaDeviceSynchronize());

    film->save_ldr("test.png");

    checkCudaErrors(cudaDestroyTextureObject(tex_obj));
    checkCudaErrors(cudaFreeArray(cu_array));
}

int main()
{
    const int width = 1920, height = 1080;

    std::filesystem::path current_path(__FILE__);
    std::filesystem::path model_path;
    model_path = current_path.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Original.obj";

    Transform transform = LookAt({ 0.0f, 1.0f, 5.5f }, { 0.0f,1.0f,0.5f }, { 0.0f,1.0f,0.0f });
    float aspect = (float)width / (float)height;
    float fov = 60.0f;
    shared_ptr<Camera> camera = make_shared<Camera>(transform.get_matrix(), aspect, fov);

    std::filesystem::path skybox_path;
    skybox_path = current_path.parent_path().parent_path() / "data" / "skybox";
    vector<string> faces = {
        skybox_path / "right.jpg",
        skybox_path / "left.jpg",
        skybox_path / "top.jpg",
        skybox_path / "bottom.jpg",
        skybox_path / "front.jpg",
        skybox_path / "back.jpg"
    };

    shared_ptr<Scene> scene = make_shared<Scene>();
    scene->set_camera(camera);
    scene->load_from_model(model_path.string());
    scene->load_environment_map(faces);

    Renderer renderer(width, height, scene);
    renderer.run();

    // test();

    return 0;
}
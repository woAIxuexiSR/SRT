#include <iostream>
#include "renderer.h"
#include "parser.h"
#include "integrator.h"

void test_render()
{
    RenderParams params;
    params.width = 1920;
    params.height = 1080;
    params.spp = 256;
    params.method = "path";
    params.transform = LookAt(make_float3(0.0f, 1.0f, 5.5f), make_float3(0.0f, 1.0f, 0.5f), make_float3(0.0f, 1.0f, 0.0f));
    params.fov = 60.0f;


    std::filesystem::path currentPath(__FILE__);
    std::filesystem::path modelPath;
    // modelPath = currentPath.parent_path().parent_path() / "data" / "sphere" / "sphere.obj";
    modelPath = currentPath.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Original.obj";
    // modelPath = currentPath.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Glossy.obj";
    // modelPath = currentPath.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Mirror.obj";
    // modelPath = currentPath.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Water.obj";
    // modelPath = currentPath.parent_path().parent_path() / "data" / "SimpleSphere" / "sphere.obj";
    // modelPath = currentPath.parent_path().parent_path() / "data" / "sponza" / "sponza.obj";
    // modelPath = currentPath.parent_path().parent_path() / "data" / "bathroom" / "salle_de_bain.obj";

    std::filesystem::path out_path;
    out_path = currentPath.parent_path().parent_path() / "data" / "out.exr";

    auto scene = make_shared<Scene>();
    scene->load_from_model(modelPath.string());

    ImageRender image_render(params, scene, out_path.string());
    TICK(time);
    image_render.render();
    TOCK(time);

    // InteractiveRender interactive_render(params, scene);
    // interactive_render.render();

    // std::filesystem::path video_path = out_path.parent_path() / "out.avi";
    // VideoRender video_render(params, scene, video_path.string());
    // video_render.render();

    // MaterialAdjustRender material_adjust_render(params.width, params.height);
    // material_adjust_render.render();
}

void comparison_render()
{

}

__global__ void kernel()
{
}

int main()
{
    test_render();

    // kernel<<<1, 1>>>();
    // checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}
#include <iostream>
#include "renderer.h"
#include "parser.h"
#include "integrator.h"

void test_render()
{
    RenderParams params;
    params.width = 1920;
    params.height = 1080;
    params.spp = 4;
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

    // ImageRender image_render(params, scene, out_path.string());
    // TICK(time);
    // image_render.render();
    // TOCK(time);

    InteractiveRender interactive_render(params, scene);
    interactive_render.render();

    // std::filesystem::path video_path = out_path.parent_path() / "out.avi";
    // VideoRender video_render(params, scene, video_path.string());
    // video_render.render();
}

void comparison_render()
{
    ComparisonRenderParams params;
    params.width = 1920;
    params.height = 1080;
    params.spp_1 = 8;
    params.method_1 = "path";
    params.spp_2 = 1;
    params.method_2 = "path";
    params.transform = LookAt(make_float3(0.0f, 1.0f, 5.5f), make_float3(0.0f, 1.0f, 0.5f), make_float3(0.0f, 1.0f, 0.0f));
    params.fov = 60.0f;


    std::filesystem::path currentPath(__FILE__);
    std::filesystem::path modelPath;
    // modelPath = currentPath.parent_path().parent_path() / "data" / "sphere" / "sphere.glb";
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

    ComparisonRender comparison_render(params, scene);
    comparison_render.render();
}

void material_adjust()
{
    MaterialAdjustRender render(1920, 1080);
    render.render();
}

__global__ void kernel()
{
}

int main()
{
    test_render();
    // comparison_render();
    // material_adjust();

    // kernel<<<1, 1>>>();
    // checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}
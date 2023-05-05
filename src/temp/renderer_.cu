#include "renderer.h"

shared_ptr<OptixRayTracer> create_ray_tracer(string method, shared_ptr<Scene> scene)
{
    shared_ptr<OptixRayTracer> ray_tracer = nullptr;
    if (method == "simple")
        ray_tracer = make_shared<SimpleShader>(scene.get());
    else if (method == "direct")
        ray_tracer = make_shared<DirectLight>(scene.get());
    else if (method == "path")
        ray_tracer = make_shared<PathTracer>(scene.get());
    else if (method == "light")
        ray_tracer = make_shared<LightTracer>(scene.get());
    else if (method == "bdpt")
        ray_tracer = make_shared<BDPT>(scene.get());
    else
        std::cout << "Ray tracing method not supported" << std::endl;
    return ray_tracer;
}

ImageRender::ImageRender(RenderParams _params, shared_ptr<Scene> scene, string _path)
{
    image_path = _path;
    film = make_shared<Film>(_params.width, _params.height);
    camera = make_shared<Camera>(_params.transform, (float)_params.width / (float)_params.height, _params.fov);
    ray_tracer = create_ray_tracer(_params.method, scene);
    ray_tracer->set_spp(_params.spp);
}

void ImageRender::render()
{
    ray_tracer->render(camera, film);

    string type = image_path.substr(image_path.find_last_of(".") + 1);
    if (type == "exr")
        film->save_exr(image_path);
    else if (type == "jpg")
    {
        film->f_to_uchar();
        film->save_jpg(image_path);
    }
    else if (type == "png")
    {
        film->f_to_uchar();
        film->save_png(image_path);
    }
    else
        std::cout << "Image type not supported" << std::endl;
}

InteractiveRender::InteractiveRender(RenderParams _params, shared_ptr<Scene> scene): params(_params)
{
    film = make_shared<Film>(_params.width, _params.height);
    camera = make_shared<Camera>(_params.transform, (float)_params.width / (float)_params.height, _params.fov);
    ray_tracer = create_ray_tracer(_params.method, scene);
    gui = make_shared<InteractiveGui>(_params.width, _params.height, camera);
    ray_tracer->set_spp(_params.spp);
}

void InteractiveRender::render()
{
    std::fstream f("framerate.txt", std::ios::out);
    while (!gui->should_close())
    {
        ray_tracer->render(camera, film);
        film->f_to_uchar();
        gui->run((unsigned char*)film->get_uptr(), f);
    }
    f.close();
}

ComparisonRender::ComparisonRender(ComparisonRenderParams _params, shared_ptr<Scene> scene): params(_params)
{
    film_1 = make_shared<Film>(_params.width, _params.height);
    film_2 = make_shared<Film>(_params.width, _params.height);
    camera = make_shared<Camera>(_params.transform, (float)_params.width / (float)_params.height, _params.fov);
    ray_tracer_1 = create_ray_tracer(_params.method_1, scene);
    ray_tracer_2 = create_ray_tracer(_params.method_2, scene);
    gui = make_shared<ComparisonGui>(_params.width, _params.height, camera);
    ray_tracer_1->set_spp(_params.spp_1);
    ray_tracer_2->set_spp(_params.spp_2);
}

void ComparisonRender::render()
{
    while (!gui->should_close())
    {
        ray_tracer_1->render(camera, film_1);
        ray_tracer_2->render(camera, film_2);
        film_1->f_to_uchar();
        film_2->f_to_uchar();
        gui->run((unsigned char*)film_1->get_uptr(), (unsigned char*)film_2->get_uptr());
    }
}

VideoRender::VideoRender(RenderParams _params, shared_ptr<Scene> scene, string _path, int _fps)
{
    video_path = _path;
    fps = _fps;
    film = make_shared<Film>(_params.width, _params.height);
    camera = make_shared<Camera>(_params.transform, (float)_params.width / (float)_params.height, _params.fov);
    ray_tracer = create_ray_tracer(_params.method, scene);
}

void VideoRender::render()
{
    string command = "mkdir temporaryImages";
    int result = std::system(command.c_str());
    if (result != 0)
    {
        std::cout << "Error: Failed to create temporaryImages directory." << std::endl;
        exit(-1);
    }

    for (int i = 0; i < 60; i++)
    {
        ray_tracer->render(camera, film);
        film->f_to_uchar();
        film->save_png("temporaryImages/" + std::to_string(i) + ".png");
    }

    int width = film->get_width(), height = film->get_height();
    command = "ffmpeg -y -framerate " + std::to_string(fps);
    command += " -i temporaryImages/%d.png ";
    command += "-c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ";
    command += video_path;
    // std::cout << command << std::endl;

    result = std::system(command.c_str());
    if (result != 0)
    {
        std::cout << "Error: Failed to create video." << std::endl;
        exit(-1);
    }

    command = "rm -r temporaryImages";
    result = std::system(command.c_str());
    if (result != 0)
    {
        std::cout << "Error: Failed to delete temporaryImages directory." << std::endl;
        exit(-1);
    }
}

MaterialAdjustRender::MaterialAdjustRender(int _w, int _h)
{
    film = make_shared<Film>(_w, _h);
    SquareMatrix<4> transform = LookAt(make_float3(0.0f, 4.0f, 8.0f), make_float3(0.0f, 0.5f, 1.0f), make_float3(0.0f, 1.0f, 0.0f));
    camera = make_shared<Camera>(transform, (float)_w / (float)_h);

    std::filesystem::path file_path(__FILE__);
    file_path = file_path.parent_path().parent_path().parent_path() / "data" / "sphere" / "sphere.obj";
    scene = make_shared<Scene>();
    scene->load_from_model(file_path.string());

    mat = make_shared<Material>();
    mat->type = MaterialType::Disney;
    mat->color = make_float3(0.2f, 0.2f, 0.8f);

    ray_tracer = make_shared<MaterialAdjuster>(scene.get(), mat);
    ray_tracer->set_spp(32);
    ray_tracer->set_background(make_float3(1.0f));
    gui = make_shared<MaterialAdjustGui>(_w, _h, camera, mat);
}

void MaterialAdjustRender::render()
{
    while (!gui->should_close())
    {
        ray_tracer->render(camera, film);
        film->f_to_uchar();
        gui->run((unsigned char*)film->get_uptr());
    }
}
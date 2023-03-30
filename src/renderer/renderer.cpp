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

ImageRender::ImageRender(RenderParams params, shared_ptr<Scene> scene, string _path)
{
    image_path = _path;
    film = make_shared<Film>(params.width, params.height);
    camera = make_shared<Camera>(params.transform, (float)params.width / (float)params.height, params.fov);
    ray_tracer = create_ray_tracer(params.method, scene);
    ray_tracer->set_spp(params.spp);
}

void ImageRender::render()
{
    ray_tracer->render(camera, film);

    string type = image_path.substr(image_path.find_last_of(".") + 1);
    if(type == "exr")
        film->save_exr(image_path);
    else if(type == "jpg")
    {
        film->f_to_uchar();
        film->save_jpg(image_path);
    }
    else if(type == "png")
    {
        film->f_to_uchar();
        film->save_png(image_path);
    }
    else
        std::cout << "Image type not supported" << std::endl;
}

InteractiveRender::InteractiveRender(RenderParams params, shared_ptr<Scene> scene)
{
    film = make_shared<Film>(params.width, params.height);
    camera = make_shared<Camera>(params.transform, (float)params.width / (float)params.height, params.fov);
    ray_tracer = create_ray_tracer(params.method, scene);
    gui = make_shared<InteractiveGui>(params.width, params.height, camera);
    ray_tracer->set_spp(params.spp);
}

void InteractiveRender::render()
{
    while(!gui->should_close())
    {
        ray_tracer->render(camera, film);
        film->f_to_uchar();
        gui->run((unsigned char*)film->get_uptr());
    }
}

VideoRender::VideoRender(RenderParams params, shared_ptr<Scene> scene, string _path, int _fps)
{
    video_path = _path;
    fps = _fps;
    film = make_shared<Film>(params.width, params.height);
    camera = make_shared<Camera>(params.transform, (float)params.width / (float)params.height, params.fov);
    ray_tracer = create_ray_tracer(params.method, scene);
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
    camera = make_shared<Camera>();

    scene = make_shared<Scene>();

    mat = make_shared<Material>();
    mat->type = MaterialType::Disney;

    ray_tracer = make_shared<MaterialAdjuster>(scene.get(), mat);
    ray_tracer->set_spp(128);
    gui = make_shared<MaterialAdjustGui>(_w, _h, camera, mat);
}

void MaterialAdjustRender::render()
{
    while(!gui->should_close())
    {
        ray_tracer->render(camera, film);
        film->f_to_uchar();
        gui->run((unsigned char*)film->get_uptr());
    }
}
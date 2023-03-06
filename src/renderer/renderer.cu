#include "renderer.h"

void ImageRender::render()
{
    rayTracer->render(camera, film);

    if(type == "EXR")
        film->save_exr(imagePath);
    else if(type == "JPG")
    {
        film->fToUchar();
        film->save_jpg(imagePath);
    }
    else if(type == "PNG")
    {
        film->fToUchar();
        film->save_png(imagePath);
    }
    else
        std::cout << "Image type not supported" << std::endl;
}

void InteractiveRender::render()
{
    while(!gui->shouldClose())
    {
        rayTracer->render(camera, film);
        film->fToUchar();
        gui->run((unsigned char*)film->getuPtr());
    }
}

void VideoRender::render()
{
    std::string command = "mkdir temporaryImages";
    int result = std::system(command.c_str());
    if(result != 0)
    {
        std::cout << "Error: Failed to create temporaryImages directory." << std::endl;
        exit(-1);
    }

    for(int i = 0; i < 60; i++)
    {
        rayTracer->render(camera, film);
        film->fToUchar();
        film->save_png("temporaryImages/" + std::to_string(i) + ".png");
    }

    int width = film->getWidth(), height = film->getHeight();
    command = "ffmpeg -y -framerate " + std::to_string(fps);
    command += " -i temporaryImages/%d.png ";
    command += "-c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ";
    command += videoPath;
    std::cout << command << std::endl;

    result = std::system(command.c_str());
    if(result != 0)
    {
        std::cout << "Error: Failed to create video." << std::endl;
        exit(-1);
    }

    command = "rm -r temporaryImages";
    result = std::system(command.c_str());
    if(result != 0)
    {
        std::cout << "Error: Failed to delete temporaryImages directory." << std::endl;
        exit(-1);
    }
}
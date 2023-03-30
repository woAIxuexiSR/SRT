#include <iostream>
#include <filesystem>

#include "definition.h"
#include "metric.h"
#include "argparse.h"

void load_exr(float** out, int* width, int* height, const char* filename)
{
    const char* err = nullptr;
    int ret = LoadEXR(out, width, height, filename, &err);
    if (ret != TINYEXR_SUCCESS) 
    {
        std::cerr << "Load EXR err: " << err << std::endl;
        FreeEXRErrorMessage(err);
        exit(-1);
    }
}

int main(int argc, char* argv[])
{
    auto args = argparser("Compute Image Metric")
        .set_program_name("metric")
        .add_help_option()
        .add_option("-t", "--type", "metric type", "RMSE")
        .add_argument<std::string>("image", "input image")
        .add_argument<std::string>("reference", "reference image")
        .parse(argc, argv);

    auto type = args.get<std::string>("type");
    auto img_file = args.get<std::string>("image");
    auto ref_file = args.get<std::string>("reference");

    float* img;
    int i_width, i_height;
    load_exr(&img, &i_width, &i_height, img_file.c_str());

    float* ref;
    int r_width, r_height;
    load_exr(&ref, &r_width, &r_height, ref_file.c_str());

    if (i_width != r_width || i_height != r_height)
    {
        std::cerr << "Image size mismatch" << std::endl;
        exit(1);
    }

    Metric metric(i_width, i_height, img, ref);
    std::cout << type << ": " << metric.compute(type) << std::endl;

    return 0;
}
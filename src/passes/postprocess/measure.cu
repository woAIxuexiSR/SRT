#include "measure.h"

REGISTER_RENDER_PASS_CPP(Measure);

void Measure::init()
{
    Image img;
    img.load_from_file(ref_path, true, true);
    if (img.resolution.x != width || img.resolution.y != height
        || img.format != Image::Format::Float)
    {
        cout << "ERROR::Measure: reference image size or format not match" << endl;
        exit(-1);
    }
    ref_buffer.resize_and_copy_from_host(img.pixels_f);
    error_buffer.resize(width * height);
}

void Measure::render(shared_ptr<Film> film)
{
    if (!enable)
    {
        frame_count = 0;
        error_history.clear();
        return;
    }
    PROFILE("Measure");

    int pixel_num = width * height;
    float4* pixels = film->get_pixels();
    float4* ref = ref_buffer.data();
    float* error = error_buffer.data();
    Metrics::Type t = type;
    Option opt = option;
    tcnn::parallel_for_gpu(pixel_num, [=] __device__(int i) {
        float3 x = make_float3(pixels[i]);
        float3 r = make_float3(ref[i]);
        error[i] = Metrics::cal_pixel(x, r, t);

        if (opt == Option::Reference)
            pixels[i] = make_float4(r, 1.0f);
        else if (opt == Option::Error)
            pixels[i] = make_float4(error[i], error[i], error[i], 1.0f);
    });

    thrust::sort(thrust::device, error, error + pixel_num);
    pixel_num = (int)(pixel_num * (1.0f - discard / 100.0f));
    float err = thrust::reduce(thrust::device, error, error + pixel_num, 0.0f, thrust::plus<float>()) / pixel_num;
    checkCudaErrors(cudaDeviceSynchronize());
    frame_count++;

    if (!online)
        cout << type_to_string(type) << " error: " << err << endl << endl;
    else if (frame_count % record_interval == 1)
    {
        error_history.push_back(err);
        if (error_history.size() > max_history_size)
            error_history.pop_front();
    }
}

void Measure::render_ui()
{
    if (ImGui::CollapsingHeader("Measure"))
    {
        ImGui::Checkbox("Enable##Measure", &enable);
        ImGui::Combo("type##Measure", (int*)&type, "MSE\0MAPE\0SMAPE\0RelMSE\0");
        ImGui::SliderFloat("discard(%)##Measure", &discard, 0.01f, 0.2f);
        ImGui::Combo("option##Measure", (int*)&option, "None\0Reference\0Error\0");

        if (!error_history.empty())
            ImGui::Text("Error: %.6f", error_history.back());
        if (ImGui::TreeNode("Error History"))
        {
            if (ImGui::Button("Clear##Measure"))
                error_history.clear();

            vector<float> error_vec(error_history.begin(), error_history.end());
            std::for_each(error_vec.begin(), error_vec.end(), [](float& x) { x = log10f(x); });
            
            if (ImPlot::BeginPlot("##Measure", ImVec2(-1, 0)))
            {
                ImPlot::SetupAxis(ImAxis_X1, "Frame", 0);
                ImPlot::SetupAxis(ImAxis_Y1, "log(Error)", 0);
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, max_history_size);
                ImPlot::SetupAxisLimits(ImAxis_Y1, -2.5, 0.5);
                ImPlot::PlotLine("", error_vec.data(), error_vec.size());
                ImPlot::EndPlot();
            }

            ImGui::TreePop();
        }
    }
}

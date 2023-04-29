#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"
#include "implot.h"
#include "implot_internal.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "helper_cuda.h"
#include "scene/camera.h"
#include "scene/material.h"

#include "definition.h"
#include <filesystem>

class WindowUserData
{
public:
    shared_ptr<Camera> camera;
    bool first_mouse;
    float last_x, last_y, last_time;

    WindowUserData(shared_ptr<Camera> _c): camera(_c), first_mouse(true), last_x(0), last_y(0), last_time(0) {}
};


class Gui
{
protected:
    GLFWwindow* window;
    WindowUserData user_data;
    unsigned program_id;

    unsigned load_shader(GLenum type, std::string filepath);
    unsigned load_texture(float* data, int tex_w, int tex_h);
    unsigned create_program(std::string vertex_path, std::string fragment_path);
    unsigned create_vao(float* vertices, int size);

public:
    Gui(int _w, int _h, string name, shared_ptr<Camera> _c);
    ~Gui();

    bool should_close();
    void process_input();
};


class MaterialAdjustGui: public Gui
{
private:
    shared_ptr<Material> mat;
    unsigned vao_id;
    int tex_width, tex_height;
    unsigned tex_id;
    cudaGraphicsResource_t cuda_tex_resource;

public:
    MaterialAdjustGui(int _w, int _h, shared_ptr<Camera> _c, shared_ptr<Material> _m);
    ~MaterialAdjustGui();
    void run(unsigned char* data);
};


class InteractiveGui: public Gui
{
private:
    unsigned vao_id;
    int tex_width, tex_height;
    unsigned tex_id;
    cudaGraphicsResource_t cuda_tex_resource;

public:
    InteractiveGui(int _w, int _h, shared_ptr<Camera> _c);
    ~InteractiveGui();
    void run(unsigned char* data, std::fstream& file);
};


class ComparisonGui: public Gui
{
private:
    int tex_width, tex_height;
    
    unsigned vao_id_1, vao_id_2;
    unsigned tex_id_1, tex_id_2;
    cudaGraphicsResource_t cuda_tex_resource_1, cuda_tex_resource_2;

public:
    ComparisonGui(int _w, int _h, shared_ptr<Camera> _c);
    ~ComparisonGui();
    void run(unsigned char* data_1, unsigned char* data_2);
};
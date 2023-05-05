#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "definition.h"
#include "helper_cuda.h"

#include "scene/camera.h"


class WindowUserData
{
public:
    shared_ptr<Camera> camera;
    bool first_mouse;
    float last_x, last_y, last_time;

    WindowUserData(shared_ptr<Camera> camera)
        : camera(camera), first_mouse(true), last_x(0), last_y(0), last_time(0)
    {}
};

class GUI
{
protected:
    GLFWwindow* window;
    WindowUserData user_data;
    unsigned program_id;

    int num_textures;
    int tex_width, tex_height;
    vector<unsigned> vaos;
    vector<unsigned> texs;
    vector<cudaGraphicsResource_t> cuda_texs;

    unsigned load_shader(GLenum type, string filename);
    unsigned load_texture(float* data, int w, int h);
    unsigned create_program(string vertex_path, string fragment_path);
    unsigned create_vao(float* vertices, int size);

    void process_input();
public:
    GUI(int _w, int _h, shared_ptr<Camera> _c, int n = 1);
    ~GUI();

    bool should_close();
    void begin_frame();
    void write_texture(float4* data, int idx = 0);
    void end_frame();
};
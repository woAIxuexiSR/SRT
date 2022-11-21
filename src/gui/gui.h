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

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

class WindowUserData
{
public:
    std::shared_ptr<Camera> camera;
    bool firstMouse;
    float lastX, lastY, lastTime;

    WindowUserData(std::shared_ptr<Camera> _c) : camera(_c), firstMouse(true), lastX(0), lastY(0), lastTime(0) {}
};


class Gui
{
private:
    WindowUserData userData;

    GLFWwindow* window;
    unsigned programId;
    unsigned vaoId;

    int texWidth, texHeight;
    unsigned texId;
    cudaGraphicsResource_t cudaTexResource;

    unsigned loadShader(GLenum type, std::string filepath);
    unsigned loadTexture(float* data);
    unsigned createProgram(std::string vertexPath, std::string fragmentPath);
    unsigned createVAO();

public:
    Gui(int _w, int _h, std::shared_ptr<Camera> _c);
    ~Gui();

    bool shouldClose();
    void run(unsigned char* data);
};
#include "gui.h"

unsigned Gui::load_shader(GLenum type, std::string filepath)
{
    std::fstream fs(filepath, std::ios::in);
    if (!fs.is_open())
    {
        std::cout << "Failed to open shader file " << filepath << std::endl;
        exit(-1);
    }

    std::stringstream ss;
    ss << fs.rdbuf();
    std::string str = ss.str();
    fs.close();

    const char* shader_source = str.c_str();

    unsigned int shader_id;
    shader_id = glCreateShader(type);
    glShaderSource(shader_id, 1, &shader_source, nullptr);
    glCompileShader(shader_id);

    int success;
    char info_log[512];
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(shader_id, 512, nullptr, info_log);
        std::cout << "Compile shader file " << filepath << " error!" << std::endl;
        std::cout << info_log << std::endl;
        exit(-1);
    }

    return shader_id;
}

unsigned Gui::load_texture(float* data, int tex_w, int tex_h)
{
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    GLenum format = GL_RGBA;

    glTexImage2D(GL_TEXTURE_2D, 0, format, tex_w, tex_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    return texture;
}

unsigned Gui::create_program(std::string vertexPath, std::string fragmentPath)
{
    unsigned int vertex_shader, fragment_shader;
    vertex_shader = load_shader(GL_VERTEX_SHADER, vertexPath);
    fragment_shader = load_shader(GL_FRAGMENT_SHADER, fragmentPath);

    unsigned int program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    int success;
    char info_log[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(program, 512, nullptr, info_log);
        std::cout << "Failed to link program!" << std::endl;
        std::cout << info_log << std::endl;
        exit(-1);
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    return program;
}

unsigned Gui::create_vao(float* vertices, int size)
{
    unsigned int vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    unsigned int vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    glDeleteBuffers(1, &vbo);

    return vao;
}

Gui::Gui(int _w, int _h, string name, shared_ptr<Camera> _c)
    : user_data(_c)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(_w, _h, name.c_str(), nullptr, nullptr);
    if (!window)
    {
        std::cout << "Failed to create window!" << std::endl;
        glfwTerminate();
        exit(-1);
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize glad!" << std::endl;
        exit(-1);
    }

    glViewport(0, 0, _w, _h);
    glfwSetWindowUserPointer(window, &user_data);
    auto framebuffer_size_callback = [](GLFWwindow* window, int w, int h) {
        glViewport(0, 0, w, h);
    };
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    auto cursor_pos_callback = [](GLFWwindow* window, double x, double y) {
        WindowUserData* ud = (WindowUserData*)glfwGetWindowUserPointer(window);
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
        {
            if (ud->first_mouse)
            {
                ud->last_x = static_cast<float>(x);
                ud->last_y = static_cast<float>(y);
                ud->first_mouse = false;
            }
            float cursor_speed = 0.04f;
            float xoffset = (static_cast<float>(x) - ud->last_x) * cursor_speed;
            float yoffset = (ud->last_y - static_cast<float>(y)) * cursor_speed;
            ud->last_x = static_cast<float>(x);
            ud->last_y = static_cast<float>(y);
            ud->camera->process_mouse_input(xoffset, yoffset);
        }
        else ud->first_mouse = true;
    };
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    GLFWscrollfun scroll_callback = [](GLFWwindow* window, double x, double y) {
        WindowUserData* ud = (WindowUserData*)glfwGetWindowUserPointer(window);
        ud->camera->process_scroll_input(static_cast<float>(y));
    };
    glfwSetScrollCallback(window, scroll_callback);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    const char* glsl_version = "#version 460";
    ImGui_ImplOpenGL3_Init(glsl_version);

    std::filesystem::path p(__FILE__);
    auto shaderPath = p.parent_path();
    auto vertex_shaderPath = shaderPath / "hello.vert";
    auto fragment_shaderPath = shaderPath / "hello.frag";
    program_id = create_program(vertex_shaderPath.string(), fragment_shaderPath.string());
}

Gui::~Gui()
{
    glDeleteProgram(program_id);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

bool Gui::should_close()
{
    return glfwWindowShouldClose(window);
}

void Gui::process_input()
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    WindowUserData* ud = (WindowUserData*)glfwGetWindowUserPointer(window);
    float deltaTime = static_cast<float>(glfwGetTime()) - ud->last_time;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        ud->camera->process_keyboard_input(ACTION::UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        ud->camera->process_keyboard_input(ACTION::DOWN, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        ud->camera->process_keyboard_input(ACTION::LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        ud->camera->process_keyboard_input(ACTION::RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        ud->camera->process_keyboard_input(ACTION::FRONT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        ud->camera->process_keyboard_input(ACTION::BACK, deltaTime);
    ud->last_time = static_cast<float>(glfwGetTime());
}

MaterialAdjustGui::MaterialAdjustGui(int _w, int _h, shared_ptr<Camera> _c, shared_ptr<Material> _m)
    : Gui(_w, _h, "Material Adjust", _c), tex_width(_w), tex_height(_h), mat(_m)
{
    float quad_vertices[] =
    {
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    int size = sizeof(quad_vertices);

    vao_id = create_vao(quad_vertices, size);
    tex_id = load_texture(nullptr, tex_width, tex_height);
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_resource, tex_id,
        GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

MaterialAdjustGui::~MaterialAdjustGui()
{
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_tex_resource));
    glDeleteVertexArrays(1, &vao_id);
    glDeleteTextures(1, &tex_id);
}

void MaterialAdjustGui::run(unsigned char* data)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("SRT");
    ImGui::Text("Application Time %.1f s", glfwGetTime());
    ImGui::Text("Application average %.1f FPS", ImGui::GetIO().Framerate);
    // ImGui::DragFloat("radius", &user_data.camera->radius, 0.2f, 1.0f, 30.0f);

    ImGui::ColorEdit3("color", &mat->color.x);
    ImGui::SliderFloat("ior", &mat->params[0], 0.5f, 1.5f);
    ImGui::SliderFloat("metallic", &mat->params[1], 0.0f, 1.0f);
    ImGui::SliderFloat("subsurface", &mat->params[2], 0.0f, 1.0f);
    ImGui::SliderFloat("roughness", &mat->params[3], 0.0f, 1.0f);
    ImGui::SliderFloat("specular", &mat->params[4], 0.0f, 1.0f);
    ImGui::SliderFloat("specularTint", &mat->params[5], 0.0f, 1.0f);
    ImGui::SliderFloat("anisotropic", &mat->params[6], 0.0f, 1.0f);
    ImGui::SliderFloat("sheen", &mat->params[7], 0.0f, 1.0f);
    ImGui::SliderFloat("sheenTint", &mat->params[8], 0.0f, 1.0f);
    ImGui::SliderFloat("clearcoat", &mat->params[9], 0.0f, 1.0f);
    ImGui::SliderFloat("clearcoatGloss", &mat->params[10], 0.0f, 1.0f);
    ImGui::SliderFloat("specTrans", &mat->params[11], 0.0f, 1.0f);

    ImGui::End();

    // ImPlot::ShowDemoWindow();

    ImGui::Render();

    process_input();

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    cudaArray* cuda_tex_array;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
        &cuda_tex_array, cuda_tex_resource, 0, 0));
    checkCudaErrors(cudaMemcpy2DToArray(cuda_tex_array, 0, 0, data, tex_width * sizeof(uchar4),
        tex_width * sizeof(uchar4), tex_height, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));

    glUseProgram(program_id);
    glBindVertexArray(vao_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
    glfwPollEvents();
}

InteractiveGui::InteractiveGui(int _w, int _h, shared_ptr<Camera> _c)
    : Gui(_w, _h, "Interactive", _c), tex_width(_w), tex_height(_h)
{
    float quad_vertices[] =
    {
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    int size = sizeof(quad_vertices);

    vao_id = create_vao(quad_vertices, size);
    tex_id = load_texture(nullptr, tex_width, tex_height);
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_resource, tex_id,
        GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

InteractiveGui::~InteractiveGui()
{
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_tex_resource));
    glDeleteVertexArrays(1, &vao_id);
    glDeleteTextures(1, &tex_id);
}

void InteractiveGui::run(unsigned char* data)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("SRT");
    ImGui::Text("Application Time %.1f s", glfwGetTime());
    ImGui::Text("Application average %.1f FPS", ImGui::GetIO().Framerate);
    // ImGui::DragFloat("radius", &user_data.camera->radius, 0.2f, 1.0f, 30.0f);

    ImGui::End();

    // ImPlot::ShowDemoWindow();

    ImGui::Render();

    process_input();

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    cudaArray* cuda_tex_array;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
        &cuda_tex_array, cuda_tex_resource, 0, 0));
    checkCudaErrors(cudaMemcpy2DToArray(cuda_tex_array, 0, 0, data, tex_width * sizeof(uchar4),
        tex_width * sizeof(uchar4), tex_height, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));

    glUseProgram(program_id);
    glBindVertexArray(vao_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
    glfwPollEvents();
}

ComparisonGui::ComparisonGui(int _w, int _h, shared_ptr<Camera> _c)
    : Gui(_w, _h, "Comparison", _c), tex_width(_w), tex_height(_h)
{
    float quad_vertices[] =
    {
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    int size = sizeof(quad_vertices);

    vao_id_1 = create_vao(quad_vertices, size / 2);
    vao_id_2 = create_vao(quad_vertices + 12, size / 2);
    tex_id_1 = load_texture(nullptr, tex_width, tex_height);
    tex_id_2 = load_texture(nullptr, tex_width, tex_height);
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_resource_1, tex_id_1,
        GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_resource_2, tex_id_2,
        GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

ComparisonGui::~ComparisonGui()
{
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_tex_resource_1));
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_tex_resource_2));
    glDeleteVertexArrays(1, &vao_id_1);
    glDeleteVertexArrays(1, &vao_id_2);
    glDeleteTextures(1, &tex_id_1);
    glDeleteTextures(1, &tex_id_2);
}

void ComparisonGui::run(unsigned char* data_1, unsigned char* data_2)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("SRT");
    ImGui::Text("Application Time %.1f s", glfwGetTime());
    ImGui::Text("Application average %.1f FPS", ImGui::GetIO().Framerate);
    // ImGui::DragFloat("radius", &user_data.camera->radius, 0.2f, 1.0f, 30.0f);

    ImGui::End();

    // ImPlot::ShowDemoWindow();

    ImGui::Render();

    process_input();

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    cudaArray* cuda_tex_array_1;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource_1, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
        &cuda_tex_array_1, cuda_tex_resource_1, 0, 0));
    checkCudaErrors(cudaMemcpy2DToArray(cuda_tex_array_1, 0, 0, data_1, tex_width * sizeof(uchar4),
        tex_width * sizeof(uchar4), tex_height, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource_1, 0));
    cudaArray* cuda_tex_array_2;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource_2, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
        &cuda_tex_array_2, cuda_tex_resource_2, 0, 0));
    checkCudaErrors(cudaMemcpy2DToArray(cuda_tex_array_2, 0, 0, data_2, tex_width * sizeof(uchar4),
        tex_width * sizeof(uchar4), tex_height, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource_2, 0));

    glUseProgram(program_id);
    glBindVertexArray(vao_id_1);
    glBindTexture(GL_TEXTURE_2D, tex_id_1);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(vao_id_2);
    glBindTexture(GL_TEXTURE_2D, tex_id_2);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
    glfwPollEvents();
}
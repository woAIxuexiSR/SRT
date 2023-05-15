#include "gui.h"

unsigned GUI::load_shader(GLenum type, string filename)
{
    std::fstream fs(filename, std::ios::in);
    if (!fs.is_open())
    {
        std::cout << "ERROR::Failed to open shader file: " << filename << std::endl;
        exit(-1);
    }

    std::stringstream ss;
    ss << fs.rdbuf();
    std::string str = ss.str();
    fs.close();

    const char* shader_source = str.c_str();

    unsigned int shader_id = glCreateShader(type);
    glShaderSource(shader_id, 1, &shader_source, nullptr);
    glCompileShader(shader_id);

    int success;
    char info_log[512];
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(shader_id, 512, nullptr, info_log);
        std::cout << "ERROR::Compile shader failed: " << info_log << std::endl;
        exit(-1);
    }

    return shader_id;
}

unsigned GUI::load_texture(float* data, int w, int h)
{
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    GLenum format = GL_RGBA32F;

    glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, GL_RGBA, GL_FLOAT, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    return texture;
}

unsigned GUI::create_program(string vertex_path, string fragment_path)
{
    unsigned int vertex_shader = load_shader(GL_VERTEX_SHADER, vertex_path);
    unsigned int fragment_shader = load_shader(GL_FRAGMENT_SHADER, fragment_path);

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
        std::cout << "ERROR::Link program failed: " << info_log << std::endl;
        exit(-1);
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    return program;
}

unsigned GUI::create_vao(float* vertices, int size)
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

GUI::GUI(int _w, int _h, shared_ptr<Camera> _c, int n)
    : user_data(_c), num_textures(n), tex_width(_w), tex_height(_h)
{
    // initialize glfw
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(_w, _h, "SRT", nullptr, nullptr);
    if (!window)
    {
        std::cout << "ERROR::Failed to create window" << std::endl;
        glfwTerminate();
        exit(-1);
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "ERROR::Failed to initialize GLAD" << std::endl;
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

    auto scroll_callback = [](GLFWwindow* window, double x, double y) {
        WindowUserData* ud = (WindowUserData*)glfwGetWindowUserPointer(window);
        ud->camera->process_scroll_input(static_cast<float>(y));
    };
    glfwSetScrollCallback(window, scroll_callback);

    glfwSwapInterval(0);

    // initialize imgui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsLight();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460");

    // create shader
    std::filesystem::path folder(__FILE__);
    folder = folder.parent_path();
    program_id = create_program((folder / "hello.vert"), (folder / "hello.frag"));

    // create vao, texture
    vaos.resize(num_textures);
    texs.resize(num_textures);
    cuda_texs.resize(num_textures);
    for (int i = 0; i < num_textures; i++)
    {
        float tl = i / num_textures, tr = (i + 1) / num_textures;
        float vl = tl * 2.0f - 1.0f, vr = tr * 2.0f - 1.0f;
        float quad_vertices[] =
        {
            vl,  1.0f, tl, 1.0f,
            vl, -1.0f, tl, 0.0f,
            vr, -1.0f, tr, 0.0f,

            vl,  1.0f, tl, 1.0f,
            vr, -1.0f, tr, 0.0f,
            vr,  1.0f, tr, 1.0f
        };
        int size = sizeof(quad_vertices);

        vaos[i] = create_vao(quad_vertices, size);
        texs[i] = load_texture(nullptr, tex_width, tex_height);
        checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_texs[i], texs[i],
            GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    }
}

GUI::~GUI()
{
    glDeleteProgram(program_id);

    for (int i = 0; i < num_textures; i++)
    {
        cudaGraphicsUnregisterResource(cuda_texs[i]);
        glDeleteVertexArrays(1, &vaos[i]);
        glDeleteTextures(1, &texs[i]);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

bool GUI::should_close()
{
    return glfwWindowShouldClose(window);
}

void GUI::process_input()
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    WindowUserData* ud = (WindowUserData*)glfwGetWindowUserPointer(window);
    float deltaTime = static_cast<float>(glfwGetTime()) - ud->last_time;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        ud->camera->process_keyboard_input(CameraMovement::UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        ud->camera->process_keyboard_input(CameraMovement::DOWN, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        ud->camera->process_keyboard_input(CameraMovement::LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        ud->camera->process_keyboard_input(CameraMovement::RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        ud->camera->process_keyboard_input(CameraMovement::FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        ud->camera->process_keyboard_input(CameraMovement::BACKWARD, deltaTime);
    ud->last_time = static_cast<float>(glfwGetTime());
}

void GUI::begin_frame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("SRT");
    ImGui::Text("Application Time %.1f s", ImGui::GetTime());
    ImGui::Text("FPS %.1f", ImGui::GetIO().Framerate);
}

void GUI::write_texture(float4* data, int idx)
{
    cudaArray* cuda_tex_array;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_texs[idx]));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
        &cuda_tex_array, cuda_texs[idx], 0, 0));
    checkCudaErrors(cudaMemcpy2DToArray(cuda_tex_array, 0, 0, data, tex_width * sizeof(float4),
        tex_width * sizeof(float4), tex_height, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_texs[idx]));
}

void GUI::end_frame()
{
    ImGui::End();

    ImGui::Render();

    process_input();

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(program_id);
    for (int i = 0; i < num_textures; i++)
    {
        glBindVertexArray(vaos[i]);
        glBindTexture(GL_TEXTURE_2D, texs[i]);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
    glfwPollEvents();
}

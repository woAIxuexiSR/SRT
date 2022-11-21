#include "gui.h"

unsigned Gui::loadShader(GLenum type, std::string filepath)
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

    const char* shaderSource = str.c_str();

    unsigned int shaderId;
    shaderId = glCreateShader(type);
    glShaderSource(shaderId, 1, &shaderSource, nullptr);
    glCompileShader(shaderId);

    int success;
    char infoLog[512];
    glGetShaderiv(shaderId, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(shaderId, 512, nullptr, infoLog);
        std::cout << "Compile shader file " << filepath << " error!" << std::endl;
        std::cout << infoLog << std::endl;
        exit(-1);
    }

    return shaderId;
}

unsigned Gui::loadTexture(float* data)
{
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    GLenum format = GL_RGBA;

    glTexImage2D(GL_TEXTURE_2D, 0, format, texWidth, texHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    return texture;
}

unsigned Gui::createProgram(std::string vertexPath, std::string fragmentPath)
{
    unsigned int vertexShader, fragmentShader;
    vertexShader = loadShader(GL_VERTEX_SHADER, vertexPath);
    fragmentShader = loadShader(GL_FRAGMENT_SHADER, fragmentPath);

    unsigned int program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    int success;
    char infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cout << "Failed to link program!" << std::endl;
        std::cout << infoLog << std::endl;
        exit(-1);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

unsigned Gui::createVAO()
{
    float quadVertices[] =
    {
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    int size = sizeof(quadVertices);

    unsigned int vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    unsigned int vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, quadVertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    glDeleteBuffers(1, &vbo);

    return vao;
}

Gui::Gui(int _w, int _h, std::shared_ptr<Camera> _c) : texWidth(_w), texHeight(_h), userData(_c)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(_w, _h, "SRT", nullptr, nullptr);
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
    glfwSetWindowUserPointer(window, &userData);
    auto framebuffer_size_callback = [](GLFWwindow* window, int w, int h) {
        glViewport(0, 0, w, h);
    };
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    auto cursor_pos_callback = [](GLFWwindow* window, double x, double y) {
        WindowUserData* ud = (WindowUserData*)glfwGetWindowUserPointer(window);
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        {
            if (ud->firstMouse)
            {
                ud->lastX = static_cast<float>(x);
                ud->lastY = static_cast<float>(y);
                ud->firstMouse = false;
            }
            float xoffset = static_cast<float>(x) - ud->lastX;
            float yoffset = ud->lastY - static_cast<float>(y);
            ud->lastX = static_cast<float>(x);
            ud->lastY = static_cast<float>(y);
            ud->camera->ProcessMouseInput(xoffset, yoffset);
        }
        else ud->firstMouse = true;
    };
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    GLFWscrollfun scroll_callback = [](GLFWwindow* window, double x, double y) {
        WindowUserData* ud = (WindowUserData*)glfwGetWindowUserPointer(window);
        ud->camera->ProcessScrollInput(static_cast<float>(y));
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
    auto vertexShaderPath = shaderPath / "hello.vert";
    auto fragmentShaderPath = shaderPath / "hello.frag";
    programId = createProgram(vertexShaderPath.string(), fragmentShaderPath.string());

    vaoId = createVAO();

    texId = loadTexture(nullptr);
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaTexResource, texId,
        GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

Gui::~Gui()
{
    glDeleteProgram(programId);
    glDeleteVertexArrays(1, &vaoId);
    glDeleteTextures(1, &texId);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

bool Gui::shouldClose()
{
    return glfwWindowShouldClose(window);
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    WindowUserData* ud = (WindowUserData*)glfwGetWindowUserPointer(window);
    float deltaTime = static_cast<float>(glfwGetTime()) - ud->lastTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        ud->camera->ProcessKeyboardInput(ACTION::UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        ud->camera->ProcessKeyboardInput(ACTION::DOWN, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        ud->camera->ProcessKeyboardInput(ACTION::LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        ud->camera->ProcessKeyboardInput(ACTION::RIGHT, deltaTime);
    ud->lastTime = static_cast<float>(glfwGetTime());
}

void Gui::run(unsigned char* data)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("SRT");
    ImGui::Text("Application Time %.1f s", glfwGetTime());
    ImGui::Text("Application average %.1f FPS", ImGui::GetIO().Framerate);
    ImGui::DragFloat("radius", &userData.camera->radius, 0.2f, 1.0f, 10.0f);

    ImGui::End();

    // ImPlot::ShowDemoWindow();

    ImGui::Render();

    processInput(window);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    cudaArray* cudaTexArray;
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaTexResource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
        &cudaTexArray, cudaTexResource, 0, 0));
    checkCudaErrors(cudaMemcpy2DToArray(cudaTexArray, 0, 0, data, texWidth * sizeof(uchar4),
        texWidth * sizeof(uchar4), texHeight, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaTexResource, 0));

    glUseProgram(programId);
    glBindVertexArray(vaoId);
    glBindTexture(GL_TEXTURE_2D, texId);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
    glfwPollEvents();
}
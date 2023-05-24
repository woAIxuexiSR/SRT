#include "pbrtparse.h"

void PBRTState::reset()
{
    material_id = -1;
    emission = make_float3(0.0f, 0.0f, 0.0f);
    transform = Transform();
    mesh = nullptr;
}

string PBRTParser::next_quoted()
{
    string token;
    bool in_quote = false;
    while (!file.eof())
    {
        char c = file.get();
        if (c == '"')
        {
            if (in_quote) break;
            else in_quote = true;
        }
        else if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
        {
            if (in_quote) token += ' ';
        }
        else
        {
            if (!in_quote)
            {
                file.unget();
                return "";
            }
            token += c;
        }
    }
    return token;
}

string PBRTParser::next_bracketed()
{
    string token;
    bool in_bracket = false;
    while (!file.eof())
    {
        char c = file.get();
        if (c == '[')
            in_bracket = true;
        else if (c == ']')
            break;
        else if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
        {
            if (in_bracket) token += ' ';
        }
        else
        {
            if (!in_bracket)
            {
                file.unget();
                return "";
            }
            token += c;
        }
    }
    return token;
}

unordered_map<string, string> PBRTParser::next_parameter_list()
{
    unordered_map<string, string> params;
    string token = next_quoted();
    while (token != "")
    {
        string value = next_bracketed();
        if (value == "")
            value = next_quoted();
        params[token] = value;
        token = next_quoted();
    }
    return params;
}

// read file but ignore the content
void PBRTParser::ignore()
{
    next_quoted();
    next_parameter_list();
}

shared_ptr<TriangleMesh> PBRTParser::load_shape(const string& type, const unordered_map<string, string>& params)
{
    shared_ptr<TriangleMesh> mesh = make_shared<TriangleMesh>();
    if (type == "plymesh")
    {
        auto it = params.find("string filename");
        if (it == params.end())
        {
            LOG_ERROR("No filename specified for plymesh");
            return nullptr;
        }
        string mesh_path = (folderpath / dequote(it->second)).string();
        mesh->load_from_file(mesh_path);
    }
    else if (type == "trianglemesh")
    {
        auto it = params.find("point3 P");
        if (it == params.end())
        {
            LOG_ERROR("No point3 P specified for trianglemesh");
            return nullptr;
        }
        vector<float> P = parse_to_vector<float>(it->second);
        vector<float3> vertices(P.size() / 3);
        memcpy(vertices.data(), P.data(), P.size() * sizeof(float));

        it = params.find("integer indices");
        if (it == params.end())
        {
            LOG_ERROR("No integer indices specified for trianglemesh");
            return nullptr;
        }
        vector<int> I = parse_to_vector<int>(it->second);
        vector<uint3> indices(I.size() / 3);
        memcpy(indices.data(), I.data(), I.size() * sizeof(int));

        it = params.find("normal N");
        vector<float3> normals;
        if (it != params.end())
        {
            vector<float> N = parse_to_vector<float>(it->second);
            normals.resize(N.size() / 3);
            memcpy(normals.data(), N.data(), N.size() * sizeof(float));
        }

        it = params.find("point2 uv");
        vector<float2> texcoords;
        if (it != params.end())
        {
            vector<float> uv = parse_to_vector<float>(it->second);
            texcoords.resize(uv.size() / 2);
            memcpy(texcoords.data(), uv.data(), uv.size() * sizeof(float));
        }

        mesh->load_from_triangles(vertices, indices, normals, texcoords);
    }
    else
    {
        LOG_ERROR("Unsupported shape type: %s", type.c_str());
        return nullptr;
    }

    return mesh;
}

void PBRTParser::load_material(const string& name, const string& type, const unordered_map<string, string>& params)
{
    shared_ptr<Material> material = make_shared<Material>();

    auto it = params.find("rgb reflectance");
    if (it != params.end())
    {
        vector<float> color = parse_to_vector<float>(it->second);
        material->color = make_float3(color[0], color[1], color[2]);
    }
    it = params.find("texture reflectance");
    int texture_id = -1;
    if (it != params.end())
    {
        string texture_name = dequote(it->second);
        texture_id = scene->get_texture_id(texture_name);
        if (texture_id == -1)
        {
            LOG_ERROR("Texture %s not found", texture_name.c_str());
            return;
        }
    }

    material->type = Material::Type::Disney;
    int id = scene->add_material(material, name, texture_id);

    if (in_attribute) attribute_state.material_id = id;
    else global_state.material_id = id;
}

void PBRTParser::load_texture(const string& name, const unordered_map<string, string>& params)
{
    auto it = params.find("string filename");
    if (it == params.end())
    {
        LOG_ERROR("No filename specified for texture");
        return;
    }
    string texture_path = (folderpath / dequote(it->second)).string();

    shared_ptr<Texture> texture = make_shared<Texture>();
    texture->load_from_file(texture_path);
    scene->add_texture(texture, name);
}

PBRTParser::PBRTParser(const string& filename)
{
    std::filesystem::path filepath(filename);
    folderpath = filepath.parent_path();

    if (filename.substr(filename.find_last_of(".") + 1) != "pbrt")
    {
        LOG_ERROR("%s is not a pbrt file", filename.c_str());
        return;
    }

    file.open(filename);
    if (!file.is_open())
    {
        LOG_ERROR("Failed to open file: %s", filename.c_str());
        return;
    }

    global_state.reset();
    attribute_state.reset();
    in_attribute = false;
    scene = make_shared<Scene>();
}

PBRTParser::~PBRTParser()
{
    file.close();
}

void PBRTParser::parse()
{
    shared_ptr<Camera> camera = make_shared<Camera>();
    Transform camera_transform;

    string token;
    while (file >> token)
    {
        if (token == "Integrator")
            ignore();
        else if (token == "Transform")
        {
            string t = next_bracketed();
            vector<float> v = parse_to_vector<float>(t);
            SquareMatrix<4> m(v.data());

            if (in_attribute)
                attribute_state.transform = Transform::Inverse(Transpose(m));
            else
                global_state.transform = Transform::Inverse(Transpose(m));
        }
        else if (token == "Sampler")
            ignore();
        else if (token == "PixelFilter")
            ignore();
        else if (token == "Film")
        {
            next_quoted();      // only support rgb
            auto params = next_parameter_list();
            width = std::stoi(params["integer xresolution"]);
            height = std::stoi(params["integer yresolution"]);
        }
        else if (token == "Camera")
        {
            next_quoted();      // only support perspective
            auto params = next_parameter_list();
            float fov = std::stof(params["float fov"]);
            camera->set_type(Camera::Type::Perspective);
            camera->set_aspect_fov((float)width / (float)height, fov);
            camera_transform = global_state.transform;
        }
        else if (token == "WorldBegin")
        {
            global_state.reset();
        }
        else if (token == "WorldEnd") {}
        else if (token == "MakeNamedMaterial")
        {
            string name = next_quoted();
            auto params = next_parameter_list();
            auto it = params.find("string type");
            if (it == params.end())
            {
                LOG_ERROR("No type specified for material");
                return;
            }
            load_material(name, dequote(it->second), params);
        }
        else if (token == "NamedMaterial")
        {
            string name = next_quoted();
            int id = scene->get_material_id(name);

            if (in_attribute) attribute_state.material_id = id;
            else global_state.material_id = id;
        }
        else if (token == "Material")
        {
            string type = next_quoted();
            auto params = next_parameter_list();
            load_material("", type, params);

        }
        else if (token == "Texture")
        {
            string name = next_quoted();
            next_quoted(); next_quoted(); // only support spectrum imagemap
            auto params = next_parameter_list();
            load_texture(name, params);
        }
        else if (token == "Shape")
        {
            string type = next_quoted();
            auto params = next_parameter_list();
            shared_ptr<TriangleMesh> mesh = load_shape(type, params);

            if (in_attribute) attribute_state.mesh = mesh;
            else
            {
                mesh->apply_transform(global_state.transform);
                assert(global_state.material_id != -1);
                scene->add_mesh(mesh, global_state.material_id);
            }
        }
        else if (token == "AttributeBegin")
        {
            in_attribute = true;
            attribute_state = global_state;
        }
        else if (token == "AttributeEnd")
        {
            attribute_state.mesh->apply_transform(attribute_state.transform);
            scene->materials[attribute_state.material_id]->emission_color = attribute_state.emission;
            scene->materials[attribute_state.material_id]->intensity = 1.0f;
            scene->add_mesh(attribute_state.mesh, attribute_state.material_id);
            in_attribute = false;
        }
        else if (token == "AreaLightSource")
        {
            next_quoted();      // only support diffuse
            auto params = next_parameter_list();
            vector<float> v = parse_to_vector<float>(params["rgb L"]);
            assert(in_attribute == true);
            attribute_state.emission = make_float3(v[0], v[1], v[2]);
        }
        else
        {
            LOG_ERROR("Unsupported token: %s", token.c_str());
        }
    }

    AABB aabb = scene->get_aabb();
    float3 camera_pos = camera_transform.apply_point(make_float3(0.0f, 0.0f, 0.0f));
    float radius = length(aabb.center() - camera_pos);
    camera->set_controller(camera_transform, radius);
    scene->set_camera(camera);
}
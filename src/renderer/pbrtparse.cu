#include "pbrtparse.h"

void PBRTParser::reset_state()
{
    material_id = -1;
    texture_id = -1;
    emission = make_float3(0.0f, 0.0f, 0.0f);
    transform = Transform();
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

void PBRTParser::load_shape(const string& type, const unordered_map<string, string>& params)
{
    shared_ptr<TriangleMesh> mesh = make_shared<TriangleMesh>();
    if (type == "plymesh")
    {
        auto it = params.find("string filename");
        if (it == params.end())
        {
            cout << "ERROR:: No filename specified for plymesh" << endl;
            return;
        }
        string mesh_path = folderpath / dequote(it->second);
        // cout << "Loading mesh: " << mesh_path << endl;
        mesh->load_from_file(mesh_path);
    }
    else if (type == "trianglemesh")
    {
        auto it = params.find("point3 P");
        if (it == params.end())
        {
            cout << "ERROR:: No point3 P specified for trianglemesh" << endl;
            return;
        }
        vector<float> P = parse_to_vector<float>(it->second);
        vector<float3> vertices(P.size() / 3);
        memcpy(vertices.data(), P.data(), P.size() * sizeof(float));

        it = params.find("integer indices");
        if (it == params.end())
        {
            cout << "ERROR:: No integer indices specified for trianglemesh" << endl;
            return;
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
        cout << "ERROR:: Unsupported shape type: " << type << endl;
        return;
    }

    for (int i = 0; i < mesh->vertices.size(); i++)
        mesh->vertices[i] = transform.apply_point(mesh->vertices[i]);
    for (int i = 0; i < mesh->normals.size(); i++)
        mesh->normals[i] = transform.apply_vector(mesh->normals[i]);
    mesh->compute_aabb();

    scene->add_mesh(mesh, material_id);
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
    if (it != params.end())
    {
        string texture_name = dequote(it->second);
        texture_id = scene->get_texture_id(texture_name);
        if (texture_id == -1)
        {
            cout << "ERROR:: Texture " << texture_name << " not found" << endl;
            return;
        }
    }

    material->type = Material::Type::Diffuse;
    material_id = scene->add_material(material, name, texture_id);
}

void PBRTParser::load_texture(const string& name, const unordered_map<string, string>& params)
{
    auto it = params.find("string filename");
    if (it == params.end())
    {
        cout << "ERROR:: No filename specified for texture" << endl;
        return;
    }
    string texture_path = folderpath / dequote(it->second);

    shared_ptr<Texture> texture = make_shared<Texture>();
    texture->load_from_file(texture_path);
    texture_id = scene->add_textures(texture, name);
}

PBRTParser::PBRTParser(const string& filename)
{
    std::filesystem::path filepath(filename);
    folderpath = filepath.parent_path();

    if (filename.substr(filename.find_last_of(".") + 1) != "pbrt")
    {
        cout << "ERROR:: " << filename << " is not a pbrt file" << endl;
        return;
    }

    file.open(filename);
    if (!file.is_open())
    {
        cout << "ERROR:: Failed to open file: " << filename << endl;
        return;
    }

    scene = make_shared<Scene>();
}

PBRTParser::~PBRTParser()
{
    file.close();
}

void PBRTParser::parse()
{
    shared_ptr<Camera> camera = make_shared<Camera>();

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
            transform = Transform(Transpose(m));
            transform = Transform::Inverse(transform);
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
            camera->set_aspect_fov((float)width / (float)height, fov);
        }
        else if (token == "WorldBegin")
        {
            // the former transform is the camera transform
            camera->set_controller(transform, 1.0f);
            reset_state();
        }
        else if (token == "WorldEnd") {}
        else if (token == "MakeNamedMaterial")
        {
            string name = next_quoted();
            auto params = next_parameter_list();
            auto it = params.find("string type");
            if (it == params.end())
            {
                cout << "ERROR:: No type specified for material " << name << endl;
                return;
            }
            load_material(name, dequote(it->second), params);
        }
        else if (token == "NamedMaterial")
        {
            string name = next_quoted();
            material_id = scene->get_material_id(name);
            if (length(emission) > 0.0f)
            {
                scene->materials[material_id]->emission_color = emission;
                scene->materials[material_id]->intensity = 1.0f;
            }
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
            load_shape(type, params);
        }
        else if (token == "AttributeBegin") {}
        else if (token == "AttributeEnd") { /*reset_state();*/ }
        else if (token == "AreaLightSource")
        {
            next_quoted();      // only support diffuse
            auto params = next_parameter_list();
            vector<float> v = parse_to_vector<float>(params["rgb L"]);
            emission = make_float3(v[0], v[1], v[2]);
        }
        else
        {
            cout << "ERROR:: Unknown token: " << token << endl;
        }
    }


    scene->set_camera(camera);
}
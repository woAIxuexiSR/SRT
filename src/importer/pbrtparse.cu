#include "pbrtparse.h"

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

void PBRTParser::ignore()
{
    next_quoted();
    next_parameter_list();
}

shared_ptr<TriangleMesh> PBRTParser::load_shape(const string& type, const unordered_map<string, string>& params)
{
    shared_ptr<TriangleMesh> mesh;
    if (type == "plymesh")
    {
        auto it = params.find("string filename");
        if (it == params.end())
        {
            cout << "ERROR::No filename specified for plymesh" << endl;
            exit(-1);
        }
        string mesh_path = (folder / dequote(it->second)).string();

        std::ifstream f(mesh_path, std::ios::binary);
        if (!f.is_open())
        {
            cout << "ERROR::Failed to open file: " << mesh_path << endl;
            exit(-1);
        }

        tinyply::PlyFile file;
        file.parse_header(f);

        shared_ptr<tinyply::PlyData> vertices_data, faces_data, normals_data, texcoords_data;

        try { vertices_data = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
        catch (const std::exception& e) { cout << "tinyply exception: " << e.what() << endl; }

        try { faces_data = file.request_properties_from_element("face", { "vertex_indices" }, 3); }
        catch (const std::exception& e) { cout << "tinyply exception: " << e.what() << endl; }

        try { normals_data = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
        catch (const std::exception& e) {}

        try { texcoords_data = file.request_properties_from_element("vertex", { "u", "v" }); }
        catch (const std::exception& e) {}

        file.read(f);

        vector<float3> vertices;
        if (vertices_data)
        {
            assert(vertices_data->t == tinyply::Type::FLOAT32);
            vertices.resize(vertices_data->count);
            memcpy(vertices.data(), vertices_data->buffer.get(), vertices_data->buffer.size_bytes());
        }
        vector<uint3> indices;
        if (faces_data)
        {
            assert(faces_data->t == tinyply::Type::UINT32 || faces_data->t == tinyply::Type::INT32);
            indices.resize(faces_data->count);
            memcpy(indices.data(), faces_data->buffer.get(), faces_data->buffer.size_bytes());
        }
        vector<float3> normals;
        if (normals_data)
        {
            assert(normals_data->t == tinyply::Type::FLOAT32);
            normals.resize(normals_data->count);
            memcpy(normals.data(), normals_data->buffer.get(), normals_data->buffer.size_bytes());
        }
        vector<float2> texcoords;
        if (texcoords_data)
        {
            assert(texcoords_data->t == tinyply::Type::FLOAT32);
            texcoords.resize(texcoords_data->count);
            memcpy(texcoords.data(), texcoords_data->buffer.get(), texcoords_data->buffer.size_bytes());
        }

        f.close();

        mesh = make_shared<TriangleMesh>("", vertices, indices, normals, vector<float3>(), texcoords);
    }
    else if (type == "trianglemesh")
    {
        auto it = params.find("point3 P");
        if (it == params.end())
        {
            cout << "ERROR::No point3 P specified for trianglemesh" << endl;
            exit(-1);
        }
        vector<float> P = parse_to_vector<float>(it->second);
        vector<float3> vertices(P.size() / 3);
        memcpy(vertices.data(), P.data(), P.size() * sizeof(float));

        it = params.find("integer indices");
        if (it == params.end())
        {
            cout << "ERROR::No integer indices specified for trianglemesh" << endl;
            exit(-1);
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

        it = params.find("float3 tangent");
        vector<float3> tangents;
        if (it != params.end())
        {
            vector<float> T = parse_to_vector<float>(it->second);
            tangents.resize(T.size() / 3);
            memcpy(tangents.data(), T.data(), T.size() * sizeof(float));
        }

        mesh = make_shared<TriangleMesh>("", vertices, indices, normals, tangents, texcoords);
    }
    else
    {
        cout << "ERROR::Unsupported shape type: " << type << endl;
        exit(-1);
    }

    mesh->compute_aabb();
    return mesh;
}

shared_ptr<Material> PBRTParser::load_material(const string& name, const string& type, const unordered_map<string, string>& params)
{
    BxDF::Type bxdf_type = BxDF::Type::Disney;
    float3 base_color = make_float3(0.0f);
    float ior = 1.5f;
    float metallic = 0.0f;
    float subsurface = 0.0f;
    float roughness = 0.5f;
    float specular = 0.5f;
    float specularTint = 0.0f;
    float anisotropic = 0.0f;
    float sheen = 0.0f;
    float sheenTint = 0.0f;
    float clearcoat = 0.0f;
    float clearcoatGloss = 0.0f;
    float specTrans = 0.0f;
    int tex_id = -1;

    auto it = params.find("texture reflectance");
    if (it != params.end())
    {
        string texname = dequote(it->second);
        tex_id = scene->find_texture(texname);
    }
    it = params.find("rgb reflectance");
    if (it != params.end())
    {
        vector<float> rgb = parse_to_vector<float>(it->second);
        base_color = make_float3(rgb[0], rgb[1], rgb[2]);
    }

    if (type == "diffuse")
        bxdf_type = BxDF::Type::Diffuse;
    else if (type == "coateddiffuse")
    {
        roughness = 0.001f;
        clearcoat = 1.0f;
        clearcoatGloss = 1.0f;
    }
    else if (type == "conductor")
    {
        metallic = 1.0f;
        roughness = 0.001f;
        specularTint = 1.0f;
        base_color = make_float3(1.0f);
    }
    else if (type == "dielectric")
    {
        bxdf_type = BxDF::Type::Dielectric;
        base_color = make_float3(1.0f);
        it = params.find("float eta");
        if (it != params.end())
            ior = std::stoi(it->second);
    }
    else if (type == "diffusetransmission")
        bxdf_type = BxDF::Type::DiffuseTransmission;
    else
    {
        cout << "ERROR::Unsupported material type: " << type << endl;
        exit(-1);
    }


    shared_ptr<Material> material = make_shared<Material>();
    material->name = name;
    material->base_color = base_color;
    material->bxdf.type = bxdf_type;
    material->bxdf.ior = ior;
    material->bxdf.metallic = metallic;
    material->bxdf.subsurface = subsurface;
    material->bxdf.roughness = max(roughness, 0.001f);
    material->bxdf.specular = specular;
    material->bxdf.specularTint = specularTint;
    material->bxdf.anisotropic = anisotropic;
    material->bxdf.sheen = sheen;
    material->bxdf.sheenTint = sheenTint;
    material->bxdf.clearcoat = clearcoat;
    material->bxdf.clearcoatGloss = max(clearcoatGloss, 0.001f);
    material->bxdf.specTrans = specTrans;
    material->color_tex_id = tex_id;
    return material;
}

shared_ptr<Texture> PBRTParser::load_texture(const string& name, const unordered_map<string, string>& params)
{
    auto it = params.find("string filename");
    if (it == params.end())
    {
        cout << "ERROR::No filename specified for texture" << endl;
        exit(-1);
    }
    string tex_path = (folder / dequote(it->second)).string();

    shared_ptr<Texture> texture = make_shared<Texture>();
    texture->name = name;
    texture->image.load_from_file(tex_path);
    return texture;
}

void PBRTParser::parse_options()
{
    float fov = 60.0f;
    float aspect = 1.0f;
    Transform camera_to_world;

    string token;
    while (file >> token)
    {
        if (token == "Integrator")
            ignore();
        else if (token == "Transform")
        {
            string t = next_bracketed();
            vector<float> v = parse_to_vector<float>(t);
            SquareMatrix<4> mat(v.data());
            mat = Inverse(Transpose(mat));      // left handed
            camera_to_world = Transform(SquareMatrix<4>(
                -mat[0][0], mat[0][1], mat[0][2], mat[0][3],
                -mat[1][0], mat[1][1], mat[1][2], mat[1][3],
                -mat[2][0], mat[2][1], mat[2][2], mat[2][3],
                -mat[3][0], mat[3][1], mat[3][2], mat[3][3]
            ));
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
            aspect = float(width) / float(height);
        }
        else if (token == "Camera")
        {
            next_quoted();      // only support perspective
            auto params = next_parameter_list();
            fov = std::stof(params["float fov"]);
        }
        else if (token == "WorldBegin")
            break;
        else
        {
            cout << "ERROR::Unsupported token: " << token << endl;
            exit(-1);
        }
    }

    shared_ptr<Camera> camera = make_shared<Camera>(Camera::Type::Perspective, aspect, fov);
    camera->set_controller(camera_to_world, 1.0f);      // fix radius 1.0
    scene->set_camera(camera);
}

void PBRTParser::parse_world()
{
    string token;
    while (file >> token)
    {
        if (token == "Transform")
        {
            string t = next_bracketed();
            vector<float> v = parse_to_vector<float>(t);
            SquareMatrix<4> mat(v.data());
            if (in_attribute) attribute_state.transform = Transform(Transpose(mat));
            else global_state.transform = Transform(Transpose(mat));
        }
        else if (token == "MakeNamedMaterial")
        {
            string name = next_quoted();
            auto params = next_parameter_list();
            auto it = params.find("string type");
            if (it == params.end())
            {
                cout << "ERROR::No type specified for material" << endl;
                exit(-1);
            }
            shared_ptr<Material> mat = load_material(name, dequote(it->second), params);
            scene->add_material(mat);
        }
        else if (token == "NamedMaterial")
        {
            string name = next_quoted();
            int id = scene->find_material(name);
            if (in_attribute) attribute_state.material_id = id;
            else global_state.material_id = id;
        }
        else if (token == "Material")
        {
            string type = next_quoted();
            auto params = next_parameter_list();
            shared_ptr<Material> mat = load_material("", type, params);
            int id = scene->add_material(mat);
            if (in_attribute) attribute_state.material_id = id;
            else global_state.material_id = id;
        }
        else if (token == "Texture")
        {
            string name = next_quoted();
            next_quoted(); next_quoted();   // only support spectrum imagemap
            auto params = next_parameter_list();
            shared_ptr<Texture> tex = load_texture(name, params);
            scene->add_texture(tex);
        }
        else if (token == "Shape")
        {
            string type = next_quoted();
            auto params = next_parameter_list();
            shared_ptr<TriangleMesh> mesh = load_shape(type, params);

            if (in_attribute) attribute_state.mesh = mesh;
            else
            {
                assert(global_state.material_id != -1);
                mesh->material_id = global_state.material_id;
                scene->add_instance(global_state.transform, mesh);
            }
        }
        else if (token == "AreaLightSource")
        {
            next_quoted();      // only support diffuse
            auto params = next_parameter_list();
            vector<float> v = parse_to_vector<float>(params["rgb L"]);
            assert(in_attribute);
            attribute_state.emission = make_float3(v[0], v[1], v[2]);
        }
        else if (token == "AttributeBegin")
        {
            in_attribute = true;
            attribute_state = global_state;
        }
        else if (token == "AttributeEnd")
        {
            assert(attribute_state.material_id != -1);
            int mid = attribute_state.material_id;
            float3 emission = attribute_state.emission;
            if (dot(emission, emission) > 0.0f)
            {
                float intensity = length(emission);
                emission /= intensity;
                scene->materials[mid]->intensity = intensity;
                scene->materials[mid]->emission_color = emission;
            }
            attribute_state.mesh->material_id = mid;
            scene->add_instance(attribute_state.transform, attribute_state.mesh);
            in_attribute = false;
        }
        else
        {
            cout << "ERROR::Unsupported token: " << token << endl;
            exit(-1);
        }
    }
}

void PBRTParser::parse(const string& filename, shared_ptr<Scene> _s)
{
    std::filesystem::path filepath(filename);
    folder = filepath.parent_path();

    if (filename.substr(filename.find_last_of(".") + 1) != "pbrt")
    {
        cout << "ERROR:: " << filename << " is not a pbrt file" << endl;
        exit(-1);
    }

    file.open(filename);
    if (!file.is_open())
    {
        cout << "ERROR::Failed to open file: " << filename << endl;
        exit(-1);
    }

    scene = _s;
    parse_options();
    parse_world();
    file.close();
}
// #include "pbrtparse.h"

// void PBRTState::reset()
// {
//     material_id = -1;
//     emission = make_float3(0.0f, 0.0f, 0.0f);
//     transform = Transform();
//     mesh = nullptr;
// }

// string PBRTParser::next_quoted()
// {
//     string token;
//     bool in_quote = false;
//     while (!file.eof())
//     {
//         char c = file.get();
//         if (c == '"')
//         {
//             if (in_quote) break;
//             else in_quote = true;
//         }
//         else if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
//         {
//             if (in_quote) token += ' ';
//         }
//         else
//         {
//             if (!in_quote)
//             {
//                 file.unget();
//                 return "";
//             }
//             token += c;
//         }
//     }
//     return token;
// }

// string PBRTParser::next_bracketed()
// {
//     string token;
//     bool in_bracket = false;
//     while (!file.eof())
//     {
//         char c = file.get();
//         if (c == '[')
//             in_bracket = true;
//         else if (c == ']')
//             break;
//         else if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
//         {
//             if (in_bracket) token += ' ';
//         }
//         else
//         {
//             if (!in_bracket)
//             {
//                 file.unget();
//                 return "";
//             }
//             token += c;
//         }
//     }
//     return token;
// }

// unordered_map<string, string> PBRTParser::next_parameter_list()
// {
//     unordered_map<string, string> params;
//     string token = next_quoted();
//     while (token != "")
//     {
//         string value = next_bracketed();
//         if (value == "")
//             value = next_quoted();
//         params[token] = value;
//         token = next_quoted();
//     }
//     return params;
// }

// // read file but ignore the content
// void PBRTParser::ignore()
// {
//     next_quoted();
//     next_parameter_list();
// }

// shared_ptr<TriangleMesh> PBRTParser::load_shape(const string& type, const unordered_map<string, string>& params)
// {
//     shared_ptr<TriangleMesh> mesh = make_shared<TriangleMesh>();
//     if (type == "plymesh")
//     {
//         auto it = params.find("string filename");
//         if (it == params.end())
//         {
//             cout << "ERROR::No filename specified for plymesh" << endl;
//             exit(-1);
//         }
//         string mesh_path = (folderpath / dequote(it->second)).string();
//         mesh->load_from_file(mesh_path);
//     }
//     else if (type == "trianglemesh")
//     {
//         auto it = params.find("point3 P");
//         if (it == params.end())
//         {
//             cout << "ERROR::No point3 P specified for trianglemesh" << endl;
//             exit(-1);
//         }
//         vector<float> P = parse_to_vector<float>(it->second);
//         vector<float3> vertices(P.size() / 3);
//         memcpy(vertices.data(), P.data(), P.size() * sizeof(float));

//         it = params.find("integer indices");
//         if (it == params.end())
//         {
//             cout << "ERROR::No integer indices specified for trianglemesh" << endl;
//             exit(-1);
//         }
//         vector<int> I = parse_to_vector<int>(it->second);
//         vector<uint3> indices(I.size() / 3);
//         memcpy(indices.data(), I.data(), I.size() * sizeof(int));

//         it = params.find("normal N");
//         vector<float3> normals;
//         if (it != params.end())
//         {
//             vector<float> N = parse_to_vector<float>(it->second);
//             normals.resize(N.size() / 3);
//             memcpy(normals.data(), N.data(), N.size() * sizeof(float));
//         }

//         it = params.find("point2 uv");
//         vector<float2> texcoords;
//         if (it != params.end())
//         {
//             vector<float> uv = parse_to_vector<float>(it->second);
//             texcoords.resize(uv.size() / 2);
//             memcpy(texcoords.data(), uv.data(), uv.size() * sizeof(float));
//         }

//         mesh->load_from_triangles(vertices, indices, normals, texcoords);
//     }
//     else
//     {
//         cout << "ERROR::Unsupported shape type: " << type << endl;
//         exit(-1);
//     }

//     return mesh;
// }

// float3 get_float3(const json& j, const string& key)
// {
//     if (j.find(key) == j.end())
//         return { 0.0f, 0.0f, 0.0f };
//     vector<float> v = parse_to_vector<float>(j.at(key).get<string>());
//     return { v[0], v[1], v[2] };
// }

// float get_roughness(const json& j)
// {
//     float roughness = 0.0f;
//     if (j.find("float vroughness") != j.end())
//     {
//         float vroughness = std::stof(j.value("float vroughness", "0.1"));
//         float uroughness = std::stof(j.value("float uroughness", "0.1"));
//         roughness = (vroughness + uroughness) * 0.5f;
//     }
//     if (j.find("float roughness") != j.end())
//         roughness = std::stof(j.at("float roughness").get<string>());
//     bool remap = j.value("bool remaproughness", "false") == "true";
//     if (remap)
//         roughness = roughness * roughness;
//     return roughness;
// }

// float3 get_eta(const json& j)
// {
//     if (j.find("rgb eta") != j.end())
//         return get_float3(j, "rgb eta");
//     if (j.find("float eta") != j.end())
//         return make_float3(std::stof(j.at("float eta").get<string>()));
//     if (j.find("spectrum eta") != j.end())
//     {
//         string t = dequote(j.at("spectrum eta").get<string>());
//         if (t == "metal-Ag-eta")
//             return make_float3(0.155264f, 0.116723f, 0.13838f);
//         else if (t == "metal-Al-eta")
//             return make_float3(1.657f, 0.8803f, 0.521f);
//         else
//         {
//             cout << "ERROR::Unsupported spectrum eta: " << t << endl;
//             exit(-1);
//         }
//     }
//     return make_float3(1.45f);
// }

// float3 get_k(const json& j)
// {
//     if (j.find("rgb k") != j.end())
//         return get_float3(j, "rgb k");
//     if (j.find("spectrum k") != j.end())
//     {
//         string t = dequote(j.at("spectrum k").get<string>());
//         if (t == "metal-Ag-k")
//             return make_float3(4.08169f, 2.48668f, 1.92051f);
//         else if (t == "metal-Al-k")
//             return make_float3(7.64116f, 6.31901f, 5.95860f);
//         else
//         {
//             cout << "ERROR::Unsupported spectrum k: " << t << endl;
//             exit(-1);
//         }
//     }
//     return make_float3(1.0f);
// }

// void PBRTParser::load_material(const string& name, const string& type, const unordered_map<string, string>& params)
// {
//     shared_ptr<Material> material = make_shared<Material>();
//     int texture_id = -1;

//     json j(params);
//     // load texture
//     if (j.find("texture reflectance") != j.end())
//     {
//         string texname = dequote(j.at("texture reflectance"));
//         texture_id = scene->get_texture_id(texname);
//         if (texture_id == -1)
//         {
//             cout << "ERROR::Texture " << texname << " not found" << endl;
//             exit(-1);
//         }
//     }
//     // load parameters
//     if (type == "diffuse")
//     {
//         material->type = Material::Type::Diffuse;
//         float3 color = get_float3(j, "rgb reflectance");
//         material->color = color;
//     }
//     else if (type == "coateddiffuse")
//     {
//         material->type = Material::Type::Disney;
//         float3 color = get_float3(j, "rgb reflectance");
//         material->color = color;
//         float roughness = get_roughness(j);
//         set_material_property(material, "metallic", 0.5f);
//         set_material_property(material, "clearcoat", 1.0f);
//         set_material_property(material, "clearcoatGloss", roughness);
//     }
//     else if (type == "conductor")
//     {
//         material->type = Material::Type::Disney;
//         float roughness = get_roughness(j);
//         set_material_property(material, "metallic", 1.0f - roughness);
//         set_material_property(material, "roughness", roughness);
//         // float3 k = get_k(j);
//         // material->color = k;
//         material->color = make_float3(1.0f);
//         float3 eta = get_eta(j);
//         float ior = (eta.x + eta.y + eta.z) / 3.0f;
//         set_material_property(material, "ior", ior);
//         set_material_property(material, "specularTint", 1.0f);
//     }
//     else if (type == "dielectric")
//     {
//         float roughness = get_roughness(j);
//         float3 eta = get_eta(j);
//         float ior = (eta.x + eta.y + eta.z) / 3.0f;
//         set_material_property(material, "ior", ior);
//         if (roughness == 0.0f)
//             material->type = Material::Type::Dielectric;
//         else
//         {
//             material->type = Material::Type::Disney;
//             set_material_property(material, "roughness", roughness);
//             set_material_property(material, "specTrans", 1.0f);
//         }
//         material->color = make_float3(1.0f);
//     }
//     else if (type == "diffusetransmission")
//     {
//         material->type = Material::Type::DiffuseTransmission;
//         if (j.find("rgb reflectance") == j.end() || j.find("rgb transmittance") == j.end())
//         {
//             cout << "ERROR::DiffuseTransmission material must have both reflectance and transmittance" << endl;
//             exit(-1);
//         }
//         material->color = get_float3(j, "rgb reflectance");
//         float3 transmittance = get_float3(j, "rgb transmittance");
//         material->params[0] = transmittance.x;
//         material->params[1] = transmittance.y;
//         material->params[2] = transmittance.z;
//     }
//     else
//     {
//         cout << "ERROR::Unsupported material type: " << type << endl;
//         exit(-1);
//     }

//     int id = scene->add_material(material, name, texture_id);
//     if (in_attribute)
//         attribute_state.material_id = id;
//     else
//         global_state.material_id = id;
// }

// void PBRTParser::load_texture(const string& name, const unordered_map<string, string>& params)
// {
//     auto it = params.find("string filename");
//     if (it == params.end())
//     {
//         cout << "ERROR::No filename specified for texture" << endl;
//         exit(-1);
//     }
//     string texture_path = (folderpath / dequote(it->second)).string();

//     shared_ptr<Texture> texture = make_shared<Texture>();
//     texture->load_from_file(texture_path);
//     scene->add_texture(texture, name);
// }

// PBRTParser::PBRTParser(const string& filename)
// {
//     std::filesystem::path filepath(filename);
//     folderpath = filepath.parent_path();

//     if (filename.substr(filename.find_last_of(".") + 1) != "pbrt")
//     {
//         cout << "ERROR:: " << filename << " is not a pbrt file" << endl;
//         exit(-1);
//     }

//     file.open(filename);
//     if (!file.is_open())
//     {
//         cout << "ERROR::Failed to open file: " << filename << endl;
//         exit(-1);
//     }

//     global_state.reset();
//     attribute_state.reset();
//     in_attribute = false;
//     scene = make_shared<Scene>();
// }

// PBRTParser::~PBRTParser()
// {
//     file.close();
// }

// void PBRTParser::parse()
// {
//     shared_ptr<Camera> camera = make_shared<Camera>();
//     Transform camera_transform;

//     string token;
//     while (file >> token)
//     {
//         if (token == "Integrator")
//             ignore();
//         else if (token == "Transform")
//         {
//             string t = next_bracketed();
//             vector<float> v = parse_to_vector<float>(t);
//             SquareMatrix<4> m(v.data());
//             // Transform transform = Transform::Inverse(Transpose(m));
//             Transform transform = Transpose(m);

//             if (in_attribute)
//                 attribute_state.transform = transform;
//             else
//                 global_state.transform = transform;
//         }
//         else if (token == "Sampler")
//             ignore();
//         else if (token == "PixelFilter")
//             ignore();
//         else if (token == "Film")
//         {
//             next_quoted();      // only support rgb
//             auto params = next_parameter_list();
//             width = std::stoi(params["integer xresolution"]);
//             height = std::stoi(params["integer yresolution"]);
//         }
//         else if (token == "Camera")
//         {
//             next_quoted();      // only support perspective
//             auto params = next_parameter_list();
//             float fov = std::stof(params["float fov"]);
//             camera->set_type(Camera::Type::Perspective);
//             camera->set_aspect_fov((float)width / (float)height, fov);
//             camera_transform = Transform::Inverse(global_state.transform);
//         }
//         else if (token == "WorldBegin")
//         {
//             global_state.reset();
//         }
//         else if (token == "WorldEnd") {}
//         else if (token == "MakeNamedMaterial")
//         {
//             string name = next_quoted();
//             auto params = next_parameter_list();
//             auto it = params.find("string type");
//             if (it == params.end())
//             {
//                 cout << "ERROR::No type specified for material" << endl;
//                 exit(-1);
//             }
//             load_material(name, dequote(it->second), params);
//         }
//         else if (token == "NamedMaterial")
//         {
//             string name = next_quoted();
//             int id = scene->get_material_id(name);

//             if (in_attribute) attribute_state.material_id = id;
//             else global_state.material_id = id;
//         }
//         else if (token == "Material")
//         {
//             string type = next_quoted();
//             auto params = next_parameter_list();
//             load_material("", type, params);

//         }
//         else if (token == "Texture")
//         {
//             string name = next_quoted();
//             next_quoted(); next_quoted(); // only support spectrum imagemap
//             auto params = next_parameter_list();
//             load_texture(name, params);
//         }
//         else if (token == "Shape")
//         {
//             string type = next_quoted();
//             auto params = next_parameter_list();
//             shared_ptr<TriangleMesh> mesh = load_shape(type, params);

//             if (in_attribute) attribute_state.mesh = mesh;
//             else
//             {
//                 mesh->apply_transform(global_state.transform);
//                 assert(global_state.material_id != -1);
//                 scene->add_mesh(mesh, global_state.material_id);
//             }
//         }
//         else if (token == "AttributeBegin")
//         {
//             in_attribute = true;
//             attribute_state = global_state;
//         }
//         else if (token == "AttributeEnd")
//         {
//             attribute_state.mesh->apply_transform(attribute_state.transform);
//             scene->materials[attribute_state.material_id]->emission_color = attribute_state.emission;
//             scene->materials[attribute_state.material_id]->intensity = 1.0f;
//             scene->add_mesh(attribute_state.mesh, attribute_state.material_id);
//             in_attribute = false;
//         }
//         else if (token == "AreaLightSource")
//         {
//             next_quoted();      // only support diffuse
//             auto params = next_parameter_list();
//             vector<float> v = parse_to_vector<float>(params["rgb L"]);
//             assert(in_attribute == true);
//             attribute_state.emission = make_float3(v[0], v[1], v[2]);
//         }
//         else
//         {
//             cout << "ERROR::Unsupported token: " << token << endl;
//             exit(-1);
//         }
//     }

//     AABB aabb = scene->get_aabb();
//     float3 camera_pos = camera_transform.apply_point(make_float3(0.0f, 0.0f, 0.0f));
//     float radius = length(aabb.center() - camera_pos);
//     camera->set_controller(camera_transform, radius);
//     scene->set_camera(camera);
// }
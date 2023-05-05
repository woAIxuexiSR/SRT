// #include "parser.h"

// string PBRTParser::next_quoted(std::ifstream& file)
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

// string PBRTParser::next_bracketed(std::ifstream& file)
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

// unordered_map<string, string> PBRTParser::next_parameter_list(std::ifstream& file)
// {
//     unordered_map<string, string> params;
//     string token = next_quoted(file);
//     while (token != "")
//     {
//         string value = next_bracketed(file);
//         params[token] = value;
//         token = next_quoted(file);
//     }
//     return params;
// }

// void PBRTParser::parse(const string& filename, RenderParams renderParams)
// {
//     std::filesystem::path filepath(filename);
//     std::filesystem::path folderpath = filepath.parent_path();

//     if (filename.substr(filename.find_last_of(".") + 1) != "pbrt")
//     {
//         cout << filename << " is not a pbrt file" << endl;
//         exit(-1);
//     }

//     std::ifstream file(filename);
//     if (!file.is_open())
//     {
//         cout << "Could not open file " << filename << endl;
//         exit(-1);
//     }

//     string token;

//     // parse configurations
//     bool worldBegin = false;
//     float asp = 1.0f;
//     SquareMatrix<4> transform;
//     while (file >> token)
//     {

//         switch (token[0])
//         {

//         case 'C':
//         {
//             if (token == "Camera")
//             {
//                 string type = next_quoted(file);
//                 auto params = next_parameter_list(file);
//                 float fov = (parse_to_vector<float>(params["float fov"])[0]) * 2;
//                 renderParams.camera->set_from_matrix(transform, fov, asp);
//             }
//             break;
//         }

//         case 'F':
//         {
//             if (token == "Film")
//             {
//                 string type = next_quoted(file);
//                 auto params = next_parameter_list(file);
//                 int width = parse_to_vector<int>(params["integer xresolution"])[0];
//                 int height = parse_to_vector<int>(params["integer yresolution"])[0];
//                 asp = (float)width / (float)height;
//                 renderParams.film->resize(width, height);
//             }
//             break;
//         }

//         case 'I':
//         {
//             if (token == "Integrator")
//             {
//                 cout << "Integrator name: " << next_quoted(file) << endl;
//                 auto params = next_parameter_list(file);
//                 for (auto& param : params)
//                     cout << "Integrator parameter: " << param.first << " " << param.second << endl;
//             }
//             break;
//         }

//         case 'P':
//         {
//             if (token == "PixelFilter")
//             {
//                 cout << "PixelFilter name: " << next_quoted(file) << endl;
//                 auto params = next_parameter_list(file);
//                 for (auto& param : params)
//                     cout << "PixelFilter parameter: " << param.first << " " << param.second << endl;
//             }
//             break;
//         }

//         case 'S':
//         {
//             if (token == "Sampler")
//             {
//                 cout << "Sampler name: " << next_quoted(file) << endl;
//                 auto params = next_parameter_list(file);
//                 for (auto& param : params)
//                     cout << "Sampler parameter: " << param.first << " " << param.second << endl;
//             }
//             break;
//         }

//         case 'T':
//         {
//             if (token == "Transform")
//             {
//                 string mat = next_bracketed(file);
//                 vector<float> v = parse_to_vector<float>(mat);
//                 float m[4][4] = { { v[0], v[1], v[2], v[3] },
//                                 { v[4], v[5], v[6], v[7] },
//                                 { v[8], v[9], v[10], v[11] },
//                                 { v[12], v[13], v[14], v[15] } };
//                 transform = SquareMatrix<4>(m);
//             }
//             break;
//         }

//         case 'W':
//         {
//             if (token == "WorldBegin")
//             {
//                 cout << "WorldBegin" << endl;
//                 worldBegin = true;
//             }
//             break;
//         }

//         }

//         if (worldBegin) break;

//     }

//     // parse world objects
//     int materialId = 0;
//     bool isLight = false;
//     SquareMatrix<4> objectTransform;
//     while (file >> token)
//     {

//         switch (token[0])
//         {

//         case 'A':
//         {
//             if (token == "AttributeBegin")
//             {
//                 cout << "AttributeBegin" << endl;
//             }
//             else if (token == "AttributeEnd")
//             {
//                 isLight = false;
//                 objectTransform = SquareMatrix<4>();
//                 cout << "AttributeEnd" << endl;
//             }
//             else if (token == "AreaLightSource")
//             {
//                 string type = next_quoted(file);
//                 auto params = next_parameter_list(file);
//                 materialId = renderParams.scene->add_light_material(type, params);
//                 isLight = true;
//             }
//             break;
//         }

//         case 'L':
//         {
//             if (token == "LightSource")
//             {
//                 cout << "LightSource name: " << next_quoted(file) << endl;
//                 auto params = next_parameter_list(file);
//                 for (auto& param : params)
//                     cout << "LightSource parameter: " << param.first << " " << param.second << endl;
//             }
//             break;
//         }

//         case 'M':
//         {
//             if (token == "MakeNamedMaterial")
//             {
//                 string name = next_quoted(file);
//                 auto params = next_parameter_list(file);
//                 renderParams.scene->add_named_material(name, params);
//             }
//             else if (token == "Material")
//             {
//                 string type = next_quoted(file);
//                 auto params = next_parameter_list(file);
//                 if (!isLight)
//                 {
//                     materialId = renderParams.scene->add_material(type, params);
//                 }
//             }
//             break;
//         }

//         case 'N':
//         {
//             if (token == "NamedMaterial")
//             {
//                 string name = next_quoted(file);
//                 if (!isLight)
//                 {
//                     materialId = renderParams.scene->get_material_id(name);
//                 }
//             }
//             break;
//         }

//         case 'S':
//         {
//             if (token == "Shape")
//             {
//                 string type = next_quoted(file);
//                 auto params = next_parameter_list(file);
//                 params["folderpath"] = folderpath.string();
//                 renderParams.scene->add_mesh(type, params, materialId, objectTransform);
//                 isLight = false;
//             }
//             break;
//         }

//         case 'T':
//         {
//             if (token == "Transform")
//             {
//                 string mat = next_bracketed(file);
//                 vector<float> v = parse_to_vector<float>(mat);
//                 float m[4][4] = { { v[0], v[1], v[2], v[3] },
//                                 { v[4], v[5], v[6], v[7] },
//                                 { v[8], v[9], v[10], v[11] },
//                                 { v[12], v[13], v[14], v[15] } };
//                 objectTransform = SquareMatrix<4>(m);
//             }
//             else if (token == "Texture")
//             {
//                 string name = next_quoted(file);
//                 string type = next_quoted(file); // "spectrum"
//                 string texType = next_quoted(file); // "imagemap"
//                 auto params = next_parameter_list(file);
//                 params["folderpath"] = folderpath.string();
//                 renderParams.scene->add_texture(name, params);
//             }
//             break;
//         }

//         case 'Z':
//         {
//             if (token == "ZZZ")
//                 cout << "Debug" << endl;
//             break;
//         }

//         default:
//         {
//             cout << token << "is not catched" << endl;
//             break;
//         }

//         }
//     }

// }
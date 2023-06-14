#include <iostream>
#include <cuda_runtime.h>
#include "my_math.h"
#include "scene.h"
#include "assimp.h"

using namespace std;

int main()
{
    std::filesystem::path file(__FILE__);
    // file = file.parent_path().parent_path().parent_path() / "example" / "cornell_box_obj" / "cornell_box.obj";
    file = file.parent_path().parent_path().parent_path() / "vampire" / "dancing_vampire.dae";

    shared_ptr<Scene> scene = make_shared<Scene>();
    AssimpImporter importer;
    importer.import(file.string(), scene);

    cout << scene->meshes.size() << endl;
    cout << scene->materials.size() << endl;
    cout << scene->textures.size() << endl;
    cout << scene->bones.size() << endl;

    return 0;
}
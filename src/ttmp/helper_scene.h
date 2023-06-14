#pragma once

#include "definition.h"
#include "my_math.h"

#include "scene/camera.h"
#include "scene/material.h"

inline void set_material_property(shared_ptr<Material> material, const string& name, float value)
{
    if (name == "ior")
        material->params[0] = value;
    else if (name == "metallic")
        material->params[1] = value;
    else if (name == "subsurface")
        material->params[2] = value;
    else if (name == "roughness")
        material->params[3] = value;
    else if (name == "specular")
        material->params[4] = value;
    else if (name == "specularTint")
        material->params[5] = value;
    else if (name == "anisotropic")
        material->params[6] = value;
    else if (name == "sheen")
        material->params[7] = value;
    else if (name == "sheenTint")
        material->params[8] = value;
    else if (name == "clearcoat")
        material->params[9] = value;
    else if (name == "clearcoatGloss")
        material->params[10] = value;
    else if (name == "specTrans")
        material->params[11] = value;
    else
    {
        cout << "ERROR::Unknown material property: " << name << endl;
        exit(-1);
    }
}
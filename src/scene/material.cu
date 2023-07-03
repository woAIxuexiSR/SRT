#include "material.h"

void Material::set_type(BxDF::Type type)
{
    bxdf.type = type;
}

void Material::set_property(const string& str, float value)
{
    if (str == "ior")
        bxdf.ior = value;
    else if (str == "metallic")
        bxdf.metallic = value;
    else if (str == "subsurface")
        bxdf.subsurface = value;
    else if (str == "roughness")
        bxdf.roughness = value;
    else if (str == "specular")
        bxdf.specular = value;
    else if (str == "specularTint")
        bxdf.specularTint = value;
    else if (str == "anisotropic")
        bxdf.anisotropic = value;
    else if (str == "sheen")
        bxdf.sheen = value;
    else if (str == "sheenTint")
        bxdf.sheenTint = value;
    else if (str == "clearcoat")
        bxdf.clearcoat = value;
    else if (str == "clearcoatGloss")
        bxdf.clearcoatGloss = value;
    else if (str == "specTrans")
        bxdf.specTrans = value;
    else
    {
        cout << "ERROR::Unknown material property: " << str << endl;
        exit(-1);
    }
}
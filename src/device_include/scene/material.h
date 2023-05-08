#pragma once

#include <cuda_runtime.h>
#include "my_math.h"

#define MAX_MATERIAL_PARAMETERS 12

// wi: incoming direction  => from surface to light (sample)
// wo: outgoing direction  => from surface to camera
// n : surface normal, always in the same hemisphere with wo
// inner : whether wo is inside the surface

enum class MaterialType
{
    Diffuse,
    DiffuseTransmission,
    Dielectric,
    Disney
};

enum class ReflectionType
{
    Diffuse,
    Specular
};

class MaterialSample
{
public:
    ReflectionType type;  // reflection type
    float3 f;           // BRDF
    float3 wi;          // incoming direction
    float pdf;          // pdf of wi

    __device__ MaterialSample()
        : type(ReflectionType::Diffuse), f({ 0.0f, 0.0f, 0.0f }), wi({ 0.0f, 0.0f, 0.0f }), pdf(0.0f) {}
    __device__ MaterialSample(ReflectionType _t, float3 _f, float3 _wi, float _pdf)
        : type(_t), f(_f), wi(_wi), pdf(_pdf) {}
};


/*
Diffuse parameters:
    description - Lambertian BRDF
    number - 0
    names - none

DiffuseTransmission parameters:
    description - same as diffuse, but with transmission
                - disable texture color
    number - 3
    names - T.x, T.y, T.z

Dielectric parameters:
    description - thin dielectric
    number - 1
    names - ior

Disney parameters:
    number - 12
    names - ior
            metallic
            subsurface
            roughness
            specular
            specularTint
            anisotropic     (unused)
            sheen
            sheenTint
            clearcoat
            clearcoatGloss
            specTrans
*/

class Material
{
public:
    MaterialType type{ MaterialType::Diffuse };
    float3 color{ 0.0f, 0.0f, 0.0f };
    float3 emission_color{ 0.0f, 0.0f, 0.0f };
    float intensity{ 0.0f };

    float params[MAX_MATERIAL_PARAMETERS]
    { 1.5f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

public:
    __host__ __device__ Material() {}
    __host__ __device__ float3 get_color() const { return color; }
    __host__ __device__ float3 get_emission_color() const { return emission_color; }
    __host__ __device__ float get_intensity() const { return intensity; }

    __host__ __device__ bool is_emissive() const { return length(emission_color) > 0.0f && intensity > 0.0f; }
    __host__ __device__ bool is_specular() const { return type == MaterialType::Dielectric; }
    __host__ __device__ bool is_transmissive() const
    {
        return (type == MaterialType::DiffuseTransmission) || (type == MaterialType::Dielectric)
            || (type == MaterialType::Disney && params[11] > 0.0f);
    }

    __device__ float3 eval(float3 wi, float3 wo, float3 n, float3 tex_color, bool inner = false) const
    {
        switch (type)
        {

        case MaterialType::Diffuse:
        {
            if (dot(wi, n) <= 0.0f || dot(wo, n) <= 0.0f)
                return { 0.0f, 0.0f, 0.0f };
            return M_1_PI * tex_color;
        }

        case MaterialType::DiffuseTransmission:
        {
            if (dot(wi, wo) <= 0.0f)
                return M_1_PI * make_float3(params[0], params[1], params[2]);
            return M_1_PI * color;
        }

        case MaterialType::Dielectric:
            return { 0.0f, 0.0f, 0.0f };

        case MaterialType::Disney:
        {
            float ior = params[0], metallic = params[1], subsurface = params[2], roughness = max(params[3], 0.001f), specular = params[4], specularTint = params[5];
            float sheen = params[7], sheenTint = params[8], clearcoat = params[9], clearcoatGloss = max(params[10], 0.001f), specTrans = params[11];

            float eta = inner ? ior : 1.0f / ior;
            Onb onb(n);
            float3 V = onb.to_local(wo), L = onb.to_local(wi);
            float3 H = L.z > 0.0f ? normalize(L + V) : normalize(L + V * eta);
            if (H.z <= 0.0f) H = -H;
            float V_H = dot(V, H), L_H = dot(L, H);

            float lum = luminance(tex_color);
            float3 ctint = lum > 0.0f ? tex_color / lum : make_float3(1.0f);
            // float F0 = (1.0f - eta) / (1.0f + eta);
            // float3 spec_color = lerp(F0 * F0 * lerp(make_float3(1.0f), ctint, specularTint), tex_color, metallic);
            float3 spec_color = lerp(specular * 0.08f * lerp(make_float3(1.0f), ctint, specularTint), tex_color, metallic);
            float3 sheen_color = lerp(make_float3(1.0f), ctint, sheenTint);

            float FM = fresnel_mix(metallic, eta, V_H);
            float diffuse_w = lum * (1.0f - metallic) * (1.0f - specTrans);
            float spec_reflect_w = luminance(lerp(spec_color, make_float3(1.0f), FM));
            float spec_refract_w = (1.0f - FM) * (1.0f - metallic) * specTrans * lum;
            float clearcoat_w = clearcoat * (1.0f - metallic);

            // diffuse
            float3 f_diffuse = make_float3(0.0f);
            if (diffuse_w > 0.0f && L.z > 0.0f)
            {
                // diffuse
                float FL = schlick_fresnel(L.z), FV = schlick_fresnel(V.z);
                float FH = schlick_fresnel(L_H);
                float Fd90 = 0.5f + 2.0f * L_H * L_H * roughness;
                float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);
                // sub-surface
                float Fss90 = L_H * L_H * roughness;
                float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
                float ss = 1.25f * (Fss * (1.0f / (L.z + V.z) - 0.5f) + 0.5f);
                // sheen
                float3 Fsheen = FH * sheen * sheen_color;

                f_diffuse = (M_1_PI * lerp(Fd, ss, subsurface) * tex_color + Fsheen) * (1.0f - metallic) * (1.0f - specTrans);
            }
            // specular reflection
            float3 f_spec_reflect = make_float3(0.0f);
            if (spec_reflect_w > 0.0f && L.z > 0.0f && V.z > 0.0f)
            {
                float _FM = fresnel_mix(metallic, eta, L_H);
                float3 F = lerp(spec_color, make_float3(1.0f), _FM);
                float D = GTR2(H.z, roughness);
                float G = smithG_GGX(abs(L.z), roughness) * smithG_GGX(abs(V.z), roughness);

                f_spec_reflect = F * D * G / (4.0f * L.z * V.z);
            }
            // specular refraction
            float3 f_spec_refract = make_float3(0.0f);
            if (spec_refract_w > 0.0f && L.z < 0.0f)
            {
                float F = schlick_fresnel(abs(V_H), eta);
                float D = GTR2(H.z, roughness);
                float denom = (L_H + V_H * eta) * (L_H + V_H * eta);
                float G = smithG_GGX(abs(L.z), roughness) * smithG_GGX(abs(V.z), roughness) * abs(L_H) * abs(V_H) * eta * eta / denom;
                float3 refract_color = make_float3(sqrt(tex_color.x), sqrt(tex_color.y), sqrt(tex_color.z));

                f_spec_refract = refract_color * (1.0f - metallic) * specTrans * (1.0f - F) * D * G / (abs(L.z) * abs(V.z));
            }
            // clearcoat
            float3 f_clearcoat = make_float3(0.0f);
            if (clearcoat_w > 0.0f && L.z > 0.0f && V.z > 0.0f)
            {
                float FH = schlick_fresnel(L_H, 1.0f / 1.5f);
                float F = lerp(0.04f, 1.0f, FH);
                float D = GTR1(H.z, clearcoatGloss);
                float G = smithG_GGX(L.z, 0.25f) * smithG_GGX(V.z, 0.25f);

                f_clearcoat = make_float3(0.25f) * clearcoat * F * D * G / (4.0f * L.z * V.z);
            }

            return f_diffuse + f_spec_reflect + f_spec_refract + f_clearcoat;
        }

        default:
            return { 0.0f, 0.0f, 0.0f };

        }
    }

    __device__ MaterialSample sample(float3 wo, float3 n, float2 rnd, float3 tex_color, bool inner = false) const
    {
        switch (type)
        {

        case MaterialType::Diffuse:
        {
            float3 v = cosine_sample_hemisphere(rnd);
            float pdf = cosine_hemisphere_pdf(v.z);
            Onb onb(n);
            float3 wi = onb.to_world(v);
            return { ReflectionType::Diffuse, eval(wi, wo, n, tex_color), wi, pdf };
        }

        case MaterialType::DiffuseTransmission:
        {
            float pr = max(color.x, max(color.y, color.z));
            float pt = max(params[0], max(params[1], params[2]));
            float reflection_ratio = pr / (pr + pt);
            if (rnd.x <= reflection_ratio)
            {
                rnd.x /= reflection_ratio;
                float3 v = cosine_sample_hemisphere(rnd);
                float pdf = cosine_hemisphere_pdf(v.z);
                Onb onb(n);
                float3 wi = onb.to_world(v);
                return { ReflectionType::Diffuse, color, wi, reflection_ratio * pdf };
            }
            else
            {
                rnd.x = (rnd.x - reflection_ratio) / (1.0f - reflection_ratio);
                float3 v = cosine_sample_hemisphere(rnd);
                float pdf = cosine_hemisphere_pdf(v.z);
                Onb onb(n);
                float3 wi = onb.to_world(v);
                return { ReflectionType::Diffuse, make_float3(params[0], params[1], params[2]), -wi, (1.0f - reflection_ratio) * pdf };
            }
        }

        case MaterialType::Dielectric:
        {
            float ior = params[0];

            float eta = inner ? ior : 1.0f / ior;
            float cosi = dot(wo, n);

            float3 wi = refract(-wo, n, eta);
            float reflect_ratio = dot(wi, wi) == 0.0f ? 1.0f : schlick_fresnel(cosi, eta);
            if (rnd.x <= reflect_ratio)
            {
                wi = reflect(-wo, n);
                return { ReflectionType::Specular, tex_color, normalize(wi), 1.0f };
                // return { ReflectionType::Specular, tex_color * reflect_ratio, normalize(wi), reflect_ratio };
            }
            else
                return { ReflectionType::Specular, tex_color, normalize(wi), 1.0f };
        }

        case MaterialType::Disney:
        {
            float ior = params[0], metallic = params[1], roughness = max(params[3], 0.001f), specular = params[4], specularTint = params[5];
            float sheenTint = params[8], clearcoat = params[9], clearcoatGloss = max(params[10], 0.001f), specTrans = params[11];

            float eta = inner ? ior : 1.0f / ior;
            Onb onb(n);
            float3 V = onb.to_local(wo);

            float lum = luminance(tex_color);
            float3 ctint = lum > 0.0f ? tex_color / lum : make_float3(1.0f);
            float3 spec_color = lerp(specular * 0.08f * lerp(make_float3(1.0f), ctint, specularTint), tex_color, metallic);
            float3 sheen_color = lerp(make_float3(1.0f), ctint, sheenTint);

            float FM = fresnel_mix(metallic, eta, V.z);
            float diffuse_w = lum * (1.0f - metallic) * (1.0f - specTrans);
            float spec_reflect_w = luminance(lerp(spec_color, make_float3(1.0f), FM));
            float spec_refract_w = (1.0f - FM) * (1.0f - metallic) * specTrans * lum;
            float clearcoat_w = clearcoat * (1.0f - metallic);

            float total_w = diffuse_w + spec_reflect_w + spec_refract_w + clearcoat_w;
            diffuse_w /= total_w;
            spec_reflect_w /= total_w;
            spec_refract_w /= total_w;
            clearcoat_w /= total_w;

            float3 L = make_float3(0.0f);
            if (rnd.x < diffuse_w)
            {
                rnd.x /= diffuse_w;
                L = cosine_sample_hemisphere(rnd);
            }
            else if (rnd.x < diffuse_w + spec_reflect_w)
            {
                rnd.x = (rnd.x - diffuse_w) / spec_reflect_w;
                float3 H = sample_GTR2(roughness, rnd);
                if (dot(V, H) <= 0.0f)
                    return MaterialSample();
                L = normalize(reflect(-V, H));
            }
            else if (rnd.x < diffuse_w + spec_reflect_w + spec_refract_w)
            {
                rnd.x = (rnd.x - diffuse_w - spec_reflect_w) / spec_refract_w;
                float3 H = sample_GTR2(roughness, rnd);
                L = refract(-V, H, eta);
                if (dot(V, H) <= 0.0f || dot(L, L) == 0.0f)
                    return MaterialSample();
                L = normalize(L);
            }
            else
            {
                rnd.x = (rnd.x - diffuse_w - spec_reflect_w - spec_refract_w) / clearcoat_w;
                float3 H = sample_GTR1(clearcoatGloss, rnd);
                if (dot(V, H) <= 0.0f)
                    return MaterialSample();
                L = normalize(reflect(-V, H));
            }

            float pdf = 0.0f;

            float3 H_reflect = normalize(L + V);
            if (H_reflect.z < 0.0f) H_reflect = -H_reflect;
            float3 H_refract = normalize(L + V * eta);
            if (H_refract.z < 0.0f) H_refract = -H_refract;

            if (diffuse_w > 0.0f && L.z > 0.0f)
                pdf += diffuse_w * cosine_hemisphere_pdf(L.z);
            if (spec_reflect_w > 0.0f && dot(V, H_reflect) > 0.0f)
                pdf += spec_reflect_w * GTR2(H_reflect.z, roughness) * H_reflect.z / (4.0f * dot(V, H_reflect));
            if (spec_refract_w > 0.0f && dot(V, H_refract) > 0.0f && dot(L, H_refract) < 0.0f)
            {
                float V_H = dot(V, H_refract), L_H = dot(L, H_refract);
                float denom = (L_H + V_H * eta);
                pdf += spec_refract_w * GTR2(H_refract.z, roughness) * H_refract.z * abs(L_H) / (denom * denom);
            }
            if (clearcoat_w > 0.0f && dot(V, H_reflect) > 0.0f)
                pdf += clearcoat_w * GTR1(H_reflect.z, clearcoatGloss) * H_reflect.z / (4.0f * dot(V, H_reflect));

            float3 wi = onb.to_world(L);
            return { ReflectionType::Diffuse, eval(wi, wo, n, tex_color, inner), wi, pdf };
            // return { ReflectionType::Diffuse, eval(wi, wo, n, tex_color, inner), wi, sample_pdf(wi, wo, n, tex_color, inner) };
        }

        default:
            return MaterialSample();

        }
    }

    __device__ float sample_pdf(float3 wi, float3 wo, float3 n, float3 tex_color, bool inner = false) const
    {
        switch (type)
        {

        case MaterialType::Diffuse:
            return cosine_hemisphere_pdf(dot(wi, n));

        case MaterialType::DiffuseTransmission:
        {
            float pr = max(color.x, max(color.y, color.z));
            float pt = max(params[0], max(params[1], params[2]));
            float reflection_ratio = pr / (pr + pt);
            if (dot(wi, wo) > 0.0f)
                return reflection_ratio * cosine_hemisphere_pdf(dot(wi, n));
            else
                return (1.0f - reflection_ratio) * cosine_hemisphere_pdf(dot(wi, n));
        }

        case MaterialType::Dielectric:
            return 1.0f;

        case MaterialType::Disney:
        {
            float ior = params[0], metallic = params[1], roughness = max(params[3], 0.001f), specular = params[4], specularTint = params[5];
            float sheenTint = params[8], clearcoat = params[9], clearcoatGloss = max(params[10], 0.001f), specTrans = params[11];

            float eta = inner ? ior : 1.0f / ior;
            Onb onb(n);
            float3 V = onb.to_local(wo), L = onb.to_local(wi);
            float3 H_reflect = normalize(L + V);
            if (H_reflect.z < 0.0f) H_reflect = -H_reflect;
            float3 H_refract = normalize(L + V * eta);
            if (H_refract.z < 0.0f) H_refract = -H_refract;

            float lum = luminance(tex_color);
            float3 ctint = lum > 0.0f ? tex_color / lum : make_float3(1.0f);
            float3 spec_color = lerp(specular * 0.08f * lerp(make_float3(1.0f), ctint, specularTint), tex_color, metallic);
            float3 sheen_color = lerp(make_float3(1.0f), ctint, sheenTint);

            float FM = fresnel_mix(metallic, eta, V.z);
            float diffuse_w = lum * (1.0f - metallic) * (1.0f - specTrans);
            float spec_reflect_w = luminance(lerp(spec_color, make_float3(1.0f), FM));
            float spec_refract_w = (1.0f - FM) * (1.0f - metallic) * specTrans * lum;
            float clearcoat_w = clearcoat * (1.0f - metallic);

            float total_w = diffuse_w + spec_reflect_w + spec_refract_w + clearcoat_w;
            diffuse_w /= total_w;
            spec_reflect_w /= total_w;
            spec_refract_w /= total_w;
            clearcoat_w /= total_w;

            float pdf = 0.0f;
            if (diffuse_w > 0.0f && L.z > 0.0f)
                pdf += diffuse_w * cosine_hemisphere_pdf(L.z);
            if (spec_reflect_w > 0.0f && dot(V, H_reflect) > 0.0f)
                pdf += spec_reflect_w * GTR2(H_reflect.z, roughness) * H_reflect.z / (4.0f * dot(V, H_reflect));
            if (spec_refract_w > 0.0f && dot(V, H_refract) > 0.0f && dot(L, H_refract) < 0.0f)
            {
                float V_H = dot(V, H_refract), L_H = dot(L, H_refract);
                float denom = (L_H + V_H * eta);
                pdf += spec_refract_w * GTR2(H_refract.z, roughness) * H_refract.z * abs(L_H) / (denom * denom);
            }
            if (clearcoat_w > 0.0f && dot(V, H_reflect) > 0.0f)
                pdf += clearcoat_w * GTR1(H_reflect.z, clearcoatGloss) * H_reflect.z / (4.0f * dot(V, H_reflect));

            return pdf;
        }

        default:
            return 0.0f;

        }
    }

};
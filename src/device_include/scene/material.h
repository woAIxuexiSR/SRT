#pragma once

#include <cuda_runtime.h>
#include "my_math.h"

#define MAX_MATERIAL_PARAMETERS 12

// wi: incoming direction  => from surface to light (sample)
// wo: outgoing direction  => from surface to camera
// n : surface normal, always be the same hemisphere with wo
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
    Glossy,
    Specular
};

class MaterialSample
{
public:
    ReflectionType type;  // reflection type
    float3 f;           // BRDF
    float3 wi;          // incoming direction
    float pdf;          // pdf of wi

    __device__ MaterialSample() {}
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
            specular        （unused)
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
    float3 emission{ 0.0f, 0.0f, 0.0f };

    float params[MAX_MATERIAL_PARAMETERS]
    { 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 1.0f, 0.0f };

public:
    __host__ __device__ Material() {}
    __host__ __device__ float3 get_color() const { return color; }
    __host__ __device__ float3 get_emission() const { return emission; }

    __host__ __device__ bool is_emissive() const { return length(emission) > 0.0f; }
    __host__ __device__ bool is_transmissive() const
    {
        return type == MaterialType::DiffuseTransmission || type == MaterialType::Dielectric
            || (type == MaterialType::Disney && params[11] > 0.0f);
    }
    __host__ __device__ bool is_specular() const { return type == MaterialType::Dielectric; }

    __device__ float3 eval(float3 wi, float3 wo, float3 n, float3 tex_color, bool inner = false) const
    {
        switch (type)
        {

        case MaterialType::Diffuse:
        {
            if (dot(wi, n) <= 0.0f)
                return { 0.0f, 0.0f, 0.0f };
            return M_1_PI * tex_color;
        }

        case MaterialType::DiffuseTransmission:
        {
            if (dot(wi, n) <= 0.0f)
                return M_1_PI * make_float3(params[0], params[1], params[2]);
            return M_1_PI * color;
        }

        case MaterialType::Dielectric:
            return { 0.0f, 0.0f, 0.0f };

        case MaterialType::Disney:
        {
            float ior = params[0], metallic = params[1], subsurface = params[2], roughness = max(params[3], 0.001f), specularTint = params[5];
            float sheen = params[7], sheenTint = params[8], clearcoat = params[9], clearcoatGloss = params[10], specTrans = params[11];

            float eta = inner ? ior : 1.0f / ior;
            float3 h = (dot(wi, n) <= 0.0f) ? normalize(wi + wo * eta) : normalize(wi + wo);
            if (dot(h, n) <= 0.0f) h = -h;

            float n_wi = dot(wi, n), n_wo = dot(wo, n), n_h = dot(h, n), wi_h = dot(wi, h), wo_h = dot(wo, h);

            // calculate color
            float luminance = dot(tex_color, make_float3(0.3f, 0.59f, 0.11f));
            float3 ctint = luminance > 0.0f ? tex_color / luminance : make_float3(1.0f);
            float F0 = (1.0f - eta) / (1.0f + eta);
            float3 spec_color = lerp(F0 * F0 * lerp(make_float3(1.0f), ctint, specularTint), tex_color, metallic);
            float3 sheen_color = lerp(make_float3(1.0f), ctint, sheenTint);

            // calculate lobes' weights
            float FM = lerp(schlick_fresnel(wi_h, eta), schlick_fresnel(wi_h), metallic);
            float3 F = lerp(spec_color, make_float3(1.0f), FM);
            float diffuse_w = luminance * (1.0f - metallic) * (1.0f - specTrans);
            float spec_reflect_w = dot(F, make_float3(0.3f, 0.59f, 0.11f));
            float spec_refract_w = (1.0f - FM) * (1.0f - metallic) * specTrans * luminance;
            float clearcoat_w = clearcoat * (1.0f - metallic);

            // printf("%f %f %f %f\n", diffuse_w, spec_reflect_w, spec_refract_w, clearcoat_w);
            // printf("%f\n", diffuse_w);

            // diffuse
            float3 f_diffuse = make_float3(0.0f);
            if (diffuse_w > 0.0f && n_wi > 0.0f)
            {
                // diffuse
                float FL = schlick_fresnel(n_wi), FV = schlick_fresnel(n_wo);
                float FH = schlick_fresnel(wi_h);
                float Fd90 = 0.5f + 2.0f * wi_h * wi_h * roughness;
                float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);
                // sub-surface
                float Fss90 = wi_h * wi_h * roughness;
                float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
                float ss = 1.25f * (Fss * (1.0f / (n_wi + n_wo) - 0.5f) + 0.5f);
                // sheen
                float3 Fsheen = FH * sheen_color * sheen;

                f_diffuse = (M_1_PI * lerp(Fd, ss, subsurface) * tex_color + Fsheen) * (1.0f - metallic) * (1.0f - specTrans);
            }
            // specular reflection
            float3 f_spec_reflect = make_float3(0.0f);
            if (spec_reflect_w > 0.0f && n_wi > 0.0f)
            {
                float D = GTR2(n_h, roughness);
                float G = smithG_GGX(n_wi, roughness) * smithG_GGX(n_wo, roughness);

                f_spec_reflect = F * D * G / (4.0f * n_wi * n_wo);
            }
            // specular refraction
            float3 f_spec_refract = make_float3(0.0f);
            if (spec_refract_w > 0.0f && n_wi < 0.0f)
            {
                float F = schlick_fresnel(abs(wi_h), eta);
                float D = GTR2(n_h, roughness);
                float denom = (wi_h + wo_h * eta) * (wi_h + wo_h * eta);
                float G = smithG_GGX(n_wi, roughness) * smithG_GGX(n_wo, roughness);
                float3 refraction_color = make_float3(sqrt(tex_color.x), sqrt(tex_color.y), sqrt(tex_color.z));

                f_spec_refract = refraction_color * (1.0f - metallic) * specTrans * (1.0f - F) * D * G
                    * abs(wi_h) * abs(wo_h) * eta * eta / (denom * abs(n_wi) * abs(n_wo));
            }
            // clearcoat
            float3 f_clearcoat = make_float3(0.0f);
            if (clearcoat_w > 0.0f && n_wi > 0.0f)
            {
                float FH = schlick_fresnel(wo_h, 1.0f / 1.5f);
                float F = lerp(0.04f, 1.0f, FH);
                float D = GTR1(n_h, clearcoatGloss);
                float G = smithG_GGX(n_wi, 0.25f) * smithG_GGX(n_wo, 0.25f);

                f_clearcoat = make_float3(0.25f) * clearcoat * F * D * G / (4.0f * n_wi * n_wo);
            }

            return f_diffuse + f_spec_reflect + f_spec_refract + f_clearcoat;
        }

        default:
            return { 0.0f, 0.0f, 0.0f };

        }
    }

    __device__ MaterialSample sample(float3 wo, float3 n, float2 rnd, float3 tex_color) const
    {
        switch (type)
        {

        case MaterialType::Diffuse:
        {
            float3 v = cosine_sample_hemisphere(rnd);
            Onb onb(n);
            float3 wi = onb(v);
            return { ReflectionType::Diffuse, eval(wi, wo, n, tex_color), wi, pdf(wi, wo, n) };
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
                Onb onb(n);
                float3 wi = onb(v);
                if (dot(wi, wo) <= 0.0f) wi = -wi;
                return { ReflectionType::Diffuse, color, wi, pdf(wi, wo, n) };
            }
            else
            {
                rnd.x = (rnd.x - reflection_ratio) / (1.0f - reflection_ratio);
                float3 v = cosine_sample_hemisphere(rnd);
                Onb onb(n);
                float3 wi = onb(v);
                if (dot(wi, wo) > 0.0f) wi = -wi;
                return { ReflectionType::Diffuse, make_float3(params[0], params[1], params[2]), wi, pdf(wi, wo, n) };
            }
        }

        case MaterialType::Dielectric:
        {
            float ior = params[0];
            float eta = 1.0f / ior, cosi = dot(wo, n);
            if (cosi <= 0.0f) { eta = ior; cosi = -cosi; n = -n; }

            float sint = eta * sqrt(max(0.0f, 1.0f - cosi * cosi));
            float cost = sqrt(max(0.0f, 1.0f - sint * sint));
            float reflect_ratio = (sint >= 1.0f) ? 1.0f : fresnel(cosi, cost, eta);
            if (rnd.x <= reflect_ratio)
            {
                float3 wi = 2.0f * cosi * n - wo;
                return { ReflectionType::Specular, tex_color, wi, 1.0f };
            }
            else
            {
                float3 rperp = -eta * (wo - cosi * n);
                float3 rparl = -sqrt(max(0.0f, 1.0f - dot(rperp, rperp))) * n;
                return { ReflectionType::Specular, tex_color, rparl + rperp, 1.0f };
            }
        }

        case MaterialType::Disney:
        {
            float metallic = params[0], roughness = params[2];
            Onb onb(n);

            float diffuse_ratio = 0.5f * (1.0f - metallic);
            if (rnd.x < diffuse_ratio)
            {
                rnd.x = rnd.x / diffuse_ratio;
                float3 v = cosine_sample_hemisphere(rnd);
                float3 wi = onb(v);
                return { ReflectionType::Diffuse, eval(wi, wo, n, tex_color), wi, pdf(wi, wo, n) };
            }
            else
            {
                rnd.x = (rnd.x - diffuse_ratio) / (1.0f - diffuse_ratio);
                float a = max(0.001f, roughness);
                float phi = rnd.x * 2.0f * M_PI;

                float cos_theta = sqrt((1.0f - rnd.y) / (1.0f + (a * a - 1.0f) * rnd.y));
                float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
                float sin_phi = sin(phi);
                float cos_phi = cos(phi);

                float3 half = make_float3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
                float3 h = onb(half);
                float3 wi = 2.0 * dot(wo, h) * h - wo;
                return { ReflectionType::Specular, eval(wi, wo, n, tex_color), wi, pdf(wi, wo, n) };
            }
        }

        default:
            return { ReflectionType::Diffuse, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 0.0f };

        }
    }

    __device__ float pdf(float3 wi, float3 wo, float3 n) const
    {
        switch (type)
        {

        case MaterialType::Diffuse:
        {
            float cos_theta = dot(wi, n);
            return cosine_hemisphere_pdf(cos_theta);
        }

        case MaterialType::DiffuseTransmission:
        {
            float pr = max(color.x, max(color.y, color.z));
            float pt = max(params[0], max(params[1], params[2]));
            float reflection_ratio = pr / (pr + pt);
            if (dot(wi, wo) <= 0.0f)
                return (1.0f - reflection_ratio) * cosine_hemisphere_pdf(abs(dot(wi, n)));
            return reflection_ratio * cosine_hemisphere_pdf(abs(dot(wi, n)));
        }

        case MaterialType::Dielectric:
            return 1.0f;

        case MaterialType::Disney:
        {
            float metallic = params[0], roughness = params[2];
            float clearcoat = params[8], clearcoatGloss = params[9];

            float diffuse_ratio = 0.5f * (1.0f - metallic);
            float specular_alpha = max(0.001f, roughness);
            float clearcoat_alpha = lerp(0.1f, 0.001f, clearcoatGloss);

            float3 h = normalize(wi + wo);
            float cos_theta = dot(h, n);
            float pdfGTR2 = GTR2(cos_theta, specular_alpha) * cos_theta;
            float pdfGTR1 = GTR1(cos_theta, clearcoat_alpha) * cos_theta;

            float ratio = 1.0f / (1.0f + clearcoat);
            float pdfSpec = lerp(pdfGTR1, pdfGTR2, ratio) / (4.0f * dot(wi, h));
            float pdfDiff = dot(wi, n) * M_1_PI;

            return diffuse_ratio * pdfDiff + (1.0f - diffuse_ratio) * pdfSpec;
        }

        default:
            return 0.0f;

        }
    }

};
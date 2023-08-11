#pragma once

#include "my_math.h"

/*
local coordinate system:
    n - shading normal, (0, 0, 1)
    t - tangent, (1, 0, 0)
    b - bitangent, (0, 1, 0)
we guarantee that wo is in the upper hemisphere of n

wi: incoming direction => from surface to light (sample)
wo: outgoing direction => from surface to camera
color: albedo
inner: whether wo is inside the surface
*/

class BxDFSample
{
public:
    float3 f;               // BxDF value
    float cos_theta;        // cos(theta) of wi
    float3 wi;              // sampled direction
    float pdf{ 0.0f };      // pdf of wi
};

/*
Diffuse parameters:
    description - Lambertian BRDF
    number - 0
    names - none

Diffuse Transmission parameters:
    description - Lambertian BTDF
    number - 0
    names - none

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
            anisotropic
            sheen
            sheenTint
            clearcoat
            clearcoatGloss
            specTrans
*/

class BxDF
{
public:
    enum class Type { Diffuse, DiffuseTransmission, Dielectric, Disney };

    Type type{ Type::Diffuse };
    float ior{ 1.5f };
    float metallic{ 0.0f };
    float subsurface{ 0.0f };
    float roughness{ 0.5f };
    float specular{ 0.5f };
    float specularTint{ 0.0f };
    float anisotropic{ 0.0f };
    float sheen{ 0.0f };
    float sheenTint{ 0.0f };
    float clearcoat{ 0.0f };
    float clearcoatGloss{ 0.001f };
    float specTrans{ 0.0f };

public:
    __device__ float3 eval(float3 wi, float3 wo, float3 color, bool inner) const
    {
        switch (type)
        {

        case Type::Diffuse:
            return (wi.z <= 0.0f) ? make_float3(0.0f) : color * (float)M_1_PI;

        case Type::DiffuseTransmission:
            return color * (float)M_1_PI;

        case Type::Dielectric:
            return { 0.0f, 0.0f, 0.0f };

        case Type::Disney:
        {
            float eta = inner ? ior : 1.0f / ior;
            float3 V = wo, L = wi;
            float3 H = (L.z > 0.0f) ? normalize(L + V) : normalize(L + V * eta);
            if (H.z <= 0.0f) H = -H;
            float V_H = dot(V, H), L_H = dot(L, H);

            float lum = Luminance(color);
            float3 ctint = (lum > 0.0f) ? color / lum : make_float3(1.0f);
            // float F0 = (1.0f - eta) / (1.0f + eta);
            // float3 spec_color = lerp(F0 * F0 * lerp(make_float3(1.0f), ctint, specularTint), color, metallic);
            float3 spec_color = lerp(specular * 0.08f * lerp(make_float3(1.0f), ctint, specularTint), color, metallic);
            float3 sheen_color = lerp(make_float3(1.0f), ctint, sheenTint);

            float FM = fresnel_mix(metallic, eta, V_H);
            float diffuse_w = lum * (1.0f - metallic) * (1.0f - specTrans);
            float spec_reflect_w = Luminance(lerp(spec_color, make_float3(1.0f), FM));
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

                f_diffuse = ((float)M_1_PI * lerp(Fd, ss, subsurface) * color + Fsheen) * (1.0f - metallic) * (1.0f - specTrans);
            }
            // specular reflection
            float3 f_spec_reflect = make_float3(0.0f);
            if (spec_reflect_w > 0.0f && L.z > 0.0f && V.z > 0.0f)
            {
                float _FM = fresnel_mix(metallic, eta, L_H);
                float3 F = lerp(spec_color, make_float3(1.0f), _FM);

                float aspect = sqrt(1.0f - anisotropic * 0.9f);
                float a2 = roughness * roughness;
                float ax = max(0.001f, a2 / aspect), ay = max(0.001f, a2 * aspect);

                // float D = GTR2(H.z, roughness);
                float D = GTR2_aniso(H.z, H.x, H.y, ax, ay);
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
                float3 refract_color = make_float3(sqrt(color.x), sqrt(color.y), sqrt(color.z));

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

    __device__ BxDFSample sample(float3 wo, float2 rnd, float3 color, bool inner) const
    {
        switch (type)
        {

        case Type::Diffuse:
        {
            float3 wi = cosine_sample_hemisphere(rnd);
            float pdf = cosine_hemisphere_pdf(wi.z);
            return { eval(wi, wo, color, inner), wi.z, wi, pdf };
        }

        case Type::DiffuseTransmission:
        {
            if (rnd.x < 0.5f)
            {
                rnd.x *= 2.0f;
                float3 wi = cosine_sample_hemisphere(rnd);
                float pdf = cosine_hemisphere_pdf(wi.z);
                return { eval(wi, wo, color, inner), wi.z, wi, 0.5f * pdf };
            }
            else
            {
                rnd.x = (rnd.x - 0.5f) * 2.0f;
                float3 wi = cosine_sample_hemisphere(rnd);
                float pdf = cosine_hemisphere_pdf(wi.z);
                return { eval(-wi, wo, color, inner), wi.z, -wi, 0.5f * pdf };
            }
        }

        case Type::Dielectric:
        {
            float eta = inner ? ior : 1.0f / ior;
            float3 n = make_float3(0.0f, 0.0f, 1.0f);

            float3 wi = refract(-wo, n, eta);
            float reflect_ratio = (dot(wi, wi) == 0.0f) ? 1.0f : schlick_fresnel(wo.z, eta);

            if (rnd.x <= reflect_ratio)
                wi = reflect(-wo, n);
            wi = normalize(wi);

            return { color, 1.0f, wi, 1.0f };       // f * cos / pdf = color
            // return { color * reflect_ratio, 1.0f, wi, reflect_ratio };
        }

        case Type::Disney:
        {
            float eta = inner ? ior : 1.0f / ior;
            float3 V = wo;

            float lum = Luminance(color);
            float3 ctint = (lum > 0.0f) ? color / lum : make_float3(1.0f);
            float3 spec_color = lerp(specular * 0.08f * lerp(make_float3(1.0f), ctint, specularTint), color, metallic);
            float3 sheen_color = lerp(make_float3(1.0f), ctint, sheenTint);

            float aspect = sqrt(1.0f - anisotropic * 0.9f);
            float a2 = roughness * roughness;
            float ax = max(0.001f, a2 / aspect), ay = max(0.001f, a2 * aspect);

            float FM = fresnel_mix(metallic, eta, V.z);
            float diffuse_w = lum * (1.0f - metallic) * (1.0f - specTrans);
            float spec_reflect_w = Luminance(lerp(spec_color, make_float3(1.0f), FM));
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
                // float3 H = sample_GTR2(roughness, rnd);
                float3 H = sample_GTR2_aniso(ax, ay, rnd);
                if (dot(V, H) <= 0.0f)
                    return BxDFSample();
                L = normalize(reflect(-V, H));
            }
            else if (rnd.x < diffuse_w + spec_reflect_w + spec_refract_w)
            {
                rnd.x = (rnd.x - diffuse_w - spec_reflect_w) / spec_refract_w;
                // float3 H = sample_GTR2(roughness, rnd);
                float3 H = sample_GGXVNDF(V, roughness, rnd);
                L = refract(-V, H, eta);
                if (dot(V, H) <= 0.0f || dot(L, L) == 0.0f)
                    return BxDFSample();
                L = normalize(L);
            }
            else
            {
                rnd.x = (rnd.x - diffuse_w - spec_reflect_w - spec_refract_w) / clearcoat_w;
                float3 H = sample_GTR1(clearcoatGloss, rnd);
                if (dot(V, H) <= 0.0f)
                    return BxDFSample();
                L = normalize(reflect(-V, H));
            }
            // return { eval(L, wo, color, inner), abs(L.z), L, sample_pdf(L, wo, color, inner) };

            float pdf = 0.0f;

            float3 H_reflect = normalize(L + V);
            if (H_reflect.z < 0.0f) H_reflect = -H_reflect;
            float3 H_refract = normalize(L + V * eta);
            if (H_refract.z < 0.0f) H_refract = -H_refract;

            if (diffuse_w > 0.0f && L.z > 0.0f)
                pdf += diffuse_w * cosine_hemisphere_pdf(L.z);
            if (spec_reflect_w > 0.0f && dot(V, H_reflect) > 0.0f)
                // pdf += spec_reflect_w * GTR2(H_reflect.z, roughness) * H_reflect.z / (4.0f * dot(V, H_reflect));
                pdf += spec_reflect_w * GTR2_aniso(H_reflect.z, H_reflect.x, H_reflect.y, ax, ay) * H_reflect.z / (4.0f * dot(V, H_reflect));
            if (spec_refract_w > 0.0f && dot(V, H_refract) > 0.0f && dot(L, H_refract) < 0.0f)
            {
                float V_H = dot(V, H_refract), L_H = dot(L, H_refract);
                float denom = (L_H + V_H * eta);
                // pdf += spec_refract_w * GTR2(H_refract.z, roughness) * H_refract.z * abs(L_H) / (denom * denom);
                pdf += spec_refract_w * GTR2(H_refract.z, roughness) * smithG_GGX(V.z, roughness) * max(dot(V, H_refract), 0.0f)
                    / V.z * abs(L_H) / (denom * denom);
            }
            if (clearcoat_w > 0.0f && dot(V, H_reflect) > 0.0f)
                pdf += clearcoat_w * GTR1(H_reflect.z, clearcoatGloss) * H_reflect.z / (4.0f * dot(V, H_reflect));

            return { eval(L, wo, color, inner), abs(L.z), L, pdf };
        }

        default:
            return BxDFSample();

        }
    }

    __device__ float sample_pdf(float3 wi, float3 wo, float3 color, bool inner) const
    {
        switch (type)
        {

        case Type::Diffuse:
            return (wi.z <= 0.0f) ? 0.0f : cosine_hemisphere_pdf(wi.z);

        case Type::DiffuseTransmission:
            return 0.5f * cosine_hemisphere_pdf(abs(wi.z));

        case Type::Dielectric:
            return 0.0f;

        case Type::Disney:
        {
            float eta = inner ? ior : 1.0f / ior;
            float3 V = wo, L = wi;
            float3 H_reflect = normalize(L + V);
            if (H_reflect.z < 0.0f) H_reflect = -H_reflect;
            float3 H_refract = normalize(L + V * eta);
            if (H_refract.z < 0.0f) H_refract = -H_refract;

            float lum = Luminance(color);
            float3 ctint = (lum > 0.0f) ? color / lum : make_float3(1.0f);
            float3 spec_color = lerp(specular * 0.08f * lerp(make_float3(1.0f), ctint, specularTint), color, metallic);
            float3 sheen_color = lerp(make_float3(1.0f), ctint, sheenTint);

            float aspect = sqrt(1.0f - anisotropic * 0.9f);
            float a2 = roughness * roughness;
            float ax = max(0.001f, a2 / aspect), ay = max(0.001f, a2 * aspect);

            float FM = fresnel_mix(metallic, eta, V.z);
            float diffuse_w = lum * (1.0f - metallic) * (1.0f - specTrans);
            float spec_reflect_w = Luminance(lerp(spec_color, make_float3(1.0f), FM));
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
                pdf += spec_reflect_w * GTR2_aniso(H_reflect.z, H_reflect.x, H_reflect.y, ax, ay) * H_reflect.z / (4.0f * dot(V, H_reflect));
            if (spec_refract_w > 0.0f && dot(V, H_refract) > 0.0f && dot(L, H_refract) < 0.0f)
            {
                float V_H = dot(V, H_refract), L_H = dot(L, H_refract);
                float denom = (L_H + V_H * eta);
                pdf += spec_refract_w * GTR2(H_refract.z, roughness) * smithG_GGX(V.z, roughness) * max(dot(V, H_refract), 0.0f)
                    / V.z * abs(L_H) / (denom * denom);
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
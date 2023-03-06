#pragma once

#include <cuda_runtime.h>
#include "srt_math.h"

// wi: incoming direction  => from surface to light (sample)
// wo: outgoing direction  => from surface to camera

enum class MaterialType
{
    Emissive,
    Lambertian,
    Glass,
    Disney,
};

class MaterialSample
{
public:
    MaterialType type;  // material type
    float3 f;           // BRDF
    float3 wi;          // incoming direction
    float pdf;          // pdf of wi

    __device__ MaterialSample() {}
    __device__ MaterialSample(MaterialType _t, float3 _f, float3 _wi, float _pdf)
        : type(_t), f(_f), wi(_wi), pdf(_pdf) {}
};


class Material
{
private:
    float3 color;

public:
    __host__ __device__ Material(): color(make_float3(0.0f)) {}
    __host__ __device__ Material(float3 _c) : color(_c) {}
    __host__ __device__ float3 getColor() const { return color; }

    virtual __host__ __device__ MaterialType getType() const = 0;
    virtual __host__ __device__ bool isLight() const { return false; }
    virtual __host__ __device__ bool isGlass() const { return false; }
    virtual __host__ __device__ bool isSpecular() const { return false; }
    virtual __host__ __device__ float3 Emission() const { return color; }

    virtual __device__ float3 Eval(float3 wi, float3 wo, float3 n, float3 tex_color) const = 0;
    virtual __device__ MaterialSample Sample(float3 wo, float3 n, float2 sample, float3 tex_color, bool reverse) const = 0;
    virtual __device__ float Pdf(float3 wi, float3 wo, float3 n) const = 0;
};


class EmissionMaterial: public Material
{
public:
    __host__ __device__ EmissionMaterial(): Material() {}
    __host__ __device__ EmissionMaterial(float3 _c) : Material(_c) {}

    __host__ __device__ MaterialType getType() const override { return MaterialType::Emissive; }
    __host__ __device__ bool isLight() const override { return true; }

    __device__ float3 Eval(float3 wi, float3 wo, float3 n, float3 tex_color) const override
    {
        return tex_color;
    }

    __device__ MaterialSample Sample(float3 wo, float3 n, float2 sample, float3 tex_color, bool reverse) const override
    {
        return MaterialSample();
    }

    __device__ float Pdf(float3 wi, float3 wo, float3 n) const override
    {
        return 0.0f;
    }
};


class LambertianMaterial: public Material
{
public:
    __host__ __device__ LambertianMaterial(): Material() {}
    __host__ __device__ LambertianMaterial(float3 _c) : Material(_c) {}

    __host__ __device__ MaterialType getType() const override { return MaterialType::Lambertian; }

    __device__ float3 Eval(float3 wi, float3 wo, float3 n, float3 tex_color) const override
    {
        if (dot(wi, n) <= 0.0f || dot(wo, n) <= 0.0f)
            return make_float3(0.0f);
        return M_1_PI * tex_color;
    }

    __device__ MaterialSample Sample(float3 wo, float3 n, float2 sample, float3 tex_color, bool reverse) const override
    {
        float3 v = CosineSampleHemiSphere(sample);
        Onb onb(n);
        float3 wi = onb(v);
        float3 beta = !reverse ? Eval(wi, wo, n, tex_color) : Eval(wo, wi, n, tex_color);
        return MaterialSample(MaterialType::Lambertian, beta, wi, Pdf(wi, wo, n));
    }

    __device__ float Pdf(float3 wi, float3 wo, float3 n) const override
    {
        float cosTheta = dot(wi, n);
        // printf("%f\n", M_1_PI);
        // printf("%f %f\n", cosTheta, cosTheta * M_1_PI);
        float pdf = CosineHemiSpherePdf(cosTheta);
        printf("%f %f %f %f %f %f\n", cosTheta, M_1_PI, cosTheta * (float)M_1_PI, (float)(cosTheta * 0.318310), (float)(cosTheta * (float)M_1_PI), pdf);
        // printf("%f\n", pdf);
        // return CosineHemiSpherePdf(cosTheta);
        return cosTheta * (float)M_1_PI;
    }
};


class GlassMaterial: public Material
{
private:
    float ior;

public:
    __host__ __device__ GlassMaterial(): Material(), ior(1.5f) {}
    __host__ __device__ GlassMaterial(float3 _c, float _ior) : Material(_c), ior(_ior) {}

    __host__ __device__ MaterialType getType() const override { return MaterialType::Glass; }
    __host__ __device__ bool isGlass() const override { return true; }
    __host__ __device__ bool isSpecular() const override { return true; }

    __device__ float3 Eval(float3 wi, float3 wo, float3 n, float3 tex_color) const override
    {
        return make_float3(0.0f);
    }

    __device__ MaterialSample Sample(float3 wo, float3 n, float2 sample, float3 tex_color, bool reverse) const override
    {
        float eta = 1.0f / ior, cosi = dot(wo, n);
        if (cosi <= 0.0f)
        {
            eta = ior;
            cosi = -cosi;
            n = -n;
        }

        float sint = eta * sqrt(max(0.0f, 1.0f - cosi * cosi));
        float cost = sqrt(max(0.0f, 1.0f - sint * sint));
        float reflectRatio = sint >= 1.0f ? 1.0f : Fresnel(cosi, cost, eta);

        float3 wi;
        if (sample.x <= reflectRatio)
            wi = 2.0f * cosi * n - wo;
        else
        {
            float3 rperp = -eta * (wo - n * cosi);
            float3 rparl = -sqrt(max(0.0f, 1.0f - dot(rperp, rperp))) * n;
            wi = normalize(rperp + rparl);
        }

        return MaterialSample(MaterialType::Glass, tex_color, wi, 1.0f);
    }

    __device__ float Pdf(float3 wi, float3 wo, float3 n) const override
    {
        return 1.0f;
    }
};

class DisneyMaterial: public Material
{
public:
    float metallic { 0.0f };
    float subsurface{ 0.0f };
    float roughness{ 0.5f };
    float specular{ 0.5f };
    float specularTint{ 0.0f };
    float anisotropic{ 0.0f };
    float sheen{ 0.0f };
    float sheenTint{ 0.5f };
    float clearcoat{ 0.0f };
    float clearcoatGloss{ 1.0f };

public:
    __host__ __device__ DisneyMaterial(): Material() {}
    __host__ __device__ DisneyMaterial(float3 _c) : Material(_c) {}

    __host__ __device__ MaterialType getType() const override { return MaterialType::Disney; }
    __host__ __device__ bool isSpecular() const override { return metallic >= 1.0f; }

    __device__ float3 Eval(float3 wi, float3 wo, float3 n, float3 tex_color) const override
    {
        float3 N = n, V = wo, L = wi;
        float3 H = normalize(L + V);
        float NDotL = dot(N, L), NDotV = dot(N, V), NDotH = dot(N, H), LDotH = dot(L, H);

        if (NDotL <= 0.0f || NDotV <= 0.0f)
            return make_float3(0.0f);

        // base color
        float3 Cdlin = tex_color;

        // diffuse
        float Fd90 = 0.5f + 2.0f * LDotH * LDotH * roughness;
        float FL = SchlickFresnel(NDotL), FV = SchlickFresnel(NDotV);
        float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);

        // sub-surface
        float Fss90 = LDotH * LDotH * roughness;
        float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
        float ss = 1.25f * (Fss * (1.0f / (NDotL + NDotV) - 0.5f) + 0.5f);

        // specular
        float Cdlum = 0.3f * Cdlin.x + 0.6f * Cdlin.y + 0.1f * Cdlin.z;
        float3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : make_float3(1.0f);
        float3 Cspec0 = lerp(specular * 0.08f * lerp(make_float3(1.0f), Ctint, specularTint), Cdlin, metallic);

        // anisotropic specular need X, Y axis
        // float aspect = sqrt(1.0f - anisotropic * 0.9f);
        // float ax = max(0.001f, roughness * roughness / aspect);
        // float ay = max(0.001f, roughness * roughness * aspect);
        // float Ds = GTR2_aniso(NDotH, dot(H, X), dot(H, Y), ax, ay);

        float alpha = max(0.001f, roughness * roughness);
        float Ds = GTR2(NDotH, alpha);
        float FH = SchlickFresnel(LDotH);
        float3 Fs = lerp(Cspec0, make_float3(1.0f), FH);
        float Gs = smithG_GGX(NDotL, roughness) * smithG_GGX(NDotV, roughness);

        // clearcoat
        float Dr = GTR1(NDotH, lerp(0.1f, 0.001f, clearcoatGloss));
        float Fr = lerp(0.04f, 1.0f, FH);
        float Gr = smithG_GGX(NDotL, 0.25f) * smithG_GGX(NDotV, 0.25f);

        // sheen
        float3 Csheen = lerp(make_float3(1.0f), Ctint, sheenTint);
        float3 Fsheen = FH * sheen * Csheen;


        // result
        float3 out = ((1.0f / M_PI) * lerp(Fd, ss, subsurface) * Cdlin + Fsheen) * (1.0f - metallic)
            + Gs * Fs * Ds + 0.25f * clearcoat * Gr * Fr * Dr;

        return out;
    }

    __device__ MaterialSample Sample(float3 wo, float3 n, float2 sample, float3 tex_color, bool reverse) const override
    {
        Onb onb(n);

        float diffuseRatio = 0.5f * (1.0f - metallic);
        float3 wi;
        if (sample.x < diffuseRatio)
        {
            sample.x = sample.x / diffuseRatio;
            float3 v = CosineSampleHemiSphere(sample);
            wi = onb(v);
        }
        else
        {
            sample.x = (sample.x - diffuseRatio) / (1.0f - diffuseRatio);

            float a = max(0.001f, roughness);
            float phi = sample.x * 2.0f * M_PI;

            float cosTheta = sqrt((1.0f - sample.y) / (1.0f + (a * a - 1.0f) * sample.y));
            float sinTheta = sqrt(1.0f - (cosTheta * cosTheta));
            float sinPhi = sin(phi);
            float cosPhi = cos(phi);

            float3 half = make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
            float3 H = onb(half);

            wi = 2.0f * dot(wo, H) * H - wo;
        }
        float3 beta = !reverse ? Eval(wi, wo, n, tex_color) : Eval(wo, wi, n, tex_color);
        return MaterialSample(MaterialType::Disney, beta, wi, Pdf(wi, wo, n));
    }

    __device__ float Pdf(float3 wi, float3 wo, float3 n) const override
    {
        float3 N = n, V = wo, L = wi;

        float specularAlpha = max(0.001f, roughness);
        float clearcoatAlpha = lerp(0.1f, 0.001f, clearcoatGloss);

        float diffuseRatio = 0.5f * (1.f - metallic);
        float specularRatio = 1.f - diffuseRatio;

        float3 H = normalize(L + V);

        float cosTheta = abs(dot(H, N));
        float pdfGTR2 = GTR2(cosTheta, specularAlpha) * cosTheta;
        float pdfGTR1 = GTR1(cosTheta, clearcoatAlpha) * cosTheta;

        float ratio = 1.0f / (1.0f + clearcoat);
        float pdfSpec = lerp(pdfGTR1, pdfGTR2, ratio) / (4.0 * abs(dot(L, H)));
        float pdfDiff = abs(dot(L, N)) * (1.0f / M_PI);

        return diffuseRatio * pdfDiff + specularRatio * pdfSpec;
    }
};
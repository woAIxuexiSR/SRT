#pragma once

#include "scene/material.h"

#define BDPT_MAX_LIGHT_VERTICES 16

class BDPTLightVertex
{
public:
    float3 pos, wi, normal, color;
    float3 beta;
    const Material* mat;

    // Fwd : from light to camera, Rev : from camera to light
    float pdfFwd, pdfRev;
    float pA;

public:
    __device__ BDPTLightVertex() {}

    __device__ inline void set(float3 _p, float3 _wi, float3 _n, float3 _c, float3 _b, const Material* _m)
    {
        pos = _p; wi = _wi; normal = _n; color = _c; beta = _b; mat = _m;
    }
};


class BDPTPath
{
public:
    int length;
    BDPTLightVertex vertices[BDPT_MAX_LIGHT_VERTICES];
};
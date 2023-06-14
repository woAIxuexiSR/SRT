#pragma once

#include "definition.h"
#include "my_math.h"
#include "animation.h"

#define MAX_BONE_PER_VERTEX 4

class Bone
{
public:
    string name;
    Transform offset;
};
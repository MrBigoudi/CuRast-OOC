#pragma once

#define CUB_DISABLE_BF16_SUPPORT

// === required by GLM ===
#define GLM_FORCE_CUDA
#define GLM_FORCE_NO_CTOR_INIT
#define CUDA_VERSION 12000
namespace std {
	using size_t = ::size_t;
};
// =======================

// #include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include "./glm/glm/glm.hpp"
#include "./glm/glm/gtc/matrix_transform.hpp"
#include "./glm/glm/gtc/matrix_access.hpp"
#include "./glm/glm/gtx/transform.hpp"
#include "./glm/glm/gtc/quaternion.hpp"

#include "./GpuVersionInterface.h"
#include "./GpuVersionStructs.h"
#include "./GpuVersionAllocator.h"
#include "./GpuVersionGlobals.h"

using glm::ivec2;
using glm::i8vec4;
using glm::vec3;
using glm::vec4;

namespace cg = cooperative_groups;


///////////////////////////////////////////////////////////////////////
////////////////////////// HELPER FUNCTIONS ///////////////////////////
///////////////////////////////////////////////////////////////////////
__device__ __forceinline__ CIdAABB createNewAABB(const CAABB& aabb){
    CIdAABB id = __nv_atomic_fetch_add(&globalVariables.nbAABBs, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
    if(id == CINVALID_ID){
        printf("ERROR: reached the maximum number of nodes that can be created\n");
    }
    globalVariables.allAABBs[id] = aabb;
    globalVariables.relationshipMap[id] = {
        CINVALID_ID, CINVALID_ID, CINVALID_ID, CINVALID_ID,
        CINVALID_ID, CINVALID_ID, CINVALID_ID, CINVALID_ID
    };
    return id;
}

// https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda/51549250#51549250
__device__ __forceinline__ void atomicMinFloatRelaxedOrderSystemScope(float* addr, float value){
    if(value >= 0.f) {
        __nv_atomic_min((int *)addr, __float_as_int(value),
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM
        );
    } else {
        __nv_atomic_max((unsigned int *)addr, __float_as_uint(value),
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM
        );
    }
}
__device__ __forceinline__ float atomicMaxFloatRelaxedOrderSystemScope(float* addr, float value){
    if(value >= 0.f) {
        __nv_atomic_max((int *)addr, __float_as_int(value),
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM
        );
    } else {
        __nv_atomic_min((unsigned int *)addr, __float_as_uint(value),
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM
        );
    }
}
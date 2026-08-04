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


// https://forums.developer.nvidia.com/t/using-assert-in-cuda-code/21816
#define customAssert() { printf("Assertion failure!\n"); asm("trap;"); }


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

using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat4;

namespace cg = cooperative_groups;


//////////////////////////////////////////////////////////////////////////////
////////////////////////// OCTREE HELPER FUNCTIONS ///////////////////////////
//////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ CIdAABB createNewAABB(const CAABB& aabb){
    CIdAABB id = __nv_atomic_fetch_add(&globalVariables.nbAABBs, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    if(id >= globalVariables.maxNbAABBs || id == CINVALID_ID){
        printf("ERROR: reached the maximum number of nodes that can be created\n");
        customAssert();
    }
    globalVariables.allAABBs[id] = aabb;
    return id;
}


__device__ __forceinline__ const CAABB& getAABB(const CIdAABB& aabb_index){
    return globalVariables.allAABBs[aabb_index];
}


__device__ __forceinline__ void displayOctreeNode(const COctreeNode* node, uint32_t id = 0, uint32_t level = 0){
    if(!node){
        printf("ERROR: can't display a null node\n");
        customAssert();
    }

    CIdAABB aabb_index = node->aabb_index;

    printf("level: %d, id: %d, aabb_index: %d, nb points: %d, nb voxels: %d, "
        "children visibility: 0b%d%d%d%d%d%d%d%d, "
        "points location: 0b%d%d%d%d%d%d%d%d, children: 0b%d%d%d%d%d%d%d%d\n",
        level, id, node->aabb_index, node->points_counter, node->voxels_counter, 
        uint8_t(bool(node->children_visibility & 0x01 << 0)),
        uint8_t(bool(node->children_visibility & 0x01 << 1)),
        uint8_t(bool(node->children_visibility & 0x01 << 2)),
        uint8_t(bool(node->children_visibility & 0x01 << 3)),
        uint8_t(bool(node->children_visibility & 0x01 << 4)),
        uint8_t(bool(node->children_visibility & 0x01 << 5)),
        uint8_t(bool(node->children_visibility & 0x01 << 6)),
        uint8_t(bool(node->children_visibility & 0x01 << 7)),
        uint8_t(bool(node->children_ids & 0x01 << 0)),
        uint8_t(bool(node->children_ids & 0x01 << 1)),
        uint8_t(bool(node->children_ids & 0x01 << 2)),
        uint8_t(bool(node->children_ids & 0x01 << 3)),
        uint8_t(bool(node->children_ids & 0x01 << 4)),
        uint8_t(bool(node->children_ids & 0x01 << 5)),
        uint8_t(bool(node->children_ids & 0x01 << 6)),
        uint8_t(bool(node->children_ids & 0x01 << 7)),
        uint8_t(node->children[0] != nullptr), 
        uint8_t(node->children[1] != nullptr), 
        uint8_t(node->children[2] != nullptr), 
        uint8_t(node->children[3] != nullptr),
        uint8_t(node->children[4] != nullptr), 
        uint8_t(node->children[5] != nullptr), 
        uint8_t(node->children[6] != nullptr), 
        uint8_t(node->children[7] != nullptr)
    );
    const CAABB& aabb = getAABB(node->aabb_index);
    printf("    aabb: mins = (%f, %f, %f), maxs = (%f, %f, %f)\n",
        aabb.mins.x, aabb.mins.y, aabb.mins.z,
        aabb.maxs.x, aabb.maxs.y, aabb.maxs.z
    );
}


__device__ __forceinline__ void displayOctreeRec(const COctreeNode* node, uint32_t id = 0, uint32_t level = 0){
    displayOctreeNode(node, id, level);
    for(size_t i=0; i<8; i++){
        if(node->children[i]){
            displayOctreeRec(node->children[i], i, level+1);
        }
    }
}


__device__ __forceinline__ void displayOctreeIt(const COctreeNode* node, uint32_t id = 0, uint32_t level = 0){

    struct Tmp {
        const COctreeNode* node;
        uint32_t id;
        uint32_t level;
    };

    CDoubleLinkedList<Tmp> to_display = {};
    to_display.init();
    to_display.pushBack({node, id, level});

    uint32_t total_points = 0;
    uint32_t total_voxels = 0;

    while(!to_display.isEmpty()){
        Tmp* tmp = to_display.front();
        const COctreeNode* cur_node = tmp->node;
        uint32_t cur_id = tmp->id;
        uint32_t cur_level = tmp->level;
        to_display.popFront();

        total_points += cur_node->points_counter;
        total_voxels += cur_node->voxels_counter;

        displayOctreeNode(cur_node, cur_id, cur_level);
        for(uint32_t i=0; i<8; i++){
            if(cur_node->children[i]){
                to_display.pushBack({cur_node->children[i], i, cur_level+1});
            }
        }
    }

    printf("Total points: %d, total voxels: %d\n\n", total_points, total_voxels);
}










//////////////////////////////////////////////////////////////////////////////
////////////////////////// GLOBAL HELPER FUNCTIONS ///////////////////////////
//////////////////////////////////////////////////////////////////////////////


template<typename T>
__device__
T clamp(T value, T min, T max){

	if(value < min) return min;
	if(value > max) return max;

	return value;
}



// https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda/51549250#51549250
__device__ __forceinline__ void atomicMinFloatRelaxedOrderSystemScope(float* addr, float value){
    if(value >= 0.f) {
        __nv_atomic_min((int *)addr, __float_as_int(value),
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE
        );
    } else {
        __nv_atomic_max((unsigned int *)addr, __float_as_uint(value),
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE
        );
    }
}
__device__ __forceinline__ float atomicMaxFloatRelaxedOrderSystemScope(float* addr, float value){
    if(value >= 0.f) {
        __nv_atomic_max((int *)addr, __float_as_int(value),
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE
        );
    } else {
        __nv_atomic_min((unsigned int *)addr, __float_as_uint(value),
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE
        );
    }
}
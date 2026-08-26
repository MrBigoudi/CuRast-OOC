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
#define customAssert() { printf("Assertion failure!\n"); asm("trap;"); return;}


// #define ASSERT_ENABLED


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

__device__ __forceinline__ CIdAABB createNewNodeId(){
    CIdAABB id = __nv_atomic_fetch_add(&globalVariables.totalNbNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    if(id >= globalVariables.maxNbConcurrentNodes || id == CINVALID_ID){
        printf("ERROR: reached the maximum number of nodes that can be created\n");
        customAssert();
    }
    return id;
}


__device__ __forceinline__ void displayOctreeNode(const COctreeNode* node, uint32_t id = 0, uint32_t level = 0){
    if(!node){
        printf("ERROR: can't display a null node\n");
        customAssert();
    }

    CIdAABB aabb_index = node->aabb_index;

    printf("level: %d, id: %d, aabb_index: %d, nb points: %d, nb voxels: %d, "
        "points location: 0b%d%d%d%d%d%d%d%d, children: 0b%d%d%d%d%d%d%d%d\n",
        level, id, node->aabb_index, node->points_counter, node->voxels_counter, 
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
    const CAABB& aabb = globalVariables.relationshipMap[node->aabb_index].aabb;
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
__device__ __forceinline__ void atomicMinFloatRelaxedOrderDeviceScope(float* addr, float value){
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
__device__ __forceinline__ float atomicMaxFloatRelaxedOrderDeviceScope(float* addr, float value){
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
__device__ __forceinline__ void atomicMinFloatRelaxedOrderBlockScope(float* addr, float value){
    if(value >= 0.f) {
        __nv_atomic_min((int *)addr, __float_as_int(value),
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK
        );
    } else {
        __nv_atomic_max((unsigned int *)addr, __float_as_uint(value),
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK
        );
    }
}
__device__ __forceinline__ float atomicMaxFloatRelaxedOrderBlockScope(float* addr, float value){
    if(value >= 0.f) {
        __nv_atomic_max((int *)addr, __float_as_int(value),
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK
        );
    } else {
        __nv_atomic_min((unsigned int *)addr, __float_as_uint(value),
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK
        );
    }
}

__device__
inline uint64_t nanotime(){

	uint64_t nanotime;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime));

	return nanotime;
}


// extern "C" __global__
// void kernel_test_display(PipelineLevel level, bool display_octree){
//     if(!globalVariables.isInitialised){return;}
//     printf("\n\n\n\n\n");
//     switch (level) {
//         case LevelInit:
//             printf("Init\n\n");
//             break;
//         case LevelBottomUp:
//             printf("Bottom up\n\n");
//             break;
//         case LevelSimlodLoad:
//             printf("Simlod load\n\n");
//             break;
//         case LevelSimlodSplitCount:
//             printf("Simlod split count\n\n");
//             break;
//         case LevelSimlodVoxelSampling:
//             printf("Simlod voxel sampling\n\n");
//             break;
//         case LevelSimlodInsertion:
//             printf("Simlod insertion\n\n");
//             break;
//         case LevelSimlod:
//             printf("Simlod\n\n");
//             break;
//         case LevelCacheUpdate:
//             printf("Cache update\n\n");
//             break;
//     }
    
//     if(display_octree){
//         displayOctreeIt(globalVariables.mainOctree);
//         printf("\n\n");
//     }
// }



// uint64_t t_start = nanotime();
// cg::this_grid().sync();
// uint64_t nanos = nanotime() - t_start;
// float micros = nanos / 1000;
// if(cg::this_grid().thread_rank() == 0){
//     printf("microseconds: %f \n", micros);
// }
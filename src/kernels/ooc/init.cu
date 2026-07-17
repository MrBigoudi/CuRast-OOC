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
using glm::vec4;

namespace cg = cooperative_groups;

__device__ CGlobalVariables globalVariables;
__device__ CMemoryAllocator globalAllocator;


template<typename T>
__device__ void initAllocatorPool(void* pool){
    CAllocatorPool<T>* allocator = reinterpret_cast<CAllocatorPool<T>*>(pool);
    allocator->elements.init();
    allocator->elements_map.init(allocator->CAPACITY);

    uint32_t alignment = alignof(T);
    uint32_t aligned_size = sizeof(T) + ((alignment - (sizeof(T) % alignment)) % alignment);
    char* base = reinterpret_cast<char*>(allocator->allocated_memory);
    for (uint32_t i = 0; i < allocator->CAPACITY; i++) {
        typename CAllocatorPool<T>::Entry* entry = new CAllocatorPool<T>::Entry();
        entry->value = new (base + i*aligned_size) T();
        allocator->elements.pushFront(entry);
        allocator->elements_map.insert_or_replace(entry->value, allocator->elements.begin());
    }
}

__device__ void initChunksAllocator() {
    initAllocatorPool<CChunk>(globalAllocator.chunksAllocator);
}

__device__ void initGridsAllocator() {
    initAllocatorPool<COccupancyGrid>(globalAllocator.gridsAllocator);
}

__device__ void initNodesAllocator() {
    initAllocatorPool<COctreeNode>(globalAllocator.nodesAllocator);
}



extern "C" __global__
void kernel_init(){
    printf("\n\n\n\n\n\n\n");

    initChunksAllocator();
    printf("Chunks Allocator initiated\n");
    // initGridsAllocator();
    // printf("Grids Allocator initiated\n");
    // initNodesAllocator();
    // printf("Nodes Allocator initiated\n");

    printf("\nFROM INIT\n");
    printf("    - Nb AABBs: %d / %d\n", globalVariables.nbAABBs, globalVariables.maxNbAABBs);
    printf("    - Nb nodes to load: %d / %d\n", globalVariables.nbNodesToLoad, globalVariables.maxNbNodesToLoad);
    printf("    - Nb nodes to store: %d / %d\n", globalVariables.nbNodesToStore, globalVariables.maxNbNodesToStore);
    printf("    - Nb spilled points: %d / %d\n", globalVariables.nbSpilledPoints, globalVariables.maxNbSpilledPoints);
    printf("    - Nb backlog voxels: %d / %d\n", globalVariables.nbBacklogVoxels, globalVariables.maxNbBacklogVoxels);

    printf("\n\n\n\n\n\n\n");
}
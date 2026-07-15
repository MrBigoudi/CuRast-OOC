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


__device__ void initChunksAllocator() {
    CAllocatorPool<CChunk>* allocator = globalAllocator.chunksAllocator;
    printf("    - capacity: %d, id: %d\n", allocator->CAPACITY, allocator->ALLOCATOR_ID);

    allocator->elements.init();
    printf("    - elements initialised, size: %d\n", allocator->elements.size);
    allocator->elements.size = 1;
    printf("    - elements initialised, size: %d\n", allocator->elements.size);


    printf("    - map initialised, size: %d, capacity: %d\n", allocator->elements_map.size, allocator->elements_map.capacity);
    allocator->elements_map = {};
    printf("    - map initialised, size: %d, capacity: %d\n", allocator->elements_map.size, allocator->elements_map.capacity);
    allocator->elements_map.init(allocator->CAPACITY);
    printf("    - map initialised, size: %d, capacity: %d\n", allocator->elements_map.size, allocator->elements_map.capacity);
    allocator->elements_map.capacity = 1;
    allocator->elements_map.size = 2;
    printf("    - map initialised, size: %d, capacity: %d\n", allocator->elements_map.size, allocator->elements_map.capacity);

    printf("map addr %p\n", &allocator->elements_map);
    printf("cap addr %p\n", &allocator->elements_map.capacity);

    uint32_t alignment = alignof(CChunk);
    uint32_t aligned_size = sizeof(CChunk) + ((alignment - (sizeof(CChunk) % alignment)) % alignment);

    // for (uint32_t i = 0; i < allocator->CAPACITY; i++) {
    //     CAllocatorPool<CChunk>::Entry* entry = new CAllocatorPool<CChunk>::Entry();
    //     entry->value = new (allocator->allocated_memory + i*aligned_size) CChunk();
    //     allocator->elements.pushFront(entry);
    //     allocator->elements_map[entry->value] = allocator->elements.begin();
    // }
    printf("    - pool filled\n");
}

// __device__ void initGridsAllocator() {
//     CAllocatorPool<COccupancyGrid>* allocator = globalAllocator.gridsAllocator;
//     printf("    - capacity: %d, id: %d\n", allocator->CAPACITY, allocator->ALLOCATOR_ID);
//     allocator->elements.init();
//     printf("    - elements initialised, size: %d\n", allocator->elements.size);
//     allocator->elements_map.init(allocator->CAPACITY);
//     printf("    - map initialised, size: %d, capacity: %d\n", allocator->elements_map.size, allocator->elements_map.capacity);

//     uint32_t alignment = alignof(COccupancyGrid);
//     uint32_t aligned_size = sizeof(COccupancyGrid) + ((alignment - (sizeof(COccupancyGrid) % alignment)) % alignment);

//     for (uint32_t i = 0; i < allocator->CAPACITY; i++) {
//         CAllocatorPool<COccupancyGrid>::Entry* entry = new CAllocatorPool<COccupancyGrid>::Entry();
//         entry->value = new (allocator->allocated_memory + i*aligned_size) COccupancyGrid();
//         allocator->elements.pushFront(entry);
//         allocator->elements_map[entry->value] = allocator->elements.begin();
//     }
//     printf("    - pool filled\n");
// }

// __device__ void initNodesAllocator() {
//     CAllocatorPool<COctreeNode>* allocator = globalAllocator.nodesAllocator;
//     printf("    - capacity: %d, id: %d\n", allocator->CAPACITY, allocator->ALLOCATOR_ID);
//     allocator->elements.init();
//     printf("    - elements initialised, size: %d\n", allocator->elements.size);
//     allocator->elements_map.init(allocator->CAPACITY);
//     printf("    - map initialised, size: %d, capacity: %d\n", allocator->elements_map.size, allocator->elements_map.capacity);

//     uint32_t alignment = alignof(COctreeNode);
//     uint32_t aligned_size = sizeof(COctreeNode) + ((alignment - (sizeof(COctreeNode) % alignment)) % alignment);

//     for (uint32_t i = 0; i < allocator->CAPACITY; i++) {
//         CAllocatorPool<COctreeNode>::Entry* entry = new CAllocatorPool<COctreeNode>::Entry();
//         entry->value = new (allocator->allocated_memory + i*aligned_size) COctreeNode();
//         allocator->elements.pushFront(entry);
//         allocator->elements_map[entry->value] = allocator->elements.begin();
//     }
//     printf("    - pool filled\n");
// }



extern "C" __global__
void kernel_init(){
    printf("\n\n\n\n\n\n\n");

    initChunksAllocator();
    printf("Chunks Allocator initiated\n");
    // initGridsAllocator();
    // printf("Grids Allocator initiated\n");
    // initNodesAllocator();
    // printf("Nodes Allocator initiated\n");

    printf("FROM INIT\n");
    printf("    - Nb AABBs: %d / %d\n", globalVariables.nbAABBs, globalVariables.maxNbAABBs);
    printf("    - Nb nodes to load: %d / %d\n", globalVariables.nbNodesToLoad, globalVariables.maxNbNodesToLoad);
    printf("    - Nb nodes to store: %d / %d\n", globalVariables.nbNodesToStore, globalVariables.maxNbNodesToStore);
    printf("    - Nb spilled points: %d / %d\n", globalVariables.nbSpilledPoints, globalVariables.maxNbSpilledPoints);
    printf("    - Nb backlog voxels: %d / %d\n", globalVariables.nbBacklogVoxels, globalVariables.maxNbBacklogVoxels);

    printf("\n\n\n\n\n\n\n");
}
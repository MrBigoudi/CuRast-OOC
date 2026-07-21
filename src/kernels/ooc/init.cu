#include "utils.cuh"

// To define them only once
__device__ CGlobalVariables globalVariables;
__device__ CMemoryAllocator globalAllocator;


template<typename T>
__device__ void initAllocatorPool(void* pool){
    CAllocatorPool<T>* allocator = reinterpret_cast<CAllocatorPool<T>*>(pool);

    uint32_t alignment = alignof(T);
    uint32_t aligned_size = sizeof(T) + ((alignment - (sizeof(T) % alignment)) % alignment);
    char* base = reinterpret_cast<char*>(allocator->allocated_memory);

    auto it = allocator->elements->begin();
    uint32_t i = 0;
    while(it){
        typename CAllocatorPool<T>::Entry* entry = it->value;
        entry->value = new (base + i*aligned_size) T();
        allocator->elements_map->insert_or_replace(entry->value, it);
        i++;
        it = it->next;
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


/// Run on a single thread
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

    // Initialise empty AABB
    createNewAABB(CAABB());

    printf("\n\n\n\n\n\n\n");
}

/// Run on 1 block of X threads
/// This function should only run once after the first point cloud is done being load from disk
extern "C" __global__
void kernel_init_octree(){

    if(globalVariables.mainOctree){return;}
    if(globalVariables.batchesAddedMask[0]){return;}
    CPointBatch batch = globalVariables.batchesToAdd[0];

    // Assume 1D kernel launch
    uint32_t thread_id = cg::this_grid().thread_rank();
    if(thread_id > batch.count){return;}

    // Thread level AABB
    CAABB tmp_aabb = CAABB();
    uint32_t nb_threads = blockDim.x * gridDim.x;
    for(uint32_t i=thread_id; i<batch.count; i+=nb_threads){
        CPoint& point = batch.points[i];
		tmp_aabb.maxs.x = max(tmp_aabb.maxs.x, point.position.x);
		tmp_aabb.maxs.y = max(tmp_aabb.maxs.y, point.position.y);
		tmp_aabb.maxs.z = max(tmp_aabb.maxs.z, point.position.z);
		tmp_aabb.mins.x = min(tmp_aabb.mins.x, point.position.x);
		tmp_aabb.mins.y = min(tmp_aabb.mins.y, point.position.y);
		tmp_aabb.mins.z = min(tmp_aabb.mins.z, point.position.z);
	}

    // Block level AABB
    atomicMinFloatRelaxedOrderSystemScope(
        &globalVariables.allAABBs[0].mins.x,
        tmp_aabb.mins.x
    );
    atomicMinFloatRelaxedOrderSystemScope(
        &globalVariables.allAABBs[0].mins.y,
        tmp_aabb.mins.y
    );
    atomicMinFloatRelaxedOrderSystemScope(
        &globalVariables.allAABBs[0].mins.z,
        tmp_aabb.mins.z
    );
    atomicMaxFloatRelaxedOrderSystemScope(
        &globalVariables.allAABBs[0].maxs.x,
        tmp_aabb.maxs.x
    );
    atomicMaxFloatRelaxedOrderSystemScope(
        &globalVariables.allAABBs[0].maxs.y,
        tmp_aabb.maxs.y
    );
    atomicMaxFloatRelaxedOrderSystemScope(
        &globalVariables.allAABBs[0].maxs.z,
        tmp_aabb.maxs.z
    );

    if(thread_id == 0){
        // Sync the threads in the block
        __syncthreads();

        // Adding small 2x delta to avoid floating point issues
        float epsilon = 0.5f;
        globalVariables.allAABBs[0].mins -= epsilon * globalVariables.allAABBs[0].mins;
        globalVariables.allAABBs[0].maxs += epsilon * globalVariables.allAABBs[0].maxs;

        // Make it cubic
        vec3 size = globalVariables.allAABBs[0].getSize();
        vec3 half_sizes_x = 0.5f * (vec3(size.x) - size);
        vec3 half_sizes_y = 0.5f * (vec3(size.y) - size);
        vec3 half_sizes_z = 0.5f * (vec3(size.z) - size);
        if(size.x > size.y){
            if(size.x > size.z){
                globalVariables.allAABBs[0].mins.y -= half_sizes_x.y;
                globalVariables.allAABBs[0].maxs.y += half_sizes_x.y;
                globalVariables.allAABBs[0].mins.z -= half_sizes_x.z;
                globalVariables.allAABBs[0].maxs.z += half_sizes_x.z;
            } else {
                globalVariables.allAABBs[0].mins.y -= half_sizes_z.y;
                globalVariables.allAABBs[0].maxs.y += half_sizes_z.y;
                globalVariables.allAABBs[0].mins.x -= half_sizes_z.x;
                globalVariables.allAABBs[0].maxs.x += half_sizes_z.x;
            }
        } else {
            if(size.y > size.z){
                globalVariables.allAABBs[0].mins.x -= half_sizes_y.x;
                globalVariables.allAABBs[0].maxs.x += half_sizes_y.x;
                globalVariables.allAABBs[0].mins.z -= half_sizes_y.z;
                globalVariables.allAABBs[0].maxs.z += half_sizes_y.z;
            } else {
                globalVariables.allAABBs[0].mins.y -= half_sizes_z.y;
                globalVariables.allAABBs[0].maxs.y += half_sizes_z.y;
                globalVariables.allAABBs[0].mins.x -= half_sizes_z.x;
                globalVariables.allAABBs[0].maxs.x += half_sizes_z.x;
            }
        }

        // Create the main octree
        CIdAABB id = createNewAABB(globalVariables.allAABBs[0]);
        globalVariables.mainOctree = globalAllocator.newOctreeNode(id);
    }

}
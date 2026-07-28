#include "utils.cuh"

// To define them only once
__device__ CGlobalVariables globalVariables;
__device__ CMemoryAllocator globalAllocator;

// __device__ uint32_t globalCpt = 0;

template<typename T>
__device__ void initAllocatorPool(uint32_t thread_id, void* pool){
    CAllocatorPool<T>* allocator = reinterpret_cast<CAllocatorPool<T>*>(pool);

    uint64_t alignment = alignof(T);
    uint64_t aligned_size = sizeof(T) + ((alignment - (sizeof(T) % alignment)) % alignment);
    char* base = reinterpret_cast<char*>(allocator->allocated_memory);

    // Init empty entries
    auto it = allocator->elements->begin();
    uint64_t i = 0;
    while(it){
        T* tmp_key = (T*)(base + i*aligned_size);
        uint32_t key = (uint32_t)((allocator->elements_map->hash_murmur(tmp_key)) % allocator->CAPACITY);
        if(key == thread_id){
            // uint32_t cur_cpt = __nv_atomic_fetch_add(&globalCpt, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
            // printf("cur_cpt: %d / %d\n", cur_cpt, allocator->CAPACITY);
            typename CAllocatorPool<T>::Entry* entry = it->value;
            entry->value = new (base + i*aligned_size) T();
            allocator->elements_map->insert_or_replace(entry->value, it);
        }
        allocator->deallocated_memory[i] = nullptr;
        i++;
        it = it->next;
    }
}

__device__ void initChunksAllocator(uint32_t thread_id) {
    initAllocatorPool<CChunk>(thread_id, globalAllocator.chunksAllocator);
}

__device__ void initGridsAllocator(uint32_t thread_id) {
    initAllocatorPool<COccupancyGrid>(thread_id, globalAllocator.gridsAllocator);
}

__device__ void initNodesAllocator(uint32_t thread_id) {
    initAllocatorPool<COctreeNode>(thread_id, globalAllocator.nodesAllocator);
}


extern "C" __global__
void kernel_init_global_allocators(){
    uint32_t thread_id = cg::this_grid().thread_rank();
    uint32_t grid_size = gridDim.x;

    if(thread_id == grid_size-1){printf("\n\n");}

    initChunksAllocator(thread_id);
    if(thread_id == grid_size-1){printf("Chunks Allocator initiated\n");}
    initGridsAllocator(thread_id);
    if(thread_id == grid_size-1){printf("Grids Allocator initiated\n");}
    initNodesAllocator(thread_id);
    if(thread_id == grid_size-1){printf("Nodes Allocator initiated\n");}

    if(thread_id == grid_size-1){
        printf("\nFROM INIT\n");
        printf("    - Nb AABBs: %d / %d\n", globalVariables.nbAABBs, globalVariables.maxNbAABBs);
        printf("    - Nb nodes to load: %d / %d\n", globalVariables.nbNodesToLoad, globalVariables.maxNbNodesToLoad);
        printf("    - Nb nodes to store: %d / %d\n", globalVariables.nbNodesToStore, globalVariables.maxNbNodesToStore);
        printf("    - Nb spilled points: %d / %d\n", globalVariables.nbSpilledPoints, globalVariables.maxNbSpilledPoints);
        printf("    - Nb backlog voxels: %d / %d\n", globalVariables.nbBacklogVoxels, globalVariables.maxNbBacklogVoxels);

        printf("\n\n");
    }
}

extern "C" __global__
void kernel_init_global_buffers(){
    uint32_t thread_id = cg::this_grid().thread_rank();

    if(thread_id < globalVariables.maxNbAABBs){
        globalVariables.relationshipMap[thread_id] = CGlobalVariables::Relationship();
        globalVariables.allAABBs[thread_id] = CAABB();
        globalVariables.nodes[thread_id] = nullptr;
        globalVariables.nodesFlags[thread_id] = 0;
    }

    if(thread_id < globalVariables.maxNbNodesToLoad){
        globalVariables.nodesToLoadBuffer[thread_id] = CINVALID_ID;
    }
    if(thread_id < globalVariables.maxNbNodesToStore){
        globalVariables.nodesToStoreBuffer[thread_id] = nullptr;
    }

    if(thread_id < globalVariables.maxNbBatches){
        globalVariables.batchesAddedMask[thread_id] = true;
        globalVariables.batchesToAddCounts[thread_id] = 0;
        globalVariables.batchesToAddBottomUpCounts[thread_id] = 0;
    }

    if(thread_id < globalVariables.updatesCacheSize){
        globalVariables.updatesCache[thread_id] = CINVALID_ID;
    } 
    if(thread_id < globalVariables.visibilityCacheSize){
        globalVariables.visibilityCache[thread_id] = CINVALID_ID;
    }    
}

extern "C" __global__
void kernel_init_octree_part_1(){
    // To only run it once
    if(globalVariables.mainOctree){return;}
    // To only run it after the first batch has been loaded
    if(globalVariables.batchesAddedMask[0]){return;}

    CPoint* new_points = globalVariables.batchesToAddPoints[0];
    uint32_t new_count = globalVariables.batchesToAddCounts[0];

    // Assume 1D kernel launch
    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    if(thread_id >= new_count){return;}

    // Thread level AABB
    CAABB tmp_aabb = CAABB();
    uint32_t nb_threads = grid.num_threads();
    for(uint32_t i=thread_id; i<new_count; i+=nb_threads){
        CPoint& point = new_points[i];
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
}

extern "C" __global__
void kernel_init_octree_part_2(){
    // To only run it once
    if(globalVariables.mainOctree){return;}
    // To only run it after the first batch has been loaded
    if(globalVariables.batchesAddedMask[0]){return;}

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
    globalVariables.mainOctree = globalAllocator.newOctreeNode(id, false);
    globalVariables.nodes[0] = globalVariables.mainOctree;

    // // TODO: to remove
    // {
    //     printf("Initial Octree: \n");
    //     displayOctreeNode(globalVariables.mainOctree);
    // }
}
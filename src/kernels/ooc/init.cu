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
        uint32_t key = (uint32_t)((allocator->elements_map->hashMurmur(tmp_key)) % allocator->CAPACITY);
        if(key == thread_id){
            typename CAllocatorPool<T>::Entry* entry = it->value;
            entry->value = new (base + i*aligned_size) T();
            allocator->elements_map->insertOrReplace(entry->value, it);
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

    initChunksAllocator(thread_id);
    initGridsAllocator(thread_id);
    initNodesAllocator(thread_id);
}

extern "C" __global__
void kernel_init_global_buffers(){
    uint32_t thread_id = cg::this_grid().thread_rank();

    if(thread_id < globalVariables.maxNbConcurrentNodes){
        globalVariables.relationshipMap[thread_id] = CGlobalVariables::Relationship();
        globalVariables.packedNodes[thread_id] = nullptr;
        globalVariables.nodesFlags[thread_id] = 0;
    }

    if(thread_id < globalVariables.maxNbBatches){
        globalVariables.batchesAddedMask[thread_id] = true;
        globalVariables.batchesToAddCounts[thread_id] = 0;
    }

    if(thread_id == 0){
        globalVariables.updatesCache = new CLRUCache(globalVariables.updatesCacheSize);
        globalVariables.visibilityCache = new CLRUCache(globalVariables.visibilityCacheSize);
        
        // Create the main octree
        CIdAABB id = createNewNodeId();
        globalVariables.mainOctree = globalAllocator.newOctreeNode(id, false);
        globalVariables.mainOctree->aabb.mins = {INFINITY, INFINITY, INFINITY};
        globalVariables.mainOctree->aabb.maxs = {-INFINITY, -INFINITY, -INFINITY};
        globalVariables.packedNodes[0] = globalVariables.mainOctree;
        globalVariables.curNbNodes = 1;
    }
}






/// Run on "maxPointsPerBatches" threads
extern "C" __global__
void kernel_init_octree_part_1(){
    // To only run it once
    if(globalVariables.isInitialised){return;}

    // Assume 1D kernel launch
    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    // Thread level AABB
    CAABB tmp_aabb = CAABB();
    bool is_init = false;

    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

        for(uint32_t i=thread_id; i<nb_new_points; i+=nb_threads){
            CPoint& point = new_points[i];
            if(!is_init){
                tmp_aabb.mins = new_points[i].position;
                tmp_aabb.maxs = new_points[i].position;
                is_init = true;
                continue;
            }

            tmp_aabb.maxs.x = max(tmp_aabb.maxs.x, point.position.x);
            tmp_aabb.maxs.y = max(tmp_aabb.maxs.y, point.position.y);
            tmp_aabb.maxs.z = max(tmp_aabb.maxs.z, point.position.z);
            tmp_aabb.mins.x = min(tmp_aabb.mins.x, point.position.x);
            tmp_aabb.mins.y = min(tmp_aabb.mins.y, point.position.y);
            tmp_aabb.mins.z = min(tmp_aabb.mins.z, point.position.z);
        }
    }

    // Block level AABB
    atomicMinFloatRelaxedOrderSystemScope(
        &globalVariables.mainOctree->aabb.mins.x,
        tmp_aabb.mins.x
    );
    atomicMinFloatRelaxedOrderSystemScope(
        &globalVariables.mainOctree->aabb.mins.y,
        tmp_aabb.mins.y
    );
    atomicMinFloatRelaxedOrderSystemScope(
        &globalVariables.mainOctree->aabb.mins.z,
        tmp_aabb.mins.z
    );
    atomicMaxFloatRelaxedOrderSystemScope(
        &globalVariables.mainOctree->aabb.maxs.x,
        tmp_aabb.maxs.x
    );
    atomicMaxFloatRelaxedOrderSystemScope(
        &globalVariables.mainOctree->aabb.maxs.y,
        tmp_aabb.maxs.y
    );
    atomicMaxFloatRelaxedOrderSystemScope(
        &globalVariables.mainOctree->aabb.maxs.z,
        tmp_aabb.maxs.z
    );
}


/// Run on a single thread
extern "C" __global__
void kernel_init_octree_part_2(){
    // To only run it once
    if(globalVariables.isInitialised){return;}
    // To only run it after the first batch has been loaded
    if(globalVariables.batchesAddedMask[0]){return;}

    // Adding small 2x delta to avoid floating point issues
    float epsilon = 0.5f;
    globalVariables.mainOctree->aabb.mins -= epsilon * globalVariables.mainOctree->aabb.mins;
    globalVariables.mainOctree->aabb.maxs += epsilon * globalVariables.mainOctree->aabb.maxs;

    // Make it cubic
    vec3 size = globalVariables.mainOctree->aabb.getSize();
    vec3 half_sizes_x = 0.5f * (vec3(size.x) - size);
    vec3 half_sizes_y = 0.5f * (vec3(size.y) - size);
    vec3 half_sizes_z = 0.5f * (vec3(size.z) - size);
    if(size.x > size.y){
        if(size.x > size.z){
            globalVariables.mainOctree->aabb.mins.y -= half_sizes_x.y;
            globalVariables.mainOctree->aabb.maxs.y += half_sizes_x.y;
            globalVariables.mainOctree->aabb.mins.z -= half_sizes_x.z;
            globalVariables.mainOctree->aabb.maxs.z += half_sizes_x.z;
        } else {
            globalVariables.mainOctree->aabb.mins.y -= half_sizes_z.y;
            globalVariables.mainOctree->aabb.maxs.y += half_sizes_z.y;
            globalVariables.mainOctree->aabb.mins.x -= half_sizes_z.x;
            globalVariables.mainOctree->aabb.maxs.x += half_sizes_z.x;
        }
    } else {
        if(size.y > size.z){
            globalVariables.mainOctree->aabb.mins.x -= half_sizes_y.x;
            globalVariables.mainOctree->aabb.maxs.x += half_sizes_y.x;
            globalVariables.mainOctree->aabb.mins.z -= half_sizes_y.z;
            globalVariables.mainOctree->aabb.maxs.z += half_sizes_y.z;
        } else {
            globalVariables.mainOctree->aabb.mins.y -= half_sizes_z.y;
            globalVariables.mainOctree->aabb.maxs.y += half_sizes_z.y;
            globalVariables.mainOctree->aabb.mins.x -= half_sizes_z.x;
            globalVariables.mainOctree->aabb.maxs.x += half_sizes_z.x;
        }
    }


    globalVariables.isInitialised = true;
    
}


extern "C" __global__
void kernel_fill_new_grids(){
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_thread_per_blocks = block.num_threads();

    for(uint32_t node_id = block_id; node_id < globalVariables.nbGridsToInit; node_id += nb_blocks){
        COctreeNode* node = globalVariables.gridsToInit[node_id];
        COccupancyGrid* occupancy = node->occupancy;

        uint32_t grid_size = OocSimLodSettings::GRID_SIZE / 32;
        for(uint32_t i=thread_id; i<grid_size; i+=nb_thread_per_blocks){
            occupancy->values[i] = 0;
        }

        // Loop over it's voxels if already have some
        CChunk* cur_chunk = node->voxels;
        const CAABB& aabb = node->aabb;
        while(cur_chunk){
            for(uint32_t i=thread_id; i<cur_chunk->size; i+=nb_thread_per_blocks){
                COccupancyGrid::GridIndex index = COccupancyGrid::getCellIndices(aabb, cur_chunk->points[i]);
                occupancy->markCellAsFilled(index);
            }
            cur_chunk = cur_chunk->next;
        }
    }
}
#include "utils.cuh"

// To define them only once
__device__ CGlobalVariables globalVariables;
__device__ CMemoryAllocator globalAllocator;


/// Run on floor("NB SMs" * "Max threads per SM" / "Max threads per block") blocks of size "Max threads per block"
extern "C" __global__
void kernel_init_global_buffers(){
    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();


    for(uint32_t i = thread_id; i < globalVariables.maxNbConcurrentNodes; i += nb_threads){
        globalVariables.relationshipMap[thread_id] = CGlobalVariables::Relationship();
        globalVariables.packedNodes[thread_id] = nullptr;
        globalVariables.resetFlags(thread_id);
        globalVariables.gridsToInitExchangedIndex[thread_id] = -1;
    }


    for(uint32_t i = thread_id; i < globalVariables.maxNbBatches; i += nb_threads){
        globalVariables.batchesAddedMask[thread_id] = true;
        globalVariables.batchesToAddCounts[thread_id] = 0;
    }


    for(uint32_t i = thread_id; i < globalVariables.updatesCacheSize; i += nb_threads){
        globalVariables.updatesCache[i] = CINVALID_ID;
    }


    if(thread_id == 0){
        // Create the main octree
        CIdAABB id = createNewNodeId();
        globalVariables.mainOctree = globalAllocator.newOctreeNode(id);
        globalVariables.packedNodes[0] = globalVariables.mainOctree;
        globalVariables.mainOctree->cur_id = 0;
        globalVariables.curNbNodes = 1;
    }
}























/// Run on floor("NB SMs" * "Max threads per SM" / "Max threads per block") blocks of size "Max threads per block"
/// Each thread is filling independently it's own bounding box before combining all of them
extern "C" __global__
void kernel_init_octree_part_1_aabb_measuring(){
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    uint32_t first_point = block_id * nb_threads_per_block + thread_id;
    uint32_t step = nb_blocks * nb_threads_per_block;

    // if(block_id == 0 && thread_id == 0){
    //     printf("kernel_init_octree_part_1_aabb_measuring\n");
    // }

    __shared__ float shBlockMinX;
    __shared__ float shBlockMinY;
    __shared__ float shBlockMinZ;
    __shared__ float shBlockMaxX;
    __shared__ float shBlockMaxY;
    __shared__ float shBlockMaxZ;

    // Thread level AABB
    CAABB tmp_aabb = CAABB();
    bool is_init = false;

    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

#ifdef ASSERT_ENABLED
        if(nb_new_points > globalVariables.maxBatchSize){
            printf("ERROR: On init, batch size exceeded the limit: %d / %d\n", 
                nb_new_points, globalVariables.maxBatchSize
            );
            customAssert();
        }
#endif


        for(uint32_t i = first_point; i < nb_new_points; i += step){
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
    atomicMinFloatRelaxedOrderBlockScope(
        &shBlockMinX,
        tmp_aabb.mins.x
    );
    atomicMinFloatRelaxedOrderBlockScope(
        &shBlockMinY,
        tmp_aabb.mins.y
    );
    atomicMinFloatRelaxedOrderBlockScope(
        &shBlockMinZ,
        tmp_aabb.mins.z
    );
    atomicMaxFloatRelaxedOrderBlockScope(
        &shBlockMaxX,
        tmp_aabb.maxs.x
    );
    atomicMaxFloatRelaxedOrderBlockScope(
        &shBlockMaxY,
        tmp_aabb.maxs.y
    );
    atomicMaxFloatRelaxedOrderBlockScope(
        &shBlockMaxZ,
        tmp_aabb.maxs.z
    );
    __syncthreads();

    // Grid level AABB
    if(thread_id == 0){
        atomicMinFloatRelaxedOrderDeviceScope(
            &globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.mins.x,
            shBlockMinX
        );
        atomicMinFloatRelaxedOrderDeviceScope(
            &globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.mins.y,
            shBlockMinY
        );
        atomicMinFloatRelaxedOrderDeviceScope(
            &globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.mins.z,
            shBlockMinZ
        );
        atomicMaxFloatRelaxedOrderDeviceScope(
            &globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.maxs.x,
            shBlockMaxX
        );
        atomicMaxFloatRelaxedOrderDeviceScope(
            &globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.maxs.y,
            shBlockMaxY
        );
        atomicMaxFloatRelaxedOrderDeviceScope(
            &globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.maxs.z,
            shBlockMaxZ
        );
    }
}


/// Run on a single thread
extern "C" __global__
void kernel_init_octree_part_2_refining(){
    // printf("kernel_init_octree_part_2_refining\n");

    // Adding small 2x delta to avoid floating point issues
    float epsilon = 0.5f;
    globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.mins 
        -= epsilon * globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.mins;
    globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.maxs 
        += epsilon * globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.maxs;

    // Make it cubic
    vec3 size = globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.getSize();
    vec3 half_sizes_x = 0.5f * (vec3(size.x) - size);
    vec3 half_sizes_y = 0.5f * (vec3(size.y) - size);
    vec3 half_sizes_z = 0.5f * (vec3(size.z) - size);
    if(size.x > size.y){
        if(size.x > size.z){
            globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.mins.y -= half_sizes_x.y;
            globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.maxs.y += half_sizes_x.y;
            globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.mins.z -= half_sizes_x.z;
            globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.maxs.z += half_sizes_x.z;
        } else {
            globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.mins.y -= half_sizes_z.y;
            globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.maxs.y += half_sizes_z.y;
            globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.mins.x -= half_sizes_z.x;
            globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.maxs.x += half_sizes_z.x;
        }
    } else {
        if(size.y > size.z){
            globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.mins.x -= half_sizes_y.x;
            globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.maxs.x += half_sizes_y.x;
            globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.mins.z -= half_sizes_y.z;
            globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.maxs.z += half_sizes_y.z;
        } else {
            globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.mins.y -= half_sizes_z.y;
            globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.maxs.y += half_sizes_z.y;
            globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.mins.x -= half_sizes_z.x;
            globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.maxs.x += half_sizes_z.x;
        }
    }

    globalVariables.isInitialised = true;

    // UI values
    globalVariables.nbNewNodesThisUpdate = 1;
    globalVariables.nbTotalNewNodes = 1;
}


/// Run on "NB SMs" blocks of size min("Max threads per SM", "Max block dim")
/// Each block is filling its assigned grids
/// Each thread in a block is filling it's assigned coordinates in the current grid
extern "C" __global__
void kernel_fill_new_grids(){
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    // if(block_id == 0 && thread_id == 0){
    //     printf("kernel_fill_new_grids\n");
    // }

    for(uint32_t node_id = block_id; node_id < globalVariables.nbGridsToInit; node_id += nb_blocks){
        COctreeNode* node = globalVariables.gridsToInit[node_id];
        uint32_t exchanged_index = globalVariables.gridsToInitExchangedIndex[node_id];
        COccupancyGrid* occupancy = node->occupancy;
        
#ifdef ASSERT_ENABLED
        if(!occupancy){
            printf("ERROR: at this point the occupancy grid should have been created\n");
            customAssert();
        }
#endif

        uint32_t grid_size = OocSimLodSettings::GRID_SIZE / 32;
        for(uint32_t i=thread_id; i<grid_size; i+=nb_threads_per_block){
            occupancy->values[i] = 0;
        }

        // Refill the grid on need
        if(exchanged_index != -1){
            uint64_t* grids_indices = globalVariables.exchangedGrids[exchanged_index];
            for(uint32_t index_id = thread_id; index_id < node->voxels_last_stored; index_id += nb_threads_per_block){
                uint64_t grid_index = grids_indices[index_id];
                uint32_t word = uint32_t(grid_index >> 32);
                uint32_t bit = uint32_t(grid_index);
                __nv_atomic_fetch_or(&occupancy->values[word], (1u << bit), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK);
            }
        }
    }
}
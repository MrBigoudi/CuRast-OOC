#include "utils.cuh"

/// Prepare the nodes that need to be loaded
/// Run on "NB SMs" * "Max threads per SM" blocks of size 1
/// TODO:
/// Later, if a point is contained in a leaf that was previously stored, put the point aside
/// For now, load them directly after this kernel launch
extern "C" __global__
void kernel_simlod_load_part_1_flagging(){
    if(!globalVariables.isInitialised){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

        for(uint32_t i=thread_id; i<nb_new_points; i+=nb_threads){
            CPoint& point = new_points[i];

            // Find the leaf containing the current point
            COctreeNode* leaf = globalVariables.mainOctree;

            CNodePosition child_position = (CNodePosition)0;
            CAABB child_aabb = CAABB();

            while(true){
                child_aabb = leaf->aabb;
                child_position = child_aabb.getNextChildIndex(point.position);
                if(leaf->children[child_position]){
                    leaf = leaf->children[child_position];
                } else {
                    break;
                }
            }

            // Check if the remaining children are already stored
            uint32_t cur_aabb_index = leaf->aabb_index;
            while(true){
                CIdAABB child_aabb_index = globalVariables.relationshipMap[cur_aabb_index].children[child_position];
                if(child_aabb_index != CINVALID_ID){
                    // Flag the node as needing to be loaded and add it to the nodes to load
                    uint32_t old_flags = globalVariables.fetchSetFlagSync(child_aabb_index, CFlagToLoad);
                    if(!(old_flags & (0x01 << CFlagToLoad))){
                        uint32_t exchanged_index = __nv_atomic_fetch_add(&globalVariables.nbNodesExchanged, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                        
                        if(exchanged_index >= globalVariables.maxNbNodesExchanged){
                            printf("ERROR: Too many nodes are being loaded\n");
                            customAssert();
                        }
                        
                        globalVariables.exchangedAABBIndices[exchanged_index] = child_aabb_index;

                        // Store the original parent in the buffer
                        globalVariables.renderingPackedNodesTmp[exchanged_index] = leaf;
                    }
                    
                    cur_aabb_index = child_aabb_index;
                    child_aabb.shrink(child_position);
                    child_position = child_aabb.getNextChildIndex(point.position);
                } else {
                    break;
                }
            }
        }
    }

    // Reset previous simlod values
    globalVariables.nbSpilledPoints = 0;
    globalVariables.nbSpillingNodes = 0;
    globalVariables.nbBacklogVoxels = 0;
}



__device__ __forceinline__ CChunk* addPointToChunk(CChunk* initial_chunk, const CPoint& cur_point){
    CChunk* cur_chunk = initial_chunk;
    if(cur_chunk->size == OocSimLodSettings::NB_POINTS_PER_CHUNK){
        cur_chunk->next = globalAllocator.newChunk(true);
        cur_chunk = cur_chunk->next;
    }
    cur_chunk->points[cur_chunk->size] = cur_point;
    cur_chunk->size++;
    return cur_chunk;
}


/// Run on "maxNbNodesExchanged" threads
/// Load the newly exchanged nodes
extern "C" __global__
void kernel_simlod_load_part_2_rebuilding_nodes(){
    if(!globalVariables.isInitialised){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    for(uint32_t exchanged_index = thread_id; exchanged_index < globalVariables.nbNodesExchanged; exchanged_index += nb_threads){

        CIdAABB aabb_index = globalVariables.exchangedAABBIndices[exchanged_index];
        CAABB aabb = globalVariables.exchangedAABBs[exchanged_index];
        uint32_t children_ids = globalVariables.exchangedChildrenIds[exchanged_index];
        uint32_t nb_points = globalVariables.exchangedPointsCounters[exchanged_index];
        uint32_t nb_voxels = globalVariables.exchangedVoxelsCounters[exchanged_index];
        CPoint* points = globalVariables.exchangedPoints[exchanged_index];
        CPoint* voxels = globalVariables.exchangedVoxels[exchanged_index];

        COctreeNode* loaded_node = globalAllocator.newOctreeNode(aabb_index, true);
        uint32_t node_index = __nv_atomic_fetch_add(&globalVariables.curNbNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        globalVariables.packedNodes[node_index] = loaded_node;

        loaded_node->aabb = aabb;
        loaded_node->children_ids = children_ids;
        loaded_node->points_counter = nb_points;
        loaded_node->voxels_counter = nb_voxels;
        loaded_node->points_stored = nb_points;
        loaded_node->voxels_stored = nb_voxels;
        globalVariables.setFlagSync(aabb_index, CFlagIsUpdated);

        // Rebuild points
        if(nb_points > 0){
            loaded_node->points = globalAllocator.newChunk(true);
            CChunk* cur_chunk = loaded_node->points;
            for(uint32_t i=0; i<nb_points; i++){
                const CPoint& cur_point = points[i];
                cur_chunk = addPointToChunk(cur_chunk, cur_point);
            }
        }

        // Rebuild voxels
        if(nb_voxels > 0){
            // Rebuild occupancy
            loaded_node->occupancy = globalAllocator.newOccupancyGrid(true);
            uint32_t grid_index = __nv_atomic_fetch_add(&globalVariables.nbGridsToInit, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            globalVariables.gridsToInit[grid_index] = loaded_node;

            loaded_node->voxels = globalAllocator.newChunk(true);
            CChunk* cur_chunk = loaded_node->voxels;
            for(uint32_t i=0; i<nb_voxels; i++){
                const CPoint& cur_point = voxels[i];
                cur_chunk = addPointToChunk(cur_chunk, cur_point);
            }
        }

        // Find parent if needed
        COctreeNode* potential_parent = globalVariables.renderingPackedNodesTmp[exchanged_index];
        if(potential_parent->aabb_index == globalVariables.relationshipMap[aabb_index].parent){
            globalVariables.setFlagSync(potential_parent->aabb_index, CFlagIsUpdated);
            bool found = false;
            for(uint32_t i=0; i<8; i++){
                if(globalVariables.relationshipMap[potential_parent->aabb_index].children[i] == aabb_index){
                    if(potential_parent->children[i] != nullptr){
                        printf("At this point, the children should not exist\n");
                        customAssert();
                    }
                    potential_parent->children[i] = loaded_node;
                    found = true;
                    break;
                }
            }
            if(!found){
                printf("The parent doesn't contain the wanted child\n");
                customAssert();
            }
        }
    }

}


/// Run on "maxNbNodesExchanged" threads
/// Rebuild the relationships of the loaded nodes
extern "C" __global__
void kernel_simlod_load_part_3_rebuilding_children(){
    if(!globalVariables.isInitialised){return;}

    // Because "kernel_fill_new_grids" should be launched just before
    globalVariables.nbGridsToInit = 0;

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    uint32_t first_node = globalVariables.curNbNodes - globalVariables.nbNodesExchanged;
    for(uint32_t node_index = first_node + thread_id; node_index < globalVariables.curNbNodes; node_index += nb_threads){
        COctreeNode* cur_node = globalVariables.packedNodes[node_index];

        uint32_t child_aabbs[8] = {
            CINVALID_ID, CINVALID_ID, CINVALID_ID, CINVALID_ID,
            CINVALID_ID, CINVALID_ID, CINVALID_ID, CINVALID_ID
        };
        uint32_t to_find = 0;
        for(uint32_t i=0; i<8; i++){
            if(cur_node->children[i]){
                printf("ERROR: a newly loaded node should not have children already initialised\n");
                customAssert();
            }
            child_aabbs[i] = globalVariables.relationshipMap[cur_node->aabb_index].children[i];
            if(child_aabbs[i] != CINVALID_ID){to_find++;}
        }
        uint32_t nb_found = 0;
        for(uint32_t i=first_node; i<globalVariables.curNbNodes; i++){
            if(nb_found == to_find){break;}
            COctreeNode* potential_node = globalVariables.packedNodes[i];
            for(uint32_t child=0; child<8; child++){
                if(child_aabbs[child] == potential_node->aabb_index){
                    cur_node->children[child] = potential_node;
                    nb_found++;
                    break;
                }
            }
        }

        // Flag the node as not needing to be loaded anymore
        globalVariables.unsetFlagSync(cur_node->aabb_index, CFlagToLoad);
        globalVariables.setFlagSync(cur_node->aabb_index, CFlagIsNew);
    }

    if(thread_id==0){
        // Because "newChunk" was called in part 3
        globalAllocator.chunksAllocator->reset_temporary_allocations();

        // // Sanity check
        // for(uint32_t i = first_node; i < globalVariables.curNbNodes; i++){
        //     CIdAABB child_id = globalVariables.packedNodes[i]->aabb_index;
        //     CIdAABB parent_id = globalVariables.relationshipMap[child_id].parent;
        //     bool found = false;
        //     for(uint32_t j=0; j<globalVariables.curNbNodes; j++){
        //         CIdAABB tmp = globalVariables.packedNodes[j]->aabb_index;
        //         if(tmp == parent_id){
        //             found = true;
        //             break;
        //         }
        //     }
        //     if(!found){
        //         printf("Can't find parent %d of child %d\n", parent_id, child_id);
        //         customAssert();
        //     }
        // }
    } 
    
    // try to avoid being on the same warp
    if((nb_threads >= 32 && thread_id==32) || (nb_threads < 32 && thread_id==0)){
        // Because "newOctreeNode" was called in part 3
        globalAllocator.nodesAllocator->reset_temporary_allocations();
    }

    // try to avoid being on the same warp
    if((nb_threads >= 64 && thread_id==64) || (nb_threads < 64 && thread_id==0)){
        // Because "newOccupancyGrid" was called in part 3
        globalAllocator.gridsAllocator->reset_temporary_allocations();
    }
}



















__device__
void updateLeafCounter(CPoint* point){
    // Find the leaf containing the current point
    COctreeNode* leaf = globalVariables.mainOctree;
    uint8_t level = 1;

    while(true){
        const CAABB& aabb = leaf->aabb;
        CNodePosition child_position = aabb.getNextChildIndex(point->position);

        if(leaf->children[child_position]){
            leaf = leaf->children[child_position];
            if(level == UINT8_MAX){
                printf("ERROR: The octree has reached it's maximum depth size...\n");
                customAssert();
            }
            level++;
        } else {
            __nv_atomic_or(&leaf->children_ids, (1u << child_position), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            // Skip if the point was already accepted at this level
            if(point->getAlpha() == level){return;}
            // Flag point as accepted at this level
            point->setAlpha(level);

            // Flag the leaf as spilling
            const uint32_t max_points_per_leaf = globalVariables.maxPointsPerLeaf;
            uint32_t old_counter = __nv_atomic_fetch_add(&leaf->points_counter, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            if(old_counter == max_points_per_leaf){ 
                // Only added once with the above equality check
                uint32_t spilling_node_index = __nv_atomic_fetch_add(&globalVariables.nbSpillingNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                if(spilling_node_index >= globalVariables.maxNbSpilledPoints){
                    printf("ERROR: reached the maximum number of spilling nodes\n");
                    customAssert();
                }
                globalVariables.spillingNodes[spilling_node_index] = leaf;
            }

            return;
        }
    }
};




__device__
void simlodCount(uint32_t first_point, uint32_t step){

    // Count new points
    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

        for(uint32_t i = first_point; i < nb_new_points; i += step){
            updateLeafCounter(&new_points[i]);
        }
    }

    // Count spilled points
    for(uint32_t i = first_point; i < globalVariables.nbSpilledPoints; i += step){
        updateLeafCounter(&globalVariables.spilledPoints[i]);
    }
}


__device__
void simlodSplit(uint32_t first_point, uint32_t step){
    for(uint32_t i = first_point; i < globalVariables.nbSpillingNodes; i += step){
        COctreeNode* spilling_node = globalVariables.spillingNodes[i];
        CIdAABB spilling_node_id = spilling_node->aabb_index;
        uint32_t spilling_node_children = spilling_node->children_ids;

        spilling_node->points_counter = 0;
        spilling_node->points_stored = 0; // also reset the previous counter

        spilling_node->children_ids = 0;
        globalVariables.setFlagSync(spilling_node_id, CFlagHasSpilled);

        if(!spilling_node->occupancy){
            spilling_node->occupancy = globalAllocator.newOccupancyGrid(true);
            uint32_t grid_index = __nv_atomic_fetch_add(&globalVariables.nbGridsToInit, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            globalVariables.gridsToInit[grid_index] = spilling_node;
        }

        for(uint32_t j=0; j<8; j++){
            // Create necessary empty children
            bool can_be_spilled = (1u << j) & spilling_node_children;
            if(!spilling_node->children[j] && can_be_spilled){                
                // Create the new node
                CIdAABB new_child_id = createNewNodeId();
                COctreeNode* new_child = globalAllocator.newOctreeNode(new_child_id, true);
                uint32_t node_index = __nv_atomic_fetch_add(&globalVariables.curNbNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                globalVariables.packedNodes[node_index] = new_child;

                spilling_node->children[j] = new_child;
                globalVariables.relationshipMap[spilling_node_id].children[j] = new_child_id;
                globalVariables.relationshipMap[new_child_id].parent = spilling_node_id;

                // Create the new AABB
                new_child->aabb = spilling_node->aabb;
                new_child->aabb.shrink((CNodePosition)j);

                globalVariables.setFlagSync(new_child_id, CFlagIsNew);
            }
        }

        // Add former points to spilled points and free memory
        CChunk* current_chunk = spilling_node->points;
        if(current_chunk){
            while(current_chunk){
                uint32_t spilled_index = __nv_atomic_fetch_add(&globalVariables.nbSpilledPoints, current_chunk->size, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                for(uint32_t j=0; j<current_chunk->size; j++){
                    uint32_t real_index = spilled_index + j;
                    if(real_index >= globalVariables.maxNbSpilledPoints){
                        printf("ERROR: reached the maximum number of spilled points\n");
                        customAssert();
                    }
                    // Flag the point as not accepted
                    CPoint cur_point = current_chunk->points[j];
                    cur_point.resetAlpha();
                    globalVariables.spilledPoints[real_index].position = cur_point.position;
                    globalVariables.spilledPoints[real_index].color = cur_point.color;
                }
                current_chunk = current_chunk->next;
            }

            globalAllocator.delChunk(spilling_node->points, true);
            spilling_node->points = nullptr;
        }

    }
}



/// Run on "MaxActiveBlocksPerMultiprocessor" cooperative blocks of size "Max block size"
/// Update the nodes counters
extern "C" __global__
void kernel_simlod_count_split(){
    if(!globalVariables.isInitialised){return;}

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    uint32_t first_point = block_id * nb_threads_per_block + thread_id;
    uint32_t step = nb_blocks * nb_threads_per_block;

    while(true){

        simlodCount(first_point, step);
        grid.sync();

        if(globalVariables.nbSpillingNodes == 0){break;}
        simlodSplit(first_point, step);
        grid.sync();


        if(block_id == 0 && thread_id == 0){
            // Because "delChunk" was called in simlodSplit
            globalAllocator.chunksAllocator->reset_temporary_deallocations();
        }
        // try to avoid being on the same warp
        if((nb_blocks >= 2 && (block_id == 1 && thread_id == 0)) || (nb_blocks < 2 && (block_id == 0 && thread_id == 0))){
            // Because "newOctreeNode" was called in simlodSplit
            globalAllocator.nodesAllocator->reset_temporary_allocations();
        }
        // try to avoid being on the same warp
        if((nb_blocks >= 3 && (block_id == 2 && thread_id == 0)) || (nb_blocks < 3 && (block_id == 0 && thread_id == 0))){
            // Because "newOccupancyGrid" was called in simlodSplit
            globalAllocator.gridsAllocator->reset_temporary_allocations();
        }

        globalVariables.nbSpillingNodes = 0;
        grid.sync();
    }
}






















__device__
void sampleVoxel(const CPoint& point, uint32_t thread_id, uint32_t nb_threads){
    COctreeNode* cur_node = globalVariables.mainOctree;
    
    uint32_t level = 0;

    while(true){
        if(!cur_node->occupancy){return;}

        // Find next child
        const CAABB& aabb = cur_node->aabb;
        CNodePosition child_position = aabb.getNextChildIndex(point.position);
        if(!cur_node->children[child_position]){return;}

        if(level % nb_threads == thread_id){
            // Sample voxel occupancy grid at this location if the node is inner for this point
            COccupancyGrid::GridIndex grid_index = COccupancyGrid::getCellIndices(aabb, point);
            bool is_cell_occupied = cur_node->occupancy->markCellAsFilled(grid_index);

            if(!is_cell_occupied){
                // Create corresponding voxel using this point
                vec3 voxel_centroid = COccupancyGrid::getCellCentroid(aabb, grid_index);
                CPoint new_voxel = {};
                new_voxel.position = voxel_centroid;
                new_voxel.color = point.color;

                // Add voxel to backlog buffers
                __nv_atomic_add(&cur_node->voxels_counter, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                uint32_t backlog_index = __nv_atomic_fetch_add(&globalVariables.nbBacklogVoxels, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                
                // TODO: find better checks everywhere
                if(backlog_index >= globalVariables.maxNbBacklogVoxels){
                    printf("ERROR: Max nb backlog buffer reached: i = %d / %d\n", backlog_index, globalVariables.maxNbBacklogVoxels);
                    customAssert();
                }

                globalVariables.backlogVoxels[backlog_index] = new_voxel;
                globalVariables.backlogVoxelsNodes[backlog_index] = cur_node;
            }
        }

        cur_node = cur_node->children[child_position];
        level++;
    }
};



/// Run on "Max threads per SM" blocks of size "NB SMs"
/// Each thread in a block is responsible for its assigned levels in the tree
extern "C" __global__
void kernel_simlod_voxel_sampling(){
    if(!globalVariables.isInitialised){return;}

    // Because "kernel_fill_new_grids" should be launched just before
    globalVariables.nbGridsToInit = 0;

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    
    // Sample voxels for new points
    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

        for(uint32_t i=block_id; i<nb_new_points; i+=nb_blocks){
            CPoint& point = new_points[i];
            sampleVoxel(point, thread_id, nb_threads_per_block);
        }
    }

    // Sample voxels for spilled points
    for(uint32_t i=block_id; i<globalVariables.nbSpilledPoints; i+=nb_blocks){
        CPoint& point = globalVariables.spilledPoints[i];
        sampleVoxel(point, thread_id, nb_threads_per_block);
    }
}





















__device__
void allocateChunks(CChunk* root_chunk, uint32_t required_chunks, uint32_t total_counter){
    CChunk* cur_chunk = root_chunk;
    for(uint32_t i=1; i<required_chunks; i++){
        if(!cur_chunk->next){cur_chunk->next = globalAllocator.newChunk(true);}
        cur_chunk->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;
        cur_chunk = cur_chunk->next;
    }
    if(cur_chunk){cur_chunk->size = total_counter % OocSimLodSettings::NB_POINTS_PER_CHUNK;}
};




/// Run on "NB SMs" * "Max threads per SM" blocks of size 1
/// Allocate the necessary new chunks
extern "C" __global__
void kernel_simlod_insertion_part_1_chunks_allocations(){
    if(!globalVariables.isInitialised){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    for(uint32_t node_index = thread_id; node_index < globalVariables.curNbNodes; node_index += nb_threads){
        COctreeNode* cur_node = globalVariables.packedNodes[node_index];

        // Allocate points chunks
        uint32_t required_chunks = (cur_node->points_counter + OocSimLodSettings::NB_POINTS_PER_CHUNK - 1) /  OocSimLodSettings::NB_POINTS_PER_CHUNK;
        if(required_chunks > 0){
            if(!cur_node->points){
                cur_node->points = globalAllocator.newChunk(true);
            }
            allocateChunks(cur_node->points, required_chunks, cur_node->points_counter);
        }

        // Allocate voxels chunks
        required_chunks = (cur_node->voxels_counter + OocSimLodSettings::NB_POINTS_PER_CHUNK - 1) /  OocSimLodSettings::NB_POINTS_PER_CHUNK;
        if(required_chunks > 0){
            if(!cur_node->voxels){
                cur_node->voxels = globalAllocator.newChunk(true);
            }
            allocateChunks(cur_node->voxels, required_chunks, cur_node->voxels_counter);
        }
    }

    // Do not reset them, they contain the real number of elements in each node
    // cur_node->points_stored = 0;
    // cur_node->voxels_stored = 0;
}



__device__
void insertPoint(const CPoint& point){
    COctreeNode* cur_node = globalVariables.mainOctree;
    // Reach all corresponding leaves
    while(cur_node){
        globalVariables.setFlagSync(cur_node->aabb_index, CFlagIsUpdated);
        // Find next child
        const CAABB& aabb = cur_node->aabb;
        CNodePosition child_position = aabb.getNextChildIndex(point.position);

        // If leaf insert point in chunks
        if(cur_node->children[child_position]){
            cur_node = cur_node->children[child_position];
        } else {
            globalVariables.setFlagSync(cur_node->aabb_index, CFlagHasNewPoints);

            uint32_t point_index = __nv_atomic_fetch_add(&cur_node->points_stored, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            uint32_t chunk_index = point_index / OocSimLodSettings::NB_POINTS_PER_CHUNK;
            CChunk* cur_chunk = cur_node->points;
            for(uint32_t i=0; i<chunk_index; i++){
                if(!cur_chunk){
                    printf("ERROR: the points chunk should have been allocated in part 1\n");
                    customAssert();
                }
                cur_chunk = cur_chunk->next;
            }
            uint32_t real_index = point_index % OocSimLodSettings::NB_POINTS_PER_CHUNK;
            cur_chunk->points[real_index] = point;
            return;
        }
    }
};

__device__
void insertVoxel(const CPoint& voxel, COctreeNode* cur_node){
    globalVariables.setFlagSync(cur_node->aabb_index, CFlagIsUpdated);
    globalVariables.setFlagSync(cur_node->aabb_index, CFlagHasNewVoxels);

    uint32_t voxel_index = __nv_atomic_fetch_add(&cur_node->voxels_stored, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    uint32_t chunk_index = voxel_index / OocSimLodSettings::NB_POINTS_PER_CHUNK;

    CChunk* cur_chunk = cur_node->voxels;
    for(uint32_t i=0; i<chunk_index; i++){
        if(!cur_chunk){
            printf("ERROR: the voxels chunk should have been allocated in part 1\n");
            customAssert();
        }
        cur_chunk = cur_chunk->next;
    }
    uint32_t real_index = voxel_index % OocSimLodSettings::NB_POINTS_PER_CHUNK;
    cur_chunk->points[real_index] = voxel;
};




/// Run on "NB SMs" * "Max threads per SM" blocks of size 1
extern "C" __global__
void kernel_simlod_insertion_part_2_filling(){
    if(!globalVariables.isInitialised){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    // Insert new points
    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

        for(uint32_t i=thread_id; i<nb_new_points; i+=nb_threads){
            CPoint& point = new_points[i];
            insertPoint(point);
        }
    }

    // Insert spilled points
    for(uint32_t i=thread_id; i<globalVariables.nbSpilledPoints; i+=nb_threads){
        CPoint& point = globalVariables.spilledPoints[i];
        insertPoint(point);
    }

    // Insert new voxels
    for(uint32_t i=thread_id; i<globalVariables.nbBacklogVoxels; i+=nb_threads){
        CPoint& voxel = globalVariables.backlogVoxels[i];
        COctreeNode* node = globalVariables.backlogVoxelsNodes[i];
        insertVoxel(voxel, node);
    }


    if(thread_id == 0){
        // Because "newChunk" was called in part 1
        globalAllocator.chunksAllocator->reset_temporary_allocations();
    }
}
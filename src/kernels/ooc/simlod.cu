#include "utils.cuh"

/// Prepare the nodes that need to be loaded
/// Run on "maxPointsPerBatches" threads
/// TODO:
/// Later, if a point is contained in a leaf that was previously stored, put the point aside
/// For now, load them directly after this kernel launch
extern "C" __global__
void kernel_simlod_load_part_1(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

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
            while(true){
                const CAABB& aabb = getAABB(leaf->aabb_index);
                CNodePosition child_position = aabb.getNextChildIndex(point.position);

                if(leaf->children[child_position]){
                    leaf = leaf->children[child_position];
                } else {
                    break;
                }
            }

            // Check if the remaining children are already stored
            uint32_t cur_aabb_index = leaf->aabb_index;
            while(true){
                const CAABB& aabb = getAABB(cur_aabb_index);
                CNodePosition child_position = aabb.getNextChildIndex(point.position);
                CIdAABB child_aabb_index = globalVariables.relationshipMap[cur_aabb_index].children[child_position];
                if(child_aabb_index != CINVALID_ID){
                    globalVariables.nodesFlags[child_aabb_index] |= (1u << CFlagToLoad);
                    cur_aabb_index = child_aabb_index;
                } else {
                    break;
                }
            }
        }
    }

    // Reset previous simlod values
    globalVariables.nbNodesReceived = 0;
    globalVariables.nbNodesToLoad = 0;
    globalVariables.nbNodesToStore = 0;
    globalVariables.nbSpilledPoints = 0;
    globalVariables.nbSpillingNodes = 0;
    globalVariables.nbBacklogVoxels = 0;
}


/// Run on "maxNbAABBs" threads
extern "C" __global__
void kernel_simlod_load_part_2(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
	if(thread_id >= globalVariables.maxNbAABBs){return;}

    // Count nodes to load
    if(globalVariables.nodesFlags[thread_id] & (1u << CFlagToLoad)){
        uint32_t index = __nv_atomic_fetch_add(&globalVariables.nbNodesToLoad, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        globalVariables.nodesToLoadBuffer[index] = (CIdAABB)thread_id;
        globalVariables.nodesFlags[thread_id] &= ~(1u << CFlagToLoad);
    }

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


/// Run on "nbNodesReceived" threads
/// Load the newly received nodes
extern "C" __global__
void kernel_simlod_load_part_3(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    uint32_t thread_id = cg::this_grid().thread_rank();
	if(thread_id >= globalVariables.nbNodesReceived){return;}

    CIdAABB aabb_index = globalVariables.receivedAABBIndices[thread_id];
    uint32_t children_ids = globalVariables.receivedChildrenIds[thread_id];
    uint32_t nb_points = globalVariables.receivedPointsCounters[thread_id];
    uint32_t nb_voxels = globalVariables.receivedVoxelsCounters[thread_id];
    CPoint* points = globalVariables.receivedPoints[thread_id];
    CPoint* voxels = globalVariables.receivedVoxels[thread_id];

    // vec3 first_point = nb_points ? points[0].position : vec3(0,0,0);
    // vec3 first_voxel = nb_voxels ? voxels[0].position : vec3(0,0,0);
    // CAABB aabb = getAABB(aabb_index);

    // printf("DEVICE side %d / %d:\n    first point = (%f, %f, %f), first voxel = (%f, %f, %f)\n    id: %d, children_ids: %d, points_counter: %d, voxels_counter: %d\n    aabb = {.mins(%f, %f, %f), .maxs(%f, %f, %f)}\n",
    //     thread_id+1, globalVariables.nbNodesReceived,
    //     first_point.x, first_point.y, first_point.z,
    //     first_voxel.x, first_voxel.y, first_voxel.z,
    //     aabb_index, children_ids, nb_points, nb_voxels,
    //     aabb.mins.x, aabb.mins.y, aabb.mins.z,
    //     aabb.maxs.x, aabb.maxs.y, aabb.maxs.z
    // );

    COctreeNode* loaded_node = globalAllocator.newOctreeNode(aabb_index, true);
    if(globalVariables.nodes[aabb_index]){
        printf("ERROR: at this point the node should not exist\n");
        customAssert();
    }
    globalVariables.nodes[aabb_index] = loaded_node;
    loaded_node->children_ids = children_ids;
    loaded_node->points_counter = nb_points;
    loaded_node->voxels_counter = nb_voxels;
    loaded_node->points_stored = nb_points;
    loaded_node->voxels_stored = nb_voxels;

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

        loaded_node->voxels = globalAllocator.newChunk(true);
        CChunk* cur_chunk = loaded_node->voxels;
        for(uint32_t i=0; i<nb_voxels; i++){
            const CPoint& cur_point = voxels[i];
            cur_chunk = addPointToChunk(cur_chunk, cur_point);

            // Sample voxel occupancy grid at this location
            const CAABB& aabb = getAABB(aabb_index);
            COccupancyGrid::GridIndex index = COccupancyGrid::getCellIndices(aabb, voxels[i]);
            loaded_node->occupancy->markCellAsFilled(index);
        }
    }

}

/// Run on "maxNbAABBs" threads
/// Rebuild the relationships of the loaded nodes
extern "C" __global__
void kernel_simlod_load_part_4(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();
	if(thread_id >= globalVariables.maxNbAABBs){return;}

    COctreeNode* cur_node = globalVariables.nodes[thread_id];
    if(!cur_node){return;}

    for(uint32_t i=0; i<8; i++){
        if(cur_node->children[i]){continue;}
        CIdAABB child_aabb = globalVariables.relationshipMap[cur_node->aabb_index].children[i];
        if(child_aabb != CINVALID_ID && globalVariables.nodes[child_aabb]){
            // printf("Does this happen ?\n");
            cur_node->children[i] = globalVariables.nodes[child_aabb];
        }
    }

    if(thread_id==0){
        // Because "newChunk" was called in part 3
        globalAllocator.chunksAllocator->reset_temporary_allocations();
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
        const CAABB& aabb = getAABB(leaf->aabb_index);
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
void simlodCount(uint32_t thread_id, uint32_t nb_threads){
    // Count new points
    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

        for(uint32_t i=thread_id; i<nb_new_points; i+=nb_threads){
            updateLeafCounter(&new_points[i]);

            if(i==0){
                printf("Device side: batch = %d / %d, count = %d, first point = (%f, %f, %f), nb spilled points: %d, nb_threads: %d\n", 
                    batch, globalVariables.maxNbBatches, nb_new_points, 
                    new_points[i].position.x,
                    new_points[i].position.y,
                    new_points[i].position.z,
                    globalVariables.nbSpilledPoints, nb_threads
                );
            }
        }
    }

    // Count spilled points
    for(uint32_t i=thread_id; i<globalVariables.nbSpilledPoints; i+=nb_threads){
        updateLeafCounter(&globalVariables.spilledPoints[i]);
    }
}


__device__
void simlodSplit(uint32_t thread_id, uint32_t nb_threads){
    for(uint32_t i=thread_id; i<globalVariables.nbSpillingNodes; i+=nb_threads){
        COctreeNode* spilling_node = globalVariables.spillingNodes[i];
        CIdAABB spilling_node_id = spilling_node->aabb_index;
        uint32_t spilling_node_children = spilling_node->children_ids;

        spilling_node->points_counter = 0;
        spilling_node->points_stored = 0; // also reset the previous counter

        spilling_node->children_ids = 0;

        if(!spilling_node->occupancy){
            spilling_node->occupancy = globalAllocator.newOccupancyGrid(true);
        }

        for(uint32_t j=0; j<8; j++){
            // Create necessary empty children
            bool can_be_spilled = (1u << j) & spilling_node_children;
            if(!spilling_node->children[j] && can_be_spilled){
                // Create the new AABB
                CAABB child_aabb = CAABB(getAABB(spilling_node_id));
                child_aabb.shrink((CNodePosition)j);
                
                CIdAABB new_child_id = createNewAABB(child_aabb);
                COctreeNode* new_child = globalAllocator.newOctreeNode(new_child_id, true);
                spilling_node->children[j] = new_child;
                globalVariables.nodes[new_child_id] = new_child;
                globalVariables.relationshipMap[spilling_node_id].children[j] = new_child_id;
            }
        }

        // Add former points to spilled points and free memory
        CChunk* current_chunk = spilling_node->points;
        if(current_chunk){
            while(current_chunk){
                for(uint32_t j=0; j<current_chunk->size; j++){
                    uint32_t index = __nv_atomic_fetch_add(&globalVariables.nbSpilledPoints, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                    if(index >= globalVariables.maxNbSpilledPoints){
                        printf("ERROR: reached the maximum number of spilled points\n");
                        customAssert();
                    }
                    // Flag the point as not accepted
                    CPoint cur_point = current_chunk->points[j];
                    cur_point.resetAlpha();
                    globalVariables.spilledPoints[index].position = cur_point.position;
                    globalVariables.spilledPoints[index].color = cur_point.color;
                }
                current_chunk = current_chunk->next;
            }

            globalAllocator.delChunk(spilling_node->points, true);
            spilling_node->points = nullptr;
        }

    }
}



/// Run on "total nb SMs" * "nb block per SM" threads
/// Run on a grid of size "total nb SMs" * "nb block per SM" and blocks of size 1
/// Update the nodes counters
extern "C" __global__
void kernel_simlod_count_split(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    uint32_t cpt = 0;
    while(true){
        cpt++;

        simlodCount(thread_id, nb_threads);
        grid.sync();
        if(globalVariables.nbSpillingNodes == 0){break;}

        // TODO: temporary to remove
        {
            if(thread_id == 0){
                printf("\n\n\n\n\nSimlod count: it %d\n\n", cpt);
                displayOctreeIt(globalVariables.mainOctree);
                printf("\n\n");
            }
            grid.sync();
        }


        simlodSplit(thread_id, nb_threads);
        grid.sync();
        globalVariables.nbSpillingNodes = 0;

        // TODO: temporary to remove
        {
            if(thread_id == 0){
                printf("\n\n\n\n\nSimlod split: it %d\n\n", cpt);
                displayOctreeIt(globalVariables.mainOctree);
                printf("\n\n");
            }
            grid.sync();
        }

        if(thread_id == 0){
            // Because "delChunk" was called in simlodSplit
            globalAllocator.chunksAllocator->reset_temporary_deallocations();
        }
        // try to avoid being on the same warp
        if((nb_threads >= 32 && thread_id==32) || (nb_threads < 32 && thread_id==0)){
            // Because "newOctreeNode" was called in simlodSplit
            globalAllocator.nodesAllocator->reset_temporary_allocations();
        }
        // try to avoid being on the same warp
        if((nb_threads >= 64 && thread_id==64) || (nb_threads < 64 && thread_id==0)){
            // Because "newOccupancyGrid" was called in simlodSplit
            globalAllocator.gridsAllocator->reset_temporary_allocations();
        }

        grid.sync();
    }
}






















__device__
void sampleVoxel(const CPoint& point){
    COctreeNode* cur_node = globalVariables.mainOctree;

    while(true){
        if(!cur_node->occupancy){return;}

        // Find next child
        const CAABB& aabb = getAABB(cur_node->aabb_index);
        CNodePosition child_position = aabb.getNextChildIndex(point.position);
        if(!cur_node->children[child_position]){return;}

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
            uint32_t index = __nv_atomic_fetch_add(&globalVariables.nbBacklogVoxels, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            
            // TODO: find better checks everywhere
            if(index >= globalVariables.maxNbBacklogVoxels){
                printf("ERROR: Max nb backlog buffer reached: i = %d / %d\n", index, globalVariables.maxNbBacklogVoxels);
                customAssert();
            }

            globalVariables.backlogVoxels[index] = new_voxel;
            globalVariables.backlogVoxelsNodes[index] = cur_node;
        }

        cur_node = cur_node->children[child_position];
    }
};



/// Run on "maxPointsPerBatches" threads
extern "C" __global__
void kernel_simlod_voxel_sampling(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    

    // Sample voxels for new points
    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

        for(uint32_t i=thread_id; i<nb_new_points; i+=nb_threads){
            CPoint& point = new_points[i];
            sampleVoxel(point);
        }
    }

    // Sample voxels for spilled points
    for(uint32_t i=thread_id; i<globalVariables.nbSpilledPoints; i+=nb_threads){
        CPoint& point = globalVariables.spilledPoints[i];
        sampleVoxel(point);
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

/// Run on "maxNbAABBs" threads
/// Allocate the necessary new chunks
extern "C" __global__
void kernel_simlod_insertion_part_1(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();
	if(thread_id >= globalVariables.maxNbAABBs){return;}

    COctreeNode* cur_node = globalVariables.nodes[thread_id];
    if(!cur_node){return;}

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

    // Do not reset them, they contain the real number of elements in each node
    // cur_node->points_stored = 0;
    // cur_node->voxels_stored = 0;
}



__device__
void insertPoint(const CPoint& point){
    COctreeNode* cur_node = globalVariables.mainOctree;
    // Reach all corresponding leaves
    while(cur_node){
        cur_node->updated = true;
        // Find next child
        const CAABB& aabb = getAABB(cur_node->aabb_index);
        CNodePosition child_position = aabb.getNextChildIndex(point.position);

        // If leaf insert point in chunks
        if(cur_node->children[child_position]){
            cur_node = cur_node->children[child_position];
        } else {
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
    cur_node->updated = true;

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


/// Run on "maxPointsPerBatches" threads
extern "C" __global__
void kernel_simlod_insertion_part_2(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

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
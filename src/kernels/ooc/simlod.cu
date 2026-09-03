#include "utils.cuh"



/// Prepare the nodes that need to be loaded
/// Run on floor("NB SMs" * "Max threads per SM" / "Max threads per block") blocks of size "Max threads per block"
/// TODO:
/// Later, if a point is contained in a leaf that was previously stored, put the point aside
/// For now, load them directly after this kernel launch
extern "C" __global__
void kernel_simlod_load_part_1_flagging(){
    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    // if(thread_id == 0){
    //     printf("kernel_simlod_load_part_1_flagging\n");
    // }

    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

#ifdef ASSERT_ENABLED
        if(nb_new_points > globalVariables.maxBatchSize){
            printf("ERROR: On flagging, batch size exceeded the limit: %d / %d\n", 
                nb_new_points, globalVariables.maxBatchSize
            );
            customAssert();
        }
#endif

        for(uint32_t i=thread_id; i<nb_new_points; i+=nb_threads){
            const CPoint& point = new_points[i];

            // Find the leaf containing the current point
            COctreeNode* leaf = globalVariables.mainOctree;

            CNodePosition child_position = (CNodePosition)0;
            CAABB child_aabb = CAABB();

            bool should_skip = false;
            while(true){
                child_aabb = globalVariables.relationshipMap[leaf->aabb_index].aabb;
                if(!child_aabb.contains(point.position)){
                    new_points[i].position = vec3(0,0,0); // Flag the point as being weird
                    should_skip = true;
                    break;
                }
                child_position = child_aabb.getNextChildIndex(point.position);
                if(leaf->children[child_position]){
                    leaf = leaf->children[child_position];
                } else {
                    break;
                }
            }
            if(should_skip){continue;}

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
                            // Instead of panicking, warn everyone that you're not done
                            globalVariables.isDoneLoading = false;
                            // Reset the flag to avoid being stuck in the above check
                            globalVariables.unsetFlagSync(child_aabb_index, CFlagToLoad);
                        } else {
                            globalVariables.exchangedAABBIndices[exchanged_index] = child_aabb_index;

                            // Store the original parent in the buffer
                            // Use temporaryNodeBuffer as a temporary buffer 
                            globalVariables.temporaryNodeBuffer[exchanged_index] = leaf;

                            // UI values
                            __nv_atomic_add(&globalVariables.nbLoadedNodesThisUpdate, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                            __nv_atomic_add(&globalVariables.nbTotalLoadedNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                        }
                    }
                    
                    cur_aabb_index = child_aabb_index;
                    child_aabb = globalVariables.relationshipMap[child_aabb_index].aabb;
                    if(!child_aabb.contains(point.position)){
                        new_points[i].position = vec3(0,0,0); // Flag the point as being weird
                        break;
                    }
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
    globalVariables.chunksAllocatorCounter = 0;
}




/// Run on "maxNbNodesExchanged" blocks of size "Max threads per block"
/// Load the newly exchanged nodes
extern "C" __global__
void kernel_simlod_load_part_2_rebuilding_nodes(){
    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    // if(block_id == 0 && thread_id == 0){
    //     printf("kernel_simlod_load_part_2_rebuilding_nodes\n");
    // }

    globalVariables.nbNodesExchanged = min(globalVariables.nbNodesExchanged, globalVariables.maxNbNodesExchanged);
    if(thread_id == 0){
        globalVariables.nbNodesExchangedBeforeLoadComplete += globalVariables.nbNodesExchanged;
    }


    for(uint32_t exchanged_index = thread_id; exchanged_index < globalVariables.nbNodesExchanged; exchanged_index += nb_threads){

        CIdAABB aabb_index = globalVariables.exchangedAABBIndices[exchanged_index];
        uint32_t children_ids = globalVariables.exchangedChildrenIds[exchanged_index];
        uint32_t nb_points = globalVariables.exchangedPointsCounters[exchanged_index];
        uint32_t nb_voxels = globalVariables.exchangedVoxelsCounters[exchanged_index];

#ifdef ASSERT_ENABLED
        if(globalVariables.isInUpdatesCache(aabb_index)){
            printf("ERROR: the newly loaded point `%d' should not be contained in the cache; nbNodesExchanged = %d\n", aabb_index, globalVariables.nbNodesExchanged);
            customAssert();
        }
#endif

        COctreeNode* loaded_node = globalAllocator.newOctreeNode(aabb_index);
        uint32_t node_index = __nv_atomic_fetch_add(&globalVariables.curNbNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);

#ifdef ASSERT_ENABLED
        if(node_index >= globalVariables.maxNbConcurrentNodes){
            printf("ERROR: failed to rebuild a new node for `%d'; can't add more nodes to the octree\n", aabb_index);
            customAssert();
        }
#endif

        globalVariables.packedNodes[node_index] = loaded_node;
        loaded_node->cur_id = node_index;
        loaded_node->children_ids = children_ids;
        loaded_node->points_counter = nb_points;
        loaded_node->points_stored = 0;
        loaded_node->points_last_stored = nb_points;
        loaded_node->voxels_last_stored = nb_voxels;

        // Allocate the occupancy grid
        if(nb_voxels > 0){
            loaded_node->occupancy = globalAllocator.newOccupancyGrid();
            uint32_t grid_index = __nv_atomic_fetch_add(&globalVariables.nbGridsToInit, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            
#ifdef ASSERT_ENABLED
            if(grid_index >= globalVariables.maxNbConcurrentNodes){
                printf("ERROR: failed to rebuild a new node for `%d'; can't recreate more grids\n", aabb_index);
                customAssert();
            }
#endif

            globalVariables.gridsToInit[grid_index] = loaded_node;
            globalVariables.gridsToInitExchangedIndex[grid_index] = exchanged_index;
        }

 
        // Find parent if needed
        // Use temporaryNodeBuffer as a temporary buffer 
        COctreeNode* potential_parent = globalVariables.temporaryNodeBuffer[exchanged_index];
        
        if(potential_parent->aabb_index == globalVariables.relationshipMap[aabb_index].parent){
            bool found = false;
            for(uint32_t i=0; i<8; i++){
                if(globalVariables.relationshipMap[potential_parent->aabb_index].children[i] == aabb_index){

#ifdef ASSERT_ENABLED
                    if(potential_parent->children[i] != nullptr){
                        printf("ERROR: At this point, the child[%d] = %d of node %d should not exist: parent = %d, relationship parent = %d\n",
                            i, potential_parent->children[i]->aabb_index, aabb_index, potential_parent->aabb_index, 
                            globalVariables.relationshipMap[potential_parent->children[i]->aabb_index].parent
                        );
                        customAssert();
                    }
#endif

                    potential_parent->children[i] = loaded_node;
                    found = true;
                    break;
                }
            }

#ifdef ASSERT_ENABLED
            if(!found){
                printf("ERROR: The parent %d doesn't contain the wanted child %d; real children are [%d, %d, %d, %d, %d, %d, %d, %d], relationship children are [%d, %d, %d, %d, %d, %d, %d, %d]\n",
                    potential_parent->aabb_index, aabb_index,
                    potential_parent->children[0] ? potential_parent->children[0]->aabb_index : -1,
                    potential_parent->children[1] ? potential_parent->children[1]->aabb_index : -1,
                    potential_parent->children[2] ? potential_parent->children[2]->aabb_index : -1,
                    potential_parent->children[3] ? potential_parent->children[3]->aabb_index : -1,
                    potential_parent->children[4] ? potential_parent->children[4]->aabb_index : -1,
                    potential_parent->children[5] ? potential_parent->children[5]->aabb_index : -1,
                    potential_parent->children[6] ? potential_parent->children[6]->aabb_index : -1,
                    potential_parent->children[7] ? potential_parent->children[7]->aabb_index : -1,
                    globalVariables.relationshipMap[potential_parent->aabb_index].children[0],
                    globalVariables.relationshipMap[potential_parent->aabb_index].children[1],
                    globalVariables.relationshipMap[potential_parent->aabb_index].children[2],
                    globalVariables.relationshipMap[potential_parent->aabb_index].children[3],
                    globalVariables.relationshipMap[potential_parent->aabb_index].children[4],
                    globalVariables.relationshipMap[potential_parent->aabb_index].children[5],
                    globalVariables.relationshipMap[potential_parent->aabb_index].children[6],
                    globalVariables.relationshipMap[potential_parent->aabb_index].children[7]
                );
                customAssert();
            }
#endif

        }
    }
}


/// Run on "maxNbNodesExchanged" blocks of size "Max threads per block"
/// Rebuild the relationships of the loaded nodes
extern "C" __global__
void kernel_simlod_load_part_3_rebuilding_children(){
    // Because "kernel_fill_new_grids" should be launched just before
    globalVariables.nbGridsToInit = 0;

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    // if(thread_id == 0){
    //     printf("kernel_simlod_load_part_3_rebuilding_children\n");
    // }

    if(!globalVariables.isDoneLoading){return;}

    uint32_t first_node = globalVariables.curNbNodes - globalVariables.nbNodesExchangedBeforeLoadComplete;
    // if(thread_id == 0){
    //     printf("nb exchanged: %d\n", globalVariables.nbNodesExchangedBeforeLoadComplete);
    // }
    for(uint32_t node_index = first_node + thread_id; node_index < globalVariables.curNbNodes; node_index += nb_threads){
        COctreeNode* cur_node = globalVariables.packedNodes[node_index];
        uint32_t child_aabbs[8] = {
            CINVALID_ID, CINVALID_ID, CINVALID_ID, CINVALID_ID,
            CINVALID_ID, CINVALID_ID, CINVALID_ID, CINVALID_ID
        };
        uint32_t to_find = 0;
        for(uint32_t i=0; i<8; i++){

#ifdef ASSERT_ENABLED
            if(cur_node->children[i] && 
                (cur_node->aabb_index != globalVariables.relationshipMap[cur_node->children[i]->aabb_index].parent)
            ){
                printf("ERROR: a newly loaded node `%d' should not have children already initialised: children[%d] = %d; parent = %d\n", 
                    cur_node->aabb_index, i, cur_node->children[i]->aabb_index, globalVariables.relationshipMap[cur_node->children[i]->aabb_index].parent
                );
                customAssert();
            }
#endif

            child_aabbs[i] = globalVariables.relationshipMap[cur_node->aabb_index].children[i];
            if(child_aabbs[i] != CINVALID_ID){to_find++;}
        }

        uint32_t nb_found = 0;
        for(uint32_t i=0; i<globalVariables.curNbNodes; i++){
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
        globalVariables.unsetFlag(cur_node->aabb_index, CFlagToLoad);
    }
}



















__device__
void updateLeafCounter(
    bool force_count, 
    CPoint& point, 
    COctreeNode* root, 
    COctreeNode** memoisation_buffer, 
    uint32_t memoisation_index,
    bool from_spilled
){
    // Find the leaf containing the current point
    COctreeNode* leaf = root;
    bool moved = false;

    if(point.position == vec3(0,0,0)){
        // printf("WARN: weird point in count\n");
        memoisation_buffer[memoisation_index] = root;
        return;
    }

    uint32_t max_points_per_leaf = globalVariables.maxPointsPerLeaf;
    uint32_t max_spilling_nodes = globalVariables.maxNbSpilledPoints;

    while(true){
        CAABB aabb = globalVariables.relationshipMap[leaf->aabb_index].aabb;
        
        if(!aabb.contains(point.position)){
            memoisation_buffer[memoisation_index] = root;
            point.position = vec3(0,0,0); // Flag the point as being weird
            return;
        }
        
        globalVariables.setFlag(leaf->aabb_index, CFlagIsUpdated);
        CNodePosition child_position = aabb.getNextChildIndex(point.position);

        if(leaf->children[child_position]){
            leaf = leaf->children[child_position];
            moved = true;
        } else {

#ifdef ASSERT_ENABLED
            if(globalVariables.relationshipMap[leaf->aabb_index].children[child_position] != CINVALID_ID){
                printf("ERROR: on count; the node %d should not have an unloaded child[%d]; node %d should be loaded\n",
                    leaf->aabb_index, child_position, 
                    globalVariables.relationshipMap[leaf->aabb_index].children[child_position]
                );
                customAssert();
            }
#endif

            uint32_t mask = (1u << child_position);
            if(!(leaf->children_ids & mask)){
                __nv_atomic_or(&leaf->children_ids, mask, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            }

            // Skip if the point was already accepted at this level
            if(!moved && !force_count){
                return;
            }

            // Merge atomicAdds within warps to reduce contention
            uint64_t leafptr = uint64_t(leaf);
			auto warp = cg::coalesced_threads();
			auto group = cg::labeled_partition(warp, leafptr);

			if(group.thread_rank() == 0){
				uint32_t old_counter = __nv_atomic_fetch_add(&leaf->points_counter, group.num_threads(), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);

                if((old_counter <= max_points_per_leaf)
                    && ((old_counter + group.num_threads()) > max_points_per_leaf)
                ){ 
                    // Only added once with the above equality check
                    uint32_t spilling_node_index = __nv_atomic_fetch_add(&globalVariables.nbSpillingNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                    if(spilling_node_index >= max_spilling_nodes){
                        printf("ERROR: reached the maximum number of spilling nodes\n");
                        customAssert();
                    }
                    globalVariables.spillingNodes[spilling_node_index] = leaf;
                }
			}

            memoisation_buffer[memoisation_index] = leaf;
            return;
        }
    }
};




__device__
void simlodCount(uint32_t first_point, uint32_t step, uint32_t iteration, bool is_first_count_split){
    uint32_t max_nb_batches = globalVariables.maxNbBatches;
    uint32_t max_batch_size = globalVariables.maxBatchSize;
    uint32_t nb_spilled_points = globalVariables.nbSpilledPoints;
    COctreeNode** memoized_batch_points_nodes = globalVariables.memoizedBatchPointsNodes;
    COctreeNode** memoized_spilled_points_nodes = globalVariables.memoizedSpilledPointsNodes;
    COctreeNode* main_octree = globalVariables.mainOctree;

    // Count new points
    for(uint32_t batch = 0; batch < max_nb_batches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

#ifdef ASSERT_ENABLED
        if(nb_new_points > max_batch_size){
            printf("ERROR: On count, batch size exceeded the limit: %d / %d\n", 
                nb_new_points, max_batch_size
            );
            customAssert();
        }
#endif

        uint32_t start_index = batch * max_batch_size;
        for(uint32_t i = first_point; i < nb_new_points; i += step){
            bool is_first = (iteration == 0 && is_first_count_split);
            COctreeNode* root = is_first
                ? main_octree
                : memoized_batch_points_nodes[start_index + i]
            ;
            updateLeafCounter(
                is_first,
                new_points[i], 
                root, 
                memoized_batch_points_nodes, 
                start_index + i,
                false
            );
        }
    }

    
    // Count spilled points
    for(uint32_t i = first_point; i < nb_spilled_points; i += step){
        COctreeNode* root = memoized_spilled_points_nodes[i];
        updateLeafCounter(
            false,
            globalVariables.spilledPoints[i], 
            root,
            memoized_spilled_points_nodes, 
            i,
            true
        );
    }
}


/// Update the nodes counters
extern "C" __global__
void kernel_simlod_count_split_part_1_count(uint32_t loop, bool is_first_iteration){
    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    simlodCount(thread_id, nb_threads, loop, is_first_iteration);
}

/// Fillup the spilling nodes counter and creating children
extern "C" __global__
void kernel_simlod_count_split_part_1_5_prefix_sum(){
    uint32_t last = globalVariables.nbSpilledPoints;
    uint32_t nb_chunks = 0;

    for(uint32_t i=0; i<globalVariables.nbSpillingNodes; i++){
        COctreeNode* spilling_node = globalVariables.spillingNodes[i];

        CChunk* cur_chunk = spilling_node->points;
        while(cur_chunk){
            uint32_t new_value = last + cur_chunk->size;
            if(new_value >= globalVariables.maxNbSpilledPoints){
                printf("ERROR: reached the maximum of spilling points\n");
                customAssert();
            }
            globalVariables.spilledChunksCounter[nb_chunks] = last;
            globalVariables.spillingChunks[nb_chunks] = cur_chunk;
            last = new_value;
            nb_chunks++;
            cur_chunk = cur_chunk->next;
        }

        CIdAABB spilling_node_id = spilling_node->aabb_index;
        uint32_t spilling_node_children = spilling_node->children_ids;

        // UI values
        __nv_atomic_add(&globalVariables.nbSplitNodesThisUpdate, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        __nv_atomic_add(&globalVariables.nbTotalSplitNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);

        // Rebuild the occupancy
        if(!spilling_node->occupancy){
            spilling_node->occupancy = globalAllocator.newOccupancyGrid();
            uint32_t grid_index = __nv_atomic_fetch_add(&globalVariables.nbGridsToInit, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);

#ifdef ASSERT_ENABLED
            if(grid_index >= globalVariables.maxNbConcurrentNodes){
                printf("ERROR: failed to split the node %d; can't recreate more grids\n", spilling_node->aabb_index);
                customAssert();
            }
#endif

            globalVariables.gridsToInit[grid_index] = spilling_node;
            globalVariables.gridsToInitExchangedIndex[grid_index] = -1;
        }
        
        // Create the needed children
        for(uint32_t child_index = 0; child_index < 8; child_index++){
            // Create necessary empty children
            bool can_be_spilled = (1u << child_index) & spilling_node_children;
            if(can_be_spilled && (globalVariables.relationshipMap[spilling_node_id].children[child_index] == CINVALID_ID)){
                // Create the new node
                CIdAABB new_child_id = createNewNodeId();
                COctreeNode* new_child = globalAllocator.newOctreeNode(new_child_id);
                uint32_t node_index = __nv_atomic_fetch_add(&globalVariables.curNbNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);

#ifdef ASSERT_ENABLED
                if(node_index >= globalVariables.maxNbConcurrentNodes){
                    printf("ERROR: failed to split the node %d; can't add more nodes to the octree\n");
                    customAssert();
                }
#endif

                globalVariables.packedNodes[node_index] = new_child;
                new_child->cur_id = node_index;

                spilling_node->children[child_index] = new_child;
                globalVariables.relationshipMap[spilling_node_id].children[child_index] = new_child_id;
                globalVariables.relationshipMap[new_child_id].parent = spilling_node_id;

                // Create the new AABB
                globalVariables.relationshipMap[new_child_id].aabb = globalVariables.relationshipMap[spilling_node_id].aabb;
                globalVariables.relationshipMap[new_child_id].aabb.shrink((CNodePosition)child_index);
                new_child->level = spilling_node->level + 1;
            }
        }
    }

    globalVariables.nbSpilledPoints = last;
    globalVariables.nbSpillingChunks = nb_chunks;
}

/// Split the nodes
extern "C" __global__
void kernel_simlod_count_split_part_2_split(){
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    uint32_t nb_spilling_nodes = globalVariables.nbSpillingNodes;

    for(uint32_t i = block_id; i < globalVariables.nbSpillingChunks; i += nb_blocks){
        uint32_t spilled_index = globalVariables.spilledChunksCounter[i];
        CChunk* current_chunk = globalVariables.spillingChunks[i];
        CPoint* current_chunk_points = current_chunk->points;
        CPoint* spilled_points = globalVariables.spilledPoints;
        uint32_t size = current_chunk->size;

        // Find the corresponding spilling node
        uint32_t total_nb_chunks = 0;
        COctreeNode* cur_spilling_node = nullptr;
        for(uint32_t j=0; j<nb_spilling_nodes; j++){
            COctreeNode* spilling_node = globalVariables.spillingNodes[j];
            uint32_t nb_chunks_in_node = 
                (spilling_node->points_stored + OocSimLodSettings::NB_POINTS_PER_CHUNK - 1)
                    / OocSimLodSettings::NB_POINTS_PER_CHUNK
            ;
            total_nb_chunks += nb_chunks_in_node;
            if(total_nb_chunks > i){
                cur_spilling_node = spilling_node;
                break;
            }
        }

        // Add former points to spilled points and free memory
        for(uint32_t j = thread_id; j < size; j += nb_threads_per_block){
            uint32_t real_index = spilled_index + j;
            spilled_points[real_index] = current_chunk_points[j];
            globalVariables.memoizedSpilledPointsNodes[real_index] = cur_spilling_node;
        }
    }
}


extern "C" __global__
void kernel_simlod_count_split_part_3_chunks_delete(){

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    uint32_t nb_spilling_chunks = globalVariables.nbSpillingChunks;
    CChunk** spilling_chunks = globalVariables.spillingChunks;
    for(uint32_t i=thread_id; i < nb_spilling_chunks; i += nb_threads){
        globalAllocator.delChunkSingle(spilling_chunks[i]);
    }
}

// Run on a single thread
extern "C" __global__
void kernel_simlod_count_split_part_4_reset(){
    for(uint32_t i=0; i < globalVariables.nbSpillingNodes; i++){
        COctreeNode* spilling_node = globalVariables.spillingNodes[i];
        spilling_node->points = nullptr;
        // spilling_node->points_counter = 0;
        spilling_node->points_counter = spilling_node->points_last_stored;
        spilling_node->points_stored = 0; // also reset the previous counter
    }

    if(globalVariables.nbSpillingNodes == 0){
        globalVariables.isDoneIterating = true;
    }
    globalVariables.nbSpillingNodes = 0;
}




















__device__
void sampleVoxel(const CPoint& point){
    COctreeNode* cur_node = globalVariables.mainOctree;
    
    if(point.position == vec3(0,0,0)){
        return;
    }

    while(true){
        if(!cur_node->occupancy){return;}

        // Find next child
        const CAABB aabb = globalVariables.relationshipMap[cur_node->aabb_index].aabb;
        
#ifdef ASSERT_ENABLED
        if(!aabb.contains(point.position)){
            printf("ERROR: weird point in voxel sampling, at this stage, the point should have been flagged\n");
            customAssert();
        }
#endif

        CNodePosition child_position = aabb.getNextChildIndex(point.position);
        COctreeNode* next_child = cur_node->children[child_position];
        if(!next_child){return;}

        // Sample voxel occupancy grid at this location if the node is inner for this point
        COccupancyGrid::GridIndex grid_index = COccupancyGrid::getCellIndices(aabb, point);
        uint32_t* values = cur_node->occupancy->values;
        uint32_t bitmask = 1 << grid_index.bit;
        uint32_t old_nonatomic = values[grid_index.word];
        if((old_nonatomic & bitmask) == 0){

            uint32_t old_value = __nv_atomic_fetch_or(&values[grid_index.word], bitmask, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            bool is_cell_occupied = (old_value & bitmask);
            
            if(!is_cell_occupied){
                // Avoid uncoalsced atomic
                uint64_t nodeptr = uint64_t(cur_node);
                auto warp = cg::coalesced_threads();
                auto group = cg::labeled_partition(warp, nodeptr);

                uint32_t backlog_base = 0;
                if(group.thread_rank() == 0){
                    __nv_atomic_add(&cur_node->voxels_counter, group.num_threads(), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                    
                    backlog_base = __nv_atomic_fetch_add(&globalVariables.nbBacklogVoxels, group.num_threads(), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                    if(backlog_base + group.num_threads() > globalVariables.maxNbBacklogVoxels){
                        printf("ERROR: Max nb backlog buffer reached\n");
                        customAssert();
                    }
                }

                // Create corresponding voxel using this point
                vec3 voxel_centroid = COccupancyGrid::getCellCentroid(aabb, grid_index);
                CPoint new_voxel = {};
                new_voxel.position = voxel_centroid;
                new_voxel.color = point.color;

                backlog_base = group.shfl(backlog_base, 0);
                uint32_t backlog_index = backlog_base + group.thread_rank();

                globalVariables.backlogVoxels[backlog_index] = new_voxel;
                globalVariables.backlogVoxelsNodes[backlog_index] = cur_node;
            }
        }

        cur_node = next_child;
    }
};



/// Run on floor("NB SMs" * "Max threads per SM" / 256) blocks of size 256
extern "C" __global__
void kernel_simlod_voxel_sampling(){
    // To reset before "kernel_simlod_insertion_part_1_chunks_allocations"
    globalVariables.chunksAllocatorCounter = 0;

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    // if(thread_id == 0){
    //     printf("kernel_simlod_voxel_sampling\n");
    // }
    
    // Sample voxels for new points
    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

#ifdef ASSERT_ENABLED
        if(nb_new_points > globalVariables.maxBatchSize){
            printf("ERROR: On voxel sampling, batch size exceeded the limit: %d / %d\n", 
                nb_new_points, globalVariables.maxBatchSize
            );
            customAssert();
        }
#endif

        for(uint32_t i=thread_id; i<nb_new_points; i+=nb_threads){
            const CPoint& point = new_points[i];
            sampleVoxel(point);
        }
    }

    // Sample voxels for spilled points
    for(uint32_t i=thread_id; i<globalVariables.nbSpilledPoints; i+=nb_threads){
        const CPoint& point = globalVariables.spilledPoints[i];
        sampleVoxel(point);
    }
}












extern "C" __global__
void kernel_simlod_insertion_part_1_1_chunks_allocations_plan(){
    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    for(uint32_t node_index = thread_id; node_index < globalVariables.curNbNodes; node_index += nb_threads){
        COctreeNode* cur_node = globalVariables.packedNodes[node_index];
#ifdef ASSERT_ENABLED
        if(!cur_node){
            printf("ERROR: In allocation, the node should exist\n");
            customAssert();
        }
#endif

        CAllocationPlan plan = {
            .points_attach = nullptr,
            .points_first_new = 0,
            .points_nb_new = 0,
            .points_last_size = 0,
            .points_existing_count = 0,
            .voxels_attach = nullptr,
            .voxels_first_new = 0,
            .voxels_nb_new = 0,
            .voxels_last_size = 0,
            .voxels_existing_count = 0
        };

        // Points
        uint32_t real_points_counter = cur_node->points_counter - cur_node->points_last_stored;
        uint32_t required_chunks = (real_points_counter + OocSimLodSettings::NB_POINTS_PER_CHUNK - 1) / OocSimLodSettings::NB_POINTS_PER_CHUNK;
        if(required_chunks > 0){
            if(!cur_node->points){
                cur_node->points = globalAllocator.newChunk();
            }

            // Reserve the full range for this node upfront (existing + new)
            uint32_t first_slot = __nv_atomic_fetch_add(&globalVariables.chunksAllocatorCounter, required_chunks, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            plan.points_first_new = first_slot;  // base slot for this node's chunks

            CChunk* cur_chunk = cur_node->points;
            uint32_t existing_count = 0;
            while(cur_chunk){
                cur_chunk->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;
                globalVariables.allocatedChunks[first_slot + existing_count] = cur_chunk;
                existing_count++;
                if(!cur_chunk->next) break;
                cur_chunk = cur_chunk->next;
            }

            uint32_t last_size = real_points_counter % OocSimLodSettings::NB_POINTS_PER_CHUNK;
            last_size = (last_size == 0) ? OocSimLodSettings::NB_POINTS_PER_CHUNK : last_size;

            uint32_t nb_new_chunks = required_chunks - existing_count;
            if(nb_new_chunks > 0){
                cur_chunk->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;
                plan.points_attach = cur_chunk;
                plan.points_nb_new = nb_new_chunks;
                plan.points_last_size = last_size;
            } else {
                cur_chunk->size = last_size;
                plan.points_attach = nullptr;
                plan.points_nb_new = 0;
                plan.points_last_size = last_size;
            }
            plan.points_existing_count = existing_count;
        }

        // Voxels
        required_chunks = (cur_node->voxels_counter + OocSimLodSettings::NB_POINTS_PER_CHUNK - 1) / OocSimLodSettings::NB_POINTS_PER_CHUNK;
        if(required_chunks > 0){
            if(!cur_node->voxels){
                cur_node->voxels = globalAllocator.newChunk();
            }

            uint32_t first_slot = __nv_atomic_fetch_add(&globalVariables.chunksAllocatorCounter, required_chunks, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            plan.voxels_first_new = first_slot;

            CChunk* cur_chunk = cur_node->voxels;
            uint32_t existing_count = 0;
            while(cur_chunk){
                cur_chunk->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;
                globalVariables.allocatedChunks[first_slot + existing_count] = cur_chunk;
                existing_count++;
                if(!cur_chunk->next) break;
                cur_chunk = cur_chunk->next;
            }

            uint32_t last_size = cur_node->voxels_counter % OocSimLodSettings::NB_POINTS_PER_CHUNK;
            last_size = (last_size == 0) ? OocSimLodSettings::NB_POINTS_PER_CHUNK : last_size;

            uint32_t nb_new_chunks = required_chunks - existing_count;
            if(nb_new_chunks > 0){
                cur_chunk->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;
                plan.voxels_attach = cur_chunk;
                plan.voxels_nb_new = nb_new_chunks;
                plan.voxels_last_size = last_size;
            } else {
                cur_chunk->size = last_size;
                plan.voxels_attach = nullptr;
                plan.voxels_nb_new = 0;
                plan.voxels_last_size = last_size;
            }
            plan.voxels_existing_count = existing_count;
        }

        globalVariables.allocationPlans[node_index] = plan;
    }
}



extern "C" __global__
void kernel_simlod_insertion_part_1_2_chunks_allocations_alloc(){
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    for(uint32_t node_index = block_id; node_index < globalVariables.curNbNodes; node_index += nb_blocks){
        COctreeNode* cur_node = globalVariables.packedNodes[node_index];
        CAllocationPlan plan = globalVariables.allocationPlans[node_index];

        // Allocate the missing chunks in parallel
        for(uint32_t i = thread_id; i < plan.points_nb_new; i += nb_threads_per_block){
            uint32_t slot = plan.points_first_new + plan.points_existing_count + i;
            CChunk* new_chunk = globalAllocator.newChunk();
            new_chunk->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;
            globalVariables.allocatedChunks[slot] = new_chunk;
        }
        for(uint32_t i = thread_id; i < plan.voxels_nb_new; i += nb_threads_per_block){
            uint32_t slot = plan.voxels_first_new + plan.voxels_existing_count + i;
            CChunk* new_chunk = globalAllocator.newChunk();
            new_chunk->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;
            globalVariables.allocatedChunks[slot] = new_chunk;
        }
    }
}



extern "C" __global__
void kernel_simlod_insertion_part_1_3_chunks_allocations_link(){
    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    for(uint32_t node_index = thread_id; node_index < globalVariables.curNbNodes; node_index += nb_threads){
        COctreeNode* cur_node = globalVariables.packedNodes[node_index];
        const CAllocationPlan& plan = globalVariables.allocationPlans[node_index];

        // Attach the new range to the existing chain, and fix the last chunk's size
        if(plan.points_nb_new > 0){
            uint32_t new_base = plan.points_first_new + plan.points_existing_count;
            plan.points_attach->next = globalVariables.allocatedChunks[new_base];
            globalVariables.allocatedChunks[new_base + plan.points_nb_new - 1]->size = plan.points_last_size;
            for(uint32_t i = 0; i < plan.points_nb_new - 1; i++){
                globalVariables.allocatedChunks[new_base + i]->next = 
                    globalVariables.allocatedChunks[new_base + i + 1];
            }
        }
        if(plan.voxels_nb_new > 0){
            uint32_t new_base = plan.voxels_first_new + plan.voxels_existing_count;
            plan.voxels_attach->next = globalVariables.allocatedChunks[new_base];
            globalVariables.allocatedChunks[new_base + plan.voxels_nb_new - 1]->size = plan.voxels_last_size;
            for(uint32_t i = 0; i < plan.voxels_nb_new - 1; i++){
                globalVariables.allocatedChunks[new_base + i]->next = 
                    globalVariables.allocatedChunks[new_base + i + 1];
            }
        }
    }
}





__device__
void insertPoint(const CPoint& point, COctreeNode* cur_node){
    if(point.position == vec3(0,0,0)){
        return;
    }

    // Group threads writing to the same node
    uint64_t nodeptr = uint64_t(cur_node);
    auto warp = cg::coalesced_threads();
    auto group = cg::labeled_partition(warp, nodeptr);

    uint32_t base_index = 0;
    if(group.thread_rank() == 0){
        base_index = __nv_atomic_fetch_add(&cur_node->points_stored, group.num_threads(),
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    }

    uint32_t first_new = globalVariables.allocationPlans[cur_node->cur_id].points_first_new
    base_index = group.shfl(base_index, 0);
    uint32_t point_index = base_index + group.thread_rank();
            
#ifdef ASSERT_ENABLED
    uint32_t real_points_counter = (cur_node->points_counter - cur_node->points_last_stored);
    if((point_index+1) > real_points_counter){
        printf("ERROR: weird points inserted, skip for safety\n");
        customAssert();
    }
#endif

    uint32_t chunk_index = point_index / OocSimLodSettings::NB_POINTS_PER_CHUNK;
    CChunk* cur_chunk = globalVariables.allocatedChunks[first_new + chunk_index];

#ifdef ASSERT_ENABLED
    if(!cur_chunk){
        printf("ERROR: Failed to insert point in node %d; the first chunk should exist: point index = %d, points counter = %d, real points counter = %d\n", 
            cur_node->aabb_index, point_index, cur_node->points_counter, real_points_counter
        );
        customAssert();
    }
#endif

    uint32_t real_index = point_index % OocSimLodSettings::NB_POINTS_PER_CHUNK;
    cur_chunk->points[real_index] = point;

    return;
};

__device__
void insertVoxel(const CPoint& voxel, COctreeNode* cur_node){

#ifdef ASSERT_ENABLED
    if(!globalVariables.relationshipMap[cur_node->aabb_index].aabb.contains(voxel.position)){
        printf("ERROR: a voxel should always be contained in its node: node %d, voxel (%f, %f, %f), aabb = .mins(%f, %f, %f), .maxs(%f, %f, %f)\n",
            cur_node->aabb_index, voxel.position.x, voxel.position.y, voxel.position.z,
            globalVariables.relationshipMap[cur_node->aabb_index].aabb.mins.x,
            globalVariables.relationshipMap[cur_node->aabb_index].aabb.mins.y,
            globalVariables.relationshipMap[cur_node->aabb_index].aabb.mins.z,
            globalVariables.relationshipMap[cur_node->aabb_index].aabb.maxs.x,
            globalVariables.relationshipMap[cur_node->aabb_index].aabb.maxs.y,
            globalVariables.relationshipMap[cur_node->aabb_index].aabb.maxs.z
        );
        customAssert();
    }
#endif

    // Group threads writing to the same node
    uint64_t nodeptr = uint64_t(cur_node);
    auto warp = cg::coalesced_threads();
    auto group = cg::labeled_partition(warp, nodeptr);

    uint32_t base_index = 0;
    if(group.thread_rank() == 0){
        base_index = __nv_atomic_fetch_add(&cur_node->voxels_stored, group.num_threads(),
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    }

    uint32_t first_new = globalVariables.allocationPlans[cur_node->cur_id].voxels_first_new
    base_index = group.shfl(base_index, 0);
    uint32_t voxel_index = base_index + group.thread_rank();

    uint32_t chunk_index = voxel_index / OocSimLodSettings::NB_POINTS_PER_CHUNK;
    CChunk* cur_chunk = globalVariables.allocatedChunks[first_new + chunk_index];

#ifdef ASSERT_ENABLED
    if(!cur_chunk){
        printf("ERROR: Failed to insert voxel in node %d; the first chunk should exist: point index = %d, points counter = %d\n", 
            cur_node->aabb_index, voxel_index, cur_node->voxels_counter
        );
        customAssert();
    }
#endif

    uint32_t real_index = voxel_index % OocSimLodSettings::NB_POINTS_PER_CHUNK;
    cur_chunk->points[real_index] = voxel;
};




/// Run on floor("NB SMs" * "Max threads per SM" / "Max threads per block") blocks of size "Max threads per block"
extern "C" __global__
void kernel_simlod_insertion_part_2_filling(){
    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    COctreeNode** memoized_batch_points_nodes = globalVariables.memoizedBatchPointsNodes;
    COctreeNode** memoized_spilled_points_nodes = globalVariables.memoizedSpilledPointsNodes;

    // if(thread_id == 0){
    //     printf("kernel_simlod_insertion_part_2_filling\n");
    // }

    // Insert new points
    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

#ifdef ASSERT_ENABLED
        if(nb_new_points > globalVariables.maxBatchSize){
            printf("ERROR: On insertion, batch size exceeded the limit: %d / %d\n", 
                nb_new_points, globalVariables.maxBatchSize
            );
            customAssert();
        }
#endif

        uint32_t start_index = batch * globalVariables.maxBatchSize;
        for(uint32_t i=thread_id; i<nb_new_points; i+=nb_threads){
            const CPoint point = new_points[i];
            COctreeNode* cur_node = memoized_batch_points_nodes[start_index + i];
            insertPoint(point, cur_node);

            // UI values
            __nv_atomic_add(&globalVariables.nbNewPointsThisUpdate, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_add(&globalVariables.nbTotalPoints, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_add(&globalVariables.currentNbPoints, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        }
    }

    // Insert spilled points
    for(uint32_t i=thread_id; i<globalVariables.nbSpilledPoints; i+=nb_threads){
        const CPoint point = globalVariables.spilledPoints[i];
        COctreeNode* cur_node = memoized_spilled_points_nodes[i];
        insertPoint(point, cur_node);
    }

    // Insert new voxels
    for(uint32_t i=thread_id; i<globalVariables.nbBacklogVoxels; i+=nb_threads){
        const CPoint voxel = globalVariables.backlogVoxels[i];
        COctreeNode* node = globalVariables.backlogVoxelsNodes[i];
        insertVoxel(voxel, node);

        // UI values
        __nv_atomic_add(&globalVariables.nbNewVoxelsThisUpdate, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        __nv_atomic_add(&globalVariables.nbTotalVoxels, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        __nv_atomic_add(&globalVariables.currentNbVoxels, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    }
}
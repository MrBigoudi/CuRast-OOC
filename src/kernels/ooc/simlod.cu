#include "utils.cuh"


/// Run on a single thread
extern "C" __global__
void kernel_simlod_load_part_0_reset(){
    if(!globalVariables.isInitialised){return;}
    if(!globalVariables.isUpdating){return;}
    if(!globalVariables.isDoneStoring || !globalVariables.isDoneIterating){return;}

    if(globalVariables.isDoneLoading){
        globalVariables.nbNodesExchangedBeforeLoadComplete = 0;
    }
    globalVariables.isDoneLoading = true;
    // printf("kernel_simlod_load_part_0_reset\n");

    // // Sanity check
    // for(uint32_t i=0; i<globalVariables.curNbNodes; i++){
    //     COctreeNode* node = globalVariables.packedNodes[i];
    //     if(!node){continue;}
    //     if(node->points_counter != node->points_stored){
    //         printf("ERROR part 0: Wtf, got %d / %d for node %d\n",
    //             node->points_counter, node->points_stored, node->aabb_index
    //         );
    //     }
    //     for(uint32_t j=0; j<8; j++){
    //         if(node->children[j] && globalVariables.relationshipMap[node->children[j]->aabb_index].parent != node->aabb_index){
    //             printf("ERROR part 0: child[%d] of node %d has parent %d\n",
    //                 j, node->aabb_index, globalVariables.relationshipMap[node->children[j]->aabb_index].parent
    //             );
    //             customAssert();
    //         }
    //         if(node->children[j] && globalVariables.relationshipMap[node->aabb_index].children[j] != node->children[j]->aabb_index){
    //             printf("ERROR part 0: child[%d] = %d of node %d should be %d\n",
    //                 j, node->children[j]->aabb_index, node->aabb_index,
    //                 globalVariables.relationshipMap[node->aabb_index].children[j]
    //             );
    //             customAssert();
    //         }
    //     }
    // }
}

/// Prepare the nodes that need to be loaded
/// Run on floor("NB SMs" * "Max threads per SM" / "Max threads per block") blocks of size "Max threads per block"
/// TODO:
/// Later, if a point is contained in a leaf that was previously stored, put the point aside
/// For now, load them directly after this kernel launch
extern "C" __global__
void kernel_simlod_load_part_1_flagging(){
    if(!globalVariables.isInitialised){return;}
    if(!globalVariables.isUpdating){return;}
    if(!globalVariables.isDoneStoring || !globalVariables.isDoneIterating){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

        if(nb_new_points > globalVariables.maxBatchSize){
            printf("ERROR: On flagging, batch size exceeded the limit: %d / %d\n", 
                nb_new_points, globalVariables.maxBatchSize
            );
            customAssert();
        }

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

    // if(thread_id == 0){
    //     printf("kernel_simlod_load_part_1_flagging\n");
    // }
}


__device__
void allocateChunks(CChunk* root_chunk, uint32_t required_chunks, uint32_t total_counter){
    CChunk* cur_chunk = root_chunk;
    for(uint32_t i=1; i<required_chunks; i++){
        if(!cur_chunk->next){cur_chunk->next = globalAllocator.newChunk(true);}
        cur_chunk->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;
        cur_chunk = cur_chunk->next;
    }
    if(cur_chunk){
        uint32_t last_size = total_counter % OocSimLodSettings::NB_POINTS_PER_CHUNK;
        last_size = (last_size == 0) ? OocSimLodSettings::NB_POINTS_PER_CHUNK : last_size;
        cur_chunk->size = last_size;
    }
};


/// Run on "maxNbNodesExchanged" blocks of size "Max threads per block"
/// Load the newly exchanged nodes
extern "C" __global__
void kernel_simlod_load_part_2_rebuilding_nodes(){
    if(!globalVariables.isInitialised){return;}
    if(!globalVariables.isUpdating){return;}
    if(!globalVariables.isDoneStoring || !globalVariables.isDoneIterating){return;}

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    __shared__ COctreeNode* shLoadedNode;
    __shared__ uint32_t shFirstPointsChunks;
    __shared__ uint32_t shNbPointsChunks;
    __shared__ uint32_t shFirstVoxelsChunks;
    __shared__ uint32_t shNbVoxelsChunks;

    globalVariables.nbNodesExchanged = min(globalVariables.nbNodesExchanged, globalVariables.maxNbNodesExchanged);
    if(block_id == 0 && thread_id == 0){
        globalVariables.nbNodesExchangedBeforeLoadComplete += globalVariables.nbNodesExchanged;
    }


    for(uint32_t exchanged_index = block_id; exchanged_index < globalVariables.nbNodesExchanged; exchanged_index += nb_blocks){

        CIdAABB aabb_index = globalVariables.exchangedAABBIndices[exchanged_index];
        uint32_t children_ids = globalVariables.exchangedChildrenIds[exchanged_index];
        uint32_t nb_points = globalVariables.exchangedPointsCounters[exchanged_index];
        uint32_t nb_voxels = globalVariables.exchangedVoxelsCounters[exchanged_index];
        CPoint* points = globalVariables.exchangedPoints[exchanged_index];
        CPoint* voxels = globalVariables.exchangedVoxels[exchanged_index];

        if(globalVariables.updatesCache->contains(aabb_index)){
            printf("ERROR: the newly loaded point `%d' should not be contained in the cache; nbNodesExchanged = %d\n", aabb_index, globalVariables.nbNodesExchanged);
            customAssert();
        }

        // Allocate the chunks
        __syncthreads(); // Needed to not update shLoadedNode before all threads are done
        if(thread_id == 0){ // If first thread of block
            // UI values
            __nv_atomic_add(&globalVariables.currentNbPoints, nb_points, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_add(&globalVariables.currentNbVoxels, nb_voxels, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            
            // TODO: parallelise
            shLoadedNode = globalAllocator.newOctreeNode(aabb_index, true);
            uint32_t node_index = __nv_atomic_fetch_add(&globalVariables.curNbNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);

            if(node_index >= globalVariables.maxNbConcurrentNodes){
                printf("ERROR: failed to rebuild a new node for `%d'; can't add more nodes to the octree\n", aabb_index);
                customAssert();
            }

            globalVariables.packedNodes[node_index] = shLoadedNode;

            shLoadedNode->children_ids = children_ids;
            shLoadedNode->points_counter = nb_points;
            shLoadedNode->voxels_counter = nb_voxels;
            shLoadedNode->points_stored = nb_points;
            shLoadedNode->voxels_stored = nb_voxels;

            shFirstPointsChunks = 0;
            shNbPointsChunks = 0;
            shFirstVoxelsChunks = 0;
            shNbVoxelsChunks = 0;

            // Allocate the occupancy grid
            if(nb_voxels > 0){
                shLoadedNode->occupancy = globalAllocator.newOccupancyGrid(true);
                uint32_t grid_index = __nv_atomic_fetch_add(&globalVariables.nbGridsToInit, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                
                if(grid_index >= globalVariables.maxNbConcurrentNodes){
                    printf("ERROR: failed to rebuild a new node for `%d'; can't recreate more grids\n", aabb_index);
                    customAssert();
                }

                globalVariables.gridsToInit[grid_index] = shLoadedNode;
            }

            // Allocate the first and last chunks for the points and the voxels
            if(nb_points > 0){
                shLoadedNode->points = globalAllocator.newChunk(true);
                uint32_t required_chunks = (nb_points + OocSimLodSettings::NB_POINTS_PER_CHUNK - 1) /  OocSimLodSettings::NB_POINTS_PER_CHUNK;
                uint32_t first_chunk = __nv_atomic_fetch_add(&globalVariables.chunksAllocatorCounter, required_chunks, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                uint32_t last_chunk = first_chunk + required_chunks - 1;
                if(last_chunk >= globalVariables.maxAllocatedChunks){
                    printf("ERROR: failed to rebuild a new node for `%d'; can't allocate more chunks for the points\n", aabb_index);
                    customAssert();
                }

                // Put first and last chunks in the chunk array
                globalVariables.allocatedChunks[first_chunk] = shLoadedNode->points;
                if(required_chunks > 1){
                    globalVariables.allocatedChunks[last_chunk] = globalAllocator.newChunk(true);
                }
                globalVariables.allocatedChunks[first_chunk]->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;
                uint32_t last_size = nb_points % OocSimLodSettings::NB_POINTS_PER_CHUNK;
                last_size = (last_size == 0) ? OocSimLodSettings::NB_POINTS_PER_CHUNK : last_size;
                globalVariables.allocatedChunks[last_chunk]->size = last_size;
               
                // Update shared variables
                shFirstPointsChunks = first_chunk;
                shNbPointsChunks = required_chunks;
            }
            if(nb_voxels > 0){
                shLoadedNode->voxels = globalAllocator.newChunk(true);
                uint32_t required_chunks = (nb_voxels + OocSimLodSettings::NB_POINTS_PER_CHUNK - 1) /  OocSimLodSettings::NB_POINTS_PER_CHUNK;
                uint32_t first_chunk = __nv_atomic_fetch_add(&globalVariables.chunksAllocatorCounter, required_chunks, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                uint32_t last_chunk = first_chunk + required_chunks - 1;
                if(last_chunk >= globalVariables.maxAllocatedChunks){
                    printf("ERROR: failed to rebuild a new node for `%d'; can't allocate more chunks for the voxels\n", aabb_index);
                    customAssert();
                }

                // Put first and last chunks in the chunk array
                globalVariables.allocatedChunks[first_chunk] = shLoadedNode->voxels;
                if(required_chunks > 1){
                    globalVariables.allocatedChunks[last_chunk] = globalAllocator.newChunk(true);
                }
                globalVariables.allocatedChunks[first_chunk]->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;
                uint32_t last_size = nb_voxels % OocSimLodSettings::NB_POINTS_PER_CHUNK;
                last_size = (last_size == 0) ? OocSimLodSettings::NB_POINTS_PER_CHUNK : last_size;
                globalVariables.allocatedChunks[last_chunk]->size = last_size;

                // Update shared variables
                shFirstVoxelsChunks = first_chunk;
                shNbVoxelsChunks = required_chunks;
            }
        }
        __syncthreads();


        // Allocate the chunks for the points and the voxels
        uint32_t first_points_chunk = shFirstPointsChunks;
        uint32_t nb_points_chunk = shNbPointsChunks;
        uint32_t first_voxels_chunk = shFirstVoxelsChunks;
        uint32_t nb_voxels_chunk = shNbVoxelsChunks;

        // -2 for the already two allocated chunks
        if(nb_points_chunk > 2){
            for(uint32_t allocation_id = thread_id; allocation_id < nb_points_chunk - 2; allocation_id += nb_threads_per_block){
                uint32_t real_id = first_points_chunk + allocation_id + 1;
                globalVariables.allocatedChunks[real_id] = globalAllocator.newChunk(true);
                globalVariables.allocatedChunks[real_id]->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;
            }
        }
        if(nb_voxels_chunk > 2){
            for(uint32_t allocation_id = thread_id; allocation_id < nb_voxels_chunk - 2; allocation_id += nb_threads_per_block){
                uint32_t real_id = first_voxels_chunk + allocation_id + 1;
                globalVariables.allocatedChunks[real_id] = globalAllocator.newChunk(true);
                globalVariables.allocatedChunks[real_id]->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;
            }
        }
        __syncthreads();


        // Link the chunks
        if(nb_points_chunk > 1){
            for(uint32_t allocation_id = thread_id; allocation_id < nb_points_chunk - 1; allocation_id += nb_threads_per_block){
                uint32_t real_id = first_points_chunk + allocation_id;
                globalVariables.allocatedChunks[real_id]->next = globalVariables.allocatedChunks[real_id+1];
            }
        }
        if(nb_voxels_chunk > 1){
            for(uint32_t allocation_id = thread_id; allocation_id < nb_voxels_chunk - 1; allocation_id += nb_threads_per_block){
                uint32_t real_id = first_voxels_chunk + allocation_id;
                globalVariables.allocatedChunks[real_id]->next = globalVariables.allocatedChunks[real_id+1];
            }
        }
        __syncthreads();


        // Fill up the chunks
        if(nb_points > 0){
            for(uint32_t i = thread_id; i < nb_points; i += nb_threads_per_block){
                const CPoint& cur_point = points[i];
                uint32_t chunk_index = i / OocSimLodSettings::NB_POINTS_PER_CHUNK;
                uint32_t point_index = i % OocSimLodSettings::NB_POINTS_PER_CHUNK;
                globalVariables.allocatedChunks[first_points_chunk + chunk_index]->points[point_index] = cur_point;
            }
        }

        // Rebuild voxels
        if(nb_voxels > 0){
            for(uint32_t i = thread_id; i < nb_voxels; i += nb_threads_per_block){
                const CPoint& cur_voxel = voxels[i];
                uint32_t chunk_index = i / OocSimLodSettings::NB_POINTS_PER_CHUNK;
                uint32_t point_index = i % OocSimLodSettings::NB_POINTS_PER_CHUNK;
                globalVariables.allocatedChunks[first_voxels_chunk + chunk_index]->points[point_index] = cur_voxel;
            }
        }


        if(thread_id == 0){
            // Find parent if needed
            // Use temporaryNodeBuffer as a temporary buffer 
            COctreeNode* potential_parent = globalVariables.temporaryNodeBuffer[exchanged_index];
            
            if(potential_parent->aabb_index == globalVariables.relationshipMap[aabb_index].parent){
                bool found = false;
                for(uint32_t i=0; i<8; i++){
                    if(globalVariables.relationshipMap[potential_parent->aabb_index].children[i] == aabb_index){
                        if(potential_parent->children[i] != nullptr){
                            printf("ERROR: At this point, the child[%d] = %d of node %d should not exist: parent = %d, relationship parent = %d\n",
                                i, potential_parent->children[i]->aabb_index, aabb_index, potential_parent->aabb_index, 
                                globalVariables.relationshipMap[potential_parent->children[i]->aabb_index].parent
                            );
                            customAssert();
                        }
                        potential_parent->children[i] = shLoadedNode;
                        found = true;
                        break;
                    }
                }
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
            }
        }
    }

    // if(block_id==0 && thread_id == 0){
    //     printf("kernel_simlod_load_part_2_rebuilding_nodes\n");
    // }
}


/// Run on "maxNbNodesExchanged" blocks of size "Max threads per block"
/// Rebuild the relationships of the loaded nodes
extern "C" __global__
void kernel_simlod_load_part_3_rebuilding_children(){
    if(!globalVariables.isInitialised){return;}
    if(!globalVariables.isUpdating){return;}
    if(!globalVariables.isDoneStoring || !globalVariables.isDoneIterating){return;}

    // Because "kernel_fill_new_grids" should be launched just before
    globalVariables.nbGridsToInit = 0;

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    if(block_id == 0 && thread_id == 0){
        // Because "newChunk" was called in part 3
        globalAllocator.chunksAllocator->reset_temporary_allocations();
        // Because "newOctreeNode" was called in part 3
        globalAllocator.nodesAllocator->reset_temporary_allocations();
        // Because "newOccupancyGrid" was called in part 3
        globalAllocator.gridsAllocator->reset_temporary_allocations();
    }

    if(!globalVariables.isDoneLoading){return;}

    uint32_t first_node = globalVariables.curNbNodes - globalVariables.nbNodesExchangedBeforeLoadComplete;
    // if(thread_id == 0 && block_id == 0){
    //     printf("nb exchanged: %d\n", globalVariables.nbNodesExchangedBeforeLoadComplete);
    // }
    for(uint32_t node_index = first_node + block_id; node_index < globalVariables.curNbNodes; node_index += nb_blocks){

        if(thread_id == 0){ // If first thread of block
            // TODO: parallelise
            COctreeNode* cur_node = globalVariables.packedNodes[node_index];

            uint32_t child_aabbs[8] = {
                CINVALID_ID, CINVALID_ID, CINVALID_ID, CINVALID_ID,
                CINVALID_ID, CINVALID_ID, CINVALID_ID, CINVALID_ID
            };
            uint32_t to_find = 0;
            for(uint32_t i=0; i<8; i++){
                if(cur_node->children[i] && 
                    (cur_node->aabb_index != globalVariables.relationshipMap[cur_node->children[i]->aabb_index].parent)
                ){
                    printf("ERROR: a newly loaded node `%d' should not have children already initialised: children[%d] = %d; parent = %d\n", 
                        cur_node->aabb_index, i, cur_node->children[i]->aabb_index, globalVariables.relationshipMap[cur_node->children[i]->aabb_index].parent
                    );
                    customAssert();
                }
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
            globalVariables.unsetFlagSync(cur_node->aabb_index, CFlagToLoad);
        }
    }

    // if(block_id == 0 && thread_id == 0){
    //     printf("kernel_simlod_load_part_3_rebuilding_children\n");
    //     // Sanity check
    //     printf("Sanity check\n");
    //     for(uint32_t i = 0; i < globalVariables.curNbNodes; i++){
    //         CIdAABB child_id = globalVariables.packedNodes[i]->aabb_index;
    //         CIdAABB parent_id = globalVariables.relationshipMap[child_id].parent;
    //         if(parent_id == CINVALID_ID){continue;}
    //         bool found = false;
    //         for(uint32_t j=0; j<globalVariables.curNbNodes; j++){
    //             CIdAABB tmp = globalVariables.packedNodes[j]->aabb_index;
    //             if(tmp == CINVALID_ID){
    //                 printf("WTFFFFF\n");
    //                 customAssert();
    //             }
    //             if(tmp == parent_id){
    //                 found = true;
    //                 break;
    //             }
    //         }
    //         if(!found){
    //             printf("ERROR: Can't find parent %d of child %d\n", parent_id, child_id);
    //             customAssert();
    //         }
    //     }
    // }
}



















__device__
void updateLeafCounter(
    bool force_count, 
    CPoint* point, 
    COctreeNode* root, 
    COctreeNode** memoisation_buffer, 
    uint32_t memoisation_index,
    bool from_spilled
){
    // Find the leaf containing the current point
    COctreeNode* leaf = root;
    bool moved = false;

    if(point->position == vec3(0,0,0)){
        // printf("WARN: weird point in count\n");
        memoisation_buffer[memoisation_index] = root;
        return;
    }

    while(true){
        const CAABB& aabb = globalVariables.relationshipMap[leaf->aabb_index].aabb;
        
        if(!aabb.contains(point->position)){
            memoisation_buffer[memoisation_index] = root;
            point->position = vec3(0,0,0); // Flag the point as being weird
            return;
        }
        
        CNodePosition child_position = aabb.getNextChildIndex(point->position);

        if(leaf->children[child_position]){
            leaf = leaf->children[child_position];
            moved = true;
        } else {
            if(globalVariables.relationshipMap[leaf->aabb_index].children[child_position] != CINVALID_ID){
                printf("ERROR: on count; the node %d should not have an unloaded child[%d]; node %d should be loaded\n",
                    leaf->aabb_index, child_position, 
                    globalVariables.relationshipMap[leaf->aabb_index].children[child_position]
                );
                customAssert();
            }

            // __nv_atomic_or(&leaf->children_ids, (1u << child_position), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            uint32_t mask = (1u << child_position);
            if(!(leaf->children_ids & mask)){
                __nv_atomic_or(&leaf->children_ids, mask, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            }


            // Skip if the point was already accepted at this level
            if(!moved && !force_count){
                return;
            }

            // // Flag the leaf as spilling
            // uint32_t old_counter = __nv_atomic_fetch_add(&leaf->points_counter, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            // if(old_counter == globalVariables.maxPointsPerLeaf){ 
            //     // Only added once with the above equality check
            //     uint32_t spilling_node_index = __nv_atomic_fetch_add(&globalVariables.nbSpillingNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            //     if(spilling_node_index >= globalVariables.maxNbSpilledPoints){
            //         printf("ERROR: reached the maximum number of spilling nodes\n");
            //         customAssert();
            //     }
            //     globalVariables.spillingNodes[spilling_node_index] = leaf;
            // }

            // Merge atomicAdds within warps to reduce contention
            uint64_t leafptr = uint64_t(leaf);
			auto warp = cg::coalesced_threads();
			auto group = cg::labeled_partition(warp, leafptr);

			if(group.thread_rank() == 0){
				uint32_t old_counter = __nv_atomic_fetch_add(&leaf->points_counter, group.num_threads(), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);

                if((old_counter <= globalVariables.maxPointsPerLeaf)
                    && ((old_counter + group.num_threads()) > globalVariables.maxPointsPerLeaf)
                ){ 
                    // Only added once with the above equality check
                    uint32_t spilling_node_index = __nv_atomic_fetch_add(&globalVariables.nbSpillingNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                    if(spilling_node_index >= globalVariables.maxNbSpilledPoints){
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

    // uint64_t t_start = nanotime();

    // Count new points
    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

        if(nb_new_points > globalVariables.maxBatchSize){
            printf("ERROR: On count, batch size exceeded the limit: %d / %d\n", 
                nb_new_points, globalVariables.maxBatchSize
            );
            customAssert();
        }

        uint32_t start_index = batch * globalVariables.maxBatchSize;
        for(uint32_t i = first_point; i < nb_new_points; i += step){
            bool is_first = (iteration == 0 && is_first_count_split);
            COctreeNode* root = is_first
                ? globalVariables.mainOctree
                : globalVariables.memoizedBatchPointsNodes[start_index + i]
            ;
            updateLeafCounter(
                is_first,
                &new_points[i], 
                root, 
                globalVariables.memoizedBatchPointsNodes, 
                start_index + i,
                false
            );
        }
    }

    
    // cg::this_grid().sync();
    
    // uint64_t nanos = nanotime() - t_start;
    // float micros = nanos / 1000;
    // if(cg::this_grid().thread_rank() == 0){
    //     printf("microseconds: %f \n", micros);
    // }

    // Count spilled points
    for(uint32_t i = first_point; i < globalVariables.nbSpilledPoints; i += step){
        COctreeNode* root = globalVariables.memoizedSpilledPointsNodes[i];
        updateLeafCounter(
            false,
            &globalVariables.spilledPoints[i], 
            root,
            globalVariables.memoizedSpilledPointsNodes, 
            i,
            true
        );
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

        // spilling_node->children_ids = 0;

        // UI values
        __nv_atomic_add(&globalVariables.nbSplitNodesThisUpdate, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        __nv_atomic_add(&globalVariables.nbTotalSplitNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        

        if(!spilling_node->occupancy){
            spilling_node->occupancy = globalAllocator.newOccupancyGrid(true);
            uint32_t grid_index = __nv_atomic_fetch_add(&globalVariables.nbGridsToInit, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);

            if(grid_index >= globalVariables.maxNbConcurrentNodes){
                printf("ERROR: failed to split the node %d; can't recreate more grids\n", spilling_node->aabb_index);
                customAssert();
            }

            globalVariables.gridsToInit[grid_index] = spilling_node;
        }

        for(uint32_t j=0; j<8; j++){
            // Create necessary empty children
            bool can_be_spilled = (1u << j) & spilling_node_children;
            if(can_be_spilled && (globalVariables.relationshipMap[spilling_node_id].children[j] == CINVALID_ID)){
                // if(globalVariables.relationshipMap[spilling_node_id].children[j] != CINVALID_ID){
                //     printf("ERROR: on split; the node %d should not have an unloaded child[%d]; node %d should be loaded\n",
                //         spilling_node_id, j, globalVariables.relationshipMap[spilling_node_id].children[j]
                //     );
                //     customAssert();
                // }
                
                // Create the new node
                CIdAABB new_child_id = createNewNodeId();
                COctreeNode* new_child = globalAllocator.newOctreeNode(new_child_id, true);
                uint32_t node_index = __nv_atomic_fetch_add(&globalVariables.curNbNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);

                if(node_index >= globalVariables.maxNbConcurrentNodes){
                    printf("ERROR: failed to split the node %d; can't add more nodes to the octree\n");
                    customAssert();
                }

                globalVariables.packedNodes[node_index] = new_child;

                spilling_node->children[j] = new_child;
                globalVariables.relationshipMap[spilling_node_id].children[j] = new_child_id;
                globalVariables.relationshipMap[new_child_id].parent = spilling_node_id;

                // Create the new AABB
                globalVariables.relationshipMap[new_child_id].aabb = globalVariables.relationshipMap[spilling_node_id].aabb;

                globalVariables.relationshipMap[new_child_id].aabb.shrink((CNodePosition)j);
                new_child->level = spilling_node->level + 1;
            }
        }

        // Add former points to spilled points and free memory
        CChunk* current_chunk = spilling_node->points;
        while(current_chunk){
            uint32_t spilled_index = __nv_atomic_fetch_add(&globalVariables.nbSpilledPoints, current_chunk->size, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            for(uint32_t j=0; j<current_chunk->size; j++){
                uint32_t real_index = spilled_index + j;
                if(real_index >= globalVariables.maxNbSpilledPoints){
                    printf("ERROR: reached the maximum number of spilled points\n");
                    customAssert();
                }

                CPoint cur_point = current_chunk->points[j];
                globalVariables.spilledPoints[real_index] = cur_point;
                globalVariables.memoizedSpilledPointsNodes[real_index] = spilling_node;
            }
            current_chunk = current_chunk->next;
        }

        globalAllocator.delChunk(spilling_node->points, true);
        spilling_node->points = nullptr;
    }
}



/// Run on "MaxActiveBlocksPerMultiprocessor" cooperative blocks of size "Max block size"
/// Update the nodes counters
extern "C" __global__
void kernel_simlod_count_split(){
    if(!globalVariables.isInitialised){return;}
    if(!globalVariables.isUpdating){return;}
    if(!globalVariables.isDoneLoading || !globalVariables.isDoneStoring){
        return;
    }

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    bool is_first_count_split = globalVariables.isDoneIterating;
    grid.sync();
    globalVariables.isDoneIterating = false;

    // while(true){
    for(uint32_t loop = 0; loop < globalVariables.maxCountSplitIterations; loop++){

        // Sanity check
        if(thread_id == 0){
            for(uint32_t i=0; i<globalVariables.nbSpillingNodes; i++){
                for(uint32_t j=i+1; j<globalVariables.nbSpillingNodes; j++){
                    if(globalVariables.spillingNodes[i]->aabb_index == globalVariables.spillingNodes[j]->aabb_index){
                        printf("ERROR: duplicate spilling node %d\n", globalVariables.spillingNodes[i]->aabb_index);
                        customAssert();
                    }
                }
            }
        }
        grid.sync();

        simlodCount(thread_id, nb_threads, loop, is_first_count_split);
        grid.sync();

        if(globalVariables.nbSpillingNodes == 0){
            globalVariables.isDoneIterating = true;
            break;
        }
        
        simlodSplit(thread_id, nb_threads);
        grid.sync();


        if(thread_id == 0){
            // Because "delChunk" was called in simlodSplit
            globalAllocator.chunksAllocator->reset_temporary_deallocations();
            // Because "newOctreeNode" was called in simlodSplit
            globalAllocator.nodesAllocator->reset_temporary_allocations();
            // Because "newOccupancyGrid" was called in simlodSplit
            globalAllocator.gridsAllocator->reset_temporary_allocations();
        }

        globalVariables.nbSpillingNodes = 0;
        grid.sync();
    }

    // if(thread_id == 0){
    //     if(!globalVariables.isDoneIterating){
    //         printf("DEBUG: Too many count/split iterations\n");
    //     } else {
    //         printf("DEBUG: count/split iterations done\n");
    //     }
    // }

    if(thread_id == 0){
        // Because "delChunk" was called in simlodSplit
        globalAllocator.chunksAllocator->reset_temporary_deallocations();
        // Because "newOctreeNode" was called in simlodSplit
        globalAllocator.nodesAllocator->reset_temporary_allocations();
        // Because "newOccupancyGrid" was called in simlodSplit
        globalAllocator.gridsAllocator->reset_temporary_allocations();
    }
}






















__device__
// void sampleVoxel(const CPoint& point, uint32_t thread_id, uint32_t nb_threads){
void sampleVoxel(const CPoint& point){
    COctreeNode* cur_node = globalVariables.mainOctree;
    
    uint32_t level = 0;

    if(point.position == vec3(0,0,0)){
        return;
    }

    while(true){
        if(!cur_node->occupancy){return;}

        // Find next child
        const CAABB& aabb = globalVariables.relationshipMap[cur_node->aabb_index].aabb;
        if(!aabb.contains(point.position)){
            printf("ERROR: weird point in voxel sampling, at this stage, the point should have been flagged\n");
            customAssert();
        }
        CNodePosition child_position = aabb.getNextChildIndex(point.position);
        if(!cur_node->children[child_position]){return;}

        // if(level % nb_threads == thread_id){
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
                
                if(backlog_index >= globalVariables.maxNbBacklogVoxels){
                    printf("ERROR: Max nb backlog buffer reached: i = %d / %d\n", backlog_index, globalVariables.maxNbBacklogVoxels);
                    customAssert();
                }

                globalVariables.backlogVoxels[backlog_index] = new_voxel;
                globalVariables.backlogVoxelsNodes[backlog_index] = cur_node;
            }
        // }

        cur_node = cur_node->children[child_position];
        level++;
    }
};



/// Run on floor("NB SMs" * "Max threads per SM" / 32) blocks of size 32
extern "C" __global__
void kernel_simlod_voxel_sampling(){
    if(!globalVariables.isInitialised){return;}
    if(!globalVariables.isUpdating){return;}
    if(!globalVariables.isDoneLoading || !globalVariables.isDoneStoring || !globalVariables.isDoneIterating){
        return;
    }

    // Because "kernel_fill_new_grids" should be launched just before
    globalVariables.nbGridsToInit = 0;

    // To reset before "kernel_simlod_insertion_part_1_chunks_allocations"
    globalVariables.chunksAllocatorCounter = 0;

    auto grid = cg::this_grid();
    // auto block = cg::this_thread_block();
    // uint32_t nb_blocks = grid.num_blocks();

    // uint32_t block_id = grid.block_rank();
    // uint32_t thread_id = block.thread_rank();
    // uint32_t nb_threads_per_block = block.num_threads();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    
    // Sample voxels for new points
    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

        if(nb_new_points > globalVariables.maxBatchSize){
            printf("ERROR: On voxel sampling, batch size exceeded the limit: %d / %d\n", 
                nb_new_points, globalVariables.maxBatchSize
            );
            customAssert();
        }

        // for(uint32_t i=block_id; i<nb_new_points; i+=nb_blocks){
        for(uint32_t i=thread_id; i<nb_new_points; i+=nb_threads){
            const CPoint& point = new_points[i];
            // sampleVoxel(point, thread_id, nb_threads_per_block);
            sampleVoxel(point);
        }
    }

    // Sample voxels for spilled points
    // for(uint32_t i=block_id; i<globalVariables.nbSpilledPoints; i+=nb_blocks){
    for(uint32_t i=thread_id; i<globalVariables.nbSpilledPoints; i+=nb_threads){
        const CPoint& point = globalVariables.spilledPoints[i];
        // sampleVoxel(point, thread_id, nb_threads_per_block);
        sampleVoxel(point);
    }
}






















/// Run on floor("NB SMs" * "Max threads per SM" / "Max threads per block") blocks of size "Max threads per block"
/// Allocate the necessary new chunks
extern "C" __global__
void kernel_simlod_insertion_part_1_chunks_allocations(){
    if(!globalVariables.isInitialised){return;}
    if(!globalVariables.isUpdating){return;}
    if(!globalVariables.isDoneLoading || !globalVariables.isDoneStoring || !globalVariables.isDoneIterating){
        return;
    }

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    // Used Claude to implement similar parallel allocation strategy as in "kernel_simlod_load_part_2_rebuilding_nodes"

    __shared__ CChunk* shPointsAttach;   // chunk to append new points chunks to (nullptr if none needed)
    __shared__ uint32_t shPointsFirstNew;
    __shared__ uint32_t shPointsNbNew;
    __shared__ uint32_t shPointsLastSize;

    __shared__ CChunk* shVoxelsAttach;   // chunk to append new voxels chunks to (nullptr if none needed)
    __shared__ uint32_t shVoxelsFirstNew;
    __shared__ uint32_t shVoxelsNbNew;
    __shared__ uint32_t shVoxelsLastSize;

    for(uint32_t node_index = block_id; node_index < globalVariables.curNbNodes; node_index += nb_blocks){

        COctreeNode* cur_node = globalVariables.packedNodes[node_index];
        if(!cur_node){
            printf("ERROR: In allocation, the node should exist\n");
        }

        // Walk the existing chains (points & voxels), figure out how many new
        // chunks are needed, fix up the size of chunks that are no longer last,
        // and reserve a contiguous range in the shared chunk pool for the rest.
        __syncthreads(); // Needed to not overwrite shared vars before previous iteration is done reading them
        if(thread_id == 0){
            shPointsAttach = nullptr;
            shPointsNbNew = 0;
            shVoxelsAttach = nullptr;
            shVoxelsNbNew = 0;

            // ----- Points -----
            uint32_t required_chunks = (cur_node->points_counter + OocSimLodSettings::NB_POINTS_PER_CHUNK - 1) / OocSimLodSettings::NB_POINTS_PER_CHUNK;
            if(required_chunks > 0){
                if(!cur_node->points){
                    cur_node->points = globalAllocator.newChunk(true);
                }

                CChunk* cur_chunk = cur_node->points;
                // if(!cur_chunk){printf("WTTTFFF points\n"); customAssert();}
                uint32_t existing_count = 1;
                while(cur_chunk->next && existing_count < required_chunks){
                    cur_chunk->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;
                    cur_chunk = cur_chunk->next;
                    existing_count++;
                }

                uint32_t last_size = cur_node->points_counter % OocSimLodSettings::NB_POINTS_PER_CHUNK;
                last_size = (last_size == 0) ? OocSimLodSettings::NB_POINTS_PER_CHUNK : last_size;

                if(existing_count < required_chunks){
                    cur_chunk->size = OocSimLodSettings::NB_POINTS_PER_CHUNK; // no longer the last chunk

                    uint32_t nb_new_chunks = required_chunks - existing_count;
                    uint32_t first_chunk = __nv_atomic_fetch_add(&globalVariables.chunksAllocatorCounter, nb_new_chunks, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);

                    shPointsAttach = cur_chunk;
                    shPointsFirstNew = first_chunk;
                    shPointsNbNew = nb_new_chunks;
                    shPointsLastSize = last_size;
                } else {
                    cur_chunk->size = last_size;
                }
            }

            // ----- Voxels -----
            required_chunks = (cur_node->voxels_counter + OocSimLodSettings::NB_POINTS_PER_CHUNK - 1) / OocSimLodSettings::NB_POINTS_PER_CHUNK;
            if(required_chunks > 0){
                if(!cur_node->voxels){
                    cur_node->voxels = globalAllocator.newChunk(true);
                }

                CChunk* cur_chunk = cur_node->voxels;
                // if(!cur_chunk){printf("WTTTFFF voxels\n"); customAssert();}

                uint32_t existing_count = 1;
                while(cur_chunk->next && existing_count < required_chunks){
                    cur_chunk->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;
                    cur_chunk = cur_chunk->next;
                    existing_count++;
                }

                uint32_t last_size = cur_node->voxels_counter % OocSimLodSettings::NB_POINTS_PER_CHUNK;
                last_size = (last_size == 0) ? OocSimLodSettings::NB_POINTS_PER_CHUNK : last_size;

                if(existing_count < required_chunks){
                    cur_chunk->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;

                    uint32_t nb_new_chunks = required_chunks - existing_count;
                    uint32_t first_chunk = __nv_atomic_fetch_add(&globalVariables.chunksAllocatorCounter, nb_new_chunks, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);

                    shVoxelsAttach = cur_chunk;
                    shVoxelsFirstNew = first_chunk;
                    shVoxelsNbNew = nb_new_chunks;
                    shVoxelsLastSize = last_size;
                } else {
                    cur_chunk->size = last_size;
                }
            }
        }
        __syncthreads();

        // Allocate the missing chunks in parallel
        uint32_t points_first_new = shPointsFirstNew;
        uint32_t points_nb_new = shPointsNbNew;
        uint32_t voxels_first_new = shVoxelsFirstNew;
        uint32_t voxels_nb_new = shVoxelsNbNew;

        for(uint32_t i = thread_id; i < points_nb_new; i += nb_threads_per_block){
            globalVariables.allocatedChunks[points_first_new + i] = globalAllocator.newChunk(true);
            globalVariables.allocatedChunks[points_first_new + i]->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;
        }
        for(uint32_t i = thread_id; i < voxels_nb_new; i += nb_threads_per_block){
            globalVariables.allocatedChunks[voxels_first_new + i] = globalAllocator.newChunk(true);
            globalVariables.allocatedChunks[voxels_first_new + i]->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;
        }
        __syncthreads();

        // Attach the new range to the existing chain, and fix the last chunk's size
        if(thread_id == 0){
            if(points_nb_new > 0){
                shPointsAttach->next = globalVariables.allocatedChunks[points_first_new];
                globalVariables.allocatedChunks[points_first_new + points_nb_new - 1]->size = shPointsLastSize;
            }
            if(voxels_nb_new > 0){
                shVoxelsAttach->next = globalVariables.allocatedChunks[voxels_first_new];
                globalVariables.allocatedChunks[voxels_first_new + voxels_nb_new - 1]->size = shVoxelsLastSize;
            }
        }

        // Link the new chunks together
        if(points_nb_new > 1){
            for(uint32_t i = thread_id; i < points_nb_new - 1; i += nb_threads_per_block){
                uint32_t real_id = points_first_new + i;
                globalVariables.allocatedChunks[real_id]->next = globalVariables.allocatedChunks[real_id + 1];
            }
        }
        if(voxels_nb_new > 1){
            for(uint32_t i = thread_id; i < voxels_nb_new - 1; i += nb_threads_per_block){
                uint32_t real_id = voxels_first_new + i;
                globalVariables.allocatedChunks[real_id]->next = globalVariables.allocatedChunks[real_id + 1];
            }
        }
        __syncthreads();
    }
}



__device__
void insertPoint(const CPoint& point){
    COctreeNode* cur_node = globalVariables.mainOctree;

    if(point.position == vec3(0,0,0)){
        return;
    }

    // Reach all corresponding leaves
    while(true){
        globalVariables.setFlagSync(cur_node->aabb_index, CFlagIsUpdated);
        // Find next child
        const CAABB& aabb = globalVariables.relationshipMap[cur_node->aabb_index].aabb;
        if(!aabb.contains(point.position)){
            printf("ERROR: weird point in point insertion, at this stage, the point should have been flagged\n");
            customAssert();
        }
        CNodePosition child_position = aabb.getNextChildIndex(point.position);

        // If leaf insert point in chunks
        if(cur_node->children[child_position]){
            cur_node = cur_node->children[child_position];
        } else {
           
            if(globalVariables.relationshipMap[cur_node->aabb_index].children[child_position] != CINVALID_ID){
                printf("ERROR: on insert point; the node %d should not have an unloaded child[%d]; node %d should be loaded, AABB = .mins(%f, %f, %f), .maxs(%f, %f, %f), points = (%f, %f, %f), child AABB = .mins(%f, %f, %f), .maxs(%f, %f, %f)\n",
                    cur_node->aabb_index, child_position, 
                    globalVariables.relationshipMap[cur_node->aabb_index].children[child_position],
                    globalVariables.relationshipMap[cur_node->aabb_index].aabb.mins.x,
                    globalVariables.relationshipMap[cur_node->aabb_index].aabb.mins.y,
                    globalVariables.relationshipMap[cur_node->aabb_index].aabb.mins.z,
                    globalVariables.relationshipMap[cur_node->aabb_index].aabb.maxs.x,
                    globalVariables.relationshipMap[cur_node->aabb_index].aabb.maxs.y,
                    globalVariables.relationshipMap[cur_node->aabb_index].aabb.maxs.z,
                    point.position.x, point.position.y, point.position.z,
                    globalVariables.relationshipMap[globalVariables.relationshipMap[cur_node->aabb_index].children[child_position]].aabb.mins.x,
                    globalVariables.relationshipMap[globalVariables.relationshipMap[cur_node->aabb_index].children[child_position]].aabb.mins.y,
                    globalVariables.relationshipMap[globalVariables.relationshipMap[cur_node->aabb_index].children[child_position]].aabb.mins.z,
                    globalVariables.relationshipMap[globalVariables.relationshipMap[cur_node->aabb_index].children[child_position]].aabb.maxs.x,
                    globalVariables.relationshipMap[globalVariables.relationshipMap[cur_node->aabb_index].children[child_position]].aabb.maxs.y,
                    globalVariables.relationshipMap[globalVariables.relationshipMap[cur_node->aabb_index].children[child_position]].aabb.maxs.z
                );
                customAssert();
            }

            uint32_t point_index = __nv_atomic_fetch_add(&cur_node->points_stored, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            if((point_index+1) > cur_node->points_counter){
                printf("ERROR: weird points inserted, skip for safety\n");
                customAssert();
            }

            uint32_t chunk_index = point_index / OocSimLodSettings::NB_POINTS_PER_CHUNK;
            CChunk* cur_chunk = cur_node->points;
            if(!cur_chunk){
                printf("ERROR: Failed to insert point in node %d; the first chunk should exist: point index = %d, points counter = %d\n", 
                    cur_node->aabb_index, point_index, cur_node->points_counter
                );
                customAssert();
            }

            for(uint32_t i=0; i<chunk_index; i++){
                cur_chunk = cur_chunk->next;
                if(!cur_chunk){
                    printf("ERROR: the points chunk should have been allocated in part 1 for node %d: point index = %d, points counter = %d\n", 
                        cur_node->aabb_index, point_index, cur_node->points_counter
                    );
                    customAssert();
                }
            }
            uint32_t real_index = point_index % OocSimLodSettings::NB_POINTS_PER_CHUNK;
            cur_chunk->points[real_index] = point;

            return;
        }
    }
};

__device__
void insertVoxel(const CPoint& voxel, COctreeNode* cur_node){
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


    // CIdAABB id = cur_node->aabb_index;
    // while(id != CINVALID_ID){
    //     globalVariables.setFlagSync(id, CFlagIsUpdated);
    //     id = globalVariables.relationshipMap[id].parent;
    // }

    uint32_t voxel_index = __nv_atomic_fetch_add(&cur_node->voxels_stored, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    uint32_t chunk_index = voxel_index / OocSimLodSettings::NB_POINTS_PER_CHUNK;

    CChunk* cur_chunk = cur_node->voxels;
    if(!cur_chunk){
        printf("ERROR: Failed to insert voxel in node %d; the first chunk should exist: point index = %d, points counter = %d\n", 
            cur_node->aabb_index, voxel_index, cur_node->voxels_counter
        );
        customAssert();
    }

    for(uint32_t i=0; i<chunk_index; i++){
        cur_chunk = cur_chunk->next;
        if(!cur_chunk){
            printf("ERROR: the voxels chunk should have been allocated in part 1 for node %d: point index = %d, points counter = %d\n", 
                cur_node->aabb_index, voxel_index, cur_node->voxels_counter
            );
            customAssert();
        }
    }
    uint32_t real_index = voxel_index % OocSimLodSettings::NB_POINTS_PER_CHUNK;
    cur_chunk->points[real_index] = voxel;
};




/// Run on floor("NB SMs" * "Max threads per SM" / "Max threads per block") blocks of size "Max threads per block"
extern "C" __global__
void kernel_simlod_insertion_part_2_filling(){
    if(!globalVariables.isInitialised){return;}
    if(!globalVariables.isUpdating){return;}
    if(!globalVariables.isDoneLoading || !globalVariables.isDoneStoring || !globalVariables.isDoneIterating){
        return;
    }

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    // Insert new points
    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

        if(nb_new_points > globalVariables.maxBatchSize){
            printf("ERROR: On insertion, batch size exceeded the limit: %d / %d\n", 
                nb_new_points, globalVariables.maxBatchSize
            );
            customAssert();
        }

        for(uint32_t i=thread_id; i<nb_new_points; i+=nb_threads){
            const CPoint& point = new_points[i];
            insertPoint(point);

            // UI values
            __nv_atomic_add(&globalVariables.nbNewPointsThisUpdate, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_add(&globalVariables.nbTotalPoints, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_add(&globalVariables.currentNbPoints, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        }
    }

    // Insert spilled points
    for(uint32_t i=thread_id; i<globalVariables.nbSpilledPoints; i+=nb_threads){
        const CPoint& point = globalVariables.spilledPoints[i];
        insertPoint(point);
    }

    // Insert new voxels
    for(uint32_t i=thread_id; i<globalVariables.nbBacklogVoxels; i+=nb_threads){
        const CPoint& voxel = globalVariables.backlogVoxels[i];
        COctreeNode* node = globalVariables.backlogVoxelsNodes[i];
        insertVoxel(voxel, node);

        // UI values
        __nv_atomic_add(&globalVariables.nbNewVoxelsThisUpdate, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        __nv_atomic_add(&globalVariables.nbTotalVoxels, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        __nv_atomic_add(&globalVariables.currentNbVoxels, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    }


    if(thread_id == 0){
        // Because "newChunk" was called in part 1
        globalAllocator.chunksAllocator->reset_temporary_allocations();
    }
}
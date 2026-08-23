#include "utils.cuh"


extern "C" __global__
void kernel_update_updates_cache_part_1_counting(){
    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    // if(thread_id == 0){
    //     printf("kernel_update_updates_cache_part_1_counging\n");
    // }

    // Count updated children
    for(uint32_t node_index = thread_id; node_index < globalVariables.curNbNodes; node_index += nb_threads){
        COctreeNode* node = globalVariables.packedNodes[node_index];
        if(!globalVariables.isUpdated(node->aabb_index)){continue;}

        uint8_t nb_updated_children = 0;
        for(uint32_t i=0; i<8; i++){
            if(node->children[i] && globalVariables.isUpdated(node->children[i]->aabb_index)){
                nb_updated_children++;
            }
        }
        globalVariables.temporaryIdBuffer[node->aabb_index] = uint32_t(nb_updated_children);
        globalVariables.setCounterFlag(node->aabb_index, nb_updated_children);
    }
    
    globalVariables.nbOrderedNodes = 0;
}

extern "C" __global__
void kernel_update_updates_cache_part_2_sorting(){
    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    // if(thread_id == 0){
    //     printf("kernel_update_updates_cache_part_2_sorting\n");
    // }

    // Loop to add nodes
    for(uint32_t node_index = thread_id; node_index < globalVariables.curNbNodes; node_index += nb_threads){
        COctreeNode* node = globalVariables.packedNodes[node_index];
        CIdAABB cur_id = node->aabb_index;
        if(!globalVariables.isUpdated(cur_id)){continue;}
        globalVariables.setFlagSync(cur_id, CFlagWillBeInUpdatesCache);
        if(globalVariables.getCounterFlag(cur_id) != 0){continue;}

        while(true){
            uint32_t position = __nv_atomic_fetch_add(
                &globalVariables.nbOrderedNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE
            );
            globalVariables.temporaryIdBuffer2[position] = cur_id;

            CIdAABB parent_id = globalVariables.relationshipMap[cur_id].parent;
            if(parent_id == CINVALID_ID){break;}

            uint32_t old_counter = __nv_atomic_fetch_sub(
                &globalVariables.temporaryIdBuffer[parent_id], 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE
            );

#ifdef ASSERT_ENABLED
            if(old_counter == 0){
                printf("ERROR: cache-ordering counter underflow for parent %d of node %d\n", parent_id, cur_id);
                customAssert();
            }
#endif

            // Continue only if I was the last child to update the parent
            if(old_counter != 1){break;}
            cur_id = parent_id;
        }
    }

    uint32_t cache_size = globalVariables.updatesCacheSize;
    // Flag new elements that will be in the cache
    for(uint32_t i = thread_id; i < cache_size; i += nb_threads){
        CIdAABB old_id = globalVariables.updatesCache[i];
        if(old_id == CINVALID_ID){break;}
        globalVariables.unsetFlagSync(old_id, CFlagIsInUpdatesCache);
    }
}

/// TODO: make the prefix scan parallel
/// Run on a single thread
extern "C" __global__
void kernel_update_updates_cache_part_3_prefix_sum(){
    // printf("kernel_update_updates_cache_part_3_prefix_sum\n");

    uint32_t nb_ordered_nodes = globalVariables.nbOrderedNodes;
    uint32_t cache_size = globalVariables.updatesCacheSize;
    uint32_t loop_cap = min(nb_ordered_nodes, cache_size);
    uint32_t scan_cpt = loop_cap;

    if(scan_cpt < cache_size){
        // Prefix scan
        for(uint32_t i = 0; i < cache_size; i++){
            CIdAABB cur_id = globalVariables.updatesCache[i];
            if(cur_id == CINVALID_ID){break;} // Reached the end of the current cache
            if(!globalVariables.willBeInUpdatesCache(cur_id)){
                globalVariables.setFlag(cur_id, CFlagIsInUpdatesCache);
                globalVariables.updatesCache[cache_size + scan_cpt] = cur_id;
                scan_cpt++;
                if(scan_cpt == cache_size){break;}
            }
        }

        // Move non duplicates at the end of the cache
        for(uint32_t i = loop_cap; i < scan_cpt; i++){
            CIdAABB cur_id = globalVariables.updatesCache[cache_size + i];
            globalVariables.updatesCache[i] = cur_id;
        }
    }

    // Move new nodes at the beginning of the cache
    for(uint32_t i = 0; i < loop_cap; i++){
        uint32_t buffer_position = nb_ordered_nodes - i - 1;
        CIdAABB cur_id = globalVariables.temporaryIdBuffer2[buffer_position];
        globalVariables.setFlag(cur_id, CFlagIsInUpdatesCache);
        globalVariables.updatesCache[i] = cur_id;
    }
}



/// Run on floor("NB SMs" * "Max threads per SM" / "Max threads per block") blocks of size "Max threads per block"
extern "C" __global__
void kernel_prepare_store_part_1_filling_buffers(){
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    // if(block_id == 0 && thread_id == 0){
    //     printf("kernel_prepare_store_part_1_filling_buffers\n");
    // }

    __shared__ uint32_t shExchangedIndex;

    for(uint32_t node_index = block_id; node_index < globalVariables.curNbNodes; node_index += nb_blocks){

        COctreeNode* node = globalVariables.packedNodes[node_index];

#ifdef ASSERT_ENABLED
        if(!node){
            printf("At this point, no node should be null\n");
            customAssert();
        }
#endif

        bool is_in_cache = globalVariables.isInUpdatesCache(node->aabb_index);
        // Unset all the flags except the isInUpdatesCache flag
        if(thread_id == 0){ // Only one thread per block
            // globalVariables.resetFlags(node->aabb_index);
            uint32_t flags = uint32_t(is_in_cache) << CFlagIsInUpdatesCache;
            globalVariables.setFlags(node->aabb_index, flags);
        }

        // if(!globalVariables.updatesCache->contains(node->aabb_index)){
        if(!is_in_cache){
            __syncthreads(); // Needed to sync after break
            if(thread_id == 0){
                shExchangedIndex = __nv_atomic_fetch_add(&globalVariables.nbNodesExchanged, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            
                if(shExchangedIndex >= globalVariables.maxNbNodesExchanged){
                    // printf("WARN: Too many nodes are being stored\n");
                    globalVariables.isDoneStoring = false;
                }

            }
            __syncthreads(); // Needed to sync before break condition

            if(shExchangedIndex >= globalVariables.maxNbNodesExchanged){
                break; // To avoid skipping thread sync
            }


            // Storing node properties
            globalVariables.exchangedAABBIndices[shExchangedIndex] = node->aabb_index;
            globalVariables.exchangedAABBParentsIndices[shExchangedIndex] = globalVariables.relationshipMap[node->aabb_index].parent;
            globalVariables.exchangedChildrenIds[shExchangedIndex] = node->children_ids;
            globalVariables.exchangedPointsCounters[shExchangedIndex] = node->points_counter;
            globalVariables.exchangedVoxelsCounters[shExchangedIndex] = node->voxels_counter;

            // UI values
            if(thread_id == 0){
                __nv_atomic_add(&globalVariables.nbStoredNodesThisUpdate, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                __nv_atomic_add(&globalVariables.nbTotalStoredNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                __nv_atomic_sub(&globalVariables.currentNbPoints, node->points_counter, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                __nv_atomic_sub(&globalVariables.currentNbVoxels, node->voxels_counter, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            }
            
            const uint32_t MAX_NB_POINTS = globalVariables.maxNbPointsChunksPerExchangedNode * OocSimLodSettings::NB_POINTS_PER_CHUNK;
            const uint32_t MAX_NB_VOXELS = globalVariables.maxNbVoxelsChunksPerExchangedNode * OocSimLodSettings::NB_POINTS_PER_CHUNK;
            
            // Storing node points
            CChunk* cur_chunk = node->points;
            uint32_t cur_point_index = thread_id;
            while(cur_chunk){
                for(uint32_t i = thread_id; i < cur_chunk->size; i += nb_threads_per_block){

#ifdef ASSERT_ENABLED
                    if(cur_point_index >= MAX_NB_POINTS){
                        printf("ERROR: Too many points in the node, some will be skipped to store it: index %d / %d\n",
                            cur_point_index, MAX_NB_POINTS
                        );
                        customAssert();
                    }
#endif

                    const CPoint& cur_point = cur_chunk->points[i];
                    CPoint* exchangedPoints = globalVariables.exchangedPoints[shExchangedIndex];
                    exchangedPoints[cur_point_index] = cur_point;
                    cur_point_index += nb_threads_per_block;
                }
                cur_chunk = cur_chunk->next;
            }

            // Storing node voxels
            cur_chunk = node->voxels;
            cur_point_index = thread_id;
            while(cur_chunk){
                for(uint32_t i = thread_id; i < cur_chunk->size; i += nb_threads_per_block){

#ifdef ASSERT_ENABLED
                    if(cur_point_index >= MAX_NB_VOXELS){
                        printf("ERROR: Too many voxels in the node, some will be skipped to store it: index %d / %d\n",
                            cur_point_index, MAX_NB_VOXELS
                        );
                        customAssert();
                    }
#endif

                    const CPoint& cur_voxel = cur_chunk->points[i];
                    CPoint* exchangedVoxels = globalVariables.exchangedVoxels[shExchangedIndex];
                    exchangedVoxels[cur_point_index] = cur_voxel;
                    cur_point_index += nb_threads_per_block;
                }
                cur_chunk = cur_chunk->next;
            }

            __syncthreads(); // Needed to sync before deletion
            if(thread_id == 0){
                // Deleting the node
                globalAllocator.delOctreeNode(node, true, true);
                globalVariables.packedNodes[node_index] = nullptr;
            }
            __syncthreads(); // Needed to sync after deletion

        }
    }
}



/// Run on floor("NB SMs" * "Max threads per SM" / "Max threads per block") blocks of size "Max threads per block"
extern "C" __global__
void kernel_prepare_store_part_2_resetting_children(){
    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    // if(thread_id == 0){
    //     printf("kernel_prepare_store_part_2_resetting_children\n");
    // }

    for(uint32_t node_index = thread_id; node_index < globalVariables.curNbNodes; node_index += nb_threads){
        COctreeNode* node = globalVariables.packedNodes[node_index];
        if(!node){continue;}

        for(uint32_t i=0; i<8; i++){
            CIdAABB child_index = globalVariables.relationshipMap[node->aabb_index].children[i];

            if(child_index == CINVALID_ID){
                continue;
            }
            
            // if(!globalVariables.updatesCache->contains(child_index)){
            if(!globalVariables.isInUpdatesCache(child_index)){
                node->children[i] = nullptr;
                continue;
            }
        }
    }
}





__device__
void packNodes(){
    // Better packing strategy with children close to parents
    uint32_t nb_nodes = 0;
    uint32_t begin = 0;
    uint32_t end = globalVariables.curNbNodes-1;

    while(begin <= end){
        if(globalVariables.packedNodes[begin]){
            begin++;
            nb_nodes++;
        } else {
            COctreeNode* last_non_empty = globalVariables.packedNodes[end];
            while(!last_non_empty){
                end--;
                last_non_empty = globalVariables.packedNodes[end];
            }
            if(end==0){break;}
            if(end < begin){break;}

#ifdef ASSERT_ENABLED
            if(!last_non_empty){
                printf("ERROR: at this point a non empty node should have been found\n");
                customAssert();
            }
#endif

            globalVariables.packedNodes[begin] = last_non_empty;
            globalVariables.packedNodes[end] = nullptr;
        }
    }

    globalVariables.curNbNodes = nb_nodes;
    globalVariables.octreeDepth = 0;
    globalVariables.batchesToAddBottomUpCount = 0;
}



/// Run on "MaxActiveBlocksPerMultiprocessor" cooperative blocks of size "Max block size"
extern "C" __global__
void kernel_prepare_store_part_3_updating_levels(){
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    uint32_t first_point = block_id * nb_threads_per_block + thread_id;
    uint32_t step = nb_blocks * nb_threads_per_block;
    bool is_first = (block_id == 0 && thread_id == 0);

    // if(is_first){
    //     printf("kernel_prepare_store_part_3_updating_levels\n");
    // }
    
    if(is_first){
        packNodes();
        // Because "delOctreeNode" was called in kernel_prepare_store_part_1_filling_buffers
        globalAllocator.chunksAllocator->reset_temporary_deallocations();
        globalAllocator.gridsAllocator->reset_temporary_deallocations();
        globalAllocator.nodesAllocator->reset_temporary_deallocations();
    }
    grid.sync();

    for(uint32_t node_index = first_point; node_index < globalVariables.curNbNodes; node_index += step){
        // Reset the correct node levels
        // No need to sync "counterFlag" accesses because of this specific kernel
        COctreeNode* node = globalVariables.packedNodes[node_index];
        node->level = 0;
        globalVariables.resetCounterFlag(node->aabb_index);
    }

    // The number of grid.sync() should be the same for each thread in a cooperative group
    while(true){
        // Update all child levels
        for(uint32_t node_index = first_point; node_index < globalVariables.curNbNodes; node_index += step){
            COctreeNode* node = globalVariables.packedNodes[node_index];
            uint32_t new_child_level = globalVariables.getCounterFlag(node->aabb_index) + 1;
            // Update max depth
            __nv_atomic_max(&globalVariables.octreeDepth, new_child_level, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            for(uint32_t i=0; i<8; i++){
                CIdAABB child_id = globalVariables.relationshipMap[node->aabb_index].children[i];
                // if((child_id != CINVALID_ID) && globalVariables.updatesCache->contains(child_id)){
                if((child_id != CINVALID_ID) && globalVariables.isInUpdatesCache(child_id)){
                    
#ifdef ASSERT_ENABLED
                    if(!node->children[i]){
                        printf("ERROR: when updating the children levels, the child should exist\n");
                        customAssert();
                    }
#endif

                    node->children[i]->level = new_child_level;
                }
            }
        }
        grid.sync();

        // Fetch new node level
        bool is_updated = false;
        for(uint32_t node_index = first_point; node_index < globalVariables.curNbNodes; node_index += step){
            COctreeNode* node = globalVariables.packedNodes[node_index];
            uint32_t saved_level = globalVariables.getCounterFlag(node->aabb_index); 
            if(saved_level < node->level){
                globalVariables.increaseCounterFlag(node->aabb_index);
                is_updated = true;
            }
        }

        if(is_updated){
            // Using batchesToAddBottomUpCount as a flag
            __nv_atomic_or(&globalVariables.batchesToAddBottomUpCount, true, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        }
        grid.sync();

        // Using batchesToAddBottomUpCount as a flag
        if(!__nv_atomic_load_n(&globalVariables.batchesToAddBottomUpCount, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE)){
            break;
        }
        grid.sync();

        if(is_first){
            // Using batchesToAddBottomUpCount as a flag
            globalVariables.batchesToAddBottomUpCount = false;
        }
    }

    globalVariables.batchesToAddBottomUpCount = 0;

    if(globalVariables.isDoneIterating
        && globalVariables.isDoneLoading
        && globalVariables.isDoneStoring // To run only on complete updates
        && block_id == 0 && thread_id == 0 // To run only once
    ){
        // UI values
        __nv_atomic_add(&globalVariables.nbTotalUpdates, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    }
}



extern "C" __global__
void kernel_reset_batches(){
    // printf("kernel_reset_batches\n");
    for(uint32_t i=0; i<globalVariables.maxNbBatches; i++){
        globalVariables.batchesAddedMask[i] = true;
    }
}
#include "utils.cuh"

__device__ 
void fillUpdatesCacheRecursive(COctreeNode* cur_node){
    if(!cur_node){return;}
    for(uint32_t i=0; i<8; i++){fillUpdatesCacheRecursive(cur_node->children[i]);}
    if(globalVariables.isUpdated(cur_node->aabb_index)){
        globalVariables.unsetFlagSync(cur_node->aabb_index, CFlagIsUpdated);
        globalVariables.updatesCache->add(cur_node->aabb_index);
    }
}

__device__ 
void fillUpdatesCacheIterative(COctreeNode* root_node){
    uint32_t cpt = 1;
    // use temporaryIdBuffer as a temporary stack
    globalVariables.temporaryIdBuffer[0] = root_node->aabb_index;

    // To be sure it has not been flagged in another stack loop
    globalVariables.unsetFlagSync(root_node->aabb_index, CFlagIsFirstVisitedInStack);

    while(cpt > 0){
        const CIdAABB& cur_node = globalVariables.temporaryIdBuffer[cpt-1];

        if(globalVariables.isFirstVisitedInStack(cur_node)){
            // Add the node to the cache after all its children
            globalVariables.unsetFlagSync(cur_node, CFlagIsUpdated);
            globalVariables.updatesCache->add(cur_node);

            globalVariables.unsetFlagSync(cur_node, CFlagIsFirstVisitedInStack);
            cpt--;
            continue;
        }

        globalVariables.setFlagSync(cur_node, CFlagIsFirstVisitedInStack);

        for(uint32_t i=0; i<8; i++){
            // Reversed order traversal
            uint32_t child_index = 7-i;
            const CIdAABB& child_node = globalVariables.relationshipMap[cur_node].children[child_index];
            if(child_node != CINVALID_ID && globalVariables.isUpdated(child_node)){
                cpt++;
                if(cpt > globalVariables.temporaryBufferSize){
                    printf("ERROR: Can't update the updates cache, the stack is full\n");
                    customAssert();
                }
                globalVariables.temporaryIdBuffer[cpt-1] = child_node;

                // To be sure it has not been flagged in another stack loop
                globalVariables.unsetFlagSync(child_node, CFlagIsFirstVisitedInStack);
            }
        }
    }
}

/// Run on a single thread
extern "C" __global__
void kernel_update_updates_cache(){
    if(!globalVariables.isInitialised){return;}

    // // Sanity check unicity in cache
    // CDoubleLinkedList<CIdAABB>::Iterator* entry = globalVariables.updatesCache->cache.begin();
    // while(entry){
    //     CIdAABB cur_id = entry->value;
    //     uint32_t cpt = 0;
    //     CDoubleLinkedList<CIdAABB>::Iterator* comp = globalVariables.updatesCache->cache.begin();
    //     while(comp){
    //         if(comp->value == cur_id){cpt++;}
    //         comp = comp->next;
    //     }
    //     if(cpt != 1){
    //         printf("ERROR: the cache should contain a single entry for `%d' but it contains %d entries\n", 
    //             cur_id, cpt
    //         );
    //         customAssert();
    //     }
    //     entry = entry->next;
    // }
    // uint32_t expected_size = globalVariables.updatesCache->getSize();
    // uint32_t cpt = 0;
    // for(uint32_t i=0; i<globalVariables.curNbNodes; i++){
    //     COctreeNode* node = globalVariables.packedNodes[i];
    //     if(!node){continue;}
    //     if(globalVariables.updatesCache->contains(node->aabb_index)){
    //         cpt++;
    //     }
    // }
    // if(cpt != expected_size){
    //     printf("ERROR: there are %d nodes in the cache but cache size is %d\n", cpt, expected_size);
    //     customAssert();
    // }
    // for(uint32_t i=0; i<globalVariables.curNbNodes; i++){
    //     COctreeNode* node_i = globalVariables.packedNodes[i];
    //     for(uint32_t j=i+1; j<globalVariables.curNbNodes; j++){
    //         COctreeNode* node_j = globalVariables.packedNodes[j];
    //         if(node_i->aabb_index == node_j->aabb_index){
    //             printf("ERROR: node %d appears multiple times in the packed nodes\n", node_i->aabb_index);
    //             customAssert();
    //         }
    //     }
    // }





    if(!globalVariables.isUpdating){return;}
    globalVariables.nbNodesExchanged = 0;

    if(!globalVariables.isDoneIterating){return;}
    if(!globalVariables.isDoneLoading){return;}

    if(!globalVariables.isDoneStoring){
        globalVariables.isDoneStoring = true;
        return;
    }

    // fillUpdatesCacheRecursive(globalVariables.mainOctree);
    fillUpdatesCacheIterative(globalVariables.mainOctree);

    // // Sanity check unicity in cache
    // entry = globalVariables.updatesCache->cache.begin();
    // while(entry){
    //     CIdAABB cur_id = entry->value;
    //     CIdAABB parent_id = globalVariables.relationshipMap[cur_id].parent;
    //     if(parent_id != CINVALID_ID && !globalVariables.updatesCache->contains(parent_id)){
    //         printf("ERROR: cache invalid, node `%d' is in cache but not its parent `%d'\n",
    //             cur_id, parent_id
    //         );
    //         customAssert();
    //     }
    //     entry = entry->next;
    // }

    // // Sanity check
    // for(uint32_t i=0; i<globalVariables.curNbNodes; i++){
    //     COctreeNode* node = globalVariables.packedNodes[i];
    //     if(!node){continue;}
    //     if(node->points_counter != node->points_stored){
    //         printf("ERROR: Wtf, got %d / %d for node %d\n",
    //             node->points_counter, node->points_stored, node->aabb_index
    //         );
    //     }
    //     for(uint32_t j=0; j<8; j++){
    //         if(node->children[j] && globalVariables.relationshipMap[node->children[j]->aabb_index].parent != node->aabb_index){
    //             printf("ERROR: child[%d] of node %d has parent %d\n",
    //                 j, node->aabb_index, globalVariables.relationshipMap[node->children[j]->aabb_index].parent
    //             );
    //             customAssert();
    //         }
    //         if(node->children[j] && globalVariables.relationshipMap[node->aabb_index].children[j] != node->children[j]->aabb_index){
    //             printf("ERROR: child[%d] = %d of node %d should be %d\n",
    //                 j, node->children[j]->aabb_index, node->aabb_index,
    //                 globalVariables.relationshipMap[node->aabb_index].children[j]
    //             );
    //             customAssert();
    //         }
    //     }
    // }
}


/// Run on floor("NB SMs" * "Max threads per SM" / "Max threads per block") blocks of size "Max threads per block"
extern "C" __global__
void kernel_prepare_store_part_1_filling_buffers(){
    if(!globalVariables.isInitialised){return;}
    if(!globalVariables.isUpdating){return;}
    if(!globalVariables.isDoneIterating){return;}
    if(!globalVariables.isDoneLoading){return;}

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    __shared__ uint32_t shExchangedIndex;

    for(uint32_t node_index = block_id; node_index < globalVariables.curNbNodes; node_index += nb_blocks){

        COctreeNode* node = globalVariables.packedNodes[node_index];
        if(!node){continue;}

        // Unset flags
        if(thread_id == 0){ // Only one thread per block
            globalVariables.resetFlagsSync(node->aabb_index);
        }

        if(!globalVariables.updatesCache->contains(node->aabb_index)){
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
                continue;
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

                // globalVariables.setFlagSync(node->aabb_index, CFlagToStore);
            }
            
            const uint32_t MAX_NB_POINTS = globalVariables.maxNbPointsChunksPerExchangedNode * OocSimLodSettings::NB_POINTS_PER_CHUNK;
            const uint32_t MAX_NB_VOXELS = globalVariables.maxNbVoxelsChunksPerExchangedNode * OocSimLodSettings::NB_POINTS_PER_CHUNK;
            
            // Storing node points
            CChunk* cur_chunk = node->points;
            uint32_t cur_point_index = thread_id;
            while(cur_chunk){
                for(uint32_t i = thread_id; i < cur_chunk->size; i += nb_threads_per_block){
                    if(cur_point_index >= MAX_NB_POINTS){
                        printf("ERROR: Too many points in the node, some will be skipped to store it: index %d / %d\n",
                            cur_point_index, MAX_NB_POINTS
                        );
                        customAssert();
                    }
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
                    if(cur_point_index >= MAX_NB_VOXELS){
                        printf("ERROR: Too many voxels in the node, some will be skipped to store it: index %d / %d\n",
                            cur_point_index, MAX_NB_VOXELS
                        );
                        customAssert();
                    }
                    const CPoint& cur_voxel = cur_chunk->points[i];
                    CPoint* exchangedVoxels = globalVariables.exchangedVoxels[shExchangedIndex];
                    exchangedVoxels[cur_point_index] = cur_voxel;
                    cur_point_index += nb_threads_per_block;
                }
                cur_chunk = cur_chunk->next;
            }

            __syncthreads(); // Needed to sync before deletion
            if(thread_id == 0){
                // globalVariables.unsetFlagSync(node->aabb_index, CFlagToStore);
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
    if(!globalVariables.isInitialised){return;}
    if(!globalVariables.isUpdating){return;}
    if(!globalVariables.isDoneIterating){return;}
    if(!globalVariables.isDoneLoading){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    for(uint32_t node_index = thread_id; node_index < globalVariables.curNbNodes; node_index += nb_threads){
        COctreeNode* node = globalVariables.packedNodes[node_index];
        if(!node){continue;}

        node->children_visibility = 0b00000000;

        for(uint32_t i=0; i<8; i++){
            CIdAABB child_index = globalVariables.relationshipMap[node->aabb_index].children[i];

            if(child_index == CINVALID_ID){
                continue;
            }
            
            if(!globalVariables.updatesCache->contains(child_index)){
                node->children[i] = nullptr;
                continue;
            }
            node->children_visibility |= ((0x01) << i);
        }
    }
}





__device__
void packNodes(){
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
            if(!last_non_empty){
                printf("ERROR: at this point a non empty node should have been found\n");
                customAssert();
            }
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
    if(!globalVariables.isInitialised){return;}
    if(!globalVariables.isUpdating){return;}
    if(!globalVariables.isDoneIterating){return;}
    if(!globalVariables.isDoneLoading){return;}

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    uint32_t first_point = block_id * nb_threads_per_block + thread_id;
    uint32_t step = nb_blocks * nb_threads_per_block;
    bool is_first = (block_id == 0 && thread_id == 0);
    
    if(is_first){
        // printf("step %d:\n", globalVariables.nbTotalUpdates);

        // uint32_t cpt = 0;
        // uint32_t old_cpt = globalVariables.curNbNodes;
        // for(uint32_t i=0; i<old_cpt; i++){
        //     if(globalVariables.packedNodes[i]){
        //         cpt++;
        //     }
        // }

        packNodes();
        // Because "delOctreeNode" was called in kernel_prepare_store_part_1_filling_buffers
        globalAllocator.chunksAllocator->reset_temporary_deallocations();
        globalAllocator.gridsAllocator->reset_temporary_deallocations();
        globalAllocator.nodesAllocator->reset_temporary_deallocations();

        // printf("    - exchanged = %d,\n    - cache size: %d /  %d\n    - %d nodes;\n    - non null nodes: %d / %d\n", 
        //     globalVariables.nbNodesExchanged,
        //     globalVariables.updatesCache->getSize(), globalVariables.updatesCacheSize,
        //     globalVariables.curNbNodes, 
        //     cpt, old_cpt
        // );
    }
    grid.sync();

    for(uint32_t node_index = first_point; node_index < globalVariables.curNbNodes; node_index += step){
        // Reset the correct node levels
        // No need to sync "counterFlag" accesses because of this specific kernel
        COctreeNode* node = globalVariables.packedNodes[node_index];
        node->level = 0;
        globalVariables.resetCounterFlagSync(node->aabb_index);
    }
    // The number of grid.sync() should be the same for each thread in a cooperative group
    while(true){
        // Update all child levels
        for(uint32_t node_index = first_point; node_index < globalVariables.curNbNodes; node_index += step){
            COctreeNode* node = globalVariables.packedNodes[node_index];
            uint32_t new_child_level = globalVariables.getCounterFlagSync(node->aabb_index) + 1;
            // Update max depth
            __nv_atomic_max(&globalVariables.octreeDepth, new_child_level, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            for(uint32_t i=0; i<8; i++){
                CIdAABB child_id = globalVariables.relationshipMap[node->aabb_index].children[i];
                if((child_id != CINVALID_ID) && globalVariables.updatesCache->contains(child_id)){
                    if(!node->children[i]){
                        printf("ERROR: when updating the children levels, the child should exist\n");
                        customAssert();
                    }
                    node->children[i]->level = new_child_level;
                }
            }
        }
        grid.sync();

        // Fetch new node level
        bool is_updated = false;
        for(uint32_t node_index = first_point; node_index < globalVariables.curNbNodes; node_index += step){
            COctreeNode* node = globalVariables.packedNodes[node_index];
            uint32_t saved_level = globalVariables.getCounterFlagSync(node->aabb_index); 
            if(saved_level < node->level){
                globalVariables.increaseCounterFlagSync(node->aabb_index);
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
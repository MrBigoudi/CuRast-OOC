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
    // use exchangedAABBIndices as a stack
    globalVariables.exchangedAABBIndices[0] = root_node->aabb_index;

    // To be sure it has not been flagged in another stack loop
    globalVariables.unsetFlagSync(root_node->aabb_index, CFlagIsFirstVisitedInStack);

    while(cpt > 0){
        const CIdAABB& cur_node = globalVariables.exchangedAABBIndices[cpt-1];

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
                if(cpt > globalVariables.maxNbNodesExchanged){
                    printf("ERROR: Can't update the updates cache, the stack is full\n");
                    customAssert();
                }
                globalVariables.exchangedAABBIndices[cpt-1] = child_node;

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
    if(!globalVariables.isUpdating){return;}
    globalVariables.nbNodesExchanged = 0;

    // fillUpdatesCacheRecursive(globalVariables.mainOctree);
    fillUpdatesCacheIterative(globalVariables.mainOctree);

}


/// Run on "NB SMs" * "Max threads per SM" blocks of size 1
extern "C" __global__
void kernel_prepare_store_part_1_filling_buffers(){
    if(!globalVariables.isInitialised){return;}
    if(!globalVariables.isUpdating){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    for(uint32_t node_index = thread_id; node_index < globalVariables.curNbNodes; node_index += nb_threads){

        COctreeNode* node = globalVariables.packedNodes[node_index];
        for(uint32_t i=0; i<CFlagIsVisible; i++){
            globalVariables.unsetFlagSync(node->aabb_index, (CNodeFlagType)i);
        }

        if(!globalVariables.updatesCache->contains(node->aabb_index)
            && !globalVariables.visibilityCache->contains(node->aabb_index)    
        ){
            uint32_t exchanged_index = __nv_atomic_fetch_add(&globalVariables.nbNodesExchanged, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);

            if(exchanged_index >= globalVariables.maxNbNodesExchanged){
                printf("ERROR: Too many nodes are being stored\n");
                customAssert();
            }

            globalVariables.exchangedAABBIndices[exchanged_index] = node->aabb_index;
            globalVariables.exchangedChildrenIds[exchanged_index] = node->children_ids;
            globalVariables.exchangedPointsCounters[exchanged_index] = node->points_counter;
            globalVariables.exchangedVoxelsCounters[exchanged_index] = node->voxels_counter;
            
            const uint32_t MAX_NB_POINTS = globalVariables.maxNbPointsChunksPerExchangedNode * OocSimLodSettings::NB_POINTS_PER_CHUNK;
            const uint32_t MAX_NB_VOXELS = globalVariables.maxNbVoxelsChunksPerExchangedNode * OocSimLodSettings::NB_POINTS_PER_CHUNK;
            
            CChunk* cur_chunk = node->points;
            uint32_t cur_point_index = 0;
            while(cur_chunk){
                for(uint32_t i=0; i<cur_chunk->size; i++){
                    if(cur_point_index >= MAX_NB_POINTS){
                        printf("ERROR: Too many points in the node, some will be skipped to store it: index %d / %d\n",
                            cur_point_index, MAX_NB_POINTS
                        );
                        break;
                    }
                    globalVariables.exchangedPoints[exchanged_index][cur_point_index] = cur_chunk->points[i];
                    cur_point_index++;
                }
                cur_chunk = cur_chunk->next;
            }

            cur_chunk = node->voxels;
            cur_point_index = 0;
            while(cur_chunk){
                for(uint32_t i=0; i<cur_chunk->size; i++){
                    if(cur_point_index >= MAX_NB_VOXELS){
                        printf("ERROR: Too many voxels in the node, some will be skipped to store it: index %d / %d\n",
                            cur_point_index, MAX_NB_VOXELS
                        );
                        break;
                    }
                    globalVariables.exchangedVoxels[exchanged_index][cur_point_index] = cur_chunk->points[i];
                    cur_point_index++;
                }
                cur_chunk = cur_chunk->next;
            }

            
            globalVariables.setFlagSync(node->aabb_index, CFlagToStore);
            
            globalAllocator.delOctreeNode(node, true, true);
            globalVariables.packedNodes[node_index] = nullptr;
        } else {
            globalVariables.setFlagSync(node->aabb_index, CFlagIsOnUpdatesCache);
        }
    }
}



/// Run on "NB SMs" * "Max threads per SM" blocks of size 1
extern "C" __global__
void kernel_prepare_store_part_2_resetting_children(){
    if(!globalVariables.isInitialised){return;}
    if(!globalVariables.isUpdating){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    for(uint32_t node_index = thread_id; node_index < globalVariables.curNbNodes; node_index += nb_threads){

        COctreeNode* node = globalVariables.packedNodes[node_index];
        if(!node){continue;} // Skip if this is one of the node that has just been removed

        for(uint32_t i=0; i<8; i++){
            CIdAABB child_index = globalVariables.relationshipMap[node->aabb_index].children[i];
            if(child_index != CINVALID_ID && globalVariables.isToStore(child_index)){
                node->children[i] = nullptr;
            }
        }
    }

    if(thread_id == 0){
        // Because "delOctreeNode" was called in simlodSplit
        globalAllocator.chunksAllocator->reset_temporary_deallocations();
        globalAllocator.gridsAllocator->reset_temporary_deallocations();
        globalAllocator.nodesAllocator->reset_temporary_deallocations();
    }
}













__device__ 
void computeRealLevelsRec(COctreeNode* cur_node, uint32_t cur_level){
    if(!cur_node){return;}
    cur_node->level = cur_level;

    globalVariables.renderingOctreeDepth = max(globalVariables.renderingOctreeDepth, cur_level);
    for(uint32_t i=0; i<8; i++){
        computeRealLevelsRec(cur_node->children[i], cur_level+1);
    }
}





/// Run on "MaxActiveBlocksPerMultiprocessor" cooperative blocks of size "Max block size"
extern "C" __global__
void kernel_create_rendereable_octree(){
    if(!globalVariables.isInitialised){return;}
    if(!globalVariables.isUpdating){return;}

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    uint32_t first_point = block_id * nb_threads_per_block + thread_id;
    uint32_t step = nb_blocks * nb_threads_per_block;

    bool is_first = (block_id == 0 && thread_id == 0);

    // Pack the nodes
    if(is_first){
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
        globalVariables.renderingOctreeDepth = 0;
        globalVariables.batchesToAddBottomUpCount = 0;   
    }
    grid.sync();



    // Compute the correct node levels
    // No need to sync "counterFlag" accesses because of this specific kernel
    for(uint32_t node_index = first_point; node_index < globalVariables.curNbNodes; node_index += step){
        COctreeNode* node = globalVariables.packedNodes[node_index];
        node->level = 0;
        globalVariables.resetCounterFlag(node->aabb_index);
    }

    // TODO: fix that

    for(uint32_t i=0; i<2; i++){
    // while(true){
        // Update all child levels
        for(uint32_t node_index = first_point; node_index < globalVariables.curNbNodes; node_index += step){
            COctreeNode* node = globalVariables.packedNodes[node_index];
            uint32_t new_child_level = globalVariables.getCounterFlag(node->aabb_index) + 1;
            // Update max depth
            __nv_atomic_max(&globalVariables.renderingOctreeDepth, new_child_level, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);

            for(uint32_t child_id = 0; child_id < 8; child_id++){
                COctreeNode* child = node->children[child_id];
                if(child){
                    printf("Old: %d, new: %d\n", child->level, new_child_level);
                    child->level = new_child_level;
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
                uint32_t old_mask = globalVariables.nodesFlags[node->aabb_index];
                uint32_t old_level = globalVariables.increaseCounterFlag(node->aabb_index);
                printf("old mask: %d, new mask: %d, old level: %d, new level: %d\n",
                    old_mask,
                    globalVariables.nodesFlags[node->aabb_index],
                    old_level,
                    globalVariables.getCounterFlag(node->aabb_index)
                );
                is_updated = true;
            }
        }
        if(!is_updated){break;}
    }




    // Deallocate old rendering nodes
    for(uint32_t node_index = first_point; node_index < globalVariables.renderingNbNodes; node_index += step){
        COctreeNode* node = globalVariables.renderingPackedNodes[node_index];
        if(!node){continue;}
        globalAllocator.delOctreeNode(node, true, true);
        globalVariables.renderingPackedNodes[node_index] = nullptr;
    }
    grid.sync();



    // Reset the allocator
    if(is_first){
        // Because "delOctreeNodeChunk" was called
        globalAllocator.chunksAllocator->reset_temporary_deallocations();
        globalAllocator.nodesAllocator->reset_temporary_deallocations();
        // No grids should be destroyed because none was allocated
    }
    grid.sync();




    // Copy the correct nodes
    for(uint32_t node_index = first_point; node_index < globalVariables.curNbNodes; node_index += step){
        COctreeNode* node = globalVariables.packedNodes[node_index];
        globalVariables.renderingPackedNodes[node_index] = globalAllocator.newOctreeNodePartialCpy(node, true);

        // Set and unset the CFlagIsStored here to avoid unsync issues between rendering and update
        globalVariables.unsetFlagSync(node->aabb_index, CFlagIsStored);
        for(uint32_t i=0; i<8; i++){
            CIdAABB child_index = globalVariables.relationshipMap[node->aabb_index].children[i];
            if(child_index != CINVALID_ID && globalVariables.isToStore(child_index)){
                node->children[i] = nullptr;
                globalVariables.setFlagSync(child_index, CFlagIsStored);
            }
        }
    }
    grid.sync();



    if(is_first){
        // Reset the allocator
        // Because "newOctreeNodePartialCpy" was called
        globalAllocator.chunksAllocator->reset_temporary_allocations();
        globalAllocator.nodesAllocator->reset_temporary_allocations();
        // No grids should be allocated

        globalVariables.renderingNbNodes = globalVariables.curNbNodes;        
    }
}



// /// Run on a single thread
// extern "C" __global__
// void kernel_prepare_rendereable_octree(){
//     if(!globalVariables.isInitialised){return;}

//     // Compute the correct node levels
//     globalVariables.renderingOctreeDepth = 0;
//     computeRealLevelsRec(globalVariables.mainOctree, 0);

//     // Remove nodes that have been deleted
//     for(uint32_t i=0; i<globalVariables.renderingNbNodes; i++){
//         if(!globalVariables.packedNodes[i]){
//             COctreeNode* to_remove = globalVariables.renderingPackedNodes[i];
//             globalAllocator.delOctreeNode(to_remove, true, false);
//             globalVariables.renderingPackedNodes[i] = nullptr;
//         }
//     }

//     // Repack the nodes
//     globalVariables.renderingNbNodes = 0;
//     uint32_t begin = 0;
//     uint32_t end = globalVariables.curNbNodes-1;
//     bool should_be_new = false;

//     while(begin <= end){
//         if(globalVariables.packedNodes[begin]){
//             COctreeNode* cur_node_cpy = globalVariables.packedNodes[begin];
//             CIdAABB node_index = cur_node_cpy->aabb_index;
            
//             COctreeNode* cur_node = globalVariables.renderingPackedNodes[begin];
//             if(globalVariables.isNew(node_index)){
//                 globalVariables.renderingPackedNodes[globalVariables.renderingNbNodes] = globalAllocator.newOctreeNodePartialCpy(
//                     cur_node_cpy, false
//                 );

//                 begin++;
//                 globalVariables.renderingNbNodes++;
//                 should_be_new = false;
//                 continue;
//             }
//             if(should_be_new){
//                 printf("WTFFF\n");
//                 customAssert();
//             }

//             if(globalVariables.hasSpilled(node_index) || globalVariables.hasNewPoints(node_index)){
//                 // Copy all points and create new chunks if needed
//                 // If still some chunks left at the end, delete them
//                 CChunk* cur_chunk_cpy = cur_node_cpy->points;
//                 CChunk* cur_chunk = cur_node->points;
//                 if(cur_chunk_cpy && !cur_chunk){
//                     cur_node->points = globalAllocator.newChunkPartialCpy(cur_chunk_cpy, false);
//                 } else {
//                     while(cur_chunk_cpy){
//                         cur_chunk->size = cur_chunk_cpy->size;
//                         for(uint32_t i=0; i<cur_chunk_cpy->size; i++){
//                             cur_chunk->points[i].position = cur_chunk_cpy->points[i].position;
//                             cur_chunk->points[i].color = cur_chunk_cpy->points[i].color;
//                         }
//                         if(cur_chunk_cpy->next){
//                             if(!cur_chunk->next){
//                                 cur_chunk->next = globalAllocator.newChunkPartialCpy(cur_chunk_cpy->next, false);
//                                 break;
//                             } else {
//                                 cur_chunk_cpy = cur_chunk_cpy->next;
//                                 cur_chunk = cur_chunk->next;
//                             }
//                         } else {
//                             globalAllocator.delChunk(cur_chunk->next, false);
//                             cur_chunk->next = nullptr;
//                         }
//                     }
//                 }
//             }

//             if(globalVariables.hasNewVoxels(node_index)){
//                 // Only copy the last new voxels
//                 CChunk* cur_chunk_cpy = cur_node_cpy->voxels;
//                 CChunk* cur_chunk = cur_node->voxels;
//                 if(cur_chunk_cpy && !cur_chunk){
//                     cur_node->voxels = globalAllocator.newChunkPartialCpy(cur_chunk_cpy, false);
//                 } else {
//                     while(cur_chunk_cpy){
//                         for(uint32_t i=cur_chunk->size; i<cur_chunk_cpy->size; i++){
//                             cur_chunk->points[i].position = cur_chunk_cpy->points[i].position;
//                             cur_chunk->points[i].color = cur_chunk_cpy->points[i].color;
//                         }
//                         cur_chunk->size = cur_chunk_cpy->size;
//                         if(cur_chunk_cpy->next){
//                             if(!cur_chunk->next){
//                                 cur_chunk->next = globalAllocator.newChunkPartialCpy(cur_chunk_cpy->next, false);
//                                 break;
//                             } else {
//                                 cur_chunk_cpy = cur_chunk_cpy->next;
//                                 cur_chunk = cur_chunk->next;
//                             }
//                         }
//                     }
//                 }
//             }

//             begin++;
//             globalVariables.renderingNbNodes++;
//             globalVariables.resetFlags(node_index);
            
//         } else {
//             COctreeNode* last_non_empty = globalVariables.packedNodes[end];
//             while(!last_non_empty){
//                 end--;
//                 last_non_empty = globalVariables.packedNodes[end];
//             }
//             if(end < begin){break;}
//             if(!last_non_empty){
//                 printf("ERROR: at this point a non empty node should have been found\n");
//                 customAssert();
//             }
//             globalVariables.packedNodes[begin] = last_non_empty;
//             globalVariables.packedNodes[end] = nullptr;

//             globalVariables.renderingPackedNodes[begin] = globalVariables.renderingPackedNodes[end];
//             globalVariables.renderingPackedNodes[end] = nullptr;
//             should_be_new = true;
//         }
//     }

//     globalVariables.curNbNodes = globalVariables.renderingNbNodes;
//     globalVariables.renderingNbNodes = 0;
// }
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

    if(!globalVariables.isDoneLoading || !globalVariables.isDoneStoring){
        return;
    }

    // fillUpdatesCacheRecursive(globalVariables.mainOctree);
    fillUpdatesCacheIterative(globalVariables.mainOctree);

}


/// Run on floor("NB SMs" * "Max threads per SM" / "Max threads per block") blocks of size "Max threads per block"
extern "C" __global__
void kernel_prepare_store_part_1_filling_buffers(){
    if(!globalVariables.isInitialised){return;}
    if(!globalVariables.isUpdating){return;}
    if(!globalVariables.isDoneLoading){
        return;
    }

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    for(uint32_t node_index = thread_id; node_index < globalVariables.curNbNodes; node_index += nb_threads){

        COctreeNode* node = globalVariables.packedNodes[node_index];
        // Unset falgs
        constexpr uint32_t clear_mask = (1u << CFlagIsVisible) - 1u;
        __nv_atomic_and(&globalVariables.nodesFlags[node->aabb_index], ~clear_mask, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);

        if(!globalVariables.updatesCache->contains(node->aabb_index)){
            uint32_t exchanged_index = __nv_atomic_fetch_add(&globalVariables.nbNodesExchanged, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);

            if(exchanged_index >= globalVariables.maxNbNodesExchanged){
                printf("WARN: Too many nodes are being stored\n");
                // customAssert();

                // Instead of panicking, warn everyone that you're not done
                globalVariables.isDoneStoring = false;
                break;
            }

            // Storing node properties
            globalVariables.exchangedAABBIndices[exchanged_index] = node->aabb_index;
            globalVariables.exchangedAABBParentsIndices[exchanged_index] = globalVariables.relationshipMap[node->aabb_index].parent;
            globalVariables.exchangedAABBs[exchanged_index] = node->aabb;
            globalVariables.exchangedChildrenIds[exchanged_index] = node->children_ids;
            globalVariables.exchangedPointsCounters[exchanged_index] = node->points_counter;
            globalVariables.exchangedVoxelsCounters[exchanged_index] = node->voxels_counter;

            // UI values
            __nv_atomic_add(&globalVariables.nbStoredNodesThisUpdate, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_add(&globalVariables.nbTotalStoredNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_sub(&globalVariables.currentNbPoints, node->points_counter, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_sub(&globalVariables.currentNbVoxels, node->voxels_counter, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            
            const uint32_t MAX_NB_POINTS = globalVariables.maxNbPointsChunksPerExchangedNode * OocSimLodSettings::NB_POINTS_PER_CHUNK;
            const uint32_t MAX_NB_VOXELS = globalVariables.maxNbVoxelsChunksPerExchangedNode * OocSimLodSettings::NB_POINTS_PER_CHUNK;
            
            // Storing node points
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

            // Storing node voxels
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

            // Resetting flags
            uint32_t clear_store_mask = 0
                | CFlagHasNewPoints
                | CFlagHasNewVoxels
                | CFlagIsNew
                | CFlagHasSpilled
            ;
            __nv_atomic_and(&globalVariables.nodesFlags[node->aabb_index], ~clear_store_mask, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            globalVariables.setFlagSync(node->aabb_index, CFlagToStore);

            // Deleting the node
            globalAllocator.delOctreeNode(node, true, true);
            globalVariables.packedNodes[node_index] = nullptr;

        } else {
            globalVariables.setFlagSync(node->aabb_index, CFlagIsInUpdatesCache);
        }
    }
}



/// Run on floor("NB SMs" * "Max threads per SM" / "Max threads per block") blocks of size "Max threads per block"
extern "C" __global__
void kernel_prepare_store_part_2_resetting_children(){
    if(!globalVariables.isInitialised){return;}
    if(!globalVariables.isUpdating){return;}
    if(!globalVariables.isDoneLoading){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    for(uint32_t node_index = thread_id; node_index < globalVariables.curNbNodes; node_index += nb_threads){
        COctreeNode* node = globalVariables.packedNodes[node_index];
        if(!node){continue;} // Skip if this is one of the node that has just been removed
        node->children_visibility = 0b00000000;

        for(uint32_t i=0; i<8; i++){
            CIdAABB child_index = globalVariables.relationshipMap[node->aabb_index].children[i];

            if(child_index == CINVALID_ID){
                continue;
            }
            // if(globalVariables.isToStore(child_index)){
            if(!globalVariables.isInUpdatesCache(child_index)){
                node->children[i] = nullptr;
                continue;
            }
            node->children_visibility |= ((0x01) << i);
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
        packNodes();
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
                if(node->children[i]){node->children[i]->level = new_child_level;}
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

    globalVariables.batchesToAddBottomUpCount = false;

    if(globalVariables.isDoneLoading
        && globalVariables.isDoneStoring // To run only on complete updates
        && block_id == 0 && thread_id == 0 // To run only once
    ){
        // UI values
        __nv_atomic_add(&globalVariables.nbTotalUpdates, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    }
}











// __device__
// void deallocateOldNodesV2(uint32_t first_point, uint32_t step){
//     for(uint32_t node_index = first_point; node_index < globalVariables.renderingNbNodes; node_index += step){
//         COctreeNode* node_to_remove = globalVariables.renderingPackedNodes[node_index];
//         globalAllocator.delOctreeNode(node_to_remove, true, true, false);
//     }
// }

// __device__
// void createNewNodesV2(uint32_t first_point, uint32_t step){
//     for(uint32_t node_index = first_point; node_index < globalVariables.curNbNodes; node_index += step){
//         COctreeNode* real_node = globalVariables.packedNodes[node_index];
//         globalVariables.renderingPackedNodes[node_index] = globalAllocator.newOctreeNodePartialCpy(real_node, true, false);
//         COctreeNode* node = globalVariables.renderingPackedNodes[node_index];
//         node->level = real_node->level;
//         node->points_counter = real_node->points_counter;
//         node->voxels_counter = real_node->voxels_counter;
//         node->children_visibility = real_node->children_visibility;

//         globalVariables.unsetFlagSync(node->aabb_index, CFlagHasNewPoints);
//         globalVariables.unsetFlagSync(node->aabb_index, CFlagHasNewVoxels);
//         globalVariables.unsetFlagSync(node->aabb_index, CFlagIsNew);
//         globalVariables.unsetFlagSync(node->aabb_index, CFlagHasSpilled);
//     }
// }


// __device__
// void deallocateOldNodes(uint32_t first_point, uint32_t step){
//     for(uint32_t node_index = first_point; node_index < globalVariables.renderingNbNodes; node_index += step){
//         COctreeNode* node_to_remove = globalVariables.renderingPackedNodes[node_index];
//         if(!node_to_remove){continue;}

//         // Node should be deleted
//         if(!globalVariables.isInUpdatesCache(node_to_remove->aabb_index)){
//             globalAllocator.delOctreeNode(node_to_remove, true, true, false);
//             globalVariables.renderingPackedNodes[node_index] = nullptr;
//             continue;
//         }

//         // Node points should be deleted
//         if(globalVariables.hasSpilled(node_to_remove->aabb_index)){
//             globalAllocator.delChunk(node_to_remove->points, true, false);
//         }
//     }
// }





// __device__
// void createNewNodes(uint32_t first_point, uint32_t step){
//     for(uint32_t node_index = first_point; node_index < globalVariables.curNbNodes; node_index += step){
//         COctreeNode* node = globalVariables.packedNodes[node_index];
//         if(globalVariables.isNew(node->aabb_index)){
//             uint32_t index = __nv_atomic_fetch_add(&globalVariables.renderingNbNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
//             globalVariables.renderingPackedNodes[index] = globalAllocator.newOctreeNodePartialCpy(node, true, false);
//         }
//     }
// }

// __device__
// void packRenderingNodes(uint32_t first_point, uint32_t step){
//     for(uint32_t rendering_node_index = first_point; rendering_node_index < globalVariables.renderingNbNodes; rendering_node_index += step){
//         COctreeNode* rendering_node = globalVariables.renderingPackedNodes[rendering_node_index];
//         globalVariables.renderingPackedNodes[rendering_node_index] = nullptr;
//         if(!rendering_node){continue;}
//         const CIdAABB& current_node = rendering_node->aabb_index;
//         bool found = false;
//         for(uint32_t node_index = 0; node_index < globalVariables.curNbNodes; node_index++){
//             const CIdAABB& target = globalVariables.packedNodes[node_index]->aabb_index;
//             if(current_node != target){continue;}
//             globalVariables.renderingPackedNodesTmp[node_index] = rendering_node;
//             found = true;
//             break;
//         }
//         if(!found){
//             printf("Failed to find a corresponding rendering node on packing\n");
//             customAssert();
//         }
//     }
// }


// __device__
// void updateRenderingNodes(uint32_t first_point, uint32_t step){
//     for(uint32_t node_index = first_point; node_index < globalVariables.renderingNbNodes; node_index += step){
//         globalVariables.renderingPackedNodes[node_index] = globalVariables.renderingPackedNodesTmp[node_index];   

//         COctreeNode* node = globalVariables.renderingPackedNodes[node_index];
//         COctreeNode* real_node = globalVariables.packedNodes[node_index];
//         node->level = real_node->level;
//         node->points_counter = real_node->points_counter;
//         node->voxels_counter = real_node->voxels_counter;
//         node->children_visibility = real_node->children_visibility;

//         // Copy the points if necessary
//         if(globalVariables.hasSpilled(node->aabb_index)){
//             node->points = globalAllocator.newChunkPartialCpy(real_node->points, true, false);
//         } else if(globalVariables.hasNewPoints(node->aabb_index)){
//             if(!node->points){
//                 node->points = globalAllocator.newChunk(true, false);
//             }
//             CChunk* real_node_points = real_node->points;
//             CChunk* cur_node_points = node->points;
//             while(real_node_points){
//                 for(uint32_t i=cur_node_points->size; i<real_node_points->size; i++){
//                     cur_node_points->points[i] = real_node_points->points[i];
//                 }
//                 cur_node_points->size = real_node_points->size;
//                 if(real_node_points->next && !cur_node_points->next){
//                     cur_node_points->next = globalAllocator.newChunk(true, false);
//                 }
//                 real_node_points = real_node_points->next;
//                 cur_node_points = cur_node_points->next;
//             }
//         }
//         // Copy the voxels if necessary
//         if(globalVariables.hasNewVoxels(node->aabb_index)){
//             if(!node->voxels){
//                 node->voxels = globalAllocator.newChunk(true, false);
//             }
//             CChunk* real_node_voxels = real_node->voxels;
//             CChunk* cur_node_voxels = node->voxels;
//             while(real_node_voxels){
//                 for(uint32_t i=cur_node_voxels->size; i<real_node_voxels->size; i++){
//                     cur_node_voxels->points[i] = real_node_voxels->points[i];
//                 }
//                 cur_node_voxels->size = real_node_voxels->size;
//                 if(real_node_voxels->next && !cur_node_voxels->next){
//                     cur_node_voxels->next = globalAllocator.newChunk(true, false);
//                 }
//                 real_node_voxels = real_node_voxels->next;
//                 cur_node_voxels = cur_node_voxels->next;
//             }
//         }

//         globalVariables.unsetFlagSync(node->aabb_index, CFlagHasNewPoints);
//         globalVariables.unsetFlagSync(node->aabb_index, CFlagHasNewVoxels);
//         globalVariables.unsetFlagSync(node->aabb_index, CFlagIsNew);
//         globalVariables.unsetFlagSync(node->aabb_index, CFlagHasSpilled);
//     }
// }


// /// Run on "MaxActiveBlocksPerMultiprocessor" cooperative blocks of size "Max block size"
// extern "C" __global__
// void kernel_create_rendereable_octree(){
//     if(!globalVariables.isInitialised){return;}
//     if(!globalVariables.isUpdating){return;}

//     auto grid = cg::this_grid();
//     auto block = cg::this_thread_block();
//     uint32_t nb_blocks = grid.num_blocks();

//     uint32_t block_id = grid.block_rank();
//     uint32_t thread_id = block.thread_rank();
//     uint32_t nb_threads_per_block = block.num_threads();

//     uint32_t first_point = block_id * nb_threads_per_block + thread_id;
//     uint32_t step = nb_blocks * nb_threads_per_block;

//     bool is_first = (block_id == 0 && thread_id == 0);

//     // Deallocate old rendering nodes and old chunks
//     deallocateOldNodes(first_point, step);
//     // deallocateOldNodesV2(first_point, step);
//     grid.sync();


//     if(is_first){
//         // Reset allocator
//         globalAllocator.chunksAllocator->reset_temporary_deallocations();
//         globalAllocator.nodesAllocator->reset_temporary_deallocations();
//     }
//     grid.sync();
    

//     // Create new nodes
//     createNewNodes(first_point, step);
//     // createNewNodesV2(first_point, step);
//     grid.sync();


//     // Make nodes match the real nodes order
//     packRenderingNodes(first_point, step);
//     grid.sync();


//     // Finish building new nodes
//     globalVariables.renderingNbNodes = globalVariables.curNbNodes;
//     updateRenderingNodes(first_point, step);
//     grid.sync();


//     if(is_first){
//         // Reset the allocator
//         // Because "newOctreeNodePartialCpy" was called
//         globalAllocator.chunksAllocator->reset_temporary_allocations();
//         globalAllocator.nodesAllocator->reset_temporary_allocations();
//         // No grids should be allocated

//         globalVariables.renderingOctreeDepth = globalVariables.octreeDepth;       

//         // UI values
//         __nv_atomic_add(&globalVariables.nbTotalUpdates, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
//     }

    // // Sanity check
    // grid.sync();
    // for(uint32_t node_index = first_point; node_index < globalVariables.renderingNbNodes; node_index += step){
    //     COctreeNode* node = globalVariables.renderingPackedNodes[node_index];
    //     COctreeNode* real_node = globalVariables.packedNodes[node_index];
        
    //     CChunk* node_points = node->points;
    //     CChunk* real_node_points = real_node->points;
    //     if((node_points && !real_node_points) || (!node_points && real_node_points)){
    //         printf("Non existing points\n");
    //         customAssert();
    //     }
    //     while(node_points){
    //         if(node_points->size != real_node_points->size){
    //             printf("Wrong points chunk size\n");
    //             customAssert();
    //         }
    //         for(uint32_t point_id = 0; point_id < node_points->size; point_id++){
    //             if(node_points->points[point_id] != real_node_points->points[point_id]){
    //                 printf("Wrong point\n");
    //                 customAssert();
    //             }
    //         }
    //         node_points = node_points->next;
    //         real_node_points = real_node_points->next;
    //     }

    //     CChunk* node_voxels = node->voxels;
    //     CChunk* real_node_voxels = real_node->voxels;
    //     if((node_voxels && !real_node_voxels) || (!node_voxels && real_node_voxels)){
    //         printf("Non existing voxels\n");
    //         customAssert();
    //     }
    //     while(node_voxels){
    //         if(node_voxels->size != real_node_voxels->size){
    //             printf("Wrong voxels chunk size\n");
    //             customAssert();
    //         }
    //         for(uint32_t point_id = 0; point_id < node_voxels->size; point_id++){
    //             if(node_voxels->points[point_id] != real_node_voxels->points[point_id]){
    //                 printf("Wrong point\n");
    //                 customAssert();
    //             }
    //         }
    //         node_voxels = node_voxels->next;
    //         real_node_voxels = real_node_voxels->next;
    //     }

    //     if(node->aabb != real_node->aabb){
    //         printf("Wrong AABB\n");
    //         customAssert();
    //     }
    //     if(node->aabb_index != real_node->aabb_index){
    //         printf("Wrong index\n");
    //         customAssert();
    //     }
    // }
// }
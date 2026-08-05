#include "utils.cuh"


__device__ 
void fillUpdatesCacheRecursive(COctreeNode* cur_node){
    if(!cur_node){return;}
    for(uint32_t i=0; i<8; i++){fillUpdatesCacheRecursive(cur_node->children[i]);}
    if(globalVariables.isUpdated(cur_node->aabb_index)){
        globalVariables.unsetFlag(cur_node->aabb_index, CFlagIsUpdated);
        globalVariables.updatesCache->add(cur_node->aabb_index);
    }
}

/// Run on a single thread
extern "C" __global__
void kernel_update_updates_cache(){
    if(!globalVariables.isInitialised){return;}

    globalVariables.nbNodesExchanged = 0;
    fillUpdatesCacheRecursive(globalVariables.mainOctree);
}


/// Run on min("curNbNodes", "NB SMs" * "Max threads per SM") blocks of size 1
extern "C" __global__
void kernel_prepare_store_part_1_filling_buffers(){
    if(!globalVariables.isInitialised){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    for(uint32_t node_index = thread_id; node_index < globalVariables.curNbNodes; node_index += nb_threads){

        COctreeNode* node = globalVariables.packedNodes[node_index];

        if(!globalVariables.updatesCache->contains(node->aabb_index)
            && !globalVariables.visibilityCache->contains(node->aabb_index)    
        ){
            uint32_t exchanged_index = __nv_atomic_fetch_add(&globalVariables.nbNodesExchanged, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);

            if(exchanged_index >= globalVariables.maxNbNodesExchanged){
                printf("ERROR: Too many nodes are being stored, skipping this one\n");
                return;
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

            // Do not require to be synced because it's the only flag type changed in the kernel
            globalVariables.setFlag(node->aabb_index, CFlagToStore);
            globalAllocator.delOctreeNode(node, true, true);
            globalVariables.packedNodes[node_index] = nullptr;
        } else {
            globalVariables.unsetFlag(node->aabb_index, CFlagToStore);
        }
    }
}



/// Run on min("curNbNodes", "NB SMs" * "Max threads per SM") blocks of size 1
extern "C" __global__
void kernel_prepare_store_part_2_resetting_children(){
    if(!globalVariables.isInitialised){return;}

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
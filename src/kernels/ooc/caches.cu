#include "utils.cuh"


__device__ 
void fillUpdatesCacheRecursive(COctreeNode* cur_node){
    if(!cur_node){return;}
    for(uint32_t i=0; i<8; i++){fillUpdatesCacheRecursive(cur_node->children[i]);}
    if(cur_node->updated){globalVariables.updatesCache->add(cur_node->aabb_index);}
}

/// Run on a single thread
extern "C" __global__
void kernel_update_updates_cache(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}
    fillUpdatesCacheRecursive(globalVariables.mainOctree);
}

/// Run on "maxNbAABBs" threads
extern "C" __global__
void kernel_prepare_store_part_1(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
	if(thread_id >= globalVariables.maxNbAABBs){return;}

    COctreeNode* node = globalVariables.nodes[thread_id];
    if(!node){return;}

    if(!globalVariables.updatesCache->contains(node->aabb_index)
        && !globalVariables.visibilityCache->contains(node->aabb_index)    
    ){
        uint32_t index = __nv_atomic_fetch_add(&globalVariables.nbNodesToStore, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);

        if(index >= globalVariables.maxNbNodesReceived){
            printf("Too many nodes are being stored, skipping this one\n");
            return;
        }

        globalVariables.receivedAABBIndices[index] = node->aabb_index;
        globalVariables.receivedChildrenIds[index] = node->children_ids;
        globalVariables.receivedPointsCounters[index] = node->points_counter;
        globalVariables.receivedVoxelsCounters[index] = node->voxels_counter;
        
        const uint32_t MAX_NB_POINTS = globalVariables.maxNbPointsChunksPerReceivedNode * OocSimLodSettings::NB_POINTS_PER_CHUNK;
        const uint32_t MAX_NB_VOXELS = globalVariables.maxNbVoxelsChunksPerReceivedNode * OocSimLodSettings::NB_POINTS_PER_CHUNK;
        
        CChunk* cur_chunk = node->points;
        uint32_t cur_point_index = 0;
        while(cur_chunk){
            for(uint32_t i=0; i<cur_chunk->size; i++){
                if(cur_point_index >= MAX_NB_POINTS){
                    printf("Too many points in the node, some will be skipped to store it: index %d / %d\n",
                        cur_point_index, MAX_NB_POINTS
                    );
                    break;
                }
                globalVariables.receivedPoints[index][cur_point_index] = cur_chunk->points[i];
                cur_point_index++;
            }
            cur_chunk = cur_chunk->next;
        }

        cur_chunk = node->voxels;
        cur_point_index = 0;
        while(cur_chunk){
            for(uint32_t i=0; i<cur_chunk->size; i++){
                if(cur_point_index >= MAX_NB_VOXELS){
                    printf("Too many voxels in the node, some will be skipped to store it: index %d / %d\n",
                        cur_point_index, MAX_NB_VOXELS
                    );
                    break;
                }
                globalVariables.receivedVoxels[index][cur_point_index] = cur_chunk->points[i];
                cur_point_index++;
            }
            cur_chunk = cur_chunk->next;
        }

        globalVariables.nodesFlags[node->aabb_index] |= (1u << CFlagToStore);
    }

}

/// Run on "maxNbAABBs" threads
extern "C" __global__
void kernel_prepare_store_part_2(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
	if(thread_id >= globalVariables.maxNbAABBs){return;}

    COctreeNode* node = globalVariables.nodes[thread_id];
    if(!node){return;}

    for(uint32_t i=0; i<8; i++){
        COctreeNode* child = node->children[i];
        if(child && (globalVariables.nodesFlags[child->aabb_index] & (1u << CFlagToStore))){
            node->children[i] = nullptr;
        }
    }
}

/// Run on "maxNbAABBs" threads
extern "C" __global__
void kernel_prepare_store_part_3(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
	if(thread_id >= globalVariables.maxNbAABBs){return;}

    if(globalVariables.nodesFlags[thread_id] & (1u << CFlagToStore)){
        globalAllocator.delOctreeNode(globalVariables.nodes[thread_id], true);
        globalVariables.nodes[thread_id] = nullptr;
        globalVariables.nodesFlags[thread_id] &= (0u << CFlagToStore);
    }

}

/// Run on a single thread
extern "C" __global__
void kernel_prepare_store_part_4(){
    globalVariables.nbNodesToStore = 0;
    
    // Because "delOctreeNode" was called in simlodSplit
    globalAllocator.chunksAllocator->reset_temporary_deallocations();
    globalAllocator.gridsAllocator->reset_temporary_deallocations();
    globalAllocator.nodesAllocator->reset_temporary_deallocations();
}
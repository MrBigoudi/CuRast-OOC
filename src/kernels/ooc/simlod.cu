#include "utils.cuh"

/// Prepare the nodes that need to be loaded
/// TODO:
/// Later, if a point is contained in a leaf that was previously stored, put the point aside
/// For now, load them directly after this kernel launch
extern "C" __global__
void kernel_simlod_load_part_1(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();

    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

        if(thread_id > nb_new_points){continue;}
        CPoint& point = new_points[thread_id];

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
                cur_aabb_index = child_aabb_index;
                globalVariables.nodesFlags[cur_aabb_index] |= (1u << CFlagToLoad);
            } else {
                break;
            }
        }
    }
}


/// Run on a single thread
extern "C" __global__
void kernel_simlod_load_part_2(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    globalVariables.nbNodesToLoad = 0;
    for(uint32_t i=0; i<globalVariables.nbAABBs; i++){
        if(globalVariables.nodesFlags[i] & (0u | 1u << CFlagToLoad)){
            globalVariables.nodesToLoadBuffer[globalVariables.nbNodesToLoad] = (CIdAABB)i;
            globalVariables.nbNodesToLoad++;
            globalVariables.nodesFlags[i] &= (0u << CFlagToLoad);
        }
    }
}


/// Run on "nb nodes received" threads
/// Load the newly received nodes
extern "C" __global__
void kernel_simlod_load_part_3(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    uint32_t thread_id = cg::this_grid().thread_rank();
	if(thread_id >= globalVariables.nbNodesReceived){return;}

    CIdAABB aabb_index = globalVariables.receivedAABBIndices[thread_id];
    uint8_t children_ids = globalVariables.receivedChildrenIds[thread_id];
    uint32_t nb_points = globalVariables.receivedPointsCounters[thread_id];
    uint32_t nb_voxels = globalVariables.receivedVoxelsCounters[thread_id];
    CPoint* points = globalVariables.receivedPoints[thread_id];
    CPoint* voxels = globalVariables.receivedVoxels[thread_id];

    COctreeNode* loaded_node = globalAllocator.newOctreeNode(aabb_index);
    loaded_node->children_ids = children_ids;
    loaded_node->counter = nb_points;
    loaded_node->updated = true;

    // Rebuild points
    if(nb_points > 0){
        loaded_node->points = globalAllocator.newChunk();
        CChunk* cur_chunk = loaded_node->points;
        for(uint32_t i=0; i<nb_points; i++){
            addPointToChunk(cur_chunk, points[i]);
        }
    }

    // Rebuild voxels
    if(nb_voxels > 0){
        loaded_node->voxels = globalAllocator.newChunk();
        CChunk* cur_chunk = loaded_node->voxels;
        for(uint32_t i=0; i<nb_points; i++){
            addPointToChunk(cur_chunk, points[i]);
        }
    }

    // Rebuild occupancy
    // TODO:

    globalVariables.nodes[aabb_index] = loaded_node;
}

/// Run on "maxNbAABBs" threads
/// Rebuild the relationships of the loaded nodes
extern "C" __global__
void kernel_simlod_load_part_4(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    uint32_t thread_id = cg::this_grid().thread_rank();
	if(thread_id >= globalVariables.maxNbAABBs){return;}

    const CGlobalVariables::Relationship& relationship = globalVariables.relationshipMap[thread_id];
    COctreeNode* cur_node = globalVariables.nodes[thread_id];
    for(uint32_t i=0; i<8; i++){
        const CIdAABB& child_aabb = relationship.children[i];
        cur_node->children[i] = globalVariables.nodes[child_aabb];
    }
}
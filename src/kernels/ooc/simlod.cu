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
        if(globalVariables.nodesFlags[i] & (1u << CFlagToLoad)){
            globalVariables.nodesToLoadBuffer[globalVariables.nbNodesToLoad] = (CIdAABB)i;
            globalVariables.nbNodesToLoad++;
            globalVariables.nodesFlags[i] &= (0u << CFlagToLoad);
        }
    }
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
    uint8_t children_ids = globalVariables.receivedChildrenIds[thread_id];
    uint32_t nb_points = globalVariables.receivedPointsCounters[thread_id];
    uint32_t nb_voxels = globalVariables.receivedVoxelsCounters[thread_id];
    CPoint* points = globalVariables.receivedPoints[thread_id];
    CPoint* voxels = globalVariables.receivedVoxels[thread_id];

    // TODO: handle sync in allocator
    COctreeNode* loaded_node = globalAllocator.newOctreeNode(aabb_index);
    loaded_node->children_ids = children_ids;
    loaded_node->counter = nb_points;
    loaded_node->updated = true;

    // Rebuild points
    if(nb_points > 0){
        // TODO: handle sync in allocator
        loaded_node->points = globalAllocator.newChunk();
        CChunk* cur_chunk = loaded_node->points;
        for(uint32_t i=0; i<nb_points; i++){
            addPointToChunk(cur_chunk, points[i]);
        }
    }

    // Rebuild voxels
    if(nb_voxels > 0){
        // TODO: handle sync in allocator
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












__device__
void simlodCount(uint32_t thread_id, uint32_t nb_threads){
    auto updateLeafCounter = [&](CPoint& point){
        // Find the leaf containing the current point
        COctreeNode* leaf = globalVariables.mainOctree;
        uint8_t level = 1;

        while(true){
            const CAABB& aabb = getAABB(leaf->aabb_index);
            CNodePosition child_position = aabb.getNextChildIndex(point.position);

            if(leaf->children[child_position]){
                leaf = leaf->children[child_position];
                if(level == UINT8_MAX){
					printf("The octree has reached it's maximum depth size...");
					// TODO: panic
				}
                level++;
            } else {
                leaf->children_ids |= (1u << child_position);
                // Skip if the point was already accepted at this level
				if(point.getAlpha() == level){return;}
				// Flag point as accepted at this level
				point.setAlpha(level);

                // Flag the leaf as spilling
                uint32_t old_counter = __nv_atomic_fetch_add(&leaf->counter, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
				if(old_counter == 65'536){ // TODO: load from settings
					globalVariables.nodesFlags[leaf->aabb_index] |= (1u << CFlagIsSpilling);
                    globalVariables.hasSpillingNodes = true;
				}

                return;
            }
        }
    };

    // Count new points
    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

        for(uint32_t i=thread_id; i<nb_new_points; i+=nb_threads){
            CPoint& point = new_points[thread_id];
            updateLeafCounter(point);
        }
    }

    // Count spilled points
    for(uint32_t i=thread_id; i<globalVariables.nbSpilledPoints; i+=nb_threads){
        CPoint& point = globalVariables.spilledPoints[i];
        updateLeafCounter(point);
    }
}


__device__
void simlodSplit(uint32_t thread_id, uint32_t nb_threads){
    if(thread_id > globalVariables.maxNbAABBs){return;}
    if(globalVariables.nodesFlags[thread_id] & (1u << CFlagIsSpilling)){
        COctreeNode* spilling_node = globalVariables.nodes[thread_id];
		uint8_t spilling_node_children = spilling_node->children_ids;

		spilling_node->counter = 0;
		spilling_node->children_ids = 0;

		if(!spilling_node->occupancy){
            // TODO: handle sync in allocator
			spilling_node->occupancy = globalAllocator.newOccupancyGrid();
		}


        // TODO: rethink split

		for(uint32_t j=0; j<8; j++){
			// Create necessary empty children
			bool can_be_spilled = (1u << j) & spilling_node_children;
			if(!spilling_node->children[j] && can_be_spilled){
				// Create the new AABB
				CAABB child_aabb = CAABB(getAABB(spilling_node->aabb_index));
				child_aabb.shrink((CNodePosition)j);
			}
		}



        globalVariables.nodesFlags[thread_id] &= (0u << CFlagIsSpilling);
    }
}



/// Run on "maxNbAABBs" threads
/// Update the nodes counters
extern "C" __global__
void kernel_simlod_count_split(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    while(true){
        simlodCount(thread_id, nb_threads);
        cg::sync(grid);
        if(!globalVariables.hasSpillingNodes){break;}

        simlodSplit(thread_id, nb_threads);
        globalVariables.hasSpillingNodes = false;
        cg::sync(grid);
    }
}
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

        if(thread_id >= nb_new_points){continue;}
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

    // Reset previous simlod values
    globalVariables.nbNodesReceived = 0;
    globalVariables.nbNodesToLoad = 0;
    globalVariables.nbSpilledPoints = 0;
    globalVariables.nbBacklogVoxels = 0;
}


/// Run on "maxNbAABBs" threads
extern "C" __global__
void kernel_simlod_load_part_2(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
	if(thread_id >= globalVariables.maxNbAABBs){return;}

    // Count nodes to load
    if(globalVariables.nodesFlags[thread_id] & (1u << CFlagToLoad)){
        uint32_t index = __nv_atomic_fetch_add(&globalVariables.nbNodesToLoad, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        globalVariables.nodesToLoadBuffer[index] = (CIdAABB)thread_id;
        globalVariables.nodesFlags[thread_id] &= (0u << CFlagToLoad);
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
    uint32_t children_ids = globalVariables.receivedChildrenIds[thread_id];
    // uint8_t children_ids = globalVariables.receivedChildrenIds[thread_id];
    uint32_t nb_points = globalVariables.receivedPointsCounters[thread_id];
    uint32_t nb_voxels = globalVariables.receivedVoxelsCounters[thread_id];
    CPoint* points = globalVariables.receivedPoints[thread_id];
    CPoint* voxels = globalVariables.receivedVoxels[thread_id];

    COctreeNode* loaded_node = globalAllocator.newOctreeNode(aabb_index, true);
    globalVariables.nodes[aabb_index] = loaded_node;
    loaded_node->children_ids = children_ids;
    loaded_node->points_counter = nb_points;
    loaded_node->updated = true;

    // Rebuild points
    if(nb_points > 0){
        loaded_node->points = globalAllocator.newChunk(true);
        CChunk* cur_chunk = loaded_node->points;
        for(uint32_t i=0; i<nb_points; i++){
            addPointToChunk(cur_chunk, points[i], true);
        }
    }

    // Rebuild voxels
    if(nb_voxels > 0){
        // Rebuild occupancy
        loaded_node->occupancy = globalAllocator.newOccupancyGrid(true);

        loaded_node->voxels = globalAllocator.newChunk(true);
        CChunk* cur_chunk = loaded_node->voxels;
        for(uint32_t i=0; i<nb_voxels; i++){
            addPointToChunk(cur_chunk, voxels[i], true);

            // Sample voxel occupancy grid at this location
            const CAABB& aabb = getAABB(aabb_index);
            COccupancyGrid::GridIndex index = COccupancyGrid::getCellIndices(aabb, voxels[i]);
            loaded_node->occupancy->markCellAsFilled(index);
        }
    }

}

/// Run on "maxNbAABBs" threads
/// Rebuild the relationships of the loaded nodes
extern "C" __global__
void kernel_simlod_load_part_4(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();
	if(thread_id >= globalVariables.maxNbAABBs){return;}

    COctreeNode* cur_node = globalVariables.nodes[thread_id];
    if(!cur_node){return;}
    const CGlobalVariables::Relationship& relationship = globalVariables.relationshipMap[thread_id];

    for(uint32_t i=0; i<8; i++){
        const CIdAABB& child_aabb = relationship.children[i];
        if(child_aabb != CINVALID_ID){
            cur_node->children[i] = globalVariables.nodes[child_aabb];
        }
    }

    if(thread_id==0){
        // Because "newChunk" was called in part 3
        globalAllocator.chunksAllocator->reset_temporary_allocations();
    } 
    
    // try to avoid being on the same warp
    if((nb_threads >= 32 && thread_id==32) || (nb_threads < 32 && thread_id==0)){
        // Because "newOctreeNode" was called in part 3
        globalAllocator.nodesAllocator->reset_temporary_allocations();
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
                __nv_atomic_or(&leaf->children_ids, (1u << child_position), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                // Skip if the point was already accepted at this level
				if(point.getAlpha() == level){return;}
				// Flag point as accepted at this level
				point.setAlpha(level);

                // Flag the leaf as spilling
                uint32_t max_points_per_leaf = 65'536; // TODO: load MAX_POINTS_PER_LEAF from settings
                uint32_t old_counter = __nv_atomic_fetch_add(&leaf->points_counter, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
				if(old_counter == max_points_per_leaf){ 
                    // No need for atomic here because lhs is always the same
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
            CPoint& point = new_points[i];
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
    if(thread_id >= globalVariables.maxNbAABBs){return;}

    for(uint32_t i=thread_id; i<globalVariables.maxNbAABBs; i+=nb_threads){
        if(globalVariables.nodesFlags[i] & (1u << CFlagIsSpilling)){
            COctreeNode* spilling_node = globalVariables.nodes[i];
            CGlobalVariables::Relationship& relationship = globalVariables.relationshipMap[i];
            uint32_t spilling_node_children = spilling_node->children_ids;
            // uint8_t spilling_node_children = spilling_node->children_ids;

            spilling_node->points_counter = 0;
            spilling_node->children_ids = 0;

            if(!spilling_node->occupancy){
                spilling_node->occupancy = globalAllocator.newOccupancyGrid(true);
            }

            for(uint32_t j=0; j<8; j++){
                // Create necessary empty children
                bool can_be_spilled = (1u << j) & spilling_node_children;
                if(!spilling_node->children[j] && can_be_spilled){
                    // Create the new AABB
                    CAABB child_aabb = CAABB(getAABB(spilling_node->aabb_index));
                    child_aabb.shrink((CNodePosition)j);
                    
                    CIdAABB id = createNewAABB(child_aabb);
                    COctreeNode* new_child = globalAllocator.newOctreeNode(id, true);
                    spilling_node->children[j] = new_child;
                    globalVariables.nodes[id] = new_child;
                    relationship.children[j] = id;
                }
            }

            CChunk* current_chunk = spilling_node->points;
            if(current_chunk){
                while(current_chunk){
                    for(uint32_t j=0; j<current_chunk->size; j++){
                        uint32_t index = __nv_atomic_fetch_add(&globalVariables.nbSpilledPoints, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                        // Flag the point as not accepted
                        current_chunk->points[j].resetAlpha();
                        globalVariables.spilledPoints[index] = current_chunk->points[j];
                        globalVariables.spillingNodes[index] = spilling_node;
                    }
                    current_chunk = current_chunk->next;
                }

                globalAllocator.delChunk(spilling_node->points, true);
                spilling_node->points = nullptr;
            }

            globalVariables.nodesFlags[thread_id] &= (0u << CFlagIsSpilling);
        }
    }
}



/// Run on "total nb SMs" * "nb block per SM" threads
/// Run on a grid of size "total nb SMs" * "nb block per SM" and blocks of size 1
/// Update the nodes counters
extern "C" __global__
void kernel_simlod_count_split(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    uint32_t cpt = 0;
    while(true){
        cpt++;

        simlodCount(thread_id, nb_threads);
        grid.sync();

        // // TODO: temporary to remove
        // {
        //     if(thread_id == 0){
        //         printf("\n\n\n\n\nSimlod count: it %d\n\n", cpt);
        //         displayOctreeNode(globalVariables.mainOctree);
        //         printf("\n\n");
        //     }
        //     grid.sync();
        // }

        if(!globalVariables.hasSpillingNodes){break;}

        simlodSplit(thread_id, nb_threads);
        globalVariables.hasSpillingNodes = false;
        grid.sync();

        // // TODO: temporary to remove
        // {
        //     if(thread_id == 0){
        //         printf("\n\n\n\n\nSimlod split: it %d\n\n", cpt);
        //         displayOctreeNode(globalVariables.mainOctree);
        //         printf("\n\n");
        //     }
        //     grid.sync();
        // }

        if(thread_id == 0){
            // Because "delChunk" was called in simlodSplit
            globalAllocator.chunksAllocator->reset_temporary_deallocations();
        }
        // try to avoid being on the same warp
        if((nb_threads >= 32 && thread_id==32) || (nb_threads < 32 && thread_id==0)){
            // Because "newOctreeNode" was called in simlodSplit
            globalAllocator.nodesAllocator->reset_temporary_allocations();
        }
        // try to avoid being on the same warp
        if((nb_threads >= 64 && thread_id==64) || (nb_threads < 64 && thread_id==0)){
            // Because "newOccupancyGrid" was called in simlodSplit
            globalAllocator.gridsAllocator->reset_temporary_allocations();
        }

        grid.sync();
    }
}

























/// Run on "maxPointsPerBatches" threads
extern "C" __global__
void kernel_simlod_voxel_sampling(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    auto sampleVoxel = [&](const CPoint& point){
        COctreeNode* cur_node = globalVariables.mainOctree;

        while(true){
            if(!cur_node->occupancy){return;}

            // Find next child
            const CAABB& aabb = getAABB(cur_node->aabb_index);
            CNodePosition child_position = aabb.getNextChildIndex(point.position);
            if(!cur_node->children[child_position]){return;}

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
                uint32_t index = __nv_atomic_fetch_add(&globalVariables.nbBacklogVoxels, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                
                // TODO: find better checks everywhere
                if(index >= globalVariables.maxNbBacklogVoxels){
                    printf("Max nb backlog buffer reached: i = %d / %d\n", index, globalVariables.maxNbBacklogVoxels);
                }

                globalVariables.backlogVoxels[index] = new_voxel;
                globalVariables.backlogVoxelsNodes[index] = cur_node;
			}

			cur_node = cur_node->children[child_position];
        }
    };

    // Sample voxels for new points
    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

        for(uint32_t i=thread_id; i<nb_new_points; i+=nb_threads){
            CPoint& point = new_points[i];
            sampleVoxel(point);
        }
    }

    // Sample voxels for spilled points
    for(uint32_t i=thread_id; i<globalVariables.nbSpilledPoints; i+=nb_threads){
        CPoint& point = globalVariables.spilledPoints[i];
        sampleVoxel(point);
    }
}
























/// Run on "maxNbAABBs" threads
/// Allocate the necessary new chunks
extern "C" __global__
void kernel_simlod_insertion_part_1(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();
	if(thread_id >= globalVariables.maxNbAABBs){return;}

    COctreeNode* cur_node = globalVariables.nodes[thread_id];
    if(!cur_node){return;}

    auto allocateChunks = [&](CChunk* root_chunk, uint32_t required_chunks, uint32_t total_counter){
        CChunk* cur_chunk = root_chunk;
        for(uint32_t i=1; i<required_chunks; i++){
            if(!cur_chunk->next){cur_chunk->next = globalAllocator.newChunk(true);}
            cur_chunk->size = OocSimLodSettings::NB_POINTS_PER_CHUNK;
            cur_chunk = cur_chunk->next;
        }
        if(cur_chunk){cur_chunk->size = total_counter % OocSimLodSettings::NB_POINTS_PER_CHUNK;}
    };

    // Allocate points chunks
    uint32_t required_chunks = (cur_node->points_counter + OocSimLodSettings::NB_POINTS_PER_CHUNK - 1) /  OocSimLodSettings::NB_POINTS_PER_CHUNK;
    if(required_chunks > 0){
        if(!cur_node->points){
            cur_node->points = globalAllocator.newChunk(true);
        }
        allocateChunks(cur_node->points, required_chunks, cur_node->points_counter);
    }


    // Allocate voxels chunks
    required_chunks = (cur_node->voxels_counter + OocSimLodSettings::NB_POINTS_PER_CHUNK - 1) /  OocSimLodSettings::NB_POINTS_PER_CHUNK;
    if(required_chunks > 0){
        if(!cur_node->voxels){
            cur_node->voxels = globalAllocator.newChunk(true);
        }
        allocateChunks(cur_node->voxels, required_chunks, cur_node->voxels_counter);
    }


    cur_node->points_stored = 0;
    cur_node->voxels_stored = 0;
}



/// Run on "maxPointsPerBatches" threads
extern "C" __global__
void kernel_simlod_insertion_part_2(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    auto insertPoint = [&](const CPoint& point){
		COctreeNode* cur_node = globalVariables.mainOctree;
		// Reach all corresponding leaves
		while(true){
			cur_node->updated = true;
			// Find next child
			const CAABB& aabb = getAABB(cur_node->aabb_index);
            CNodePosition child_position = aabb.getNextChildIndex(point.position);

			// If leaf insert point in chunks
			if(cur_node->children[child_position]){
				cur_node = cur_node->children[child_position];
			} else {
				uint32_t point_index = __nv_atomic_fetch_add(&cur_node->points_stored, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                uint32_t chunk_index = point_index / OocSimLodSettings::NB_POINTS_PER_CHUNK;
                CChunk* cur_chunk = cur_node->points;
                for(uint32_t i=0; i<chunk_index; i++){
                    cur_chunk = cur_chunk->next;
                }
                uint32_t real_index = point_index % OocSimLodSettings::NB_POINTS_PER_CHUNK;
                cur_chunk->points[real_index] = point;
                return;
			}
		}
	};

    auto insertVoxel = [&](const CPoint& voxel, COctreeNode* cur_node){
		cur_node->updated = true;

        uint32_t voxel_index = __nv_atomic_fetch_add(&cur_node->voxels_stored, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        uint32_t chunk_index = voxel_index / OocSimLodSettings::NB_POINTS_PER_CHUNK;

		CChunk* cur_chunk = cur_node->voxels;
        for(uint32_t i=0; i<chunk_index; i++){
            cur_chunk = cur_chunk->next;
        }
        uint32_t real_index = voxel_index % OocSimLodSettings::NB_POINTS_PER_CHUNK;
        cur_chunk->points[real_index] = voxel;
	};


    // Insert new points
    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}
        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

        for(uint32_t i=thread_id; i<nb_new_points; i+=nb_threads){
            CPoint& point = new_points[i];
            insertPoint(point);
        }
    }

    // Insert spilled points
    for(uint32_t i=thread_id; i<globalVariables.nbSpilledPoints; i+=nb_threads){
        CPoint& point = globalVariables.spilledPoints[i];
        insertPoint(point);
    }

    // Insert new voxels
    for(uint32_t i=thread_id; i<globalVariables.nbBacklogVoxels; i+=nb_threads){
        CPoint& voxel = globalVariables.backlogVoxels[i];
        COctreeNode* node = globalVariables.backlogVoxelsNodes[i];
        insertVoxel(voxel, node);
    }


    if(thread_id == 0){
        // Because "newChunk" was called in part 1
        globalAllocator.chunksAllocator->reset_temporary_allocations();
    }
}
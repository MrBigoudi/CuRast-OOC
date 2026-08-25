#include "utils.cuh"


/// Run on floor("NB SMs" * "Max threads per SM" / "Max threads per block") blocks of size "Max threads per block"
/// Each thread is filling independently it's own counter before combining all of them
extern "C" __global__
void kernel_bottom_up_update_part_1_counting(){
    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    // if(nb_threads == 0){
    //     printf("kernel_bottom_up_update_part_1_counting\n");
    // }

    uint32_t nb_new_levels = 0;
    CNodePosition node_position = CFrontTopLeft;
    CAABB new_aabb = globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb;

    for(uint32_t batch = 0; batch < globalVariables.maxNbBatches; batch++){
        if(globalVariables.batchesAddedMask[batch]){continue;}

        CPoint* new_points = globalVariables.batchesToAddPoints[batch];
        uint32_t nb_new_points = globalVariables.batchesToAddCounts[batch];

#ifdef ASSERT_ENABLED
        if(nb_new_points > globalVariables.maxBatchSize){
            printf("ERROR: On bottom up update, batch size exceeded the limit: %d / %d\n", 
                nb_new_points, globalVariables.maxBatchSize
            );
            customAssert();
        }
#endif

        for(uint32_t i=thread_id; i<nb_new_points; i+=nb_threads){
            const CPoint& point = new_points[i];
            while(!new_aabb.contains(point.position)){
                nb_new_levels++;
                new_aabb.extend(node_position);
                updateNodePosition(node_position);
            }
        }

    }

    // Combine max new level per block
    __nv_atomic_max(
        &globalVariables.batchesToAddBottomUpCount,
        nb_new_levels,
        __NV_ATOMIC_RELAXED, 
        __NV_THREAD_SCOPE_DEVICE
    );


    // Reset UI values
    globalVariables.nbNewPointsThisUpdate = 0;
    globalVariables.nbNewVoxelsThisUpdate = 0;
    globalVariables.nbNewNodesThisUpdate = 0;
    globalVariables.nbLoadedNodesThisUpdate = 0;
    globalVariables.nbStoredNodesThisUpdate = 0;
    globalVariables.nbSplitNodesThisUpdate = 0;
    globalVariables.nbDeletedNodesThisUpdate = 0;
    globalVariables.nbDeletedChunksThisUpdate = 0;
    globalVariables.nbDeletedGridsThisUpdate = 0;
    globalVariables.nbNewChunksThisUpdate = 0;
    globalVariables.nbNewGridsThisUpdate = 0;
}


__device__ void addNewVoxels(
    COctreeNode* new_parent,
    const CAABB& parent_aabb, 
    const CChunk* child_chunk_list
){
    while(child_chunk_list){
        for(uint32_t j=0; j<child_chunk_list->size; j++){
            const CPoint& point = child_chunk_list->points[j];

            // Sample voxel occupancy grid at this location
            COccupancyGrid::GridIndex index = COccupancyGrid::getCellIndices(parent_aabb, point);
            bool is_cell_occupied = new_parent->occupancy->markCellAsFilled(index);

            // Fill up voxels chunks
            // The occupancy grid will be erased later anyways so don't bother filling it here
            if(!is_cell_occupied){
                // Create corresponding voxel using this point
                vec3 voxel_centroid = COccupancyGrid::getCellCentroid(parent_aabb, index);
                CPoint new_voxel = {};
                new_voxel.position = voxel_centroid;
                new_voxel.color = point.color;
                
                // Add voxel to voxels chunk list
                if(!new_parent->voxels){new_parent->voxels = globalAllocator.newChunk(false);}

                CChunk* parent_chunk_list = new_parent->voxels;
                while(parent_chunk_list->next){parent_chunk_list = parent_chunk_list->next;}

                if(parent_chunk_list->size == OocSimLodSettings::NB_POINTS_PER_CHUNK){
                    parent_chunk_list->next =  globalAllocator.newChunk(false);
                    parent_chunk_list = parent_chunk_list->next;
                }
                parent_chunk_list->points[parent_chunk_list->size] = new_voxel;
                parent_chunk_list->size++;
                new_parent->voxels_counter++;
            }
        }
        child_chunk_list = child_chunk_list->next;
    }
    new_parent->voxels_stored = new_parent->voxels_counter;
};

/// Run on a single thread
extern "C" __global__
void kernel_bottom_up_update_part_2_instancing(){
    // printf("kernel_bottom_up_update_part_2_instancing\n");

    uint32_t nb_new_levels = globalVariables.batchesToAddBottomUpCount;

    // Bottom up update of the main octree
    COctreeNode* cur_child = globalVariables.mainOctree;
    CNodePosition node_position = CFrontTopLeft;

    for(uint32_t i=0; i<nb_new_levels; i++){
		// Create the new parent node
		CIdAABB parent_aabb_index = createNewNodeId();

        // printf("\n\n\n\n\n\n\nFrom bottom up: %d\n\n\n\n\n\n\n", parent_aabb_index);

		COctreeNode* new_parent = globalAllocator.newOctreeNode(parent_aabb_index, false);
		uint32_t node_index = globalVariables.curNbNodes;
        globalVariables.curNbNodes++;

#ifdef ASSERT_ENABLED
        if(node_index >= globalVariables.maxNbConcurrentNodes){
            printf("ERROR: Can't create more nodes in the bottom up update\n");
            customAssert();
        }
#endif

        globalVariables.packedNodes[node_index] = new_parent;

        // Create the new AABB
        CAABB new_parent_aabb = globalVariables.relationshipMap[cur_child->aabb_index].aabb;
		new_parent_aabb.extend(node_position);
        
        // Create the occupancy
		new_parent->occupancy = globalAllocator.newOccupancyGrid(false);
        uint32_t grid_index = globalVariables.nbGridsToInit;
        globalVariables.nbGridsToInit++;
        globalVariables.gridsToInit[grid_index] = new_parent;

        globalVariables.setFlag(new_parent->aabb_index, CFlagIsUpdated);
        globalVariables.setFlag(cur_child->aabb_index, CFlagIsUpdated);
		new_parent->children[node_position] = cur_child;

		// Sample voxels to fill new occupancy grid
		addNewVoxels(new_parent, new_parent_aabb, cur_child->points);
		addNewVoxels(new_parent, new_parent_aabb, cur_child->voxels);

		// Update the AABB maps
        globalVariables.relationshipMap[parent_aabb_index].aabb = new_parent_aabb;
		globalVariables.relationshipMap[parent_aabb_index].children[node_position] = cur_child->aabb_index;
        globalVariables.relationshipMap[cur_child->aabb_index].parent = parent_aabb_index;

		cur_child = new_parent;
		updateNodePosition(node_position);
	}

    globalVariables.mainOctree = cur_child;
    globalVariables.mainOctree->level = 0;
    globalVariables.batchesToAddBottomUpCount = 0;
}
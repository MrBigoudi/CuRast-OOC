#include "utils.cuh"

extern "C" __global__
void kernel_bottom_up_update_part_1(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads = block.num_threads();

    // printf("nb blocks: %llu, nb theads: %llu, nb theads per block: %u\n", grid.num_blocks(), grid.num_threads(), nb_threads);

    if(block_id >= globalVariables.maxNbBatches){return;}
    if(globalVariables.batchesAddedMask[block_id]){return;}

    CPoint* new_points = globalVariables.batchesToAddPoints[block_id];
    uint32_t nb_new_points = globalVariables.batchesToAddCounts[block_id];

    // Find max new level per thread
    CNodePosition node_position = CFrontTopLeft;
    CAABB new_aabb = getAABB(globalVariables.mainOctree->aabb_index);
    uint32_t nb_new_levels = 0;

    for(uint32_t i = thread_id; i < nb_new_points; i += nb_threads){
        CPoint& point = new_points[i];
        while(!new_aabb.contains(point.position)){
            nb_new_levels++;
            new_aabb.extend(node_position);
            updateNodePosition(node_position);
        }
    }

    // Combine max new level per block
    __nv_atomic_max(
        &globalVariables.batchesToAddBottomUpCounts[block_id],
        nb_new_levels,
        __NV_ATOMIC_RELAXED, 
        __NV_THREAD_SCOPE_SYSTEM
    );
}


__device__ void fillOccupancyGrid(
    COctreeNode* new_parent,
    const CAABB& parent_aabb, 
    const CChunk* child_chunk_list
){
    while(child_chunk_list){
        for(uint32_t j=0; j<child_chunk_list->size; j++){
            const CPoint& point = child_chunk_list->points[j];

            // Sample voxel occupancy grid at this location
            COccupancyGrid::GridIndex index = COccupancyGrid::getCellIndices(parent_aabb, point);
            bool is_cell_occupied = new_parent->occupancy->isCellOcupied(index);

            // Fill up occupancy grid
            if(!is_cell_occupied){
                new_parent->occupancy->markCellAsFilled(index);
                // Create corresponding voxel using this point
                vec3 voxel_centroid = COccupancyGrid::getCellCentroid(parent_aabb, index);
                CPoint new_voxel = {};
                new_voxel.position = voxel_centroid;
                new_voxel.color = point.color;
                
                // Add voxel to voxels chunk list
                if(!new_parent->voxels){new_parent->voxels =  globalAllocator.newChunk();}

                CChunk* parent_chunk_list = new_parent->voxels;
                while(parent_chunk_list->next){parent_chunk_list = parent_chunk_list->next;}

                if(parent_chunk_list->size == OocSimLodSettings::NB_POINTS_PER_CHUNK){
                    parent_chunk_list->next =  globalAllocator.newChunk();
                    parent_chunk_list = parent_chunk_list->next;
                }
                parent_chunk_list->points[parent_chunk_list->size] = new_voxel;
                parent_chunk_list->size++;
            }
        }
        child_chunk_list = child_chunk_list->next;
    }
};

/// Run on a single thread
extern "C" __global__
void kernel_bottom_up_update_part_2(){
    // TODO: rethink this safeguard
    if(!globalVariables.mainOctree){return;}

    // Combine max new level in total
    uint32_t nb_new_levels = 0;
    for(uint32_t i=0; i<globalVariables.maxNbBatches; i++){
        nb_new_levels = max(nb_new_levels, globalVariables.batchesToAddBottomUpCounts[i]);
        globalVariables.batchesToAddBottomUpCounts[i] = 0;
    }

    // Bottom up update of the main octree
    COctreeNode* cur_child = globalVariables.mainOctree;
    CNodePosition node_position = CFrontTopLeft;

    for(uint32_t i=0; i<nb_new_levels; i++){
		// Create the new AABB
		CAABB parent_aabb = getAABB(cur_child->aabb_index);
		parent_aabb.extend(node_position);

		// Create the new parent node
		CIdAABB parent_aabb_index = createNewAABB(parent_aabb);
		// OctreeNode* new_parent = new OctreeNode(parent_aabb_index);
		COctreeNode* new_parent = globalAllocator.newOctreeNode(parent_aabb_index);

		new_parent->occupancy = globalAllocator.newOccupancyGrid();
		new_parent->updated = true;
		cur_child->updated = true;
		new_parent->children[node_position] = cur_child;

		// Sample voxels to fill new occupancy grid
		fillOccupancyGrid(new_parent, parent_aabb, cur_child->points);
		fillOccupancyGrid(new_parent, parent_aabb, cur_child->voxels);

		// Update the AABB maps
		globalVariables.relationshipMap[parent_aabb_index].children[node_position] = cur_child->aabb_index;

		cur_child = new_parent;
		updateNodePosition(node_position);
	}

    globalVariables.mainOctree = cur_child;
}
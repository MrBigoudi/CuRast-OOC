#include "simLod.h"

void simLodUpdate(std::shared_ptr<OctreeNode>& main_root, std::shared_ptr<AABB>& main_aabb, std::shared_ptr<vector<Point>>& points){
	std::shared_ptr<Timing> count_split_timing = addTiming("simlod count/split loop", true, 1);

	uint32_t inner_cpt = 0;
	while(true){

		std::shared_ptr<Timing> timing = addTiming("simlod count", true, 2);
		simLodCount(main_root, main_aabb, points, spilledPoints, spillingNodes);
		timing->stop_clock();

		if(spillingNodes->size() == 0){
			break;
		}
		
		// println("//////////////////////////////////////////////////");
		// println("////////// Octree after simlod counting //////////");
		// println("//////////////////////////////////////////////////");
		// main_root->display();

		timing = addTiming("simlod split", true, 2);
		simLodSplit(spilledPoints, spillingNodes);
		timing->stop_clock();

		// println("//////////////////////////////////////////////////");
		// println("////////// Octree after simlod splitting /////////");
		// println("//////////////////////////////////////////////////");
		// main_root->display();
		
		inner_cpt++;
	}
	count_split_timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("//////// Octree after simlod count/splits ////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();


	std::shared_ptr<Timing> timing = addTiming("simlod voxel sampling", true, 1);
	simLodVoxelSampling(main_root, main_aabb, points, spilledPoints, backlogVoxels, backlogVoxelsNodes);
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("//////// Octree after simlod voxel sample ////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();
	

	timing = addTiming("simlod insertion", true, 1);
	simLodInsertion(main_root, main_aabb, points, spilledPoints, backlogVoxels, backlogVoxelsNodes);
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("///////// Octree after simlod insertions /////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();


	// Clean buffers
	timing = addTiming("simlod buffer cleaning", true, 1);
	spilledPoints->clear();
	spillingNodes->clear();
	backlogVoxels->clear();
	backlogVoxelsNodes->clear();
	timing->stop_clock();

}



void simLodCount(
    std::shared_ptr<OctreeNode>& main_root, 
    std::shared_ptr<AABB>& main_aabb, 
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<OctreeNode*>>& spilling_nodes
){

	mutex mtx;

	auto countPoint = [&](Point& point){
		// Reach corresponding leaf
		std::shared_ptr<OctreeNode> leaf = main_root;
		AABB current_aabb = AABB(*main_aabb);

		uint8_t level = 1;

		while(true){
			// Find next child
			NodePosition child_index = current_aabb.getNextChildIndex(point.position);
			AABB old_aabb = current_aabb;
			current_aabb.shrink(child_index);
			leaf->children_ids |= 0x01 << child_index;

			if(!current_aabb.contains(point.position)){
				println("Error SimLOD count: the point should always be contained in the next AABB");
				println("\tpoint position: ({}, {}, {})", point.position.x, point.position.y, point.position.z);
				println("\tpoint new index: {}", uint32_t(child_index));
				println("\told AABB: mins({}, {}, {}), maxs({}, {}, {})", 
					old_aabb.mins.x, old_aabb.mins.y, old_aabb.mins.z,
					old_aabb.maxs.x, old_aabb.maxs.y, old_aabb.maxs.z
				);
				println("\tnew AABB: mins({}, {}, {}), maxs({}, {}, {})", 
					current_aabb.mins.x, current_aabb.mins.y, current_aabb.mins.z,
					current_aabb.maxs.x, current_aabb.maxs.y, current_aabb.maxs.z
				);
				println("skipping point...");
				return;
			}

			// If not leaf continue
			if(leaf->children[child_index]){
				leaf = leaf->children[child_index];
				// Get node level
				if(level == UINT8_MAX){
					println("The octree has reached it's maximum depth size...");
					exit(EXIT_FAILURE);
				}
				level++;
			} else {
				// Skip if the point was already accepted at this level
				if(point.color[3] == level){return;}

				// Flag point as accepted at this level
				point.color[3] = level;

				{
					// Sync read / write to counter
					// uint16_t old_counter = leaf->counter.fetch_add(1u);
					uint16_t old_counter = 0;
					
					{
						std::lock_guard<std::mutex> lock(mtx);
						old_counter = leaf->counter;
					}
					leaf->counter++;
					if(old_counter == MAX_POINTS_PER_LEAF){
						// leaf->counter.store(MAX_POINTS_PER_LEAF + 1u);
						leaf->counter = MAX_POINTS_PER_LEAF + 1u;
						spilling_nodes->push_back(leaf.get());
					}
				}

				return;
			}
		}
	};

	if(!CPU_PARALLELISED){
		for(Point& point : *points){
			countPoint(point);
		}
		for(Point& point : *spilled_points){
			countPoint(point);		
		}
	} else {
		vector<uint32_t> first_indices = {};
		uint32_t step_size = 100'000u;
		uint32_t nb_points = points->size();
		uint32_t nb_spilled_points = spilled_points->size();
		uint32_t max_nb_points = max(nb_points, nb_spilled_points);
		for(uint32_t i=0; i<max_nb_points; i+=step_size){first_indices.push_back(i);}

		std::for_each(std::execution::par, first_indices.begin(), first_indices.end(), [&](uint32_t index){
			for(uint32_t i=0; i<step_size; i++){
				if((index + i) >= nb_points){break;}
				countPoint((*points)[index + i]);
			}
			for(uint32_t i=0; i<step_size; i++){
				if((index + i) >= nb_spilled_points){break;}
				countPoint((*spilled_points)[index + i]);
			}
		});
	}
}






void simLodSplit(
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<OctreeNode*>>& spilling_nodes
){
	for(uint32_t node_id = 0; node_id < spilling_nodes->size(); node_id++){
		OctreeNode*& spilling_node = (*spilling_nodes)[node_id];
		uint8_t spilling_node_children = spilling_node->children_ids;

		// spilling_node->is_leaf = false;
		spilling_node->counter = 0;
		if(!spilling_node->occupancy){
			spilling_node->occupancy = std::make_shared<OccupancyGrid>();
		}

		for(uint32_t j=0; j<8; j++){
			// Create necessary empty children
			if(!spilling_node->children[j] && (0x01 << j) & spilling_node_children){
				std::shared_ptr<OctreeNode> empty_child = std::make_shared<OctreeNode>();
				// empty_child->is_leaf = true;
				spilling_node->children[j] = empty_child;
			}
		}

		// Add former points to spilled points and free memory
		std::vector<std::shared_ptr<Chunk>> old_chunks = {spilling_node->points};
		std::shared_ptr<Chunk> current_chunk = spilling_node->points;
		if(!current_chunk){
			continue;
		}
		
		while(current_chunk->next){
			current_chunk = current_chunk->next;
			old_chunks.push_back(current_chunk);
		}
		for(int32_t i = old_chunks.size()-1; i >= 0; i--){
			current_chunk = old_chunks[i];
			for(uint32_t j=0; j<current_chunk->size; j++){

				// Flag the point as not accepted
				current_chunk->points[j].color[3] = 0;

				spilled_points->push_back(current_chunk->points[j]);
			}
			current_chunk->next = nullptr;
			current_chunk = nullptr;
			old_chunks[i] = nullptr;
		}

		spilling_node->points = nullptr;
	}
	spilling_nodes->clear();
}


void simLodVoxelSampling(
    std::shared_ptr<OctreeNode>& main_root, 
    std::shared_ptr<AABB>& main_aabb, 
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<Point>>& backlog_voxels,
    std::shared_ptr<vector<OctreeNode*>>& backlog_voxels_nodes
){
	auto sampleVoxel = [&](Point& point){
		// Reach all corresponding inner nodes
		std::shared_ptr<OctreeNode> node = main_root;
		AABB current_aabb = AABB(*main_aabb);

		while(true){
			// if(node->is_leaf){return;}
			if(!node->occupancy){return;}

			// Sample voxel occupancy grid at this location
			vec3 normalized_coordinates = current_aabb.getPointNormalizedCoordinates(point.position);
			uint32_t grid_x = clamp(uint32_t(floor(GRID_SIZE * normalized_coordinates.x)), 0u, GRID_SIZE - 1u);
			uint32_t grid_y = clamp(uint32_t(floor(GRID_SIZE * normalized_coordinates.y)), 0u, GRID_SIZE - 1u);
			uint32_t grid_z = clamp(uint32_t(floor(GRID_SIZE * normalized_coordinates.z)), 0u, GRID_SIZE - 1u);
			uint32_t index = grid_x + GRID_SIZE * (grid_y + GRID_SIZE * grid_z);
			uint32_t word_index = index >> 5u;
			uint32_t bit_index = index & 31u;
			bool is_cell_occupied = (node->occupancy->values[word_index] & (1u << bit_index)) != 0;

			if(!is_cell_occupied){
				// Fill up occupancy grid
				node->occupancy->values[word_index] |= (1u << bit_index);
				// Create corresponding voxel using this point
				vec3 world_grid_size = current_aabb.getSize() / float(GRID_SIZE);
				vec3 voxel_centroid = current_aabb.mins + world_grid_size * vec3(grid_x, grid_y, grid_z);
				Point new_voxel = {};
				new_voxel.position = voxel_centroid;
				new_voxel.color[0] = point.color[0];
				new_voxel.color[1] = point.color[1];
				new_voxel.color[2] = point.color[2];
				// new_voxel.color[0] = 0x00;
				// new_voxel.color[1] = 0xff;
				// new_voxel.color[2] = 0xff;

				// Add voxel to backlog buffers
				// node->counter++;
				backlog_voxels->push_back(new_voxel);
				backlog_voxels_nodes->push_back(node.get());
			}

			// Find next child
			NodePosition child_index = current_aabb.getNextChildIndex(point.position);
			AABB old_aabb = current_aabb;
			current_aabb.shrink(child_index);

			if(!current_aabb.contains(point.position)){
				println("Error SimLOD voxel sampling: the point should always be contained in the next AABB");
				println("\tpoint position: ({}, {}, {})", point.position.x, point.position.y, point.position.z);
				println("\tpoint new index: {}", uint32_t(child_index));
				println("\told AABB: mins({}, {}, {}), maxs({}, {}, {})", 
					old_aabb.mins.x, old_aabb.mins.y, old_aabb.mins.z,
					old_aabb.maxs.x, old_aabb.maxs.y, old_aabb.maxs.z
				);
				println("\tnew AABB: mins({}, {}, {}), maxs({}, {}, {})", 
					current_aabb.mins.x, current_aabb.mins.y, current_aabb.mins.z,
					current_aabb.maxs.x, current_aabb.maxs.y, current_aabb.maxs.z
				);
				println("skipping point...");
				return;
			}

			if(!node->children[child_index]){return;}
			node = node->children[child_index];
		}
	};

	for(Point& point : *points){
		sampleVoxel(point);
	}
	for(Point& point : *spilled_points){
		sampleVoxel(point);		
	}
}

void simLodInsertion(
    std::shared_ptr<OctreeNode>& main_root, 
    std::shared_ptr<AABB>& main_aabb, 
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<Point>>& backlog_voxels,
    std::shared_ptr<vector<OctreeNode*>>& backlog_voxels_nodes
){
	auto insertPoint = [&](Point& point){
		// Reach all corresponding leaves
		std::shared_ptr<OctreeNode> node = main_root;
		AABB current_aabb = AABB(*main_aabb);

		while(true){
			node->updated = true;

			// Find next child
			NodePosition child_index = current_aabb.getNextChildIndex(point.position);
			AABB old_aabb = current_aabb;
			current_aabb.shrink(child_index);

			if(!current_aabb.contains(point.position)){
				println("Error SimLOD insertion: the point should always be contained in the next AABB");
				println("\tpoint position: ({}, {}, {})", point.position.x, point.position.y, point.position.z);
				println("\tpoint new index: {}", uint32_t(child_index));
				println("\told AABB: mins({}, {}, {}), maxs({}, {}, {})", 
					old_aabb.mins.x, old_aabb.mins.y, old_aabb.mins.z,
					old_aabb.maxs.x, old_aabb.maxs.y, old_aabb.maxs.z
				);
				println("\tnew AABB: mins({}, {}, {}), maxs({}, {}, {})", 
					current_aabb.mins.x, current_aabb.mins.y, current_aabb.mins.z,
					current_aabb.maxs.x, current_aabb.maxs.y, current_aabb.maxs.z
				);
				println("skipping point...");
				return;
			}

			// If leaf insert point in chunks
			if(node->children[child_index]){
				node = node->children[child_index];
			} else {
				if(!node->points){node->points = std::make_shared<Chunk>();}
				std::shared_ptr<Chunk> chunk_list = node->points;
				while(chunk_list->next){chunk_list = chunk_list->next;}
				if(chunk_list->size == POINTS_PER_CHUNK){
					chunk_list->next = std::make_shared<Chunk>();
					chunk_list = chunk_list->next;
				}
				chunk_list->points[chunk_list->size] = point;
				chunk_list->size++;
				return;
			}
		}
	};

	auto insertVoxel = [&](Point& voxel, OctreeNode* node){
		node->updated = true;

		if(!node->voxels){node->voxels = std::make_shared<Chunk>();}
		std::shared_ptr<Chunk> chunk_list = node->voxels;
		while(chunk_list->next){chunk_list = chunk_list->next;}
		if(chunk_list->size == POINTS_PER_CHUNK){
			chunk_list->next = std::make_shared<Chunk>();
			chunk_list = chunk_list->next;
		}
		chunk_list->points[chunk_list->size] = voxel;
		chunk_list->size++;
		return;
	};

	if(!CPU_PARALLELISED){
		for(Point& point : *points){
			insertPoint(point);
		}
		for(Point& point : *spilled_points){
			insertPoint(point);		
		}
		uint32_t nb_new_voxels = backlog_voxels->size();
		for(uint32_t i=0; i<nb_new_voxels; i++){
			insertVoxel((*backlog_voxels)[i], (*backlog_voxels_nodes)[i]);
		}
	} else {
		std::thread parallel_thread([&](){
			uint32_t nb_new_voxels = backlog_voxels->size();
			for(uint32_t i=0; i<nb_new_voxels; i++){
				insertVoxel((*backlog_voxels)[i], (*backlog_voxels_nodes)[i]);
			}
		});

		for(Point& point : *points){
			insertPoint(point);
		}
		for(Point& point : *spilled_points){
			insertPoint(point);		
		}

		parallel_thread.join();
	}
}
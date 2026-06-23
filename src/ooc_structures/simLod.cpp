#include "simLod.h"
#include "outOfCore.h"

#include <unordered_set>

void simLodUpdate(OctreeNode* main_root, std::shared_ptr<vector<Point>>& points){
	std::shared_ptr<Timing> count_split_timing = addTiming("simlod count/split loop", true, 1);

	// println("//////////////////////////////////////////////////");
	// println("////////// Octree before simlod update ///////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();

	std::shared_ptr<Timing> timing = addTiming("simlod load", true, 1);
	simLodLoad(main_root, points, spilledPoints);
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("////////// Octree after simlod load //////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();

	while(true){
		std::shared_ptr<Timing> timing = addTiming("simlod count", true, 2);
		simLodCount(main_root, points, spilledPoints, spillingNodes);
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
	}
	count_split_timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("//////// Octree after simlod count/splits ////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();


	timing = addTiming("simlod voxel sampling", true, 1);
	simLodVoxelSampling(main_root, points, spilledPoints, backlogVoxels, backlogVoxelsNodes);
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("//////// Octree after simlod voxel sample ////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();
	

	timing = addTiming("simlod insertion", true, 1);
	simLodInsertion(main_root, points, spilledPoints, backlogVoxels, backlogVoxelsNodes);
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
    OctreeNode* main_root,
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<OctreeNode*>>& spilling_nodes
){

	std::mutex mtx_counter, mtx_spilling_nodes;

	auto countPoint = [&](Point& point, std::mutex& mtx_counter, std::mutex& mtx_spilling_nodes){
		// Reach corresponding leaf
		OctreeNode* leaf = main_root;

		uint8_t level = 1;

		while(true){
			// Find next child
			NodePosition child_index = leaf->aabb->getNextChildIndex(point.position);
			leaf->children_ids |= 0x01 << child_index;

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
					uint32_t old_counter = 0;
					
					{
						std::lock_guard<std::mutex> lock_counter(mtx_counter);
						old_counter = leaf->counter;
						leaf->counter = min(old_counter + 1u, MAX_POINTS_PER_LEAF + 1u);
					}
					if(old_counter == MAX_POINTS_PER_LEAF){
						std::lock_guard<std::mutex> lock_spilling(mtx_spilling_nodes);
						spilling_nodes->push_back(leaf);
					}
				}

				return;
			}
		}
	};

	if(!CPU_PARALLELISED){
		for(Point& point : *points){
			countPoint(point, mtx_counter, mtx_spilling_nodes);
		}
		for(Point& point : *spilled_points){
			countPoint(point, mtx_counter, mtx_spilling_nodes);		
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
				countPoint((*points)[index + i], mtx_counter, mtx_spilling_nodes);
			}
			for(uint32_t i=0; i<step_size; i++){
				if((index + i) >= nb_spilled_points){break;}
				countPoint((*spilled_points)[index + i], mtx_counter, mtx_spilling_nodes);
			}
		});
	}
}






void simLodSplit(
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<OctreeNode*>>& spilling_nodes
){

	mutex mtx_spilled_points;

	auto lambda = [&](OctreeNode*& spilling_node){
	// for(uint32_t node_id = 0; node_id < spilling_nodes->size(); node_id++){
		// OctreeNode*& spilling_node = (*spilling_nodes)[node_id];
		uint8_t spilling_node_children = spilling_node->children_ids;

		// spilling_node->is_leaf = false;
		spilling_node->counter = 0;
		if(!spilling_node->occupancy){
			spilling_node->occupancy = new OccupancyGrid();
		}

		for(uint32_t j=0; j<8; j++){
			// Create necessary empty children
			if(!spilling_node->children[j] && (0x01 << j) & spilling_node_children){
				OctreeNode* empty_child = new OctreeNode();
				empty_child->aabb = new AABB(*spilling_node->aabb);
				empty_child->aabb->shrink((NodePosition)j);
				empty_child->from_split = true;
				// empty_child->is_leaf = true;
				spilling_node->children[j] = empty_child;

				// TODO: temporary code
				{
					std::lock_guard<std::mutex> lock(aabb_relationship_map_mtx);
					if(!aabb_relationship_map.contains(*spilling_node->aabb)){
						aabb_relationship_map[*spilling_node->aabb] = {nullopt};
					}
					aabb_relationship_map[*spilling_node->aabb][j] = *empty_child->aabb;
					aabb_parent_map[*empty_child->aabb] = *spilling_node->aabb;
				}
			}
		}

		// Add former points to spilled points and free memory
		Chunk* current_chunk = spilling_node->points;
		if(!current_chunk){
			// continue;
			return;
		}
		
		while(current_chunk){
			for(uint32_t j=0; j<current_chunk->size; j++){
				// Flag the point as not accepted
				current_chunk->points[j].color[3] = 0;
				std::lock_guard<std::mutex> lock_spilled(mtx_spilled_points);
				spilled_points->push_back(current_chunk->points[j]);
			}
			current_chunk = current_chunk->next;
		}

		delete(spilling_node->points);
		spilling_node->points = nullptr;
	};


	// // Sanity check
	// std::unordered_set<OctreeNode*> set = {};
	// for(OctreeNode*& node : *spilling_nodes){
	// 	if(set.contains(node)){
	// 		println("Duplicate spilling nodes should not happen");
	// 		exit(EXIT_FAILURE);
	// 	}
	// 	set.insert(node);
	// }

	if(!CPU_PARALLELISED){
		std::for_each(spilling_nodes->begin(), spilling_nodes->end(), [&](OctreeNode*& spilling_node){
			lambda(spilling_node);
		});
	} else {
		std::for_each(std::execution::par, spilling_nodes->begin(), spilling_nodes->end(), [&](OctreeNode*& spilling_node){
			lambda(spilling_node);
		});
	}

	spilling_nodes->clear();
}


void simLodVoxelSampling(
    OctreeNode* main_root,
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<Point>>& backlog_voxels,
    std::shared_ptr<vector<OctreeNode*>>& backlog_voxels_nodes
){
	auto sampleVoxel = [&](Point& point){
		// Reach all corresponding inner nodes
		OctreeNode* node = main_root;

		while(true){
			// if(node->is_leaf){return;}
			if(!node->occupancy){return;}

			// Find next child
			NodePosition child_index = node->aabb->getNextChildIndex(point.position);
			if(!node->children[child_index]){return;}

			// Sample voxel occupancy grid at this location if the node is inner for this point
			vec3 normalized_coordinates = node->aabb->getPointNormalizedCoordinates(point.position);
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
				vec3 world_grid_size = node->aabb->getSize() / float(GRID_SIZE);
				vec3 voxel_centroid = node->aabb->mins + world_grid_size * vec3(grid_x, grid_y, grid_z) + 0.5f*world_grid_size;
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
				backlog_voxels_nodes->push_back(node);
			}

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
    OctreeNode* main_root,
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<Point>>& backlog_voxels,
    std::shared_ptr<vector<OctreeNode*>>& backlog_voxels_nodes
){

	auto insertPoint = [&](Point& point, OctreeNode* main_node){
		OctreeNode* cur_node = main_node;
		// Reach all corresponding leaves
		while(true){
			cur_node->updated = true;
			// Find next child
			NodePosition child_index = cur_node->aabb->getNextChildIndex(point.position);
			// If leaf insert point in chunks
			if(cur_node->children[child_index]){
				cur_node = cur_node->children[child_index];
			} else {
				if(!cur_node->points){cur_node->points = new Chunk();}
				Chunk* chunk_list = cur_node->points;
				while(chunk_list->next){chunk_list = chunk_list->next;}
				if(chunk_list->size == POINTS_PER_CHUNK){
					chunk_list->next = new Chunk();
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
		if(!node->voxels){node->voxels = new Chunk();}
		Chunk* chunk_list = node->voxels;
		while(chunk_list->next){chunk_list = chunk_list->next;}
		if(chunk_list->size == POINTS_PER_CHUNK){
			chunk_list->next = new Chunk();
			chunk_list = chunk_list->next;
		}
		chunk_list->points[chunk_list->size] = voxel;
		chunk_list->size++;
		return;
	};

	if(!CPU_PARALLELISED){
		for(Point& point : *points){
			insertPoint(point, main_root);
		}
		for(Point& point : *spilled_points){
			insertPoint(point, main_root);		
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
			insertPoint(point, main_root);
		}
		for(Point& point : *spilled_points){
			insertPoint(point, main_root);		
		}

		parallel_thread.join();
	}

}



void simLodLoad(
    OctreeNode* main_root,
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points
){

	// tmp_ser is here to avoid loading a node multiple time in the parallel context
	mutex mtx_set;
	std::unordered_set<AABB, AABB::Hash> tmp_set = {};


	// Try to insert all points
	auto tryInsertPoint = [&](Point& point, OctreeNode* main_root){
		// Reach corresponding leaf
		OctreeNode* leaf = main_root;

		uint8_t level = 0;

		while(true){
			// Find next child
			NodePosition child_index = leaf->aabb->getNextChildIndex(point.position);

			// If current node is not a leaf continue, else current node becomes child
			OctreeNode* child = leaf->children[child_index];

			if(child){
				leaf = child;
				// Get node level
				if(level == UINT8_MAX){
					println("The octree has reached it's maximum depth size...");
					exit(EXIT_FAILURE);
				}
				level++;
			} else {
				// Check if the child has been stored
				bool has_been_stored = false;
				// AABB child_aabb = *leaf->aabb;
				// child_aabb.shrink((NodePosition)child_index);
				AABB child_aabb = {};
				// TODO: temporary code
				{
					std::lock_guard<std::mutex> lock(aabb_relationship_map_mtx);
					if(aabb_relationship_map[*leaf->aabb][child_index].has_value()){
						child_aabb = aabb_relationship_map[*leaf->aabb][child_index].value();
					} else {
						return;
					}
				}
				
				
				{
					std::lock_guard<std::mutex> lock(mtx_set);
					has_been_stored = LRUCache::hasBeenStored(child_aabb) || tmp_set.contains(child_aabb);

					// If the child has not been stored, we've reached the end of the loop
					if(!has_been_stored){return;}

					// Else, we load the child and make it the current node
					tmp_set.insert(child_aabb);
					leaf->children[child_index] = loadOctree(child_aabb, true);
				}

				leaf = leaf->children[child_index];

				if(!leaf){
					println("At this point in the SimLodLoad, the leaf should never be null");
					exit(EXIT_FAILURE);
				}

				// Get node level
				if(level == UINT8_MAX){
					println("The octree has reached it's maximum depth size...");
					exit(EXIT_FAILURE);
				}
				level++;
			}
		}
	};

	if(!CPU_PARALLELISED){
		for(Point& point : *points){
			tryInsertPoint(point, main_root);
		}
		for(Point& point : *spilled_points){
			tryInsertPoint(point, main_root);		
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
				tryInsertPoint((*points)[index + i], main_root);
			}
			for(uint32_t i=0; i<step_size; i++){
				if((index + i) >= nb_spilled_points){break;}
				tryInsertPoint((*spilled_points)[index + i], main_root);
			}
		});
	}

	// if(!tmp_set.empty()){
	// 	println("nb loaded nodes = {}", tmp_set.size());
	// }
}
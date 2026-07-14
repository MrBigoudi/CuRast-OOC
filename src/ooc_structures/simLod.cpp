#include "simLod.h"
#include "outOfCore.h"

#include "allocator.h"

void SimLod::update(
	OctreeNode* main_root, 
	std::shared_ptr<vector<Point>>& points,
	std::shared_ptr<AABBRelationshipMap> relationship_map_ref
){

	if(!main_root){return;}

	// println("//////////////////////////////////////////////////");
	// println("////////// Octree before simlod update ///////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();
	// GlobalVariables::displayCpuMemoryUsage();

	std::shared_ptr<Timing> timing = Timing::addTiming("simlod load", true, 1);
	load(main_root, points, relationship_map_ref);
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("////////// Octree after simlod load //////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();
	// GlobalVariables::displayCpuMemoryUsage();

	std::shared_ptr<Timing> count_split_timing = Timing::addTiming("simlod count/split loop", true, 1);
	while(true){
		std::shared_ptr<Timing> timing = Timing::addTiming("simlod count", true, 2);
		count(main_root, points, GlobalVariables::spilledPoints, GlobalVariables::spillingNodes);
		timing->stop_clock();

		if(GlobalVariables::spillingNodes->size() == 0){
			break;
		}
		
		// println("//////////////////////////////////////////////////");
		// println("////////// Octree after simlod counting //////////");
		// println("//////////////////////////////////////////////////");
		// main_root->display();
		// println("after count: {} spilled nodes, {} spilled points", GlobalVariables::spillingNodes->size(), GlobalVariables::spilledPoints->size());

		timing = Timing::addTiming("simlod split", true, 2);
		split(GlobalVariables::spilledPoints, GlobalVariables::spillingNodes, relationship_map_ref);
		timing->stop_clock();

		// println("//////////////////////////////////////////////////");
		// println("////////// Octree after simlod splitting /////////");
		// println("//////////////////////////////////////////////////");
		// main_root->display();
		// println("after split: {} spilled nodes, {} spilled points", GlobalVariables::spillingNodes->size(), GlobalVariables::spilledPoints->size());
	}
	count_split_timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("//////// Octree after simlod count/splits ////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();
	// GlobalVariables::displayCpuMemoryUsage();



	timing = Timing::addTiming("simlod voxel sampling", true, 1);
	voxelSampling(main_root, points, GlobalVariables::spilledPoints, GlobalVariables::backlogVoxels, GlobalVariables::backlogVoxelsNodes);
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("//////// Octree after simlod voxel sample ////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();
	// GlobalVariables::displayCpuMemoryUsage();
	

	timing = Timing::addTiming("simlod insertion", true, 1);
	insertion(main_root, points, GlobalVariables::spilledPoints, GlobalVariables::backlogVoxels, GlobalVariables::backlogVoxelsNodes);
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("///////// Octree after simlod insertions /////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();
	// GlobalVariables::displayCpuMemoryUsage();


	// Clean buffers
	timing = Timing::addTiming("simlod buffer cleaning", true, 1);
	GlobalVariables::spilledPoints->clear();
	GlobalVariables::spillingNodes->clear();
	GlobalVariables::backlogVoxels->clear();
	GlobalVariables::backlogVoxelsNodes->clear();
	timing->stop_clock();

}






void SimLod::load(
    OctreeNode* main_root,
    std::shared_ptr<vector<Point>>& points,
	std::shared_ptr<AABBRelationshipMap> relationship_map_ref
){

	// tmp_set is here to avoid loading a node multiple time in the parallel context
	mutex mtx_set;
	std::unordered_set<IdAABB> tmp_set = {};


	// Try to insert all points
	auto tryInsertPoint = [&](Point& point, OctreeNode* main_root){
		// Reach corresponding leaf
		OctreeNode* leaf = main_root;
		if(!leaf){return;}

		uint8_t level = 0;

		while(true){
			// Find next child
			const AABB& aabb = GlobalVariables::getAABB(leaf->aabb_index);
			NodePosition child_index = aabb.getNextChildIndex(point.position);

			// If current node is not a leaf continue, else current node becomes child
			OctreeNode* child = leaf->children[child_index];

			if(child){
				leaf = child;
				// Get node level
				if(level == UINT8_MAX){
					println("The octree has reached it's maximum depth size...");
					throw(EXIT_FAILURE);
				}
				level++;
			} else {
				// Check if the child has been stored
				if((*relationship_map_ref)[leaf->aabb_index][child_index] == INVALID_ID){return;}
				IdAABB child_aabb_index = (*relationship_map_ref)[leaf->aabb_index][child_index];
				
				{
					std::lock_guard<std::mutex> lock(mtx_set);
					if(!tmp_set.contains(child_aabb_index)){
						// Load the child and make it the current node
						tmp_set.insert(child_aabb_index);
						leaf->children[child_index] = loadOctree(child_aabb_index);
					}
				}

				leaf = leaf->children[child_index];

				if(!leaf){
					println("At this point in the SimLodLoad, the leaf should never be null");
					throw(EXIT_FAILURE);
				}

				// Get node level
				if(level == UINT8_MAX){
					println("The octree has reached it's maximum depth size...");
					throw(EXIT_FAILURE);
				}
				level++;
			}
		}
	};

	// // To sync loads / stores to CPU cache
    // std::lock_guard<std::mutex> lock(LRUCache::caches_sync_mtx);

	if(!OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		std::for_each(points->begin(), points->end(), [&](Point& point){
			tryInsertPoint(point, main_root);
		});
	} else {
		std::for_each(std::execution::par, points->begin(), points->end(), [&](Point& point){
			tryInsertPoint(point, main_root);
		});
	}
}





void SimLod::count(
    OctreeNode* main_root,
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<OctreeNode*>>& spilling_nodes
){

	auto countPoint = [&](Point& point) -> OctreeNode* {
		// Reach corresponding leaf
		OctreeNode* leaf = main_root;
		uint8_t level = 1;

		while(true){
			// Find next child
			const AABB& aabb = GlobalVariables::getAABB(leaf->aabb_index);
			NodePosition child_index = aabb.getNextChildIndex(point.position);

			// If not leaf continue
			if(leaf->children[child_index]){
				leaf = leaf->children[child_index];
				// Get node level
				if(level == UINT8_MAX){
					println("The octree has reached it's maximum depth size...");
					throw(EXIT_FAILURE);
				}
				level++;
			} else {
				leaf->children_ids |= 0x01 << child_index;

				// Skip if the point was already accepted at this level
				if(point.color[3] == level){return nullptr;}

				// Flag point as accepted at this level
				point.color[3] = level;

				uint32_t old_counter = leaf->counter.fetch_add(1u);

				if(old_counter == OocSimLodSettings::MAX_POINTS_PER_LEAF){
					return leaf;
				}

				return nullptr;
			}
		}
	};

	uint32_t nb_points = points->size();
	uint32_t nb_spilled_points = spilled_points->size();
	std::vector<uint32_t> indices(nb_points + nb_spilled_points);
	std::iota(indices.begin(), indices.end(), 0);
	std::vector<OctreeNode*> tmp_spilled = std::vector<OctreeNode*>(nb_points + nb_spilled_points, nullptr);

	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		// Count points in parallel
		std::for_each(std::execution::par, indices.begin(), indices.end(), [&](uint32_t index){
			Point* point = nullptr;
			if(index >= nb_points){
				point = &(*spilled_points)[index - nb_points];
			} else {
				point = &(*points)[index];
			}
			tmp_spilled[index] = countPoint(*point);
		});
		
	} else {
		// Count points sequentially
		std::for_each(indices.begin(), indices.end(), [&](uint32_t index){
			Point* point = nullptr;
			if(index >= nb_points){
				point = &(*spilled_points)[index - nb_points];
			} else {
				point = &(*points)[index];
			}
			tmp_spilled[index] = countPoint(*point);
		});
	}

	// Add spilling nodes sequentially
	for(OctreeNode* node : tmp_spilled){
		if(node){spilling_nodes->push_back(node);}
	}
}

void SimLod::split(
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<OctreeNode*>>& spilling_nodes,
    std::shared_ptr<AABBRelationshipMap> relationship_map_ref
){
	typedef std::vector<Point> SpilledPoints;
	typedef std::vector<std::pair<NodePosition, AABB>> NewNodes;

	auto lambda = [&](OctreeNode* spilling_node) -> std::pair<SpilledPoints, NewNodes> {
		NewNodes new_nodes = {};
		SpilledPoints new_spilled_points = {};

		uint8_t spilling_node_children = spilling_node->children_ids;

		spilling_node->counter.store(0);
		spilling_node->children_ids = 0;
		if(!spilling_node->occupancy){
			// spilling_node->occupancy = new OccupancyGrid();
			spilling_node->occupancy = MemoryAllocator::newOccupancyGrid();
		}

		for(uint32_t j=0; j<8; j++){
			// Create necessary empty children
			bool can_be_spilled = (0x01 << j) & spilling_node_children;
			if(!spilling_node->children[j] && can_be_spilled){
				// Create the new AABB
				AABB child_aabb = AABB(GlobalVariables::getAABB(spilling_node->aabb_index));
				child_aabb.shrink((NodePosition)j);

				new_nodes.push_back({(NodePosition)j, child_aabb});
			}
		}

		// Add former points to spilled points and free memory
		Chunk* current_chunk = spilling_node->points;
		if(!current_chunk){
			return {new_spilled_points, new_nodes};
		}
		
		while(current_chunk){
			for(uint32_t j=0; j<current_chunk->size; j++){
				// Flag the point as not accepted
				current_chunk->points[j].color[3] = 0;
				new_spilled_points.push_back(current_chunk->points[j]);
			}
			current_chunk = current_chunk->next;
		}

		// delete(spilling_node->points);
		MemoryAllocator::delChunk(spilling_node->points);
		spilling_node->points = nullptr;

		return {new_spilled_points, new_nodes};
	};

	if(!spilling_nodes->empty()){
		uint32_t nb_spilling_nodes = spilling_nodes->size();
		std::vector<std::pair<SpilledPoints, NewNodes>> tmp_res(nb_spilling_nodes);
		std::vector<uint32_t> indices(nb_spilling_nodes);
		std::iota(indices.begin(), indices.end(), 0);

		if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
			std::for_each(std::execution::par, indices.begin(), indices.end(), [&](uint32_t index){
				tmp_res[index] = lambda((*spilling_nodes)[index]);
			});
		} else {
			std::for_each(indices.begin(), indices.end(), [&](uint32_t index){
				tmp_res[index] = lambda((*spilling_nodes)[index]);
			});
		}

		for(uint32_t i=0; i<nb_spilling_nodes; i++){
			OctreeNode* spilling_node = (*spilling_nodes)[i];
			std::pair<SpilledPoints, NewNodes>& value = tmp_res[i];

			// Sequentially add the spill points to the buffer
			for(const Point& point : value.first){
				spilled_points->push_back(point);
			}
			// Sequentially create the new nodes
			for(const auto& [child, aabb] : value.second){
				IdAABB child_aabb_index = GlobalVariables::createNewAABB(aabb);
				// spilling_node->children[child] = new OctreeNode(child_aabb_index);
				spilling_node->children[child] = MemoryAllocator::newOctreeNode(child_aabb_index);
				(*relationship_map_ref)[spilling_node->aabb_index][child] = child_aabb_index;
			}
		}

		spilling_nodes->clear();
	}
}


void SimLod::voxelSampling(
    OctreeNode* main_root,
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<Point>>& backlog_voxels,
    std::shared_ptr<vector<OctreeNode*>>& backlog_voxels_nodes
){

	typedef std::vector<std::pair<Point, OctreeNode*>> BackLog;

	auto sampleVoxel = [&](Point& point) -> BackLog {
		// Reach all corresponding inner nodes
		OctreeNode* node = main_root;

		BackLog backlog = {};

		while(true){
			if(!node->occupancy){return backlog;}

			// Find next child
			const AABB& aabb = GlobalVariables::getAABB(node->aabb_index);
			NodePosition child_index = aabb.getNextChildIndex(point.position);
			if(!node->children[child_index]){return backlog;}

			// Sample voxel occupancy grid at this location if the node is inner for this point
			OccupancyGrid::GridIndex index = OccupancyGrid::getCellIndices(aabb, point);
			bool is_cell_occupied = node->occupancy->isCellOcupied(index);

			if(!is_cell_occupied){
				// Fill up occupancy grid
				node->occupancy->markCellAsFilled(index);
				// Create corresponding voxel using this point
				vec3 voxel_centroid = OccupancyGrid::getCellCentroid(aabb, index);
				Point new_voxel = {};
				new_voxel.position = voxel_centroid;
				new_voxel.color[0] = point.color[0];
				new_voxel.color[1] = point.color[1];
				new_voxel.color[2] = point.color[2];

				// Add voxel to backlog buffers
				backlog.push_back({new_voxel, node});
			}

			node = node->children[child_index];
		}
	};


	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		uint32_t nb_points = points->size();
		uint32_t nb_spilled_points = spilled_points->size();
		std::vector<uint32_t> indices(nb_points + nb_spilled_points);
		std::iota(indices.begin(), indices.end(), 0);
		std::vector<BackLog> tmp_backlog = std::vector<BackLog>(nb_points + nb_spilled_points);

		// Create new voxels in parallel
		std::for_each(std::execution::par, indices.begin(), indices.end(), [&](uint32_t index){
			Point* point = nullptr;
			if(index >= nb_points){
				point = &(*spilled_points)[index - nb_points];
			} else {
				point = &(*points)[index];
			}
			tmp_backlog[index] = sampleVoxel(*point);
		});
		// Sequentially add all the voxels in the backlog buffer
		for(auto& backlog : tmp_backlog){
			for(auto& [voxel, node] : backlog){
				backlog_voxels->push_back(voxel);
				backlog_voxels_nodes->push_back(node);
			}
		}
	} else {
		std::for_each(points->begin(), points->end(), [&](Point& point){
			BackLog backlog = sampleVoxel(point);
			for(auto& [voxel, node] : backlog){
				backlog_voxels->push_back(voxel);
				backlog_voxels_nodes->push_back(node);
			}
		});
		std::for_each(spilled_points->begin(), spilled_points->end(), [&](Point& point){
			BackLog backlog = sampleVoxel(point);
			for(auto& [voxel, node] : backlog){
				backlog_voxels->push_back(voxel);
				backlog_voxels_nodes->push_back(node);
			}
		});
	}
}

void SimLod::insertion(
    OctreeNode* main_root,
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<Point>>& backlog_voxels,
    std::shared_ptr<vector<OctreeNode*>>& backlog_voxels_nodes
){

	std::unordered_map<OctreeNode*, std::mutex> points_mtx_map = {};
	std::function<void(OctreeNode*)> mapFill = [&](OctreeNode* root){
		if(!root){return;}
		points_mtx_map[root];
		for(uint32_t i=0; i< 8; i++){
			mapFill(root->children[i]);
		}
	};
	mapFill(main_root);

	auto insertPoint = [&](Point& point, OctreeNode* main_node){
		OctreeNode* cur_node = main_node;
		// Reach all corresponding leaves
		while(true){
			cur_node->updated = true;
			// Find next child
			const AABB& aabb = GlobalVariables::getAABB(cur_node->aabb_index);
			NodePosition child_index = aabb.getNextChildIndex(point.position);
			// If leaf insert point in chunks
			if(cur_node->children[child_index]){
				cur_node = cur_node->children[child_index];
			} else {
				std::lock_guard<std::mutex>lock(points_mtx_map[cur_node]);
				if(!cur_node->points){
					// cur_node->points = new Chunk();
					cur_node->points = MemoryAllocator::newChunk();
				}
				Chunk* chunk_list = cur_node->points;
				while(chunk_list->next){chunk_list = chunk_list->next;}
				if(chunk_list->size == OocSimLodSettings::NB_POINTS_PER_CHUNK){
					// chunk_list->next = new Chunk();
					chunk_list->next = MemoryAllocator::newChunk();
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
		if(!node->voxels){
			// node->voxels = new Chunk();
			node->voxels = MemoryAllocator::newChunk();
		}
		Chunk* chunk_list = node->voxels;
		while(chunk_list->next){chunk_list = chunk_list->next;}
		if(chunk_list->size == OocSimLodSettings::NB_POINTS_PER_CHUNK){
			// chunk_list->next = new Chunk();
			chunk_list->next = MemoryAllocator::newChunk();
			chunk_list = chunk_list->next;
		}
		chunk_list->points[chunk_list->size] = voxel;
		chunk_list->size++;
		return;
	};

	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		std::thread parallel_thread([&](){
			uint32_t nb_new_voxels = backlog_voxels->size();
			for(uint32_t i=0; i<nb_new_voxels; i++){
				insertVoxel((*backlog_voxels)[i], (*backlog_voxels_nodes)[i]);
			}
		});

		std::for_each(std::execution::par, points->begin(), points->end(), [&](Point& point){
			insertPoint(point, main_root);
		});
		std::for_each(std::execution::par, spilled_points->begin(), spilled_points->end(), [&](Point& point){
			insertPoint(point, main_root);
		});

		parallel_thread.join();
	} else {
		std::for_each(points->begin(), points->end(), [&](Point& point){
			insertPoint(point, main_root);
		});
		std::for_each(spilled_points->begin(), spilled_points->end(), [&](Point& point){
			insertPoint(point, main_root);
		});
		uint32_t nb_new_voxels = backlog_voxels->size();
		for(uint32_t i=0; i<nb_new_voxels; i++){
			insertVoxel((*backlog_voxels)[i], (*backlog_voxels_nodes)[i]);
		}
	}

}
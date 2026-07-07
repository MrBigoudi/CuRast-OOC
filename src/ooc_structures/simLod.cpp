#include "simLod.h"
#include "outOfCore.h"

void SimLod::update(
	OctreeNode* main_root, 
	std::shared_ptr<vector<Point>>& points,
	std::shared_ptr<AABBRelationshipMap> relationship_map_ref
){
	// println("//////////////////////////////////////////////////");
	// println("////////// Octree before simlod update ///////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();

	std::shared_ptr<Timing> timing = Timing::addTiming("simlod load", true, 1);
	// load(main_root, points, relationship_map_ref);
	loadWithAtomic(main_root, points, relationship_map_ref);
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("////////// Octree after simlod load //////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();

	std::shared_ptr<Timing> count_split_timing = Timing::addTiming("simlod count/split loop", true, 1);
	while(true){
		std::shared_ptr<Timing> timing = Timing::addTiming("simlod count", true, 2);
		// count(main_root, points, GlobalVariables::spilledPoints, GlobalVariables::spillingNodes);
		countWithAtomic(main_root, points, GlobalVariables::spilledPoints, GlobalVariables::spillingNodes);
		timing->stop_clock();

		if(GlobalVariables::spillingNodes->size() == 0){
			break;
		}
		
		// println("//////////////////////////////////////////////////");
		// println("////////// Octree after simlod counting //////////");
		// println("//////////////////////////////////////////////////");
		// main_root->display();

		timing = Timing::addTiming("simlod split", true, 2);
		split(GlobalVariables::spilledPoints, GlobalVariables::spillingNodes, relationship_map_ref);
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


	timing = Timing::addTiming("simlod voxel sampling", true, 1);
	// voxelSampling(main_root, points, GlobalVariables::spilledPoints, GlobalVariables::backlogVoxels, GlobalVariables::backlogVoxelsNodes);
	voxelSamplingWithAtomic(main_root, points, GlobalVariables::spilledPoints, GlobalVariables::backlogVoxels, GlobalVariables::backlogVoxelsNodes);
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("//////// Octree after simlod voxel sample ////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();
	

	timing = Timing::addTiming("simlod insertion", true, 1);
	// insertion(main_root, points, GlobalVariables::spilledPoints, GlobalVariables::backlogVoxels, GlobalVariables::backlogVoxelsNodes);
	insertionWithAtomic(main_root, points, GlobalVariables::spilledPoints, GlobalVariables::backlogVoxels, GlobalVariables::backlogVoxelsNodes);
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("///////// Octree after simlod insertions /////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();


	// Clean buffers
	timing = Timing::addTiming("simlod buffer cleaning", true, 1);
	GlobalVariables::spilledPoints->clear();
	GlobalVariables::spillingNodes->clear();
	GlobalVariables::backlogVoxels->clear();
	GlobalVariables::backlogVoxelsNodes->clear();
	timing->stop_clock();

}









void SimLod::split(
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<OctreeNode*>>& spilling_nodes,
    std::shared_ptr<AABBRelationshipMap> relationship_map_ref
){
	typedef std::vector<Point> SpilledPoints;
	typedef std::vector<std::pair<NodePosition, AABB>> NewNodes;

	auto lambda = [&](OctreeNode*& spilling_node) -> std::pair<SpilledPoints, NewNodes> {
		NewNodes new_nodes = {};
		SpilledPoints new_spilled_points = {};

		uint8_t spilling_node_children = spilling_node->children_ids;

		spilling_node->counter.store(0);
		spilling_node->children_ids = 0;
		if(!spilling_node->occupancy){
			spilling_node->occupancy = new OccupancyGrid();
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

		delete(spilling_node->points);
		spilling_node->points = nullptr;

		return {new_spilled_points, new_nodes};
	};

	std::unordered_map<OctreeNode*, std::pair<SpilledPoints, NewNodes>> tmp_map = {};
	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		std::for_each(std::execution::par, spilling_nodes->begin(), spilling_nodes->end(), [&](OctreeNode*& spilling_node){
			tmp_map[spilling_node] = lambda(spilling_node);
		});
	} else {
		std::for_each(spilling_nodes->begin(), spilling_nodes->end(), [&](OctreeNode*& spilling_node){
			tmp_map[spilling_node] = lambda(spilling_node);
		});
	}

	for(auto& [spilling_node, value] : tmp_map){
		// Sequentially add the spill points to the buffer
		for(const Point& point : value.first){
			spilled_points->push_back(point);
		}
		// Sequentially create the new nodes
		for(const auto& [child, aabb] : value.second){
			IdAABB child_aabb_index = GlobalVariables::createNewAABB(aabb);
			spilling_node->children[child] = new OctreeNode(child_aabb_index);
			(*relationship_map_ref)[spilling_node->aabb_index][child] = child_aabb_index;
		}
	}

	spilling_nodes->clear();
}





void SimLod::load(
    OctreeNode* main_root,
    std::shared_ptr<vector<Point>>& points,
	std::shared_ptr<AABBRelationshipMap> relationship_map_ref
){

	std::vector<OctreeNode*> all_nodes = GlobalVariables::getAllNodes(main_root);

	// Try to insert all points
	auto tryInsertPoints = [&](OctreeNode* cur_node, std::shared_ptr<vector<Point>>& points){
		std::vector<OctreeNode*> loaded_children = {cur_node};
		uint32_t nb_nodes = 1;

		for(const Point& point : *points){
			for(uint32_t i = 0; i < nb_nodes; i++){
				OctreeNode* node = loaded_children[i];
				const AABB& aabb = GlobalVariables::getAABB(node->aabb_index);
				if(!aabb.contains(point.position)){continue;}

				// Check if the child is already loaded
				NodePosition child_index = aabb.getNextChildIndex(point.position);
				OctreeNode* child = node->children[child_index];
				if(child){continue;}

				// Check if the child has been stored
				IdAABB child_aabb_index = (*relationship_map_ref)[node->aabb_index][child_index];
				if(child_aabb_index == INVALID_ID){continue;}

				// Load the child
				node->children[child_index] = loadOctree(child_aabb_index);
				loaded_children.push_back(node->children[child_index]);
				nb_nodes++;
			}
		}
	};

	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		std::for_each(std::execution::par, all_nodes.begin(), all_nodes.end(), [&](OctreeNode* node){
			tryInsertPoints(node, points);
		});
	} else {
		std::for_each(all_nodes.begin(), all_nodes.end(), [&](OctreeNode* node){
			tryInsertPoints(node, points);
		});
	}
}





void SimLod::count(
    OctreeNode* main_root,
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<OctreeNode*>>& spilling_nodes
){
	std::vector<std::pair<OctreeNode*, uint8_t>> all_nodes = GlobalVariables::getAllPartialLeavesWithLevels(main_root, 1);
	uint32_t nb_nodes = all_nodes.size();
	std::vector<bool> already_spilled = std::vector<bool>(nb_nodes, false);
	std::vector<uint32_t> indices(nb_nodes);
	std::iota(indices.begin(), indices.end(), 0);

	auto countPoints = [&](uint32_t index, std::shared_ptr<vector<Point>> points) {
		OctreeNode* cur_node = all_nodes[index].first;
		const AABB& aabb = GlobalVariables::getAABB(cur_node->aabb_index);

		for(Point& point : *points){
			// Skip if the node does not contain this point
			if(!aabb.contains(point.position)){continue;}
			// Skip if the node is not a leaf for this point
			NodePosition child_index = aabb.getNextChildIndex(point.position);
			if(cur_node->children[child_index]){continue;}

			cur_node->children_ids |= 0x01 << child_index;

			// Flag the point to not count it again on the next iteration
			uint8_t level = all_nodes[index].second;
			if(level == UINT8_MAX){
				println("The octree has reached it's maximum depth size...");
				exit(EXIT_FAILURE);
			}

			// Only one of the node should be able to write on the point as it only arrives to one leaf
			if(point.color[3] == level){continue;}
			point.color[3] = level;

			if(!already_spilled[index]){
				cur_node->counter++;
				if(cur_node->counter == OocSimLodSettings::MAX_POINTS_PER_LEAF){
					already_spilled[index] = true;
				}
			}
		}
	};

	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		// First count in parallel
		std::for_each(std::execution::par, indices.begin(), indices.end(), [&](uint32_t index){
			countPoints(index, points);
			countPoints(index, spilled_points);
		});
		// Then sequentially add the nodes in the spilling nodes pool
		std::for_each(indices.begin(), indices.end(), [&](uint32_t index){
			if(already_spilled[index]){
				spilling_nodes->push_back(all_nodes[index].first);
			}
		});
	} else {
		// Counting and Adding to the spilling nodes pool happen at the same time
		std::for_each(indices.begin(), indices.end(), [&](uint32_t index){
			countPoints(index, points);
			countPoints(index, spilled_points);
			if(already_spilled[index]){
				spilling_nodes->push_back(all_nodes[index].first);
			}
		});
	}
}


void SimLod::voxelSampling(
    OctreeNode* main_root,
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<Point>>& backlog_voxels,
    std::shared_ptr<vector<OctreeNode*>>& backlog_voxels_nodes
){
	std::vector<OctreeNode*> all_nodes = GlobalVariables::getAllPartialNodes(main_root);

	// Creates all the new voxels
	auto sampleVoxels = [&](OctreeNode* cur_node, std::shared_ptr<vector<Point>>& cur_points) -> std::vector<Point> {
		// Skip entirely if the node is a full leaf
		if(!cur_node->occupancy){return {};}

		std::vector<Point> new_voxels = {};
		new_voxels.reserve(cur_points->size());
		for(const Point& point : *cur_points){
			// Skip if the node does not contain this point
			const AABB& aabb = GlobalVariables::getAABB(cur_node->aabb_index);
			if(!aabb.contains(point.position)){continue;}

			// Skip if the node is a leaf for this point
			NodePosition child_index = aabb.getNextChildIndex(point.position);
			if(!cur_node->children[child_index]){continue;}

			// Else sample the voxel grid at this location
			OccupancyGrid::GridIndex index = OccupancyGrid::getCellIndices(aabb, point);
			bool is_cell_occupied = cur_node->occupancy->isCellOcupied(index);

			// Create a new voxel if the grid index was not flagged
			if(!is_cell_occupied){
				cur_node->occupancy->markCellAsFilled(index);
				vec3 voxel_centroid = OccupancyGrid::getCellCentroid(aabb, index);
				Point new_voxel = {};
				new_voxel.position = voxel_centroid;
				new_voxel.color[0] = point.color[0];
				new_voxel.color[1] = point.color[1];
				new_voxel.color[2] = point.color[2];

				new_voxels.push_back(new_voxel);
			}
		}
		return new_voxels;
	};

	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		// Create new voxels in parallel
		std::unordered_map<OctreeNode*, std::vector<std::vector<Point>>> map = {};
		std::for_each(std::execution::par, all_nodes.begin(), all_nodes.end(), [&](OctreeNode* node){
			std::vector<Point> new_voxels = sampleVoxels(node, points);
			std::vector<Point> new_voxels_spilled = sampleVoxels(node, spilled_points);
			if(new_voxels.empty() && new_voxels_spilled.empty()){return;}
			if(new_voxels.empty()){
				map[node] = {new_voxels_spilled};
				return;
			}
			if(new_voxels_spilled.empty()){
				map[node] = {new_voxels};
				return;
			}
			map[node] = {new_voxels, new_voxels_spilled};
		});
		// Sequentially add all the voxels in the backlog buffer
		for(auto& [node, lists] : map){
			for(const std::vector<Point>& new_voxels : lists){
				for(const Point& voxel : new_voxels){
					backlog_voxels->push_back(voxel);
					backlog_voxels_nodes->push_back(node);
				}
			}
		}
	} else {
		std::for_each(all_nodes.begin(), all_nodes.end(), [&](OctreeNode* node){
			std::vector<Point> new_voxels = sampleVoxels(node, points);
			for(const Point& voxel : new_voxels){
				backlog_voxels->push_back(voxel);
				backlog_voxels_nodes->push_back(node);
			}
			std::vector<Point> new_voxels_spilled = sampleVoxels(node, spilled_points);
			for(const Point& voxel : new_voxels_spilled){
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
	std::vector<OctreeNode*> all_nodes = GlobalVariables::getAllNodes(main_root);

	auto insertPoints = [&](OctreeNode* cur_node, std::shared_ptr<vector<Point>>& cur_points){
		Chunk* chunk_list = cur_node->points;
		if(chunk_list){
			while(chunk_list->next){chunk_list = chunk_list->next;}
		}

		for(const Point& point : *cur_points){
			// Skip if the node does not contain this point
			const AABB& aabb = GlobalVariables::getAABB(cur_node->aabb_index);
			if(!aabb.contains(point.position)){continue;}
			cur_node->updated = true;

			// Skip if the node is not a leaf for this point
			NodePosition child_index = aabb.getNextChildIndex(point.position);
			if(cur_node->children[child_index]){continue;}

			// Insert the point to the chunk list
			if(!chunk_list){
				cur_node->points = new Chunk();
				chunk_list = cur_node->points;
			}
			if(chunk_list->size == OocSimLodSettings::NB_POINTS_PER_CHUNK){
				chunk_list->next = new Chunk();
				chunk_list = chunk_list->next;
			}
			chunk_list->points[chunk_list->size] = point;
			chunk_list->size++;
		}
	};

	auto insertVoxels = [&](OctreeNode* cur_node){
		Chunk* chunk_list = cur_node->voxels;
		if(chunk_list){
			while(chunk_list->next){chunk_list = chunk_list->next;}
		}

		for(uint32_t i=0; i<backlog_voxels->size(); i++){
			const Point& voxel = (*backlog_voxels)[i];
			OctreeNode* voxel_node = (*backlog_voxels_nodes)[i];

			if(voxel_node != cur_node){continue;}
			cur_node->updated = true;

			// Insert the voxel to the chunk list
			if(!chunk_list){
				cur_node->voxels = new Chunk();
				chunk_list = cur_node->voxels;
			}
			if(chunk_list->size == OocSimLodSettings::NB_POINTS_PER_CHUNK){
				chunk_list->next = new Chunk();
				chunk_list = chunk_list->next;
			}
			chunk_list->points[chunk_list->size] = voxel;
			chunk_list->size++;
		}
	};

	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		std::for_each(std::execution::par, all_nodes.begin(), all_nodes.end(), [&](OctreeNode* node){
			insertPoints(node, points);
			insertPoints(node, spilled_points);
			insertVoxels(node);
		});
	} else {
		std::for_each(all_nodes.begin(), all_nodes.end(), [&](OctreeNode* node){
			insertPoints(node, points);
			insertPoints(node, spilled_points);
			insertVoxels(node);
		});
	}
}



std::unordered_map<OctreeNode*, std::vector<uint32_t>> SimLod::fillPoints(
	OctreeNode* main_root, 
	std::shared_ptr<vector<Point>>& points,
	uint32_t first_index
){
	std::unordered_map<OctreeNode*, std::vector<uint32_t>> res = {};

	for(uint32_t i=first_index; i<points->size(); i++){
		const Point& point = (*points)[i];
		OctreeNode* leaf = main_root;

		while(true){
			const AABB& aabb = GlobalVariables::getAABB(leaf->aabb_index);
			if(!aabb.contains(point.position)){
				break;
			}
			if(!res.contains(leaf)){res[leaf] = {};}
			res[leaf].push_back(i);

			NodePosition child_index = aabb.getNextChildIndex(point.position);
			// If not leaf continue
			if(leaf->children[child_index]){
				leaf = leaf->children[child_index];
			}
		}
	}

	return res;
}













































void SimLod::countWithAtomic(
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
					exit(EXIT_FAILURE);
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

	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		uint32_t nb_points = points->size();
		uint32_t nb_spilled_points = spilled_points->size();
		std::vector<uint32_t> indices(nb_points + nb_spilled_points);
		std::iota(indices.begin(), indices.end(), 0);
		std::vector<OctreeNode*> tmp_spilled = std::vector<OctreeNode*>(nb_points + nb_spilled_points, nullptr);

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
		// Add spilling nodes sequentially
		for(OctreeNode* node : tmp_spilled){
			if(node){spilling_nodes->push_back(node);}
		}
	} else {
		std::for_each(points->begin(), points->end(), [&](Point& point){
			std::optional<OctreeNode*> spilled = countPoint(point);
			if(spilled.has_value()){spilling_nodes->push_back(spilled.value());}
		});
		std::for_each(spilled_points->begin(), spilled_points->end(), [&](Point& point){
			std::optional<OctreeNode*> spilled = countPoint(point);		
			if(spilled.has_value()){spilling_nodes->push_back(spilled.value());}
		});
	}
}



void SimLod::voxelSamplingWithAtomic(
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

void SimLod::insertionWithAtomic(
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
			const AABB& aabb = GlobalVariables::getAABB(cur_node->aabb_index);
			NodePosition child_index = aabb.getNextChildIndex(point.position);
			// If leaf insert point in chunks
			if(cur_node->children[child_index]){
				cur_node = cur_node->children[child_index];
			} else {
				if(!cur_node->points){cur_node->points = new Chunk();}
				Chunk* chunk_list = cur_node->points;
				while(chunk_list->next){chunk_list = chunk_list->next;}
				if(chunk_list->size == OocSimLodSettings::NB_POINTS_PER_CHUNK){
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
		if(chunk_list->size == OocSimLodSettings::NB_POINTS_PER_CHUNK){
			chunk_list->next = new Chunk();
			chunk_list = chunk_list->next;
		}
		chunk_list->points[chunk_list->size] = voxel;
		chunk_list->size++;
		return;
	};

	if(!OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
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



void SimLod::loadWithAtomic(
    OctreeNode* main_root,
    std::shared_ptr<vector<Point>>& points,
	std::shared_ptr<AABBRelationshipMap> relationship_map_ref
){

	// tmp_ser is here to avoid loading a node multiple time in the parallel context
	mutex mtx_set;
	std::unordered_set<IdAABB> tmp_set = {};


	// Try to insert all points
	auto tryInsertPoint = [&](Point& point, OctreeNode* main_root){
		// Reach corresponding leaf
		OctreeNode* leaf = main_root;

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
					exit(EXIT_FAILURE);
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

	if(!OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		for(Point& point : *points){
			tryInsertPoint(point, main_root);
		}
	} else {
		vector<uint32_t> first_indices = {};
		uint32_t step_size = 100'000u;
		uint32_t nb_points = points->size();
		for(uint32_t i=0; i<nb_points; i+=step_size){first_indices.push_back(i);}

		std::for_each(std::execution::par, first_indices.begin(), first_indices.end(), [&](uint32_t index){
			for(uint32_t i=0; i<step_size; i++){
				if((index + i) >= nb_points){break;}
				tryInsertPoint((*points)[index + i], main_root);
			}
		});
	}

	// if(!tmp_set.empty()){
	// 	println("nb loaded nodes = {}", tmp_set.size());
	// }
}
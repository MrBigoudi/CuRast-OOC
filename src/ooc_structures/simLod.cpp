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
	load(main_root, points, relationship_map_ref);
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("////////// Octree after simlod load //////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();

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
	voxelSampling(main_root, points, GlobalVariables::spilledPoints, GlobalVariables::backlogVoxels, GlobalVariables::backlogVoxelsNodes);
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("//////// Octree after simlod voxel sample ////////");
	// println("//////////////////////////////////////////////////");
	// main_root->display();
	

	timing = Timing::addTiming("simlod insertion", true, 1);
	insertion(main_root, points, GlobalVariables::spilledPoints, GlobalVariables::backlogVoxels, GlobalVariables::backlogVoxelsNodes);
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

		spilling_node->counter = 0;
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
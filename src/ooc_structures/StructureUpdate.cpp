#include "StructureUpdate.h"

#include "laszip/laszip_api.h"
#include "globals.h"

///////////////////////////////////////////////////////////////////////////////
////////////////////////////// GLOBAL VARIABLES ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// The batches that still need to be read
std::deque<PointBatch> batchesToLoad = {};
/// The batches that are already loaded
std::deque<PointBatch> batchesLoaded = {};
/// The main octree
std::shared_ptr<OctreeNode> mainOctree = std::make_shared<OctreeNode>(OctreeNode());
/// The main bounding box
std::shared_ptr<AABB> mainAABB = std::make_shared<AABB>(AABB());

/// The buffer of spilled points
std::shared_ptr<vector<Point>> spilledPoints = std::make_shared<vector<Point>>(vector<Point>());
/// The buffer of spilling nodes
std::shared_ptr<vector<OctreeNode*>> spillingNodes = std::make_shared<vector<OctreeNode*>>(vector<OctreeNode*>());

/// The backlog buffer for new voxels
std::shared_ptr<vector<Point>> backlogVoxels = std::make_shared<vector<Point>>(vector<Point>());
/// The backlog buffer for the nodes corresponding to the new voxels
std::shared_ptr<vector<OctreeNode*>> backlogVoxelsNodes = std::make_shared<vector<OctreeNode*>>(vector<OctreeNode*>());


///////////////////////////////////////////////////////////////////////////////
////////////////////////////// HELPER FUNCTIONS ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void initLoadPointBatches(string file){
	// Basic checks
	if(!fs::exists(file)){
		println("ERROR: file '{}' does not exist", file);
		return;
	}
	if(!iEndsWith(file, "las") && !iEndsWith(file, "laz")){
		println("ERROR: file '{}' doesn't have a supported file type", file);
		return;
	}

	// Load header
	laszip_POINTER laszip_reader;
	if(laszip_create(&laszip_reader)){
		println("ERROR: creating laszip reader for '{}'", file);
		return;
	}
	laszip_BOOL is_compressed = 0;
	if(laszip_open_reader(laszip_reader, file.c_str(), &is_compressed)){
		println("ERROR: opening laszip reader for '{}'", file);
		laszip_destroy(laszip_reader);
		return;
	}
	laszip_header* header;
	if(laszip_get_header_pointer(laszip_reader, &header)){
		println("ERROR: getting laszip header pointer for '{}'", file);
		laszip_close_reader(laszip_reader);
		laszip_destroy(laszip_reader);
		return;
	}
	std::shared_ptr<laszip_header> shared_header = std::make_shared<laszip_header>(*header);
	std::shared_ptr<string> shared_file = std::make_shared<string>(file);

	// Create batches
	uint64_t num_points = header->number_of_point_records ? header->number_of_point_records : header->extended_number_of_point_records;
	for(uint64_t first_point = 0; first_point < num_points; first_point += MAX_BATCH_SIZE){
		PointBatch new_batch = {};
		new_batch.file = shared_file;
		new_batch.header = shared_header;
		new_batch.first = first_point;
		new_batch.count = std::min(num_points - first_point, MAX_BATCH_SIZE);
		batchesToLoad.push_back(new_batch);
	}
}


void loadPointsInBatches(){
	// TODO: in parallel
	while(!batchesToLoad.empty()){
		PointBatch& batch = batchesToLoad.front();

		laszip_POINTER laszip_reader;
		if(laszip_create(&laszip_reader)){
			println("ERROR: creating laszip reader for '{}'", *batch.file);
			return;
		}
		laszip_BOOL is_compressed = 0;
		if(laszip_open_reader(laszip_reader, (*batch.file).c_str(), &is_compressed)){
			println("ERROR: opening laszip reader for '{}'", *batch.file);
			laszip_destroy(laszip_reader);
			return;
		}
		laszip_point* laz_point;
		if(laszip_get_point_pointer(laszip_reader, &laz_point)){
			println("ERROR: getting laszip point pointer for '{}'", *batch.file);
			laszip_close_reader(laszip_reader);
			laszip_destroy(laszip_reader);
			return;
		}
		if(laszip_seek_point(laszip_reader, batch.first)){
			println("ERROR: seeking laszip point for for '{}'", *batch.file);
			laszip_close_reader(laszip_reader);
			laszip_destroy(laszip_reader);
			return;	
		}

		double scale_x = batch.header->x_scale_factor;
		double scale_y = batch.header->y_scale_factor;
		double scale_z = batch.header->z_scale_factor;
		double offset_x = batch.header->x_offset;
		double offset_y = batch.header->y_offset;
		double offset_z = batch.header->z_offset;

		uint8_t fmt = batch.header->point_data_format;
		bool has_rgb = (fmt == 2 || fmt == 3 || fmt == 5 || fmt == 7 || fmt == 8 || fmt == 10);
		batch.points = std::make_shared<vector<Point>>(vector<Point>());

		for (uint64_t i = 0; i < batch.count; i++) {
			if(laszip_read_point(laszip_reader)){
				println("ERROR: reading point {} for '{}'", i+batch.first, *batch.file);
				break;
			}

			Point new_point = {};
			float x = (float)(laz_point->X * scale_x + offset_x);
			float y = (float)(laz_point->Y * scale_y + offset_y);
			float z = (float)(laz_point->Z * scale_z + offset_z);
			new_point.position = {x,y,z};

			if(has_rgb){
				// LAS RGB is 16-bit; many writers use the high byte, some use the low byte
				for(size_t j=0; j<3; j++){
					new_point.color[j] = laz_point->rgb[j] > 255 ? (uint8_t)(laz_point->rgb[j] >> 8) : (uint8_t)laz_point->rgb[j];
				}
			} else {
				uint8_t intensity = (uint8_t)(laz_point->intensity >> 8);
				for(size_t j=0; j<3; j++){
					new_point.color[j] = intensity;
				}
			}

			batch.points->push_back(new_point);
		}

		println("done loading a batch of {} points", batch.count);

		laszip_close_reader(laszip_reader);
		batchesLoaded.push_back(batch);
		batchesToLoad.pop_front();
	}
}

void loadBatchesOnGPU(CuRast* editor){
	while(!batchesLoaded.empty()){
		PointBatch& batch = batchesLoaded.front();

		// Upload positions and colors to GPU
		CUdeviceptr cptr_positions, cptr_colors;
		cuMemAlloc(&cptr_positions, batch.count * sizeof(vec3));
		cuMemAlloc(&cptr_colors,    batch.count * sizeof(uint32_t));
		cuMemcpyHtoD(cptr_positions, batch.getPositions().data(), batch.count * sizeof(vec3));
		cuMemcpyHtoD(cptr_colors, batch.getColors().data(),batch.count * sizeof(uint32_t));

		auto node = make_shared<SNCPoints>("pointcloud");
		node->cptr_positions = cptr_positions;
		node->cptr_colors    = cptr_colors;
		node->numPoints      = batch.count;

		editor->scene.world->children.push_back(node);

		batchesLoaded.pop_front();
	}
}

// TODO: temporary function to load synchronously the point cloud
void loadPointcloud(string file, CuRast* editor){
	uint32_t timing_id = timingsList.size();
	timingsList.emplace_back(format("init load point batches v{}", NbLoadedClouds), true);
	initLoadPointBatches(file);
	timingsList[timing_id].stop_clock();
	

	timing_id = timingsList.size();
	timingsList.emplace_back(format("load points in batches v{}", NbLoadedClouds), true);
	loadPointsInBatches();
	timingsList[timing_id].stop_clock();


	timing_id = timingsList.size();
	timingsList.emplace_back(format("init octree v{}", NbLoadedClouds), true);
	if(NbLoadedClouds == 0){
		initOctree(mainOctree, mainAABB, batchesLoaded[0].points);
	}
	timingsList[timing_id].stop_clock();


	timing_id = timingsList.size();
	timingsList.emplace_back(format("compute max new level v{}", NbLoadedClouds), true);
	// Compute max new level needed per batch
	uint32_t nb_loaded = batchesLoaded.size();
	vector<uint32_t> tmp_new_levels = vector<uint32_t>(nb_loaded, 0);
	// In parallel
	{
		for(uint32_t i=0; i < nb_loaded; i++){
			PointBatch& batch = batchesLoaded[i];
			tmp_new_levels[i] = growOctree(mainOctree, mainAABB, batch.points);
		}
	}
	timingsList[timing_id].stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("//////////// Octree after grow octree ////////////");
	// println("//////////////////////////////////////////////////");
	// mainOctree->display();


	timing_id = timingsList.size();
	timingsList.emplace_back(format("update octree bottom up v{}", NbLoadedClouds), true);
	// In single thread
	{
		uint32_t nb_new_levels = 0;
		for(uint32_t& level : tmp_new_levels){
			nb_new_levels = max(nb_new_levels, level);
		}
		println("Max new level: {}", nb_new_levels);

		uptadeOctree(mainOctree, mainAABB, nb_new_levels);
	}
	timingsList[timing_id].stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("/////////// Octree after update octree ///////////");
	// println("//////////////////////////////////////////////////");
	// mainOctree->display();


	timing_id = timingsList.size();
	timingsList.emplace_back(format("simlod update v{}", NbLoadedClouds), true);
	// In parallel
	{
		for(uint32_t i=0; i<nb_loaded; i++){
			PointBatch& batch = batchesLoaded[i];
			simLodUpdate(mainOctree, mainAABB, batch.points);
		}
	}
	timingsList[timing_id].stop_clock();


	// println("//////////////////////////////////////////////////");
	// println("/////////// Octree after simLOD update ///////////");
	// println("//////////////////////////////////////////////////");
	// mainOctree->display();

	timing_id = timingsList.size();
	timingsList.emplace_back(format("send points to GPU v{}", NbLoadedClouds), true);
	loadBatchesOnGPU(editor);
	timingsList[timing_id].stop_clock();


	timing_id = timingsList.size();
	timingsList.emplace_back(format("send octree to GPU v{}", NbLoadedClouds), true);
	loadOctree(editor, mainOctree, mainAABB);
	timingsList[timing_id].stop_clock();

	NbLoadedClouds++;
};




void simLodCount(
    std::shared_ptr<OctreeNode>& main_root, 
    std::shared_ptr<AABB>& main_aabb, 
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<OctreeNode*>>& spilling_nodes
){
	auto countPoint = [&](Point& point){
		// Reach corresponding leaf
		std::shared_ptr<OctreeNode> leaf = main_root;
		AABB current_aabb = {.mins = main_aabb->mins, .maxs = main_aabb->maxs };

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

				uint32_t old_counter = leaf->counter;
				leaf->counter = min(uint16_t(leaf->counter + 1u), uint16_t(MAX_POINTS_PER_LEAF + 1u));

				// Flag point as accepted at this level
				point.color[3] = level;

				if(leaf->counter > MAX_POINTS_PER_LEAF && old_counter <= MAX_POINTS_PER_LEAF){
					spilling_nodes->push_back(leaf.get());
				}
				return;
			}
		}
	};

	for(Point& point : *points){
		countPoint(point);
	}
	for(Point& point : *spilled_points){
		countPoint(point);		
	}
}

void simLodSplit(
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<OctreeNode*>>& spilling_nodes
){
	for(uint32_t node_id = 0; node_id < spilling_nodes->size(); node_id++){
		OctreeNode*& spilling_node = (*spilling_nodes)[node_id];
		uint8_t spilling_node_children = spilling_node->children_ids;

	// for(OctreeNode*& spilling_node : *spilling_nodes){
		// if(spilling_node->voxels || spilling_node->occupancy){
		// 	println("An inner node should not be spliteable");
		// 	exit(EXIT_FAILURE);
		// }
		
		// // Check if the node is inner
		// bool is_inner = false;
		// for(uint32_t j=0; j<8; j++){
		// 	if(spilling_node->children[j]){
		// 		is_inner = true;
		// 		break;
		// 	}
		// }

		// spilling_node->is_leaf = false;
		spilling_node->counter = 0;
		if(!spilling_node->occupancy){
			spilling_node->occupancy = std::make_shared<OccupancyGrid>(OccupancyGrid());
		}

		for(uint32_t j=0; j<8; j++){
			// Create necessary empty children
			if(!spilling_node->children[j] && (0x01 << j) & spilling_node_children){
				std::shared_ptr<OctreeNode> empty_child = std::make_shared<OctreeNode>(OctreeNode());
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

static uint32_t staticCpt = 0;
void simLodUpdate(std::shared_ptr<OctreeNode>& main_root, std::shared_ptr<AABB>& main_aabb, std::shared_ptr<vector<Point>>& points){
	uint32_t timing_id = timingsList.size();
	timingsList.emplace_back(format("simlod count/split loop v{}-{}", NbLoadedClouds, staticCpt), true, 1);
	while(true){
		simLodCount(main_root, main_aabb, points, spilledPoints, spillingNodes);
		if(spillingNodes->size() == 0){
			break;
		}
		
		// println("//////////////////////////////////////////////////");
		// println("////////// Octree after simlod counting //////////");
		// println("//////////////////////////////////////////////////");
		// mainOctree->display();

		simLodSplit(spilledPoints, spillingNodes);

		// println("//////////////////////////////////////////////////");
		// println("////////// Octree after simlod splitting /////////");
		// println("//////////////////////////////////////////////////");
		// mainOctree->display();

	}
	timingsList[timing_id].stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("//////// Octree after simlod count/splits ////////");
	// println("//////////////////////////////////////////////////");
	// mainOctree->display();


	timing_id = timingsList.size();
	timingsList.emplace_back(format("simlod voxel sampling v{}-{}", NbLoadedClouds, staticCpt), true, 1);
	simLodVoxelSampling(main_root, main_aabb, points, spilledPoints, backlogVoxels, backlogVoxelsNodes);
	timingsList[timing_id].stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("//////// Octree after simlod voxel sample ////////");
	// println("//////////////////////////////////////////////////");
	// mainOctree->display();
	

	timing_id = timingsList.size();
	timingsList.emplace_back(format("simlod insertion v{}-{}", NbLoadedClouds, staticCpt), true, 1);
	simLodInsertion(main_root, main_aabb, points, spilledPoints, backlogVoxels, backlogVoxelsNodes);
	timingsList[timing_id].stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("///////// Octree after simlod insertions /////////");
	// println("//////////////////////////////////////////////////");
	// mainOctree->display();


	// Clean buffers
	timing_id = timingsList.size();
	timingsList.emplace_back(format("simlod buffer cleaning v{}-{}", NbLoadedClouds, staticCpt), true, 1);
	spilledPoints->clear();
	spillingNodes->clear();
	backlogVoxels->clear();
	backlogVoxelsNodes->clear();
	timingsList[timing_id].stop_clock();

	staticCpt++;
}


void uptadeOctree(std::shared_ptr<OctreeNode>& main_root, std::shared_ptr<AABB>& main_aabb, uint32_t nb_new_levels){
	NodePosition node_position = FIRST_NODE_POSITION;
	for(uint32_t i=0; i<nb_new_levels; i++){
		// Create new parent
		std::shared_ptr<OctreeNode> new_parent = std::make_shared<OctreeNode>(OctreeNode());
		// new_parent->is_leaf = false;
		new_parent->occupancy = std::make_shared<OccupancyGrid>(OccupancyGrid());

		// // Create 7 other empty children
		// for(uint32_t j=0; j<8; j++){
		// 	// Put old main node in correct place in the parent
		// 	if(j != node_position){
		// 		std::shared_ptr<OctreeNode> empty_child = std::make_shared<OctreeNode>(OctreeNode());
		// 		// empty_child->is_leaf = true;
		// 		new_parent->children[j] = empty_child;
		// 	} else {
		// 		new_parent->children[j] = main_root;
		// 	}
		// }
		// Create the correct child
		new_parent->children[node_position] = main_root;
		new_parent->children_ids |= 0x01 << node_position;

		auto fillOccupancyGrid = [&](std::shared_ptr<Chunk> child_chunk_list){
			while(child_chunk_list){
				for(uint32_t j=0; j<child_chunk_list->size; j++){
					Point point = child_chunk_list->points[j];
					// Sample voxel occupancy grid at this location
					vec3 normalized_coordinates = main_aabb->getPointNormalizedCoordinates(point.position);
					uint32_t grid_x = clamp(uint32_t(floor(GRID_SIZE * normalized_coordinates.x)), 0u, GRID_SIZE - 1u);
					uint32_t grid_y = clamp(uint32_t(floor(GRID_SIZE * normalized_coordinates.y)), 0u, GRID_SIZE - 1u);
					uint32_t grid_z = clamp(uint32_t(floor(GRID_SIZE * normalized_coordinates.z)), 0u, GRID_SIZE - 1u);
					uint32_t index = grid_x + GRID_SIZE * (grid_y + GRID_SIZE * grid_z);
					uint32_t word_index = index >> 5u;
					uint32_t bit_index = index & 31u;
					bool is_cell_occupied = (new_parent->occupancy->values[word_index] & (1u << bit_index)) != 0;

					// Fill up occupancy grid
					if(!is_cell_occupied){
						new_parent->occupancy->values[word_index] |= (1u << bit_index);
						// Create corresponding voxel using this point
						Point new_voxel = {.position = main_aabb->getCentroid() };
						new_voxel.color[0] = point.color[0];
						new_voxel.color[1] = point.color[1];
						new_voxel.color[2] = point.color[2];

						// Add voxel to voxels chunk list
						if(!new_parent->voxels){new_parent->voxels = std::make_shared<Chunk>(Chunk());}
						std::shared_ptr<Chunk> parent_chunk_list = new_parent->voxels;
						while(parent_chunk_list->next){parent_chunk_list = parent_chunk_list->next;}
						if(parent_chunk_list->size == POINTS_PER_CHUNK){
							parent_chunk_list->next = std::make_shared<Chunk>(Chunk());
							parent_chunk_list = parent_chunk_list->next;
						}
						parent_chunk_list->points[parent_chunk_list->size] = new_voxel;
						parent_chunk_list->size++;

						// // Increase parent counter
						// new_parent->counter++;			
					}
				}
				child_chunk_list = child_chunk_list->next;
			}
		};

		// Sample voxels to fill new occupancy grid
		// std::shared_ptr<Chunk> child_chunk_list = main_root->is_leaf ? main_root->points : main_root->voxels;
		fillOccupancyGrid(main_root->points);
		fillOccupancyGrid(main_root->voxels);		

		main_aabb->extend(node_position);
		main_root = new_parent;
		updateNodePosition(node_position);
	}

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
		AABB current_aabb = {.mins = main_aabb->mins, .maxs = main_aabb->maxs };

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
				Point new_voxel = {.position = current_aabb.getCentroid() };
				new_voxel.color[0] = point.color[0];
				new_voxel.color[1] = point.color[1];
				new_voxel.color[2] = point.color[2];

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
		AABB current_aabb = {.mins = main_aabb->mins, .maxs = main_aabb->maxs };

		while(true){
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
				if(!node->points){node->points = std::make_shared<Chunk>(Chunk());}
				std::shared_ptr<Chunk> chunk_list = node->points;
				while(chunk_list->next){chunk_list = chunk_list->next;}
				if(chunk_list->size == POINTS_PER_CHUNK){
					chunk_list->next = std::make_shared<Chunk>(Chunk());
					chunk_list = chunk_list->next;
				}
				chunk_list->points[chunk_list->size] = point;
				chunk_list->size++;
				return;
			}
		}
	};

	auto insertVoxel = [&](Point& voxel, OctreeNode* node){
		if(!node->voxels){node->voxels = std::make_shared<Chunk>(Chunk());}
		std::shared_ptr<Chunk> chunk_list = node->voxels;
		while(chunk_list->next){chunk_list = chunk_list->next;}
		if(chunk_list->size == POINTS_PER_CHUNK){
			chunk_list->next = std::make_shared<Chunk>(Chunk());
			chunk_list = chunk_list->next;
		}
		chunk_list->points[chunk_list->size] = voxel;
		chunk_list->size++;
		return;
	};

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
}

void mainLoop(){
	// Init everything
	// while true:
		// CPU side, in parallel:
			// - Load points on CPU
			// - Upload points to GPU
			// - Run render / grow / update kernels
		// GPU side:
			// - Render
			// - Grow, for every thread in parallel, compute nb new levels
			// - One thread creates the new children
			// - Update
}

void initOctree(std::shared_ptr<OctreeNode>& main_root, std::shared_ptr<AABB>& main_aabb, std::shared_ptr<vector<Point>>& points){
	for(Point& point : *points){
		main_aabb->maxs.x = std::max(main_aabb->maxs.x, point.position.x);
		main_aabb->maxs.y = std::max(main_aabb->maxs.y, point.position.y);
		main_aabb->maxs.z = std::max(main_aabb->maxs.z, point.position.z);
		main_aabb->mins.x = std::min(main_aabb->mins.x, point.position.x);
		main_aabb->mins.y = std::min(main_aabb->mins.y, point.position.y);
		main_aabb->mins.z = std::min(main_aabb->mins.z, point.position.z);
	}
	// Make it cubic
	vec3 size = main_aabb->getSize();
	vec3 half_size = 0.5f * size;
	if(size.x > size.y){
		if(size.x > size.z){
			main_aabb->mins.y -= half_size.y;
			main_aabb->maxs.y += half_size.y;
			main_aabb->mins.z -= half_size.z;
			main_aabb->maxs.z += half_size.z;
		} else {
			main_aabb->mins.y -= half_size.y;
			main_aabb->maxs.y += half_size.y;
			main_aabb->mins.x -= half_size.x;
			main_aabb->maxs.x += half_size.x;
		}
	} else {
		if(size.y > size.z){
			main_aabb->mins.x -= half_size.x;
			main_aabb->maxs.x += half_size.x;
			main_aabb->mins.z -= half_size.z;
			main_aabb->maxs.z += half_size.z;
		} else {
			main_aabb->mins.y -= half_size.y;
			main_aabb->maxs.y += half_size.y;
			main_aabb->mins.x -= half_size.x;
			main_aabb->maxs.x += half_size.x;
		}
	}

	// Adding small 1% delta to avoid floating point issues
	float epsilon = 0.01f;
	main_aabb->mins -= epsilon * main_aabb->mins;
	main_aabb->maxs += epsilon * main_aabb->maxs;
}

uint32_t growOctree(std::shared_ptr<OctreeNode>& main_root, std::shared_ptr<AABB>& main_aabb, std::shared_ptr<vector<Point>>& points){
	uint32_t nb_new_levels = 0;
	AABB new_aabb = *main_aabb;
	NodePosition node_position = FIRST_NODE_POSITION;
	// For each point in a batch check if fits in current AABB
	for(Point& point : *points){
		while(!new_aabb.contains(point.position)){
			// Create new roots considering main box as successively the 1st, 2nd, ..., 8th child to build octree in spiral
			vec3 size = new_aabb.getSize();
			nb_new_levels++;
			new_aabb.extend(node_position);
			updateNodePosition(node_position);
		}
	}
	return nb_new_levels;
}



// TODO: temporary function
void loadOctree(CuRast* editor, const std::shared_ptr<OctreeNode>& main_root, const std::shared_ptr<AABB>& main_aabb){
	// Create cuda memory pointers
	auto octree = make_shared<SNCOctree>("octree");
	octree->cptr_nodes = {};
	octree->cptr_aabbs = {};
	octree->cptr_chunks = {};

	// Create enough chunks
	uint32_t chunk_counter = 0;
	uint32_t nodes_counter = 0;
	uint32_t max_lod_level = 0;

	std::function<CUdeviceptr(const std::shared_ptr<OctreeNode>&, const std::shared_ptr<AABB>&, uint32_t)> recursive = [&](
		const std::shared_ptr<OctreeNode>& cur_node, const std::shared_ptr<AABB>& cur_aabb, uint8_t level
	) -> CUdeviceptr {

		CUdeviceptr child_indices[8] = {0};
		
		// if(!cur_node->is_leaf){
			for(uint32_t child = 0; child < 8; child++){
				if(cur_node->children[child]){
					std::shared_ptr<AABB> new_aabb = std::make_shared<AABB>(AABB());
					new_aabb->mins = cur_aabb->mins;
					new_aabb->maxs = cur_aabb->maxs;
					new_aabb->shrink((NodePosition)child);
					if(level == UINT8_MAX){
						println("Can't have a level greater than {}", UINT8_MAX);
						exit(EXIT_FAILURE);
					}
					child_indices[child] = recursive(cur_node->children[child], new_aabb, level+1);
				}
			}
		// }

		
		// Chunk* cur_chunk = nullptr;
		auto allocateChunks = [&](Chunk* root) -> std::optional<uint32_t> {
			uint32_t before_chunk_counter = chunk_counter;

			// Create CChunks
			vector<Chunk*> chunks = {};

			Chunk* cur_chunk = root;
			while(cur_chunk){
				chunks.push_back(cur_chunk);
				cur_chunk = cur_chunk->next.get();
			}
			for(int32_t i=chunks.size()-1; i>=0; i--){
				CChunk tmp = {};
				tmp.size = chunks[i]->size;
				tmp.next = nullptr;
				for(uint32_t j=0; j<tmp.size; j++){
					CPoint tmp_point = {
						.position = chunks[i]->points[j].position, 
						.color = (uint32_t)chunks[i]->points[j].color[0]
							| ((uint32_t)chunks[i]->points[j].color[1] << 8)
							| ((uint32_t)chunks[i]->points[j].color[2] << 16)
							| (0xFFu << 24)
					};
					tmp.points[j] = tmp_point;
				}
				if(chunks[i]->next){
					tmp.next = (CChunk*)octree->cptr_chunks[chunk_counter-1];
				}
				octree->cptr_chunks.push_back(CUdeviceptr());
				cuMemAlloc(&octree->cptr_chunks[chunk_counter], sizeof(CChunk));
				cuMemcpyHtoD(octree->cptr_chunks[chunk_counter], &tmp, sizeof(CChunk));
				chunk_counter++;
			}

			if(chunk_counter > before_chunk_counter){
				return chunk_counter-1;
			} else {
				return nullopt;
			}
		};
		std::optional<uint32_t> chunk_first_points = nullopt;
		std::optional<uint32_t> chunk_first_voxels = nullopt;
		if(cur_node->points){
			chunk_first_points = allocateChunks(cur_node->points.get());
		} 
		if(cur_node->voxels){
			chunk_first_voxels = allocateChunks(cur_node->voxels.get());
		}
		

		// Create COctreeNode
		COctreeNode new_node = {};
		// if(!cur_node->is_leaf){
			for(uint32_t child = 0; child < 8; child++){
				if(cur_node->children[child]){
					new_node.children[child] = (COctreeNode*) child_indices[child];
				}
			}
		// }
		new_node.counter = cur_node->counter;
		// new_node.is_leaf = cur_node->is_leaf;
		new_node.children_ids = cur_node->children_ids;
		new_node.level = level;

		// if(cur_node->is_leaf && !chunks.empty()){
		if(chunk_first_points.has_value()){
			new_node.points = (CChunk*)octree->cptr_chunks[chunk_first_points.value()];
		}
		// if(!cur_node->is_leaf && !chunks.empty()){
		if(chunk_first_voxels.has_value()){
			new_node.voxels = (CChunk*)octree->cptr_chunks[chunk_first_voxels.value()];
			new_node.occupancy = COccupancyGrid();
			for(uint32_t i=0; i<C_GRID_NUM_CELLS; i++){
				new_node.occupancy.values[i] = cur_node->occupancy->values[i];
			}
		}
		if(level > max_lod_level){
			max_lod_level = level;
		}

		CAABB new_aabb = {
			.mins = cur_aabb->mins,
			.maxs = cur_aabb->maxs,
		};

		// Create cuda pointers
		CUdeviceptr cptr_node, cptr_aabb;
		cuMemAlloc(&cptr_node, sizeof(COctreeNode)); 
		cuMemAlloc(&cptr_aabb, sizeof(CAABB));
		cuMemcpyHtoD(cptr_node, &new_node, sizeof(COctreeNode));
		cuMemcpyHtoD(cptr_aabb, &new_aabb, sizeof(CAABB));
		
		octree->cptr_nodes.push_back(cptr_node);
		octree->cptr_aabbs.push_back(cptr_aabb);

		// if(nodes_counter < 10){
		// 	println("- HOST: AABB[{}] = mins({}, {}, {}), maxs({}, {}, {})", nodes_counter,
		// 		new_aabb.mins.x, new_aabb.mins.y, new_aabb.mins.z,
		// 		new_aabb.maxs.x, new_aabb.maxs.y, new_aabb.maxs.z
		// 	);
		// }

		nodes_counter++;
		return cptr_node;
	};

	CUdeviceptr cptr_root_node = recursive(main_root, main_aabb, 0);	
	octree->num_nodes = nodes_counter;
	octree->max_lod_level = max_lod_level;

	println("Final octree: {} nodes, {} max level", nodes_counter, max_lod_level);
	
	editor->scene.world->children.push_back(octree);

}

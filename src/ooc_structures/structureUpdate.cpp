#include "structureUpdate.h"

#include "simLod.h"
#include "outOfCore.h"


void initOctree(OctreeNode* root_node, std::shared_ptr<vector<Point>>& points){
	for(const Point& point : *points){
		root_node->aabb.maxs.x = std::max(root_node->aabb.maxs.x, point.position.x);
		root_node->aabb.maxs.y = std::max(root_node->aabb.maxs.y, point.position.y);
		root_node->aabb.maxs.z = std::max(root_node->aabb.maxs.z, point.position.z);
		root_node->aabb.mins.x = std::min(root_node->aabb.mins.x, point.position.x);
		root_node->aabb.mins.y = std::min(root_node->aabb.mins.y, point.position.y);
		root_node->aabb.mins.z = std::min(root_node->aabb.mins.z, point.position.z);
	}

	// Adding small 2x delta to avoid floating point issues
	float epsilon = 0.5f;
	root_node->aabb.mins -= epsilon * root_node->aabb.mins;
	root_node->aabb.maxs += epsilon * root_node->aabb.maxs;

	// Make it cubic
	vec3 size = root_node->aabb.getSize();
	vec3 half_sizes_x = 0.5f * (vec3(size.x) - size);
	vec3 half_sizes_y = 0.5f * (vec3(size.y) - size);
	vec3 half_sizes_z = 0.5f * (vec3(size.z) - size);
	if(size.x > size.y){
		if(size.x > size.z){
			root_node->aabb.mins.y -= half_sizes_x.y;
			root_node->aabb.maxs.y += half_sizes_x.y;
			root_node->aabb.mins.z -= half_sizes_x.z;
			root_node->aabb.maxs.z += half_sizes_x.z;
		} else {
			root_node->aabb.mins.y -= half_sizes_z.y;
			root_node->aabb.maxs.y += half_sizes_z.y;
			root_node->aabb.mins.x -= half_sizes_z.x;
			root_node->aabb.maxs.x += half_sizes_z.x;
		}
	} else {
		if(size.y > size.z){
			root_node->aabb.mins.x -= half_sizes_y.x;
			root_node->aabb.maxs.x += half_sizes_y.x;
			root_node->aabb.mins.z -= half_sizes_y.z;
			root_node->aabb.maxs.z += half_sizes_y.z;
		} else {
			root_node->aabb.mins.y -= half_sizes_z.y;
			root_node->aabb.maxs.y += half_sizes_z.y;
			root_node->aabb.mins.x -= half_sizes_z.x;
			root_node->aabb.maxs.x += half_sizes_z.x;
		}
	}

	// TODO: temporary code
	{
		std::lock_guard<std::mutex> lock(GlobalVariables::aabbRelationshipMapMtx);
		GlobalVariables::aabbRelationshipMap[root_node->aabb] = {nullopt};
	}
}



uint32_t growOctree(OctreeNode* root_node, const std::shared_ptr<vector<Point>>& points){
	uint32_t nb_new_levels = 0;
	AABB new_aabb = AABB(root_node->aabb);
	NodePosition node_position = FrontTopLeft;
	// For each point in a batch check if fits in current AABB
	for(const Point& point : *points){
		while(!new_aabb.contains(point.position)){
			// Create new roots considering main box as successively the 1st, 2nd, ..., 8th child to build octree in spiral
			nb_new_levels++;
			new_aabb.extend(node_position);
			updateNodePosition(node_position);
		}
	}
	return nb_new_levels;
}



OctreeNode* uptadeOctree(OctreeNode* main_root, uint32_t nb_new_levels){
	OctreeNode* cur_child = main_root;
	NodePosition node_position = FrontTopLeft;
	for(uint32_t i=0; i<nb_new_levels; i++){
		// Create new parent
		OctreeNode* new_parent = new OctreeNode(cur_child->aabb);
		new_parent->aabb.extend(node_position);
		new_parent->occupancy = new OccupancyGrid();

		new_parent->updated = true;
		cur_child->updated = true;

		// Create the correct child
		new_parent->children[node_position] = cur_child;
		// new_parent->children_ids |= 0x01 << node_position;

		auto fillOccupancyGrid = [&](AABB& cur_aabb, const Chunk* child_chunk_list){
			while(child_chunk_list){
				for(uint32_t j=0; j<child_chunk_list->size; j++){
					const Point& point = child_chunk_list->points[j];

					// Sample voxel occupancy grid at this location
					vec3 normalized_coordinates = cur_aabb.getPointNormalizedCoordinates(point.position);
					uint32_t grid_x = clamp(
						uint32_t(floor(OocSimLodSettings::GRID_SIZE_PER_DIMENSION * normalized_coordinates.x)), 
						0u, 
						OocSimLodSettings::GRID_SIZE_PER_DIMENSION - 1u
					);
					uint32_t grid_y = clamp(
						uint32_t(floor(OocSimLodSettings::GRID_SIZE_PER_DIMENSION * normalized_coordinates.y)), 
						0u, 
						OocSimLodSettings::GRID_SIZE_PER_DIMENSION - 1u
					);
					uint32_t grid_z = clamp(
						uint32_t(floor(OocSimLodSettings::GRID_SIZE_PER_DIMENSION * normalized_coordinates.z)), 
						0u, 
						OocSimLodSettings::GRID_SIZE_PER_DIMENSION - 1u
					);
					uint32_t index = grid_x + OocSimLodSettings::GRID_SIZE_PER_DIMENSION * (grid_y + OocSimLodSettings::GRID_SIZE_PER_DIMENSION * grid_z);
					uint32_t word_index = index >> 5u;
					uint32_t bit_index = index & 31u;
					bool is_cell_occupied = (new_parent->occupancy->values[word_index] & (1u << bit_index)) != 0;

					// Fill up occupancy grid
					if(!is_cell_occupied){
						new_parent->occupancy->values[word_index] |= (1u << bit_index);
						// Create corresponding voxel using this point
						vec3 world_grid_size = cur_aabb.getSize() / float(OocSimLodSettings::GRID_SIZE_PER_DIMENSION);
						vec3 voxel_centroid = cur_aabb.mins + world_grid_size * vec3(grid_x, grid_y, grid_z) + 0.5f*world_grid_size;
						Point new_voxel = {};
						new_voxel.position = voxel_centroid;
						new_voxel.color[0] = point.color[0];
						new_voxel.color[1] = point.color[1];
						new_voxel.color[2] = point.color[2];

						// Add voxel to voxels chunk list
						if(!new_parent->voxels){new_parent->voxels =  new Chunk();}
						Chunk* parent_chunk_list = new_parent->voxels;
						while(parent_chunk_list->next){parent_chunk_list = parent_chunk_list->next;}
						if(parent_chunk_list->size == OocSimLodSettings::NB_POINTS_PER_CHUNK){
							parent_chunk_list->next =  new Chunk();
							parent_chunk_list = parent_chunk_list->next;
						}
						parent_chunk_list->points[parent_chunk_list->size] = new_voxel;
						parent_chunk_list->size++;
					}
				}
				child_chunk_list = child_chunk_list->next;
			}
		};

		// Sample voxels to fill new occupancy grid
		fillOccupancyGrid(new_parent->aabb, cur_child->points);
		fillOccupancyGrid(new_parent->aabb, cur_child->voxels);

		// TODO: temporary code
		{
			std::lock_guard<std::mutex> lock(GlobalVariables::aabbRelationshipMapMtx);
			GlobalVariables::aabbRelationshipMap[new_parent->aabb] = {nullopt};
			GlobalVariables::aabbRelationshipMap[new_parent->aabb][node_position] = cur_child->aabb;
			GlobalVariables::aabbParentMap[cur_child->aabb] = new_parent->aabb;
		}

		cur_child = new_parent;
		updateNodePosition(node_position);
	}
	return cur_child;
}


void freeOctreesOnGPU(CuRast* editor){
	std::string main_octree_name = GlobalVariables::getSimLodOctreeName();

	bool delete_all = CuRastSettings::freeOldOctreeMemoryOnGPU;
	std::vector<SNCOctree*> octrees = {};
	{
		std::lock_guard<std::mutex> lock_scene(GlobalVariables::updateSceneMutex);
		editor->scene.forEach<SNCOctree>([&](SNCOctree* node){
			if(node->need_to_be_executed || (delete_all && node->name != main_octree_name)){
				octrees.push_back(node);
			}
		});
	}
	if(delete_all){
		CuRastSettings::freeOldOctreeMemoryOnGPU = false;
	}

	
	for(SNCOctree* octree : octrees){
		std::lock_guard<std::mutex> lock_scene(GlobalVariables::updateSceneMutex);
		editor->scene.world->remove(octree);
	}
}

void freePreviousOctreeOnGPU(CuRast* editor, std::shared_ptr<SNCOctree> caller){
	std::optional<uint64_t> octree_id = caller ? std::optional<uint64_t>(caller->octree_id) : nullopt;
	while(caller && !caller->isDoneLoadingToGpu()){}

	std::lock_guard<std::mutex> lock_scene(GlobalVariables::updateSceneMutex);
	editor->scene.forEach<SNCOctree>([&](SNCOctree* node){
		if(octree_id.has_value() && node->octree_id == (octree_id.value()-1)){
			node->need_to_be_executed = true;
		}
	});
}


std::optional<CUdeviceptr> allocateChunks(std::shared_ptr<SNCOctree>& octree, const Chunk* root, bool is_voxel_chunk){
	// Create CChunks
	const Chunk* cur_chunk = root;
	CChunk* prev = nullptr;
	std::optional<CUdeviceptr> first = nullopt;
	while(cur_chunk){
		std::pair<CChunk*, CUdeviceptr> allocated = GlobalVariables::batchedMemory.allocate<CChunk>();
		CUdeviceptr tmp_gpu = allocated.second;

		CChunk* tmp = allocated.first;
		tmp->size = cur_chunk->size;
		tmp->next = nullptr;

		if(is_voxel_chunk){octree->nb_voxels += tmp->size;} 
		else{octree->nb_points += tmp->size;} 
		octree->nb_chunks++;

		for(uint32_t j=0; j<tmp->size; j++){
			CPoint tmp_point = {
				.position = cur_chunk->points[j].position, 
				.color = (uint32_t)cur_chunk->points[j].color[0]
					| ((uint32_t)cur_chunk->points[j].color[1] << 8)
					| ((uint32_t)cur_chunk->points[j].color[2] << 16)
					| (0xFFu << 24)
			};
			tmp->points[j] = tmp_point;
		}

		if(prev){
			prev->next = (CChunk*)tmp_gpu;
		}

		GlobalVariables::batchedMemory.addFutureCopy<CChunk>(tmp, tmp_gpu);

		cur_chunk = cur_chunk->next;
		prev = tmp;
		if(!first){first = tmp_gpu;}
	}

	return first;
};


void createCudaMemory(CuRast* editor, CUcontext* context, std::shared_ptr<OctreeNode>& input_octree){
	GlobalVariables::batchedMemory.reset();

	// Create cuda memory pointers
	std::shared_ptr<SNCOctree> octree = make_shared<SNCOctree>(
		GlobalVariables::getSimLodOctreeName(true), 
		GlobalVariables::simLodOctreeCounter
	);
	octree->cptr_nodes = {};

	cuCtxSetCurrent(*context);
	CUresult cuda_status = cuStreamCreate(&octree->stream, CU_STREAM_NON_BLOCKING);
	// CUresult cuda_status = cuStreamCreate(&octree->stream, CU_STREAM_DEFAULT);
	CURuntime::assertCudaSuccess(cuda_status);

	// Create enough chunks
	uint32_t max_lod_level = 0;

	std::function<CUdeviceptr(const OctreeNode*, uint8_t)> recursive = [&](
		const OctreeNode* cur_node, uint8_t level
	) -> CUdeviceptr {

		CUdeviceptr child_indices[8] = {0};
		
		for(uint32_t child = 0; child < 8; child++){
			if(cur_node->children[child]){
				if(level == UINT8_MAX){
					println("Can't have a level greater than {}", UINT8_MAX);
					exit(EXIT_FAILURE);
				}
				const OctreeNode* next_node = cur_node->children[child];
				child_indices[child] = recursive(next_node, level+1);
			}
		}	

		// Create COctreeNode
		std::pair<COctreeNode*, CUdeviceptr> allocated = GlobalVariables::batchedMemory.allocate<COctreeNode>();
		COctreeNode* new_node = allocated.first;

		for(uint32_t child = 0; child < 8; child++){
			if(cur_node->children[child]){
				new_node->children[child] = (COctreeNode*) child_indices[child];
			}
		}

		new_node->points = (CChunk*)allocateChunks(octree, cur_node->points).value_or(0);
		new_node->voxels = (CChunk*)allocateChunks(octree, cur_node->voxels, true).value_or(0);
		new_node->occupancy = nullptr;
		new_node->aabb = {cur_node->aabb.mins, cur_node->aabb.maxs};

		new_node->counter = cur_node->counter;
		new_node->children_ids = cur_node->children_ids;
		new_node->children_visibility = cur_node->children_visibility;
		new_node->level = level;

		new_node->updated = cur_node->updated;
		new_node->is_large = cur_node->is_large;
		new_node->is_visible = cur_node->is_visible;
		new_node->is_cut = cur_node->is_cut;

		if(level > max_lod_level){
			max_lod_level = level;
		}

		// Create cuda pointers
		CUdeviceptr cptr_node = allocated.second;

		GlobalVariables::batchedMemory.addFutureCopy<COctreeNode>(new_node, cptr_node);
		octree->cptr_nodes.push_back(cptr_node);
		octree->nb_nodes++;

		return cptr_node;
	};

	const OctreeNode* next_octree = input_octree.get();
	recursive(next_octree, 0);

	octree->max_lod_level = max_lod_level;

	// Copy arrays of pointers to GPU
	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		cuda_status = cuMemAllocAsync(&octree->nodes, octree->cptr_nodes.size() * sizeof(CUdeviceptr), octree->stream);
		cuda_status = cuMemcpyHtoDAsync(octree->nodes, octree->cptr_nodes.data(), octree->cptr_nodes.size() * sizeof(CUdeviceptr), octree->stream);
	} else {
		cuda_status = cuMemAlloc(&octree->nodes, octree->cptr_nodes.size() * sizeof(CUdeviceptr));
		cuda_status = cuMemcpyHtoD(octree->nodes, octree->cptr_nodes.data(), octree->cptr_nodes.size() * sizeof(CUdeviceptr));
	}
	CURuntime::assertCudaSuccess(cuda_status);
	GlobalVariables::batchedMemory.copyMemory(context, &octree->stream);

	{
		std::lock_guard<std::mutex> lock_scene(GlobalVariables::updateSceneMutex);
		editor->scene.world->children.push_back(octree);
	}
	// Free previous octrees
	if(CuRastSettings::autoFreeOldOctreeMemoryOnGPU){
		freePreviousOctreeOnGPU(editor, octree);
	}
};


void loadOctreeOnGPU(CuRast* editor, CUcontext* context, 
    bool bypass_semaphore
){	
	std::shared_ptr<OctreeNode> octree_ref = nullptr;
	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL && !bypass_semaphore){
		std::lock_guard<std::mutex> lock_send(GlobalVariables::isUpdatingMtx);
		octree_ref = GlobalVariables::mainOctree;
	} else {
		octree_ref = GlobalVariables::mainOctree;
	}
	if(!octree_ref){return;}

	std::shared_ptr<Timing> timing = Timing::addTiming("send octree to GPU ", true);
	createCudaMemory(editor, context, octree_ref);
	timing->stop_clock();
}



void addPointBatches(){
	std::vector<uint32_t> batches_indices(OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE, 0);
	uint32_t last_index = 0;

	for(uint32_t i=0; i<OocSimLodSettings::BATCHES_LIST_SIZE; i++){
		std::lock_guard<std::mutex> lock(GlobalVariables::batchesQueueMutexes[i]);
		if(GlobalVariables::batchesQueue[i] && GlobalVariables::batchesQueue[i]->state == BatchState::Loaded){
			batches_indices[last_index] = i;
			last_index++;
			if(last_index >= OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE){break;}
		}
	}
	if(last_index == 0){return;}
	static uint32_t lastUpdateAttempt = 0;
	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL && batches_indices.size() < OocSimLodSettings::MIN_BATCHES_PER_OCTREE_UPDATE){
		if(lastUpdateAttempt < OocSimLodSettings::MAX_ATTEMPTS_BEFORE_IGNORING_MIN_VARIABLES){
			lastUpdateAttempt++;
			return;
		} 
	}
	lastUpdateAttempt = 0;

	auto first = batches_indices.begin();
	auto last = first + last_index;

	if(!GlobalVariables::mainOctree){
		std::shared_ptr<Timing> timing = Timing::addTiming("init octree", true);
		std::lock_guard<std::mutex> lock_send(GlobalVariables::isUpdatingMtx);
		GlobalVariables::mainOctree = std::make_shared<OctreeNode>(AABB());
		uint32_t batch_index = batches_indices[0];
		std::lock_guard<std::mutex> lock(GlobalVariables::batchesQueueMutexes[batch_index]);
		initOctree(GlobalVariables::mainOctree.get(), GlobalVariables::batchesQueue[batch_index]->points);
		timing->stop_clock();

		// Copy octree once at the beginning
		timing = Timing::addTiming("copy initial octree", true);
		GlobalVariables::mainOctreeCpy = new OctreeNode(*GlobalVariables::mainOctree);
		timing->stop_clock();
	}

	// println("//////////////////////////////////////////////////");
	// println("///////////// OctreeCpy before update ////////////");
	// println("//////////////////////////////////////////////////");
	// mainOctreeCpy->display();
	// println("//////////////////////////////////////////////////");
	// println("////////////// Octree before update //////////////");
	// println("//////////////////////////////////////////////////");
	// mainOctree->display();

	// Update the temporary octree
	std::shared_ptr<Timing> timing = Timing::addTiming("compute max new level", true);
	// Compute max new level needed per batch
	vector<uint32_t> tmp_new_levels = vector<uint32_t>(last_index, 0);
	// In parallel
	{
		std::vector<uint32_t> indices(last_index);
		auto first_of_indices = indices.begin();
		auto last_of_indices = indices.end();
		std::iota(first_of_indices, last_of_indices, 0);
		if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
			std::for_each(std::execution::par, first_of_indices, last_of_indices, [&](uint32_t index){
				std::shared_ptr<PointBatch> batch = GlobalVariables::batchesQueue[batches_indices[index]];
				std::lock_guard<std::mutex> lock(GlobalVariables::batchesQueueMutexes[batches_indices[index]]);
				tmp_new_levels[index] = growOctree(GlobalVariables::mainOctreeCpy, batch->points);
			});
		} else {
			std::for_each(first_of_indices, last_of_indices, [&](uint32_t index){
				std::shared_ptr<PointBatch> batch = GlobalVariables::batchesQueue[batches_indices[index]];
				std::lock_guard<std::mutex> lock(GlobalVariables::batchesQueueMutexes[batches_indices[index]]);
				tmp_new_levels[index] = growOctree(GlobalVariables::mainOctreeCpy, batch->points);
			});
		}
	}
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("//////////// Octree after grow octree ////////////");
	// println("//////////////////////////////////////////////////");
	// mainOctreeCpy->display();


	timing = Timing::addTiming("update octree bottom up", true);
	// In single thread
	{
		uint32_t nb_new_levels = 0;
		for(uint32_t& level : tmp_new_levels){
			nb_new_levels = max(nb_new_levels, level);
		}
		// println("Max new level: {}", nb_new_levels);
		GlobalVariables::mainOctreeCpy = uptadeOctree(GlobalVariables::mainOctreeCpy, nb_new_levels);
	}
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("/////////// Octree after update octree ///////////");
	// println("//////////////////////////////////////////////////");
	// mainOctreeCpy->display();


	timing = Timing::addTiming("simlod update", true);
	// TODO: In parallel
	{
		std::for_each(first, last, [&](uint32_t index){
			std::lock_guard<std::mutex> lock(GlobalVariables::batchesQueueMutexes[index]);
			std::shared_ptr<PointBatch> batch = GlobalVariables::batchesQueue[index];
			simLodUpdate(GlobalVariables::mainOctreeCpy, batch->points);
		});
	}
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("/////////// Octree after simLOD update ///////////");
	// println("//////////////////////////////////////////////////");
	// mainOctreeCpy->display();

	timing = Timing::addTiming("update cache", true);
	updateUpdatesCache(GlobalVariables::mainOctreeCpy);
	timing->stop_clock();
	// displayCache();

	// if(!LRUCache::sanityCheckStored(*mainOctreeCpy->aabb)){
    //     println("Sanity check failed for the stored cache");
    // }

	// println("//////////////////////////////////////////////////");
	// println("/////////// Octree after cache update ///////////");
	// println("//////////////////////////////////////////////////");
	// mainOctreeCpy->display();

	// if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
	// 	// Block if the mainOctree / mainAABB are being send to the GPU (see `loadOctreeOnGPU`)
	// 	// Acquire => semaphore_counter -= 1
	// 	octreeReadyToBeUpdated.acquire();
	// 	// println("semaphore UPDATING acquired");
	// }

	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		std::lock_guard<std::mutex> lock_send(GlobalVariables::isUpdatingMtx);
		GlobalVariables::mainOctree = std::make_shared<OctreeNode>(*GlobalVariables::mainOctreeCpy);
		GlobalVariables::lodUpdated = true;
	} else {
		GlobalVariables::mainOctree = std::make_shared<OctreeNode>(*GlobalVariables::mainOctreeCpy);
	}
	
	// if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
	// 	// Release the semaphore when not using mainOctree / mainAABB anymore
	// 	// Rlease => semaphore_counter += 1
	// 	octreeReadyToBeSent.release();
	// 	// println("semaphore SENDING released");
	// }

	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		std::for_each(std::execution::par, first, last, [&](uint32_t index){
			std::lock_guard<std::mutex> lock(GlobalVariables::batchesQueueMutexes[index]);
			GlobalVariables::batchesQueue[index]->state = BatchState::Inserted;
		});
	} else {
		std::for_each(first, last, [&](uint32_t index){
			std::lock_guard<std::mutex> lock(GlobalVariables::batchesQueueMutexes[index]);
			GlobalVariables::batchesQueue[index]->state = BatchState::Inserted;
		});
	}
};


void updateOctreeRoutine(){
	while(true){
		addPointBatches();
	}
}
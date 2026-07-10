#include "structureUpdate.h"

#include "simLod.h"
#include "outOfCore.h"
#include "allocator.h"


OctreeNode* initOctree(std::shared_ptr<vector<Point>>& points){
	// Initialise the AABB
	AABB root_aabb = {};

	for(const Point& point : *points){
		root_aabb.maxs.x = std::max(root_aabb.maxs.x, point.position.x);
		root_aabb.maxs.y = std::max(root_aabb.maxs.y, point.position.y);
		root_aabb.maxs.z = std::max(root_aabb.maxs.z, point.position.z);
		root_aabb.mins.x = std::min(root_aabb.mins.x, point.position.x);
		root_aabb.mins.y = std::min(root_aabb.mins.y, point.position.y);
		root_aabb.mins.z = std::min(root_aabb.mins.z, point.position.z);
	}

	// Adding small 2x delta to avoid floating point issues
	float epsilon = 0.5f;
	root_aabb.mins -= epsilon * root_aabb.mins;
	root_aabb.maxs += epsilon * root_aabb.maxs;

	// Make it cubic
	vec3 size = root_aabb.getSize();
	vec3 half_sizes_x = 0.5f * (vec3(size.x) - size);
	vec3 half_sizes_y = 0.5f * (vec3(size.y) - size);
	vec3 half_sizes_z = 0.5f * (vec3(size.z) - size);
	if(size.x > size.y){
		if(size.x > size.z){
			root_aabb.mins.y -= half_sizes_x.y;
			root_aabb.maxs.y += half_sizes_x.y;
			root_aabb.mins.z -= half_sizes_x.z;
			root_aabb.maxs.z += half_sizes_x.z;
		} else {
			root_aabb.mins.y -= half_sizes_z.y;
			root_aabb.maxs.y += half_sizes_z.y;
			root_aabb.mins.x -= half_sizes_z.x;
			root_aabb.maxs.x += half_sizes_z.x;
		}
	} else {
		if(size.y > size.z){
			root_aabb.mins.x -= half_sizes_y.x;
			root_aabb.maxs.x += half_sizes_y.x;
			root_aabb.mins.z -= half_sizes_y.z;
			root_aabb.maxs.z += half_sizes_y.z;
		} else {
			root_aabb.mins.y -= half_sizes_z.y;
			root_aabb.maxs.y += half_sizes_z.y;
			root_aabb.mins.x -= half_sizes_z.x;
			root_aabb.maxs.x += half_sizes_z.x;
		}
	}

	// Initialise the main octree
	IdAABB root_aabb_index = GlobalVariables::createNewAABB(root_aabb);
	// return std::make_shared<OctreeNode>(root_aabb_index);
	return MemoryAllocator::newOctreeNode(root_aabb_index);
}



uint32_t growOctree(OctreeNode* root_node, const std::shared_ptr<vector<Point>>& points){
	if(!root_node){return 0;}
	uint32_t nb_new_levels = 0;
	AABB new_aabb = GlobalVariables::getAABB(root_node->aabb_index);
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
	if(!main_root){return 0;}
	
	OctreeNode* cur_child = main_root;
	NodePosition node_position = FrontTopLeft;
	for(uint32_t i=0; i<nb_new_levels; i++){
		// Create the new AABB
		AABB parent_aabb = GlobalVariables::getAABB(cur_child->aabb_index);
		parent_aabb.extend(node_position);

		// Create the new parent node
		IdAABB parent_aabb_index = GlobalVariables::createNewAABB(parent_aabb);
		// OctreeNode* new_parent = new OctreeNode(parent_aabb_index);
		OctreeNode* new_parent = MemoryAllocator::newOctreeNode(parent_aabb_index);

		NEW_COUNTER++;

		// new_parent->occupancy = new OccupancyGrid();
		new_parent->occupancy = MemoryAllocator::newOccupancyGrid();

		NEW_COUNTER++;

		new_parent->updated = true;
		cur_child->updated = true;
		// Create the correct child
		new_parent->children[node_position] = cur_child;

		auto fillOccupancyGrid = [&](const Chunk* child_chunk_list){
			while(child_chunk_list){
				for(uint32_t j=0; j<child_chunk_list->size; j++){
					const Point& point = child_chunk_list->points[j];

					// Sample voxel occupancy grid at this location
					OccupancyGrid::GridIndex index = OccupancyGrid::getCellIndices(parent_aabb, point);
					bool is_cell_occupied = new_parent->occupancy->isCellOcupied(index);

					// Fill up occupancy grid
					if(!is_cell_occupied){
						new_parent->occupancy->markCellAsFilled(index);
						// Create corresponding voxel using this point
						vec3 voxel_centroid = OccupancyGrid::getCellCentroid(parent_aabb, index);
						Point new_voxel = {};
						new_voxel.position = voxel_centroid;
						new_voxel.color[0] = point.color[0];
						new_voxel.color[1] = point.color[1];
						new_voxel.color[2] = point.color[2];

						// Add voxel to voxels chunk list
						if(!new_parent->voxels){
							// new_parent->voxels =  new Chunk();
							new_parent->voxels =  MemoryAllocator::newChunk();

							NEW_COUNTER++;
						}
						Chunk* parent_chunk_list = new_parent->voxels;
						while(parent_chunk_list->next){parent_chunk_list = parent_chunk_list->next;}
						if(parent_chunk_list->size == OocSimLodSettings::NB_POINTS_PER_CHUNK){
							// parent_chunk_list->next =  new Chunk();
							parent_chunk_list->next =  MemoryAllocator::newChunk();
							parent_chunk_list = parent_chunk_list->next;

							NEW_COUNTER++;
						}
						parent_chunk_list->points[parent_chunk_list->size] = new_voxel;
						parent_chunk_list->size++;
					}
				}
				child_chunk_list = child_chunk_list->next;
			}
		};

		// Sample voxels to fill new occupancy grid
		fillOccupancyGrid(cur_child->points);
		fillOccupancyGrid(cur_child->voxels);

		// Update the AABB maps
		(*GlobalVariables::aabbRelationshipMap)[parent_aabb_index][node_position] = cur_child->aabb_index;

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
		GlobalVariables::nbOctreesInScene--;
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


std::optional<CUdeviceptr> allocateChunks(
	std::shared_ptr<SNCOctree>& octree, 
    const Chunk* root, bool is_voxel_chunk
){
	// Create CChunks
	std::vector<const Chunk*> all_chunks = {};
	std::vector<std::pair<CChunk*, CUdeviceptr>> allocated_chunks = {};
	const Chunk* cur_chunk = root;
	while(cur_chunk){
		all_chunks.push_back(cur_chunk);
		BatchedMemory& memory = GlobalVariables::batchedMemories[GlobalVariables::currentBatchedMemoriesIndex];
		std::pair<CChunk*, CUdeviceptr> allocated = memory.allocate<CChunk>();
		memory.addFutureCopy<CChunk>(allocated.first, allocated.second);
		allocated_chunks.push_back(allocated);
		cur_chunk = cur_chunk->next;
	}

	uint32_t nb_chunks = all_chunks.size();
	octree->nb_chunks += nb_chunks;
	auto fillChunk = [&](uint32_t index){
		const Chunk* cur_chunk = all_chunks[index];

		CChunk* tmp = allocated_chunks[index].first;

		tmp->size = cur_chunk->size;
		tmp->next = nullptr;
		if(index != nb_chunks-1){
			tmp->next = (CChunk*)(allocated_chunks[index+1].second);
		} else {
			uint32_t nb_points = tmp->size + (nb_chunks - 1) * OocSimLodSettings::NB_POINTS_PER_CHUNK;
			if(is_voxel_chunk){octree->nb_voxels += nb_points;}
			else {octree->nb_points += nb_points;}
		}

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
	};

	std::vector<uint32_t> indices(all_chunks.size()); 
	std::iota(indices.begin(), indices.end(), 0);

	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		std::for_each(std::execution::par, indices.begin(), indices.end(), [&](uint32_t index){
			fillChunk(index);
		});
	} else {
		std::for_each(indices.begin(), indices.end(), [&](uint32_t index){
			fillChunk(index);
		});
	}

	return all_chunks.empty() ? nullopt : std::optional<CUdeviceptr>(allocated_chunks[0].second);
};


void createCudaMemory(CuRast* editor, CUcontext* context, 
    OctreeNode* input_octree,
    std::shared_ptr<AABBRelationshipMap> relationship_map_ref
){
	BatchedMemory& memory = GlobalVariables::batchedMemories[GlobalVariables::currentBatchedMemoriesIndex];
	memory.reset();

	// Create cuda memory pointers
	std::shared_ptr<SNCOctree> octree = make_shared<SNCOctree>(
		GlobalVariables::getSimLodOctreeName(true), 
		GlobalVariables::simLodOctreeCounter
	);
	std::vector<CUdeviceptr> cptr_nodes = {};

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
					throw(EXIT_FAILURE);
				}
				const OctreeNode* next_node = cur_node->children[child];
				child_indices[child] = recursive(next_node, level+1);
			}
		}	

		// Create COctreeNode
		std::pair<COctreeNode*, CUdeviceptr> allocated = memory.allocate<COctreeNode>();
		COctreeNode* new_node = allocated.first;

		for(uint32_t child = 0; child < 8; child++){
			if(cur_node->children[child]){
				new_node->children[child] = (COctreeNode*) child_indices[child];
			}
		}

		new_node->points = (CChunk*)allocateChunks(octree, cur_node->points).value_or(0);
		new_node->voxels = (CChunk*)allocateChunks(octree, cur_node->voxels, true).value_or(0);
		new_node->occupancy = nullptr;
		new_node->aabb_index = cur_node->aabb_index;

		new_node->counter = cur_node->counter.load();
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

		memory.addFutureCopy<COctreeNode>(new_node, cptr_node);
		cptr_nodes.push_back(cptr_node);
		octree->nb_nodes++;

		return cptr_node;
	};

	const OctreeNode* next_octree = input_octree;
	recursive(next_octree, 0);

	octree->max_lod_level = max_lod_level;

	octree->nb_aabbs = relationship_map_ref->size();

	// Copy arrays of pointers to GPU
	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		// Allocate the space for the AABBs
		cuda_status = cuMemAllocAsync(&octree->aabbs, octree->nb_aabbs * sizeof(AABB), octree->stream);
		cuda_status = cuMemcpyHtoDAsync(octree->aabbs, GlobalVariables::allAABBs.data(), octree->nb_aabbs * sizeof(AABB), octree->stream);
		// Allocate the space for the nodes pointers
		cuda_status = cuMemAllocAsync(&octree->nodes, cptr_nodes.size() * sizeof(CUdeviceptr), octree->stream);
		cuda_status = cuMemcpyHtoDAsync(octree->nodes, cptr_nodes.data(), cptr_nodes.size() * sizeof(CUdeviceptr), octree->stream);
	} else {
		// Allocate the space for the AABBs
		cuda_status = cuMemAlloc(&octree->aabbs, octree->nb_aabbs * sizeof(AABB));
		cuda_status = cuMemcpyHtoD(octree->aabbs, GlobalVariables::allAABBs.data(), octree->nb_aabbs * sizeof(AABB));
		// Allocate the space for the nodes pointers
		cuda_status = cuMemAlloc(&octree->nodes, cptr_nodes.size() * sizeof(CUdeviceptr));
		cuda_status = cuMemcpyHtoD(octree->nodes, cptr_nodes.data(), cptr_nodes.size() * sizeof(CUdeviceptr));
	}
	CURuntime::assertCudaSuccess(cuda_status);
	memory.copyMemory(context, &octree->stream);

	{
		std::lock_guard<std::mutex> lock_scene(GlobalVariables::updateSceneMutex);
		editor->scene.world->children.push_back(octree);
		GlobalVariables::nbOctreesInScene++;
	}
	// Free previous octrees
	if(CuRastSettings::autoFreeOldOctreeMemoryOnGPU){
		std::thread thread([&](CuRast* editor, std::shared_ptr<SNCOctree> octree){
			freePreviousOctreeOnGPU(editor, octree);
		}, editor, octree);
		thread.detach();
		// freePreviousOctreeOnGPU(editor, octree);
	}
	// Swap allocated memories
	GlobalVariables::currentBatchedMemoriesIndex++;
	GlobalVariables::currentBatchedMemoriesIndex = GlobalVariables::currentBatchedMemoriesIndex % GlobalVariables::batchedMemories.size();
};


void loadOctreeOnGPU(CuRast* editor, CUcontext* context, 
    OctreeNode* octree_ref,
    std::shared_ptr<AABBRelationshipMap> relationship_map_ref
){
	if(!octree_ref){return;}

	// std::shared_ptr<Timing> timing = Timing::addTiming("send octree to GPU ", true);
	createCudaMemory(editor, context, octree_ref, relationship_map_ref);
	// timing->stop_clock();
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
		GlobalVariables::aabbRelationshipMap = std::make_shared<AABBRelationshipMap>();

		std::shared_ptr<Timing> timing = Timing::addTiming("init octree", true);
		std::lock_guard<std::mutex> lock_send(GlobalVariables::isUpdatingMtx);
		// Use the first batch to generate the intial bounding box
		uint32_t batch_index = batches_indices[0];
		std::lock_guard<std::mutex> lock(GlobalVariables::batchesQueueMutexes[batch_index]);
		GlobalVariables::mainOctree = initOctree(GlobalVariables::batchesQueue[batch_index]->points);
		GlobalVariables::allOctreesRefCounter[GlobalVariables::mainOctree] = 1;
		timing->stop_clock();
		GlobalVariables::swapAABBsMaps();

		// Copy octree once at the beginning
		timing = Timing::addTiming("copy initial octree", true);
		// GlobalVariables::mainOctreeCpy = new OctreeNode(*GlobalVariables::mainOctree);
		GlobalVariables::mainOctreeCpy = MemoryAllocator::newOctreeNodeCpy(*GlobalVariables::mainOctree);
		timing->stop_clock();

		NEW_COUNTER++;
	}

	// println("//////////////////////////////////////////////////");
	// println("////////////// Octree before update //////////////");
	// println("//////////////////////////////////////////////////");
	// GlobalVariables::mainOctreeCpy->display();
	// GlobalVariables::displayCpuMemoryUsage();

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
	// GlobalVariables::mainOctreeCpy->display();
	// GlobalVariables::displayCpuMemoryUsage();


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
	// GlobalVariables::mainOctreeCpy->display();
	// GlobalVariables::displayCpuMemoryUsage();



	timing = Timing::addTiming("simlod update", true);
	// TODO: In parallel
	{
		std::for_each(first, last, [&](uint32_t index){
			std::lock_guard<std::mutex> lock(GlobalVariables::batchesQueueMutexes[index]);
			std::shared_ptr<PointBatch> batch = GlobalVariables::batchesQueue[index];
			SimLod::update(GlobalVariables::mainOctreeCpy, batch->points, GlobalVariables::aabbRelationshipMap);
		});
	}
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("/////////// Octree after simLOD update ///////////");
	// println("//////////////////////////////////////////////////");
	// GlobalVariables::mainOctreeCpy->display();
	// GlobalVariables::displayCpuMemoryUsage();


	timing = Timing::addTiming("update cache", true);
	updateUpdatesCache(GlobalVariables::mainOctreeCpy);
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("/////////// Octree after cache update ////////////");
	// println("//////////////////////////////////////////////////");
	// GlobalVariables::mainOctreeCpy->display();
	// GlobalVariables::displayCpuMemoryUsage();


	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		std::lock_guard<std::mutex> lock_send(GlobalVariables::isUpdatingMtx);
		GlobalVariables::swapAABBsMaps();
		GlobalVariables::swapOctrees();
	} else {
		GlobalVariables::swapAABBsMaps();
		GlobalVariables::swapOctrees();
	}

	// println("//////////////////////////////////////////////////");
	// println("/////////////// Octree after swap ////////////////");
	// println("//////////////////////////////////////////////////");
	// GlobalVariables::mainOctreeCpy->display();
	// GlobalVariables::displayCpuMemoryUsage();

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
		if(GlobalVariables::mainLoopIsTerminating){return;}
	}
}
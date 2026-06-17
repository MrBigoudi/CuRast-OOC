#include "structureUpdate.h"

#include "simLod.h"
#include "outOfCore.h"


void initOctree(OctreeNode* root_node, std::shared_ptr<vector<Point>>& points){
	for(const Point& point : *points){
		root_node->aabb->maxs.x = std::max(root_node->aabb->maxs.x, point.position.x);
		root_node->aabb->maxs.y = std::max(root_node->aabb->maxs.y, point.position.y);
		root_node->aabb->maxs.z = std::max(root_node->aabb->maxs.z, point.position.z);
		root_node->aabb->mins.x = std::min(root_node->aabb->mins.x, point.position.x);
		root_node->aabb->mins.y = std::min(root_node->aabb->mins.y, point.position.y);
		root_node->aabb->mins.z = std::min(root_node->aabb->mins.z, point.position.z);
	}

	// Adding small 2x delta to avoid floating point issues
	float epsilon = 0.5f;
	root_node->aabb->mins -= epsilon * root_node->aabb->mins;
	root_node->aabb->maxs += epsilon * root_node->aabb->maxs;

	// Make it cubic
	vec3 size = root_node->aabb->getSize();
	vec3 half_sizes_x = 0.5f * (vec3(size.x) - size);
	vec3 half_sizes_y = 0.5f * (vec3(size.y) - size);
	vec3 half_sizes_z = 0.5f * (vec3(size.z) - size);
	if(size.x > size.y){
		if(size.x > size.z){
			root_node->aabb->mins.y -= half_sizes_x.y;
			root_node->aabb->maxs.y += half_sizes_x.y;
			root_node->aabb->mins.z -= half_sizes_x.z;
			root_node->aabb->maxs.z += half_sizes_x.z;
		} else {
			root_node->aabb->mins.y -= half_sizes_z.y;
			root_node->aabb->maxs.y += half_sizes_z.y;
			root_node->aabb->mins.x -= half_sizes_z.x;
			root_node->aabb->maxs.x += half_sizes_z.x;
		}
	} else {
		if(size.y > size.z){
			root_node->aabb->mins.x -= half_sizes_y.x;
			root_node->aabb->maxs.x += half_sizes_y.x;
			root_node->aabb->mins.z -= half_sizes_y.z;
			root_node->aabb->maxs.z += half_sizes_y.z;
		} else {
			root_node->aabb->mins.y -= half_sizes_z.y;
			root_node->aabb->maxs.y += half_sizes_z.y;
			root_node->aabb->mins.x -= half_sizes_z.x;
			root_node->aabb->maxs.x += half_sizes_z.x;
		}
	}
}



uint32_t growOctree(OctreeNode* root_node, const std::shared_ptr<vector<Point>>& points){
	uint32_t nb_new_levels = 0;
	AABB new_aabb = AABB(*root_node->aabb);
	NodePosition node_position = FIRST_NODE_POSITION;
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
	NodePosition node_position = FIRST_NODE_POSITION;
	for(uint32_t i=0; i<nb_new_levels; i++){
		// Create new parent
		OctreeNode* new_parent = new OctreeNode();
		new_parent->occupancy = new OccupancyGrid();
		new_parent->from_bottom_up = true;

		new_parent->updated = true;
		cur_child->updated = true;

		// Create the correct child
		new_parent->children[node_position] = cur_child;
		new_parent->children_ids |= 0x01 << node_position;

		auto fillOccupancyGrid = [&](AABB& cur_aabb, const Chunk* child_chunk_list){
			while(child_chunk_list){
				for(uint32_t j=0; j<child_chunk_list->size; j++){
					const Point& point = child_chunk_list->points[j];

					// Sample voxel occupancy grid at this location
					vec3 normalized_coordinates = cur_aabb.getPointNormalizedCoordinates(point.position);
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
						vec3 world_grid_size = cur_aabb.getSize() / float(GRID_SIZE);
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
						if(parent_chunk_list->size == POINTS_PER_CHUNK){
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
		new_parent->aabb = new AABB(*cur_child->aabb);
		new_parent->aabb->extend(node_position);
		fillOccupancyGrid(*new_parent->aabb, cur_child->points);
		fillOccupancyGrid(*new_parent->aabb, cur_child->voxels);

		cur_child = new_parent;
		updateNodePosition(node_position);
	}
	return cur_child;
}


void freeOctreesOnGPU(CuRast* editor){
	std::string main_octree_name = getSimLodOctreeName();

	bool delete_all = CuRastSettings::freeOldOctreeMemoryOnGPU;
	std::vector<SNCOctree*> octrees = {};
	{
		std::lock_guard<std::mutex> lock_scene(updateSceneMutex);
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
		std::lock_guard<std::mutex> lock_scene(updateSceneMutex);
		editor->scene.world->remove(octree);
	}
}

void freePreviousOctreeOnGPU(CuRast* editor, std::shared_ptr<SNCOctree> caller){
	uint64_t octree_id = caller->octree_id;
	while(!caller->isDoneLoadingToGpu()){}

	SNCOctree* previous_octree = nullptr;	{
		std::lock_guard<std::mutex> lock_scene(updateSceneMutex);
		editor->scene.forEach<SNCOctree>([&](SNCOctree* node){
			if(node->octree_id == (octree_id-1)){
				node->need_to_be_executed = true;
			}
		});
	}
}


void loadOctreeOnGPU(OctreeNode* main_octree,
	CuRast* editor, CUcontext* context, bool bypass_semaphore
){
	if(CPU_PARALLELISED && !bypass_semaphore){
		// If the mainOctree / mainAABB are being updated (see `addPointBatches`)
		// Do not block but do not send data to the GPU
		// Acquire => semaphore_counter -= 1
		if(!octreeReadyToBeSent.try_acquire()){return;}
	}

	std::shared_ptr<Timing> timing = addTiming("send octree to GPU ", true);

	uint64_t load_id = 0;
	{
		std::lock_guard<std::mutex> lock_load_counter(loadCounterMtx);
		load_id = loadCounter + 1;
		loadCounter = load_id;
	}

	auto lambda = [&](CuRast* editor, CUcontext* context, OctreeNode* input_octree){
		// Create cuda memory pointers
		std::shared_ptr<SNCOctree> octree = make_shared<SNCOctree>(getSimLodOctreeName(true), simLodOctreeCounter);
		octree->cptr_nodes = {};
		octree->cptr_aabbs = {};
		octree->cptr_chunks = {};
		octree->cptr_occupancy_grids = {};
		cuCtxSetCurrent(*context);
		CUresult cuda_status = cuStreamCreate(&octree->stream, CU_STREAM_NON_BLOCKING);

		// Create enough chunks
		uint32_t chunk_counter = 0;
		uint32_t nodes_counter = 0;
		uint32_t occupancy_grids_counter = 0;
		uint32_t max_lod_level = 0;

		std::function<CUdeviceptr(const OctreeNode*&, uint8_t)> recursive = [&](
			const OctreeNode*& cur_node, uint8_t level
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

			
			// Chunk* cur_chunk = nullptr;
			auto allocateChunks = [&](const Chunk* root) -> std::optional<uint32_t> {
				uint32_t before_chunk_counter = chunk_counter;

				// Create CChunks
				vector<const Chunk*> chunks = {};

				const Chunk* cur_chunk = root;
				while(cur_chunk){
					chunks.push_back(cur_chunk);
					cur_chunk = cur_chunk->next;
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
					if(CPU_PARALLELISED){
						cuMemAllocAsync(&octree->cptr_chunks[chunk_counter], sizeof(CChunk), octree->stream);
						cuMemcpyHtoDAsync(octree->cptr_chunks[chunk_counter], &tmp, sizeof(CChunk), octree->stream);
					} else {
						cuMemAlloc(&octree->cptr_chunks[chunk_counter], sizeof(CChunk));
						cuMemcpyHtoD(octree->cptr_chunks[chunk_counter], &tmp, sizeof(CChunk));
					}
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
				chunk_first_points = allocateChunks(cur_node->points);
			} 
			if(cur_node->voxels){
				chunk_first_voxels = allocateChunks(cur_node->voxels);
			}
			

			// Create COctreeNode
			COctreeNode new_node = {};
			for(uint32_t child = 0; child < 8; child++){
				if(cur_node->children[child]){
					new_node.children[child] = (COctreeNode*) child_indices[child];
				}
			}
			new_node.counter = cur_node->counter;
			new_node.children_ids = cur_node->children_ids;
			new_node.level = level;

			if(chunk_first_points.has_value()){
				new_node.points = (CChunk*)octree->cptr_chunks[chunk_first_points.value()];
			}
			if(chunk_first_voxels.has_value()){
				new_node.voxels = (CChunk*)octree->cptr_chunks[chunk_first_voxels.value()];
			}
			if(cur_node->occupancy){
				// Allocate occupancy grid
				COccupancyGrid tmp = {};
				for(uint32_t i=0; i<C_GRID_NUM_CELLS; i++){
					tmp.values[i] = cur_node->occupancy->values[i];
				}
				octree->cptr_occupancy_grids.push_back(CUdeviceptr());
				if(CPU_PARALLELISED){
					cuMemAllocAsync(&octree->cptr_occupancy_grids[occupancy_grids_counter], sizeof(COccupancyGrid), octree->stream);
					cuMemcpyHtoDAsync(octree->cptr_occupancy_grids[occupancy_grids_counter], &tmp, sizeof(COccupancyGrid), octree->stream);
				} else {
					cuMemAlloc(&octree->cptr_occupancy_grids[occupancy_grids_counter], sizeof(COccupancyGrid));
					cuMemcpyHtoD(octree->cptr_occupancy_grids[occupancy_grids_counter], &tmp, sizeof(COccupancyGrid));
				}
				new_node.occupancy = (COccupancyGrid*)octree->cptr_occupancy_grids[occupancy_grids_counter];
				occupancy_grids_counter++;
			}
			if(level > max_lod_level){
				max_lod_level = level;
			}

			CAABB new_aabb = {
				.mins = cur_node->aabb->mins,
				.maxs = cur_node->aabb->maxs,
			};

			// cur_node->display(0, level, true);
			// println("\tAABB: .mins = ({}, {}, {}), .maxs = ({}, {}, {})",
			// 	new_aabb.mins.x, new_aabb.mins.y, new_aabb.mins.z,
			// 	new_aabb.maxs.x, new_aabb.maxs.y, new_aabb.maxs.z
			// );

			// Create cuda pointers
			CUdeviceptr cptr_node, cptr_aabb;
			if(CPU_PARALLELISED){
				cuMemAllocAsync(&cptr_node, sizeof(COctreeNode), octree->stream); 
				cuMemAllocAsync(&cptr_aabb, sizeof(CAABB), octree->stream);
				cuMemcpyHtoDAsync(cptr_node, &new_node, sizeof(COctreeNode), octree->stream);
				cuMemcpyHtoDAsync(cptr_aabb, &new_aabb, sizeof(CAABB), octree->stream);
			} else {
				cuMemAlloc(&cptr_node, sizeof(COctreeNode)); 
				cuMemAlloc(&cptr_aabb, sizeof(CAABB));
				cuMemcpyHtoD(cptr_node, &new_node, sizeof(COctreeNode));
				cuMemcpyHtoD(cptr_aabb, &new_aabb, sizeof(CAABB));
			}
			
			octree->cptr_nodes.push_back(cptr_node);
			octree->cptr_aabbs.push_back(cptr_aabb);

			nodes_counter++;
			return cptr_node;
		};


		
		const OctreeNode* nex_octree = input_octree;
		recursive(nex_octree, 0);

		octree->num_nodes = nodes_counter;
		octree->max_lod_level = max_lod_level;
		octree->nb_points = NB_POINTS;


		// Copy arrays of pointers to GPU
		if(CPU_PARALLELISED){
			cuMemAllocAsync(&octree->aabbs, octree->cptr_aabbs.size() * sizeof(CUdeviceptr), octree->stream);
			cuMemcpyHtoDAsync(octree->aabbs, octree->cptr_aabbs.data(), octree->cptr_aabbs.size() * sizeof(CUdeviceptr), octree->stream);
			cuMemAllocAsync(&octree->chunks, octree->cptr_chunks.size() * sizeof(CUdeviceptr), octree->stream);
			cuMemcpyHtoDAsync(octree->chunks, octree->cptr_chunks.data(), octree->cptr_chunks.size() * sizeof(CUdeviceptr), octree->stream);
			cuMemAllocAsync(&octree->nodes, octree->cptr_nodes.size() * sizeof(CUdeviceptr), octree->stream);
			cuMemcpyHtoDAsync(octree->nodes, octree->cptr_nodes.data(), octree->cptr_nodes.size() * sizeof(CUdeviceptr), octree->stream);
			cuMemAllocAsync(&octree->occupancy_grids, octree->cptr_occupancy_grids.size() * sizeof(CUdeviceptr), octree->stream);
			cuMemcpyHtoDAsync(octree->occupancy_grids, octree->cptr_occupancy_grids.data(), octree->cptr_occupancy_grids.size() * sizeof(CUdeviceptr), octree->stream);
		} else {
			cuMemAlloc(&octree->aabbs, octree->cptr_aabbs.size() * sizeof(CUdeviceptr));
			cuMemcpyHtoD(octree->aabbs, octree->cptr_aabbs.data(), octree->cptr_aabbs.size() * sizeof(CUdeviceptr));
			cuMemAlloc(&octree->chunks, octree->cptr_chunks.size() * sizeof(CUdeviceptr));
			cuMemcpyHtoD(octree->chunks, octree->cptr_chunks.data(), octree->cptr_chunks.size() * sizeof(CUdeviceptr));
			cuMemAlloc(&octree->nodes, octree->cptr_nodes.size() * sizeof(CUdeviceptr));
			cuMemcpyHtoD(octree->nodes, octree->cptr_nodes.data(), octree->cptr_nodes.size() * sizeof(CUdeviceptr));
			cuMemAlloc(&octree->occupancy_grids, octree->cptr_occupancy_grids.size() * sizeof(CUdeviceptr));
			cuMemcpyHtoD(octree->occupancy_grids, octree->cptr_occupancy_grids.data(), octree->cptr_occupancy_grids.size() * sizeof(CUdeviceptr));
		}
		{
			std::lock_guard<std::mutex> lock_scene(updateSceneMutex);
			editor->scene.world->children.push_back(octree);
		}

		if(CuRastSettings::autoFreeOldOctreeMemoryOnGPU){
			std::thread thread_free_old_octree([&](CuRast* editor, std::shared_ptr<SNCOctree> octree){
				freePreviousOctreeOnGPU(editor, octree);
			}, editor, octree);
			thread_free_old_octree.detach();
		}
		// println("Final octree: {} nodes, {} max level", nodes_counter, max_lod_level);
	};

	std::thread thread_load_to_gpu([&](CuRast* editor, CUcontext* context, 
		OctreeNode* octree, uint64_t load_id
	){
		// Acquire => semaphore_counter -= 1
		octreeNotBeingSent.acquire();

		bool send_to_gpu = false;
		{
			std::lock_guard<std::mutex> lock_load_counter(loadCounterMtx);
			if(load_id == loadCounter){
				send_to_gpu = true;
			}
		}

		if(send_to_gpu){
			lambda(editor, context, octree);
		}

		// Release => semaphore_counter += 1
		octreeNotBeingSent.release();
	}, editor, context, main_octree, load_id);
	if(!CPU_PARALLELISED){
		thread_load_to_gpu.join();
	} else {
		thread_load_to_gpu.detach();
	}

	if(CPU_PARALLELISED && !bypass_semaphore){
		// Release the semaphore when every GPU side structures is done being built
		// Rlease => semaphore_counter += 1
		octreeReadyToBeUpdated.release();
	}

	timing->stop_clock();
}









void addPointBatches(){
	std::vector<uint32_t> batches_indices(CuRastSettings::maxBatchesPerUpdate, 0);
	uint32_t last_index = 0;

	for(uint32_t i=0; i<BATCHES_QUEUE_SIZE; i++){
		std::lock_guard<std::mutex> lock(batchesQueueMutexes[i]);
		if(batchesQueue[i] && batchesQueue[i]->state == BatchState::Loaded){
			batches_indices[last_index] = i;
			last_index++;
			if(last_index >= CuRastSettings::maxBatchesPerUpdate){break;}
		}
	}
	if(last_index == 0){return;}
	static uint32_t lastUpdateAttempt = 0;
	if(CPU_PARALLELISED && batches_indices.size() < CuRastSettings::minBatchesPerUpdate){
		// println("counter: {}", lastUpdateAttempt);
		if(lastUpdateAttempt < 256){
			lastUpdateAttempt++;
			return;
		} 
	}
	lastUpdateAttempt = 0;

	auto first = batches_indices.begin();
	auto last = first + last_index;

	if(!mainOctree->aabb){
		std::shared_ptr<Timing> timing = addTiming("init octree", true);
		mainOctree->aabb = new AABB();
		uint32_t batch_index = batches_indices[0];
		std::lock_guard<std::mutex> lock(batchesQueueMutexes[batch_index]);
		initOctree(mainOctree, batchesQueue[batch_index]->points);
		timing->stop_clock();

		// Copy octree once at the beginning
		timing = addTiming("copy initial octree", true);
		delete(mainOctreeCpy);
		mainOctreeCpy = new OctreeNode(*mainOctree);
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
	std::shared_ptr<Timing> timing = addTiming("compute max new level", true);
	// Compute max new level needed per batch
	vector<uint32_t> tmp_new_levels = vector<uint32_t>(last_index, 0);
	// In parallel
	{
		std::vector<uint32_t> indices(last_index);
		auto first_of_indices = indices.begin();
		auto last_of_indices = indices.end();
		std::iota(first_of_indices, last_of_indices, 0);
		if(CPU_PARALLELISED){
			std::for_each(std::execution::par, first_of_indices, last_of_indices, [&](uint32_t index){
				std::shared_ptr<PointBatch> batch = batchesQueue[batches_indices[index]];
				std::lock_guard<std::mutex> lock(batchesQueueMutexes[batches_indices[index]]);
				tmp_new_levels[index] = growOctree(mainOctreeCpy, batch->points);
			});
		} else {
			std::for_each(first_of_indices, last_of_indices, [&](uint32_t index){
				std::shared_ptr<PointBatch> batch = batchesQueue[batches_indices[index]];
				std::lock_guard<std::mutex> lock(batchesQueueMutexes[batches_indices[index]]);
				tmp_new_levels[index] = growOctree(mainOctreeCpy, batch->points);
			});
		}
	}
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("//////////// Octree after grow octree ////////////");
	// println("//////////////////////////////////////////////////");
	// mainOctreeCpy->display();


	timing = addTiming("update octree bottom up", true);
	// In single thread
	{
		uint32_t nb_new_levels = 0;
		for(uint32_t& level : tmp_new_levels){
			nb_new_levels = max(nb_new_levels, level);
		}
		// println("Max new level: {}", nb_new_levels);
		mainOctreeCpy = uptadeOctree(mainOctreeCpy, nb_new_levels);
	}
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("/////////// Octree after update octree ///////////");
	// println("//////////////////////////////////////////////////");
	// mainOctreeCpy->display();


	timing = addTiming("simlod update", true);
	// TODO: In parallel
	{
		std::for_each(first, last, [&](uint32_t index){
			std::lock_guard<std::mutex> lock(batchesQueueMutexes[index]);
			std::shared_ptr<PointBatch> batch = batchesQueue[index];
			simLodUpdate(mainOctreeCpy, batch->points);
		});
	}
	timing->stop_clock();

	// println("//////////////////////////////////////////////////");
	// println("/////////// Octree after simLOD update ///////////");
	// println("//////////////////////////////////////////////////");
	// mainOctreeCpy->display();

	timing = addTiming("update cache", true);
	updateCache(mainOctreeCpy);
	timing->stop_clock();
	// displayCache();

	// println("//////////////////////////////////////////////////");
	// println("/////////// Octree after cache update ///////////");
	// println("//////////////////////////////////////////////////");
	// mainOctreeCpy->display();

	if(CPU_PARALLELISED){
		// Block if the mainOctree / mainAABB are being send to the GPU (see `loadOctreeOnGPU`)
		// Acquire => semaphore_counter -= 1
		octreeReadyToBeUpdated.acquire();
	}

	if(CPU_PARALLELISED){
		std::for_each(std::execution::par, first, last, [&](uint32_t index){
			std::lock_guard<std::mutex> lock(batchesQueueMutexes[index]);
			batchesQueue[index]->state = BatchState::Inserted;
		});
	} else {
		std::for_each(first, last, [&](uint32_t index){
			std::lock_guard<std::mutex> lock(batchesQueueMutexes[index]);
			batchesQueue[index]->state = BatchState::Inserted;
		});
	}

	// Delete old mainOctree before copying new octree
	delete(mainOctree);
	mainOctree = new OctreeNode(*mainOctreeCpy);
	
	if(CPU_PARALLELISED){
		// Release the semaphore when not using mainOctree / mainAABB anymore
		// Rlease => semaphore_counter += 1
		octreeReadyToBeSent.release();
	}
};


void updateOctreeRoutine(){
	while(true){
		addPointBatches();
	}
}








// void mainLoop(){
// 	// Init everything
// 	// while true:
// 		// CPU side, in parallel:
// 			// - Load points on CPU
// 			// - Upload points to GPU
// 			// - Run render / grow / update kernels
// 		// GPU side:
// 			// - Render
// 			// - Grow, for every thread in parallel, compute nb new levels
// 			// - One thread creates the new children
// 			// - Update
// }



void getCulledNodes(){
	// TODO:
	// Fetch all nodes in frustum (DFS list)
		// If node not in tree && node stored
			// Load node
			// Update LRU
	// If new nodes loaded
		// Send new octree to GPU
			
}
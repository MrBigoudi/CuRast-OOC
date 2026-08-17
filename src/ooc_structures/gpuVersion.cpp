#include "gpuVersion.h"

#include "loader.h"
#include "outOfCore.h"

// #include <list>

void GpuVersion::initHostSide(CuRast* editor, CUcontext* context) {
    // Host side data
    exchangedPointsPointers = malloc(OocSimLodSettings::MAX_NB_NODES_TO_EXCHANGE * sizeof(CUdeviceptr));
    exchangedVoxelsPointers = malloc(OocSimLodSettings::MAX_NB_NODES_TO_EXCHANGE * sizeof(CUdeviceptr));
    batchesToAddPointsPointers = malloc(OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE * sizeof(CUdeviceptr));
    CURuntime::assertCudaSuccess(cuMemAllocHost(&nbExchangedNodes, sizeof(uint32_t)));
    // CURuntime::assertCudaSuccess(cuMemAllocHost(&isTemporarySwitching, sizeof(bool)));
    *(uint32_t*)nbExchangedNodes = 0;
    // *(bool*)isTemporarySwitching = false;

    // CURuntime::assertCudaSuccess(cuEventCreate(&eventUpdateCompleted, CU_EVENT_DISABLE_TIMING));
    // CURuntime::assertCudaSuccess(cuEventCreate(&eventSwapCompleted, CU_EVENT_DISABLE_TIMING));
    // CURuntime::assertCudaSuccess(cuEventCreate(&eventRenderingStreamInformed, CU_EVENT_DISABLE_TIMING));


    // hostCache = new CLRUCache(OocSimLodSettings::LRU_CPU_CACHE_SIZE);
    // removedNodes.reserve(hostCache->CACHE_SIZE);
    // newlyVisible = std::vector<std::shared_ptr<HostStorageNode>>(hostCache->CACHE_SIZE, nullptr);
    // newlyVisibleToDelete = std::vector<bool>(hostCache->CACHE_SIZE, false);
    // previouslyVisible = std::vector<CIdAABB>(hostCache->CACHE_SIZE, CINVALID_ID);
    relationshipMap = std::vector<CIdAABB>(OocSimLodSettings::MAX_NB_NODES, CINVALID_ID);
}

void GpuVersion::initBuffers(CuRast* editor, CUcontext* context) {

    // Unbounded data
    hostStaging.maxNbConcurrentNodes = OocSimLodSettings::MAX_NB_NODES;
    hostStaging.relationshipMap = alloc<CGlobalVariables::Relationship>(hostStaging.maxNbConcurrentNodes);
    hostStaging.packedNodes = alloc<COctreeNode*>(hostStaging.maxNbConcurrentNodes);
    hostStaging.nodesFlags = alloc<uint32_t>(hostStaging.maxNbConcurrentNodes);

    // hostStaging.renderingPackedNodes = alloc<COctreeNode*>(hostStaging.maxNbConcurrentNodes);
    // hostStaging.renderingPackedNodesTmp = alloc<COctreeNode*>(hostStaging.maxNbConcurrentNodes);
    hostStaging.temporaryBufferSize = hostStaging.maxNbConcurrentNodes;
    hostStaging.temporaryIdBuffer = alloc<CIdAABB>(hostStaging.temporaryBufferSize);
    hostStaging.temporaryNodeBuffer = alloc<COctreeNode*>(hostStaging.temporaryBufferSize);


    hostStaging.maxCountSplitIterations = OocSimLodSettings::MAX_NB_COUNT_SPLIT_ITERATION;


    // Exchangeable data
    hostStaging.maxNbNodesExchanged = OocSimLodSettings::MAX_NB_NODES_TO_EXCHANGE;
    hostStaging.exchangedAABBIndices = alloc<CIdAABB>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedAABBParentsIndices = alloc<CIdAABB>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedAABBs = alloc<CAABB>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedChildrenIds = alloc<uint32_t>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedPointsCounters = alloc<uint32_t>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedVoxelsCounters = alloc<uint32_t>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedPoints = alloc<CPoint*>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedVoxels = alloc<CPoint*>(hostStaging.maxNbNodesExchanged);
    hostStaging.maxNbPointsChunksPerExchangedNode = 
        (OocSimLodSettings::MAX_POINTS_PER_LEAF + OocSimLodSettings::NB_POINTS_PER_CHUNK - 1) 
        / OocSimLodSettings::NB_POINTS_PER_CHUNK
    ;
    hostStaging.maxNbVoxelsChunksPerExchangedNode = OocSimLodSettings::MAX_NB_VOXELS_CHUNKS_TO_EXCHANGE;
    hostStaging.maxNbRenderedPoints = OocSimLodSettings::MAX_NB_RENDERED_POINTS;
    hostStaging.maxNbRenderedVoxels = OocSimLodSettings::MAX_NB_RENDERED_VOXELS;
    
    for(uint32_t i=0; i<hostStaging.maxNbNodesExchanged; i++){
        uint64_t real_size = 0;
    
        CUdeviceptr new_ptr = 0;
        real_size = OocSimLodSettings::NB_POINTS_PER_CHUNK * hostStaging.maxNbPointsChunksPerExchangedNode * sizeof(CPoint);
        totalAllocatedMemory += real_size;
        CURuntime::assertCudaSuccess(cuMemAlloc(&new_ptr, real_size));
        ((CUdeviceptr*)(exchangedPointsPointers))[i] = new_ptr;
        pointers.push_back(new_ptr);

        real_size = OocSimLodSettings::NB_POINTS_PER_CHUNK * hostStaging.maxNbVoxelsChunksPerExchangedNode * sizeof(CPoint);
        totalAllocatedMemory += real_size;
        CURuntime::assertCudaSuccess(cuMemAlloc(&new_ptr, real_size));
        ((CUdeviceptr*)(exchangedVoxelsPointers))[i] = new_ptr;
        pointers.push_back(new_ptr);

        CURuntime::assertCudaSuccess(cuMemcpyHtoD(
            (CUdeviceptr)hostStaging.exchangedPoints,
            exchangedPointsPointers,
            hostStaging.maxNbNodesExchanged * sizeof(CUdeviceptr)
        ));
        CURuntime::assertCudaSuccess(cuMemcpyHtoD(
            (CUdeviceptr)hostStaging.exchangedVoxels,
            exchangedVoxelsPointers,
            hostStaging.maxNbNodesExchanged * sizeof(CUdeviceptr)
        ));
    }

    hostStaging.gridsToInit = alloc<COctreeNode*>(hostStaging.maxNbConcurrentNodes);

    hostStaging.maxNbBatches = OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE;
    hostStaging.batchesAddedMask = alloc<uint32_t>(hostStaging.maxNbBatches);
    hostStaging.batchesToAddCounts = alloc<uint32_t>(hostStaging.maxNbBatches);
    hostStaging.batchesToAddPoints = alloc<CPoint*>(hostStaging.maxNbBatches);
    uint64_t real_size = OocSimLodSettings::MAX_POINTS_PER_BATCHES * sizeof(CPoint);
    for(uint32_t i=0; i<hostStaging.maxNbBatches; i++){
        totalAllocatedMemory += real_size;
        CUdeviceptr new_ptr = 0;
        CURuntime::assertCudaSuccess(cuMemAlloc(&new_ptr, real_size));
        ((CUdeviceptr*)(batchesToAddPointsPointers))[i] = new_ptr;
        pointers.push_back(new_ptr);
    }
    CURuntime::assertCudaSuccess(cuMemcpyHtoD(
        (CUdeviceptr)hostStaging.batchesToAddPoints,
        batchesToAddPointsPointers,
        hostStaging.maxNbBatches * sizeof(CUdeviceptr)
    ));
    


    // Visibility cache
    hostStaging.renderedPoints = alloc<CPoint>(hostStaging.maxNbRenderedPoints);
    hostStaging.renderedVoxels = alloc<CPoint>(hostStaging.maxNbRenderedVoxels);
    hostStaging.renderedVoxelsSizes = alloc<glm::vec3>(hostStaging.maxNbRenderedVoxels);
    hostStaging.renderedVoxelsNextChildIndex = alloc<CNodePosition>(hostStaging.maxNbRenderedVoxels);
    hostStaging.renderedVoxelsNodes = alloc<CIdAABB>(hostStaging.maxNbRenderedVoxels);
    

    // Lru caches
    hostStaging.updatesCacheSize = OocSimLodSettings::LRU_UPDATES_CACHE_SIZE;
    hostStaging.updatesCache = nullptr;
    hostStaging.visibilityCacheSize = OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE;
    hostStaging.visibilityCache = alloc<CIdAABB>(hostStaging.visibilityCacheSize);

    
    // Temporary buffers
    hostStaging.maxNbSpilledPoints = OocSimLodSettings::MAX_NB_SPILLING_POINTS;
    hostStaging.spilledPoints = alloc<CPoint>(hostStaging.maxNbSpilledPoints);
    hostStaging.spillingNodes = alloc<COctreeNode*>(hostStaging.maxNbSpilledPoints);

    hostStaging.maxNbBacklogVoxels = OocSimLodSettings::MAX_NB_BACKLOG_VOXELS;
    hostStaging.backlogVoxels = alloc<CPoint>(hostStaging.maxNbBacklogVoxels);
    hostStaging.backlogVoxelsNodes = alloc<COctreeNode*>(hostStaging.maxNbBacklogVoxels);

    hostStaging.maxPointsPerLeaf = OocSimLodSettings::MAX_POINTS_PER_LEAF;
    

    // Final allocation
    deviceStaging = prog->getGlobalsPointer("globalVariables");
    if (deviceStaging == 0) {
        throw std::runtime_error("globalVariables symbol not found");
    }
    CURuntime::assertCudaSuccess(cuMemcpyHtoD(deviceStaging, &hostStaging, sizeof(CGlobalVariables)));
}


void GpuVersion::initAllocators(CuRast* editor, CUcontext* context, CUstream* stream) {
    CMemoryAllocator hostStaging = {};

    hostStaging.chunksAllocator = allocAllocator<CChunk>(
        OocSimLodSettings::NB_ALLOCABLE_CHUNKS, AllocatorId::ChunkAllocator, stream
    );
    hostStaging.gridsAllocator = allocAllocator<COccupancyGrid>(
        OocSimLodSettings::NB_ALLOCABLE_GRIDS, AllocatorId::OccupancyGridAllocator, stream
    );
    hostStaging.nodesAllocator = allocAllocator<COctreeNode>(
        OocSimLodSettings::NB_ALLOCABLE_NODES, AllocatorId::OctreeNodeAllocator, stream
    );

    CUdeviceptr global_allocator_ptr = prog->getGlobalsPointer("globalAllocator");
    if (global_allocator_ptr == 0) {
        throw std::runtime_error("globalAllocator symbol not found");
    }
    CURuntime::assertCudaSuccess(cuMemcpyHtoD(global_allocator_ptr, &hostStaging, sizeof(CMemoryAllocator)));
}






void GpuVersion::init(CuRast* editor, CUcontext* context) {
    prog = new CudaModularProgram({
        "./src/kernels/ooc/init.cu",
        "./src/kernels/ooc/bottomUp.cu",
        "./src/kernels/ooc/simlod.cu",
        "./src/kernels/ooc/caches.cu",
        "./src/kernels/ooc/render.cu",
        "./src/kernels/ooc/test.cu",
    });

    CURuntime::assertCudaSuccess(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    initHostSide(editor, context);
    initBuffers(editor, context);
    initAllocators(editor, context, &stream);
    cudaDeviceSynchronize(); // Needed because of the batch copies in the init functions

    // size_t heap_size = 1024 * 1024 * 1024; // 1Gb for now
    // CURuntime::assertCudaSuccess(cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, heap_size));

    OptionalLaunchSettings launch_settings = {
        .gridsize  = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE,
        .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    };
    prog->launch("kernel_init_global_allocators", {}, launch_settings);
    prog->launch("kernel_init_global_buffers", {}, launch_settings);
    LoaderGpuVersion::init();
}


void GpuVersion::destroy(CuRast *editor, CUcontext *context){
    // if(hostCache){delete(hostCache);}

    cudaDeviceSynchronize();
    for(CUdeviceptr& ptr : pointers){
        if(ptr){
            CURuntime::assertCudaSuccess(cuMemFree(ptr));
        }
    }
    free(exchangedPointsPointers);
    free(exchangedVoxelsPointers);
    free(batchesToAddPointsPointers);
    CURuntime::assertCudaSuccess(cuMemFreeHost(nbExchangedNodes));
    // CURuntime::assertCudaSuccess(cuMemFreeHost(isTemporarySwitching));

    // cuEventDestroy(eventUpdateCompleted);
    // cuEventDestroy(eventSwapCompleted);
    // cuEventDestroy(eventRenderingStreamInformed);

    cudaDeviceSynchronize();
    CURuntime::assertCudaSuccess(cuStreamDestroy(stream));
}













void GpuVersion::octreeUpdateInit(CuRast* editor, CUcontext* context){
    OptionalLaunchSettings launch_settings = {
        .gridsize  = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE,
        .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    };
    prog->launch("kernel_init_octree_part_1_aabb_measuring", {}, launch_settings);

    launch_settings = {
        .gridsize = 1,
        .blocksize = 1,
        .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    };
    prog->launch("kernel_init_octree_part_2_refining", {}, launch_settings);
}














void GpuVersion::octreeUpdateBottomUp(CuRast* editor, CUcontext* context){
    OptionalLaunchSettings launch_settings = {
        .gridsize  = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE,
        .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    };
    prog->launch("kernel_bottom_up_update_part_1_counting", {}, launch_settings);

    launch_settings = {
        .gridsize = 1,
        .blocksize = 1,
        .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    };
    prog->launch("kernel_bottom_up_update_part_2_instancing", {}, launch_settings);
}













void GpuVersion::octreeUpdateFillNewGrids(CuRast* editor, CUcontext* context){
    uint32_t block_size = min(
        OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM,
        OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
    );
    OptionalLaunchSettings launch_settings = {
        .gridsize = OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM,
        .blocksize = block_size,
        .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    };
    prog->launch("kernel_fill_new_grids", {}, launch_settings);
}







void GpuVersion::octreeUpdateSimLODLoad(CuRast* editor, CUcontext* context){
    OptionalLaunchSettings launch_settings = {
        .gridsize  = 1,
        .blocksize = 1,
        .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    };
    prog->launch("kernel_simlod_load_part_0_reset", {}, launch_settings);

    launch_settings = {
        .gridsize  = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE,
        .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    };
    prog->launch("kernel_simlod_load_part_1_flagging", {}, launch_settings);

    // Pointers arithmetic shenanigans to get the correct device address
    uint64_t pad = uint64_t(&(hostStaging.nbNodesExchanged)) - uint64_t(&hostStaging);
    CUdeviceptr src_device = deviceStaging + pad;

    // Get the number of nodes to load
    CURuntime::assertCudaSuccess(cuMemcpyDtoHAsync(
		nbExchangedNodes, 
		src_device,
		sizeof(uint32_t),
        OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
	));

    cudaStreamSynchronize(OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0);

    uint32_t nb_nodes_to_load = min(*(uint32_t*)(nbExchangedNodes), hostStaging.maxNbNodesExchanged);
    if(nb_nodes_to_load == 0){return;}
    // println("\n\nNb nodes to load: {}\n\n\n", nb_nodes_to_load);

    // Get the ids of the nodes to load
    std::vector<CIdAABB> ids(nb_nodes_to_load, CINVALID_ID);
    CURuntime::assertCudaSuccess(cuMemcpyDtoHAsync(
		ids.data(), 
		(CUdeviceptr)hostStaging.exchangedAABBIndices,
		nb_nodes_to_load * sizeof(CIdAABB),
        OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
	));

    cudaStreamSynchronize(OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0);

    // Load the nodes from disk
    std::vector<std::shared_ptr<HostStorageNode>> loaded(nb_nodes_to_load, nullptr);
    
    // // Sync
    // {
    //     std::lock_guard<std::mutex> lock(syncHostStorageNodesAccessMtx);
    //     for(uint32_t i=0; i<nb_nodes_to_load; i++){
    //         CIdAABB aabb_index = ids[i];
    //         if(persistentStoredNodes.contains(aabb_index)){
    //             loaded[i] = persistentStoredNodes[aabb_index];            
    //         } else {
    //             std::shared_ptr<HostStorageNode> new_node = OctreeNodeSerializable::deserializeV2(aabb_index, "From simlod load");
    //             persistentStoredNodes[aabb_index] = new_node;
    //             loaded[i] = new_node;
    //         }
    //         recentlyUsedNodesFromUpdates.insert(aabb_index);
    //     }
    // }
    for(uint32_t i=0; i<nb_nodes_to_load; i++){
        CIdAABB aabb_index = ids[i];
        std::shared_ptr<HostStorageNode> new_node = OctreeNodeSerializable::deserializeV2(aabb_index);
        loaded[i] = new_node;
    }




    // Send the nodes back to the device
    if(nb_nodes_to_load > hostStaging.maxNbNodesExchanged){
        println("ERROR: can't send more than {} nodes back to the device... tried to send {}", 
            hostStaging.maxNbNodesExchanged, nb_nodes_to_load
        );
        throw(EXIT_FAILURE);
    }

    std::vector<uint32_t> children_ids(nb_nodes_to_load, 0);
    std::vector<uint32_t> nbs_points(nb_nodes_to_load, 0);
    std::vector<uint32_t> nbs_voxels(nb_nodes_to_load, 0);
    std::vector<CAABB> aabbs(nb_nodes_to_load, CAABB());

    for(uint32_t i = 0; i<nb_nodes_to_load; i++){
        ids[i] = loaded[i]->node.aabb_index;
        children_ids[i] = uint32_t(loaded[i]->node.children_ids);
        aabbs[i].maxs = loaded[i]->node.aabb.maxs;
        aabbs[i].mins = loaded[i]->node.aabb.mins;
        nbs_points[i] = loaded[i]->node.points_counter;
        nbs_voxels[i] = loaded[i]->node.voxels_counter;

        // printf("loaded: %d: .mins = (%f, %f, %f), .maxs = (%f, %f, %f)\n", i,
        //     aabbs[i].mins.x, aabbs[i].mins.y, aabbs[i].mins.z,
        //     aabbs[i].maxs.x, aabbs[i].maxs.y, aabbs[i].maxs.z
        // );

        if(nbs_points[i]){
            CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync( 
                ((CUdeviceptr*)(exchangedPointsPointers))[i],
                loaded[i]->points.data(), nbs_points[i]*sizeof(CPoint), 
                OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
            ));
        }
        if(nbs_voxels[i]){
            CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync( 
                ((CUdeviceptr*)(exchangedVoxelsPointers))[i],
                loaded[i]->voxels.data(), nbs_voxels[i]*sizeof(CPoint),
                OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
            ));
        }
    }

    CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
		(CUdeviceptr)hostStaging.exchangedAABBIndices,
		ids.data(),
		nb_nodes_to_load * sizeof(CIdAABB), OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
	));
    CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
		(CUdeviceptr)hostStaging.exchangedAABBs,
		aabbs.data(),
		nb_nodes_to_load * sizeof(CAABB), OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
	));
    CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
		(CUdeviceptr)hostStaging.exchangedChildrenIds,
		children_ids.data(),
		nb_nodes_to_load * sizeof(uint32_t), OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
	));
    CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
		(CUdeviceptr)hostStaging.exchangedPointsCounters,
		nbs_points.data(),
		nb_nodes_to_load * sizeof(uint32_t), OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
	));
    CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
		(CUdeviceptr)hostStaging.exchangedVoxelsCounters,
		nbs_voxels.data(),
		nb_nodes_to_load * sizeof(uint32_t), OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
	));

    launch_settings = {
        .gridsize = OocSimLodSettings::MAX_NB_NODES_TO_EXCHANGE,
        .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    };
    prog->launch("kernel_simlod_load_part_2_rebuilding_nodes", {}, launch_settings);

    octreeUpdateFillNewGrids(editor, context);

    prog->launch("kernel_simlod_load_part_3_rebuilding_children", {}, launch_settings);
}














void GpuVersion::octreeUpdateSimLODCountSplit(CuRast* editor, CUcontext* context){
    OptionalLaunchSettings launch_settings = {
        .gridsize = 0, // Not used with launchCoopertative
        .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    };
    prog->launchCooperative("kernel_simlod_count_split", {}, launch_settings);
}














void GpuVersion::octreeUpdateSimLODVoxelSampling(CuRast* editor, CUcontext* context){
    octreeUpdateFillNewGrids(editor, context);

    uint32_t block_size = 32;
    uint32_t num_sms = OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM;
    uint32_t max_threads_per_sm = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM;
    uint32_t grid_size = (num_sms * max_threads_per_sm + block_size - 1) / block_size;

    OptionalLaunchSettings launch_settings = {
        .gridsize = grid_size,
        .blocksize = block_size,
        .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    };
    prog->launch("kernel_simlod_voxel_sampling", {}, launch_settings);
}












void GpuVersion::octreeUpdateSimLODInsertion(CuRast* editor, CUcontext* context){
    OptionalLaunchSettings launch_settings = {
        .gridsize  = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE,
        .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    };

    prog->launch("kernel_simlod_insertion_part_1_chunks_allocations", {}, launch_settings);
    prog->launch("kernel_simlod_insertion_part_2_filling", {}, launch_settings);
}















void GpuVersion::octreeUpdateSimLOD(CuRast* editor, CUcontext* context){
    octreeUpdateSimLODLoad(editor, context);

    octreeUpdateSimLODCountSplit(editor, context);

    octreeUpdateSimLODVoxelSampling(editor, context);

    octreeUpdateSimLODInsertion(editor, context);
}















void GpuVersion::octreeUpdateCacheUpdate(CuRast* editor, CUcontext* context){
    OptionalLaunchSettings launch_settings = {
        .gridsize = 1,
        .blocksize = 1,
        .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    };
    prog->launch("kernel_update_updates_cache", {}, launch_settings);

    uint32_t grid_size = OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM 
        * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM
    ;
    launch_settings = {
        .gridsize  = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE,
        .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    };
    prog->launch("kernel_prepare_store_part_1_filling_buffers", {}, launch_settings);
    prog->launch("kernel_prepare_store_part_2_resetting_children", {}, launch_settings);

    launch_settings = {
        .gridsize = 0,
        .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0,
    };
    prog->launchCooperative("kernel_prepare_store_part_3_updating_levels", {}, launch_settings);

    // Readback the nodes and store them
    // Pointers arithmetic shenanigans to get the correct device address
    uint64_t pad = uint64_t(&(hostStaging.nbNodesExchanged)) - uint64_t(&hostStaging);
    CUdeviceptr src_device = deviceStaging + pad;

    // Get the number of nodes to store
    CURuntime::assertCudaSuccess(cuMemcpyDtoHAsync(
		nbExchangedNodes, 
		src_device,
		sizeof(uint32_t),
        OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
	));

    cudaStreamSynchronize(OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0);
    
    uint32_t nb_nodes_to_store = min(*(uint32_t*)(nbExchangedNodes), hostStaging.maxNbNodesExchanged);
    if(nb_nodes_to_store == 0){return;}
    // println("\n\nNb nodes to store: {}\n\n\n", nb_nodes_to_store);

    std::vector<CIdAABB> ids(nb_nodes_to_store, CINVALID_ID);
    std::vector<CIdAABB> parents_ids(nb_nodes_to_store, CINVALID_ID);
    std::vector<uint32_t> children_ids(nb_nodes_to_store, 0);
    std::vector<CAABB> aabbs(nb_nodes_to_store, CAABB());
    std::vector<uint32_t> nbs_points(nb_nodes_to_store, 0);
    std::vector<uint32_t> nbs_voxels(nb_nodes_to_store, 0);

    CURuntime::assertCudaSuccess(cuMemcpyDtoHAsync(
		ids.data(),
		(CUdeviceptr)hostStaging.exchangedAABBIndices, 
		nb_nodes_to_store * sizeof(CIdAABB), OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
	));
    CURuntime::assertCudaSuccess(cuMemcpyDtoHAsync(
		parents_ids.data(),
		(CUdeviceptr)hostStaging.exchangedAABBParentsIndices, 
		nb_nodes_to_store * sizeof(CIdAABB), OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
	));
    CURuntime::assertCudaSuccess(cuMemcpyDtoHAsync(
		children_ids.data(),
		(CUdeviceptr)hostStaging.exchangedChildrenIds, 
		nb_nodes_to_store * sizeof(uint32_t), OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
	));
    CURuntime::assertCudaSuccess(cuMemcpyDtoHAsync(
		aabbs.data(),
		(CUdeviceptr)hostStaging.exchangedAABBs, 
		nb_nodes_to_store * sizeof(CAABB), OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
	));
    CURuntime::assertCudaSuccess(cuMemcpyDtoHAsync(
		nbs_points.data(),
		(CUdeviceptr)hostStaging.exchangedPointsCounters, 
		nb_nodes_to_store * sizeof(uint32_t), OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
	));
    CURuntime::assertCudaSuccess(cuMemcpyDtoHAsync(
		nbs_voxels.data(),
		(CUdeviceptr)hostStaging.exchangedVoxelsCounters, 
		nb_nodes_to_store * sizeof(uint32_t), OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
	));
    cudaStreamSynchronize(OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0);


    for(uint32_t i=0; i<nb_nodes_to_store; i++){
        
        std::shared_ptr<HostStorageNode> tmp_node = std::make_shared<HostStorageNode>();
        tmp_node->node.aabb_index = ids[i];
        tmp_node->node.aabb = aabbs[i];
        tmp_node->node.children_ids = children_ids[i];
        tmp_node->node.points_counter = nbs_points[i];
        tmp_node->node.voxels_counter = nbs_voxels[i];
        tmp_node->points = std::vector<CPoint>(tmp_node->node.points_counter);
        tmp_node->voxels = std::vector<CPoint>(tmp_node->node.voxels_counter);

        if(tmp_node->node.points_counter > 0){
            CURuntime::assertCudaSuccess(cuMemcpyDtoHAsync(
                tmp_node->points.data(),
                ((CUdeviceptr*)(exchangedPointsPointers))[i],
                tmp_node->node.points_counter * sizeof(CPoint), OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
            ));
        }

        if(tmp_node->node.voxels_counter > 0){
            CURuntime::assertCudaSuccess(cuMemcpyDtoHAsync(
                tmp_node->voxels.data(),
                ((CUdeviceptr*)(exchangedVoxelsPointers))[i],
                tmp_node->node.voxels_counter * sizeof(CPoint), OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
            ));
        }

        cudaStreamSynchronize(OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0);

        
        // Update the CPU version of the relationship map
        relationshipMap[ids[i]] = parents_ids[i];

        // Store the node on disk
        OctreeNodeSerializable::serializeV2(tmp_node);

        // {
        //     std::lock_guard<std::mutex> lock(syncAABBStorageAccessMtx);
        //     if(!storedNodes.contains(node.aabb_index)){
        //         storedNodes[node.aabb_index] =  node.aabb;
        //     }
        // }
        // std::shared_ptr<HostStorageNode> new_node = new HostStorageNode();
        // new_node->node = node;
        // new_node->points = points;
        // new_node->voxels = voxels;
        
        // // Sync
        // {
        //     std::lock_guard<std::mutex> lock(syncHostStorageNodesAccessMtx);
        //     persistentStoredNodes[node.aabb_index] = new_node;
        //     recentlyUsedNodesFromUpdates.insert(node.aabb_index);
        // }

        
    }
}




void GpuVersion::updateOctree(CuRast* editor, CUcontext* context){
	cuCtxSetCurrent(*context);

    GpuVersionUI::lastUpdateStart = high_resolution_clock::now();

    LoaderGpuVersion::run(&stream, editor, context);
    octreeUpdateInit(editor, context);
    octreeUpdateBottomUp(editor, context);

    static uint32_t cpt = 0;
    cpt++;
    uint32_t max_cpt = 100;

    // // TODO: to remove, just to debug
    // {
    //     OptionalLaunchSettings launch_settings = {
    //         .gridsize = 1,
    //         .blocksize = 1,
    //         .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    //     };
    //     bool should_display = true;
    //     PipelineLevel level = PipelineLevel::LevelBottomUp;
    //     if(cpt < max_cpt){
    //         prog->launch("kernel_test_display", {&level, &should_display}, launch_settings);
    //     }
    // }

    octreeUpdateSimLOD(editor, context);

    // // TODO: to remove, just to debug
    // {
    //     OptionalLaunchSettings launch_settings = {
    //         .gridsize = 1,
    //         .blocksize = 1,
    //         .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    //     };
    //     bool should_display = true;
    //     PipelineLevel level = PipelineLevel::LevelSimlod;
    //     if(cpt < max_cpt){
    //         prog->launch("kernel_test_display", {&level, &should_display}, launch_settings);
    //     }
    // }

    octreeUpdateCacheUpdate(editor, context);

    // // TODO: to remove, just to debug
    // {
    //     OptionalLaunchSettings launch_settings = {
    //         .gridsize = 1,
    //         .blocksize = 1,
    //         .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    //     };
    //     bool should_display = true;
    //     PipelineLevel level = PipelineLevel::LevelCacheUpdate;
    //     if(cpt < max_cpt){
    //         prog->launch("kernel_test_display", {&level, &should_display}, launch_settings);
    //     }
    // }

    // // Record completion of the update kernels on the UPDATE stream (no host block needed)
    // CURuntime::assertCudaSuccess(cuEventRecord(eventUpdateCompleted, stream));

    // {
    //     // Wait if the scene is being rendered
    //     std::lock_guard<std::mutex> lock(renderSubmissionMutex);
        
    //     // Tell the rendering to switch to the packed nodes
    //     uint64_t pad = uint64_t(&(hostStaging.isTemporarySwitching)) - uint64_t(&hostStaging);
    //     CUdeviceptr dst_device = deviceStaging + pad;
    //     *(bool*)isTemporarySwitching = true;
        
    //     CURuntime::assertCudaSuccess(cuStreamWaitEvent(0, eventUpdateCompleted, 0));
    //     CURuntime::assertCudaSuccess(cuMemcpyHtoD(
    //         dst_device, isTemporarySwitching, sizeof(bool)
    //     ));
    //     CURuntime::assertCudaSuccess(cuEventRecord(eventRenderingStreamInformed, 0));
    // }

    // // Don't start the swap before the rendering stream is being informed
    // CURuntime::assertCudaSuccess(cuStreamWaitEvent(stream, eventRenderingStreamInformed, 0));
    // // Run the real swap on this stream
    // OptionalLaunchSettings launch_settings = {
    //     .gridsize = 0,
    //     .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    //     .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0,
    // };
    // prog->launchCooperative("kernel_create_rendereable_octree", {}, launch_settings);
    // CURuntime::assertCudaSuccess(cuEventRecord(eventSwapCompleted, stream));


    // {
    //     // Tell the rendering to switch back to the rendering packed nodes
    //     std::lock_guard<std::mutex> lock(renderSubmissionMutex);
        
    //     uint64_t pad = uint64_t(&(hostStaging.isTemporarySwitching)) - uint64_t(&hostStaging);
    //     CUdeviceptr dst_device = deviceStaging + pad;
    //     *(bool*)isTemporarySwitching = false;
        
    //     CURuntime::assertCudaSuccess(cuStreamWaitEvent(0, eventSwapCompleted, 0));
    //     CURuntime::assertCudaSuccess(cuMemcpyHtoD(
    //         dst_device, isTemporarySwitching, sizeof(bool)
    //     ));
    //     CURuntime::assertCudaSuccess(cuEventRecord(eventRenderingStreamInformed, 0));
    // }

    // // Don't start a new update before the rendering stream is being informed
    // CURuntime::assertCudaSuccess(cuStreamWaitEvent(stream, eventRenderingStreamInformed, 0));


    // TODO: to remove, just to flag the batches and display stuff
    {
        OptionalLaunchSettings launch_settings = {
            .gridsize = 1,
            .blocksize = 1,
            .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
        };
        prog->launch("kernel_test", {}, launch_settings);
    }

    GpuVersionUI::update();
    // updateHostCache();
}


void GpuVersion::renderOctree(RenderTarget& target){
    CRenderTarget real_target = {};
    real_target.framebuffer = target.framebuffer;
    real_target.colorbuffer = target.colorbuffer;
    real_target.width = target.width;
    real_target.height = target.height;
    real_target.view = target.view;
    real_target.proj = target.proj;
    real_target.camera_pos = target.cameraPos;

    CRenderingSettings real_settings = {};
    real_settings.debug_lod_to_render = CuRastSettings::debugLodToRender;
    real_settings.use_voxels_debug_color = CuRastSettings::voxelsDebugColor;
    real_settings.min_pixel_span = CuRastSettings::minPixelSpan;
    real_settings.voxels_nb_points_per_axis = uint32_t(CuRastSettings::voxelsPointsPerAxis);

    // // Wait if the update kernel is being added to stream 0
    // std::lock_guard<std::mutex> lock_update(renderSubmissionMutex);
    // std::lock_guard<std::mutex> lock_visibility(syncVisibilityUpdateMtx);

    // Render nodes
    {
        uint32_t block_size = min(
            OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM,
            OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
        );
        OptionalLaunchSettings launch_settings = {
            .gridsize = OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM,
            .blocksize = block_size
        };

        if(CuRastSettings::bruteForceRendering){
            // TODO: to remove
            prog->launch("kernel_test_multi_resolution", {&real_target, &real_settings}, launch_settings);
        } else {
            prog->launch("kernel_visibilityPass", {&real_target, &real_settings}, launch_settings);
            prog->launch("kernel_drawVisibilityCache", {&real_target, &real_settings}, launch_settings);
            prog->launch("kernel_drawOctreeLarge", {&real_target, &real_settings}, launch_settings);
            prog->launch("kernel_drawOctreeSmall", {&real_target, &real_settings}, launch_settings);
            // Render bounding boxes
            if(CuRastSettings::showBoundingBoxes){
                prog->launch("kernel_render_bounding_boxes", {&real_target, &real_settings}, launch_settings);
            }
        }
    }
}









/// For another project
void GpuVersion::takeRandomScreenShots(){
    const uint32_t NB_SCREENSHOTS = 1024;

    static bool oldSettingsIsLeftDown = false;
    static double oldSettingsScroll = 0;
    static double oldSettingsPosX = 0;
    static double oldSettingsPosY = 0;
    static uint32_t oldSettingsNbPointsPerAxis = 0;
    static vec4 oldSettingsBackgroundColor = {};

    static uint32_t screenshotCounter = NB_SCREENSHOTS;
    static bool buttonWasPressed = false;

    static vec4 lastBg = {};
    static double lastScroll = 0;
    static double lastPosX = 0;
    static double lastPosY = 0;

    // Begin screenshots
    if(CuRastSettings::bruteForceRendering && !buttonWasPressed){
        // Start screenshots
        buttonWasPressed = true;
        screenshotCounter = 0;
        // Save old settings
        oldSettingsBackgroundColor = CuRastSettings::background;
        oldSettingsIsLeftDown = Runtime::controls->isLeftDown;
        oldSettingsScroll = Runtime::mouseEvents.wheel_y;
        oldSettingsPosX = Runtime::mouseEvents.pos_x;
        oldSettingsPosY = Runtime::mouseEvents.pos_y;
        oldSettingsNbPointsPerAxis = uint32_t(CuRastSettings::voxelsPointsPerAxis);
    }

    // Reset screenshots
    if(buttonWasPressed && screenshotCounter == NB_SCREENSHOTS){
        Runtime::controls->isLeftDown = oldSettingsIsLeftDown;
        Runtime::mouseEvents.wheel_y = oldSettingsScroll;
        Runtime::mouseEvents.pos_x = oldSettingsPosX;
        Runtime::mouseEvents.pos_y = oldSettingsPosY;
        CuRastSettings::background = oldSettingsBackgroundColor;
        CuRastSettings::voxelsPointsPerAxis = int32_t(oldSettingsNbPointsPerAxis);
        buttonWasPressed = false;
        CuRastSettings::bruteForceRendering = false;
    }

    // If screenshots
    if(screenshotCounter < NB_SCREENSHOTS){
        fs::create_directories("./screenshots");
        Runtime::controls->isLeftDown = true;

        // Half of the screenshots are for the ground truth
        if(screenshotCounter % 2 == 1){
            CuRastSettings::requestScreenshot = std::make_shared<string>(
                format(
                    "./screenshots/id_{}_target.png", 
                    uint32_t(screenshotCounter / 2)
                )
            );
            CuRastSettings::voxelsPointsPerAxis = 1;
            // Runtime::mouseEvents.wheel_y = lastScroll;
            // Runtime::mouseEvents.pos_x = lastPosX;
            // Runtime::mouseEvents.pos_y = lastPosY;
            CuRastSettings::background = lastBg;
        } else {
            Runtime::mouseEvents.wheel_y = pow(-1, rand()%2) * (rand() % 100);
            lastScroll = Runtime::mouseEvents.wheel_y;

            double max = 8192.0;
            Runtime::mouseEvents.pos_x = (double(rand()) / double(RAND_MAX)) * max;
            Runtime::mouseEvents.pos_y = (double(rand()) / double(RAND_MAX)) * max;
            lastPosX = Runtime::mouseEvents.pos_x;
            lastPosY = Runtime::mouseEvents.pos_y;

            CuRastSettings::background.r = (double(rand()) / double(RAND_MAX));
            CuRastSettings::background.g = (double(rand()) / double(RAND_MAX));
            CuRastSettings::background.b = (double(rand()) / double(RAND_MAX));
            lastBg = CuRastSettings::background;

            CuRastSettings::voxelsPointsPerAxis = rand() % 128 + 1; 

            CuRastSettings::requestScreenshot = std::make_shared<string>(
                format("./screenshots/id_{}_perturbed_{}.png", 
                    uint32_t(screenshotCounter / 2),
                    CuRastSettings::voxelsPointsPerAxis
                )
            );
        }

        screenshotCounter++;
    }
}





// void GpuVersion::updateHostCache(){
//     std::vector<std::shared_ptr<HostStorageNode>> nodes_to_erase = {};

//     std::lock_guard<std::mutex> lock(syncHostStorageNodesAccessMtx);

//     for(const CIdAABB& index : recentlyUsedNodesFromUpdates){
//         removedNodes.erase(index);
//         CIdAABB old_index = hostCache->add(index);
//         if(old_index != CINVALID_ID){
//             removedNodes.insert(old_index);
//         }
//     }

//     for(const CIdAABB& index : removedNodes){
//         if(!persistentStoredNodes.contains(index)){
//             println("ERROR: At this point, the node `{}' should be in the persistent storage", index);
//             throw(EXIT_FAILURE);
//         }
//         nodes_to_erase.push_back(persistentStoredNodes[index]);
//         persistentStoredNodes.erase(index);
//     }


//     for(std::shared_ptr<HostStorageNode> node : nodes_to_erase){
//         OctreeNodeSerializable::serializeV2(node);
//         delete(node);
//     }
// }




// #include "visibility.h"

// void GpuVersion::visibilityUpdate(CuRast* editor, CUcontext* context){
// 	cuCtxSetCurrent(*context);
    
//     // Get the frustum
//     const mat4&  view = VKRenderer::view.view;
//     const mat4&  proj = VKRenderer::view.proj;
//     Frustum frustum = Frustum(proj * view);
//     vec3 camera_pos = vec3(glm::inverse(view) * vec4(0.0f, 0.0f, 0.0f, 1.0f));

//     // Get all visible nodes and initialise their distances to the camera
//     std::vector<std::pair<CIdAABB, float>> visible_nodes = {};
//     {
//         std::lock_guard<std::mutex> lock(syncAABBStorageAccessMtx);
//         for(const auto& [index, aabb] : storedNodes){
//             if(frustum.doesIntersect(aabb)){
//                 float dist = glm::length(aabb.getCentroid() - camera_pos);
//                 visible_nodes.push_back({index, dist});
//             }
//         }
//     }

//     // Order the nodes with respect to the camera
//     std::sort(visible_nodes.begin(), visible_nodes.end(), 
//         [](const std::pair<CIdAABB, float>& lhs, const std::pair<CIdAABB, float>& rhs){
//             return lhs.second < rhs.second; // From closest to furthest 
//         }
//     );


//     // Gather the correct number of nodes to send to the device
//     std::unordered_set<CIdAABB> visibility_cache_set(hostStaging.visibilityCacheSize);
//     std::vector<CPoint> points_to_send(hostStaging.maxNbRenderedPoints, CPoint());
//     std::vector<CPoint> voxels_to_send(hostStaging.maxNbRenderedVoxels, CPoint());
//     std::vector<glm::vec3> voxels_sizes_to_send(hostStaging.maxNbRenderedVoxels, glm::vec3());
//     std::vector<CNodePosition> voxels_next_child_indices_to_send(hostStaging.maxNbRenderedVoxels);
//     std::vector<CIdAABB> voxels_nodes_to_send(hostStaging.maxNbRenderedVoxels, CINVALID_ID);
//     uint32_t cpt = 0;
//     uint32_t point_cpt = 0;
//     uint32_t voxel_cpt = 0;


//     std::unordered_set<CIdAABB> visited_nodes = {};
//     std::list<CIdAABB> to_visit_node = {};

//     {
//         std::lock_guard<std::mutex> lock(syncHostStorageNodesAccessMtx);

//         // Get the LRU_VISIBILTY_CACHE closest nodes
//         for(const std::pair<CIdAABB, float>& visible_node : visible_nodes){
//             to_visit_node.push_back(visible_node.first);
//         }

//         while(!to_visit_node.empty()){
//             if(cpt == OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE){break;}
//             const CIdAABB node_index = to_visit_node.front();
//             if(visited_nodes.contains(node_index)){
//                 to_visit_node.pop_front();
//                 continue;
//             }
//             CIdAABB parent = relationshipMap[node_index];
//             if(parent != CINVALID_ID // if has parent
//                 && !visited_nodes.contains(parent) // if parent not already in cache 
//                 && (relationshipMap[parent] != CINVALID_ID) // if parent exist in memory
//             ){
//                 to_visit_node.push_front(parent);
//                 continue;
//             }
//             to_visit_node.pop_front();
//             visited_nodes.insert(node_index);

//             // Check if the node is in the persistent storage
//             if(persistentStoredNodes.contains(node_index)){
//                 newlyVisible[cpt] = persistentStoredNodes[node_index];
//                 cpt++;
//                 continue;
//             }

//             // Load the stored node
//             std::shared_ptr<HostStorageNode> new_node = OctreeNodeSerializable::deserializeV2(node_index, "From vis update");
//             newlyVisible[cpt] = new_node;
//             newlyVisibleToDelete[cpt] = true;
//             cpt++;
//             continue;
//         }

//         // Get the maxNbRenderedPoints closest points and the maxNbRenderedVoxels closest voxels
//         for(uint32_t i=0; i<cpt; i++){
//             const std::shared_ptr<HostStorageNode> cur_node = newlyVisible[i];
//             bool should_be_in_cache_points = true;
//             bool should_be_in_cache_voxels = true;

//             if(point_cpt < hostStaging.maxNbRenderedPoints){
//                 uint32_t cur_nb_points = cur_node->points.size();
//                 for(uint32_t point_id = 0; point_id < cur_nb_points; point_id++){
//                     const CPoint& point = cur_node->points[point_id];
//                     points_to_send[point_cpt] = point;
//                     point_cpt++;
//                     if(point_cpt == hostStaging.maxNbRenderedPoints){
//                         should_be_in_cache_points = (point_id == cur_nb_points);
//                         break;
//                     }
//                 }
//             }
//             if(voxel_cpt < hostStaging.maxNbRenderedVoxels){
//                 uint32_t cur_nb_voxels = cur_node->voxels.size();
//                 for(uint32_t voxel_id = 0; voxel_id < cur_nb_voxels; voxel_id++){
//                     const CPoint& voxel = cur_node->voxels[voxel_id];
//                     CAABB voxel_aabb = cur_node->node.aabb;
//                     vec3 voxel_size = (voxel_aabb.maxs - voxel_aabb.mins) / float(OocSimLodSettings::GRID_SIZE_PER_DIMENSION);
//                     voxels_sizes_to_send[voxel_cpt] = voxel_size;
//                     voxels_next_child_indices_to_send[voxel_cpt] = voxel_aabb.getNextChildIndex(voxel.position);
//                     voxels_nodes_to_send[voxel_cpt] = cur_node->node.aabb_index;
//                     voxels_to_send[voxel_cpt] = voxel;

//                     voxel_cpt++;
//                     if(voxel_cpt == hostStaging.maxNbRenderedVoxels){
//                         should_be_in_cache_voxels = (voxel_id == cur_nb_voxels);
//                         break;
//                     }
//                 }
//             }

//             if(should_be_in_cache_points && should_be_in_cache_voxels){
//                 visibility_cache_set.insert(cur_node->node.aabb_index);
//             }

//             if(voxel_cpt == hostStaging.maxNbRenderedVoxels && point_cpt == hostStaging.maxNbRenderedPoints){break;}

//             if(newlyVisibleToDelete[i]){
//                 delete(newlyVisible[i]);
//                 newlyVisibleToDelete[i] = false;
//             }
//         }
//     }

//     std::vector<CIdAABB> visibility_cache(visibility_cache_set.begin(), visibility_cache_set.end());
//     uint32_t nb_rendered_nodes = visibility_cache.size();
//     GpuVersionUI::visNbNodes = nb_rendered_nodes;
//     GpuVersionUI::visNbPoints = point_cpt;
//     GpuVersionUI::visNbVoxels = voxel_cpt;

//     // Send the nodes to the device
//     {
//         std::lock_guard<std::mutex> lock(syncVisibilityUpdateMtx);

//         // Sending points
//         uint64_t pad = uint64_t(&(hostStaging.nbRenderedPoints)) - uint64_t(&hostStaging);
//         CUdeviceptr dst_device = deviceStaging + pad;
//         CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
//             dst_device,
//             &point_cpt,
//             sizeof(uint32_t), 0
//         ));
//         CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
//             (CUdeviceptr)hostStaging.renderedPoints,
//             points_to_send.data(),
//             point_cpt * sizeof(CPoint), 0
//         ));

//         // Sending voxels
//         pad = uint64_t(&(hostStaging.nbRenderedVoxels)) - uint64_t(&hostStaging);
//         dst_device = deviceStaging + pad;
//         CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
//             dst_device,
//             &voxel_cpt,
//             sizeof(uint32_t), 0
//         ));
//         CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
//             (CUdeviceptr)hostStaging.renderedVoxels,
//             voxels_to_send.data(),
//             voxel_cpt * sizeof(CPoint), 0
//         ));
//         CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
//             (CUdeviceptr)hostStaging.renderedVoxelsSizes,
//             voxels_sizes_to_send.data(),
//             voxel_cpt * sizeof(glm::vec3), 0
//         ));
//         CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
//             (CUdeviceptr)hostStaging.renderedVoxelsNextChildIndex,
//             voxels_next_child_indices_to_send.data(),
//             voxel_cpt * sizeof(CNodePosition), 0
//         ));
//         CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
//             (CUdeviceptr)hostStaging.renderedVoxelsNodes,
//             voxels_nodes_to_send.data(),
//             voxel_cpt * sizeof(CIdAABB), 0
//         ));

//         // Sending nodes ids
//         pad = uint64_t(&(hostStaging.visibilityCacheCurrentSize)) - uint64_t(&hostStaging);
//         dst_device = deviceStaging + pad;
//         CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
//             dst_device,
//             &nb_rendered_nodes,
//             sizeof(uint32_t), 0
//         ));
//         CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
//             (CUdeviceptr)hostStaging.visibilityCache,
//             visibility_cache.data(),
//             nb_rendered_nodes * sizeof(CIdAABB), 0
//         ));
//         // cudaStreamSynchronize(0);
//     }

// }






void GpuVersionUI::update() {
    uint64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        high_resolution_clock::now() - lastUpdateStart
    ).count();

#ifndef COPY_FROM_GPU
#define COPY_FROM_GPU(member, value)                                           \
    {                                                                          \
        const uint64_t pad =                                                   \
            reinterpret_cast<uintptr_t>(&(GpuVersion::hostStaging.member)) -   \
            reinterpret_cast<uintptr_t>(&GpuVersion::hostStaging);             \
        const CUdeviceptr src_device = GpuVersion::deviceStaging + pad;        \
        CURuntime::assertCudaSuccess(                                          \
            cuMemcpyDtoH(&(value), src_device, sizeof(value))                  \
        );                                                                     \
    }
#endif
    COPY_FROM_GPU(nbTotalUpdates, nbTotalUpdates);
    if(lastNbTotalUpdates == nbTotalUpdates){
        return;
    }
    lastNbTotalUpdates = nbTotalUpdates;

    COPY_FROM_GPU(curNbNodes, currentNbNodes);
    COPY_FROM_GPU(currentNbChunks, currentNbChunks);
    COPY_FROM_GPU(currentNbGrids, currentNbGrids);
    COPY_FROM_GPU(currentNbPoints, currentNbPoints);
    COPY_FROM_GPU(currentNbVoxels, currentNbVoxels);

    COPY_FROM_GPU(nbTotalPoints, nbTotalPoints);
    COPY_FROM_GPU(nbTotalVoxels, nbTotalVoxels);
    COPY_FROM_GPU(nbTotalNewNodes, nbTotalNewNodes);
    COPY_FROM_GPU(nbTotalNewGrids, nbTotalNewGrids);
    COPY_FROM_GPU(nbTotalNewChunks, nbTotalNewChunks);
    COPY_FROM_GPU(nbTotalDeletedNodes, nbTotalDeletedNodes);
    COPY_FROM_GPU(nbTotalDeletedGrids, nbTotalDeletedGrids);
    COPY_FROM_GPU(nbTotalDeletedChunks, nbTotalDeletedChunks);
    COPY_FROM_GPU(nbTotalLoadedNodes, nbTotalLoadedNodes);
    COPY_FROM_GPU(nbTotalSplitNodes, nbTotalSplitNodes);
    COPY_FROM_GPU(nbTotalStoredNodes, nbTotalStoredNodes);
    
    COPY_FROM_GPU(nbNewPointsThisUpdate, nbNewPointsThisUpdate);
    COPY_FROM_GPU(nbNewVoxelsThisUpdate, nbNewVoxelsThisUpdate);
    COPY_FROM_GPU(nbNewNodesThisUpdate, nbNewNodesThisUpdate);
    COPY_FROM_GPU(nbLoadedNodesThisUpdate, nbLoadedNodesThisUpdate);
    COPY_FROM_GPU(nbStoredNodesThisUpdate, nbStoredNodesThisUpdate);
    COPY_FROM_GPU(nbSplitNodesThisUpdate, nbSplitNodesThisUpdate);
    COPY_FROM_GPU(nbDeletedNodesThisUpdate, nbDeletedNodesThisUpdate);
    COPY_FROM_GPU(nbDeletedChunksThisUpdate, nbDeletedChunksThisUpdate);
    COPY_FROM_GPU(nbDeletedGridsThisUpdate, nbDeletedGridsThisUpdate);
    COPY_FROM_GPU(nbNewChunksThisUpdate, nbNewChunksThisUpdate);
    COPY_FROM_GPU(nbNewGridsThisUpdate, nbNewGridsThisUpdate);

    auto updateStats = [](uint32_t value, uint32_t& minValue, uint32_t& maxValue, uint32_t& avgValue) {
        if (nbTotalUpdates == 1) {
            minValue = value; maxValue = value; avgValue = value;
        } else {
            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
            avgValue = static_cast<uint32_t>((static_cast<uint64_t>(avgValue) * (nbTotalUpdates - 1) + value) / nbTotalUpdates);
        }
    };

    updateStats(nbNewPointsThisUpdate, minNbNewPointsPerUpdate, maxNbNewPointsPerUpdate, avgNbNewPointsPerUpdate);
    updateStats(nbNewVoxelsThisUpdate, minNbNewVoxelsPerUpdate, maxNbNewVoxelsPerUpdate,avgNbNewVoxelsPerUpdate);
    updateStats(nbNewNodesThisUpdate, minNbNewNodesPerUpdate, maxNbNewNodesPerUpdate, avgNbNewNodesPerUpdate);
    updateStats(nbLoadedNodesThisUpdate, minNbLoadedNodesPerUpdate, maxNbLoadedNodesPerUpdate, avgNbLoadedNodesPerUpdate);
    updateStats(nbStoredNodesThisUpdate, minNbStoredNodesPerUpdate, maxNbStoredNodesPerUpdate, avgNbStoredNodesPerUpdate);
    updateStats(nbSplitNodesThisUpdate, minNbSplitNodesPerUpdate, maxNbSplitNodesPerUpdate, avgNbSplitNodesPerUpdate);
    updateStats(nbDeletedNodesThisUpdate, minNbDeletedNodesPerUpdate, maxNbDeletedNodesPerUpdate, avgNbDeletedNodesPerUpdate);
    updateStats(nbDeletedChunksThisUpdate, minNbDeletedChunksPerUpdate, maxNbDeletedChunksPerUpdate, avgNbDeletedChunksPerUpdate);
    updateStats(nbDeletedGridsThisUpdate, minNbDeletedGridsPerUpdate, maxNbDeletedGridsPerUpdate, avgNbDeletedGridsPerUpdate);
    updateStats(nbNewChunksThisUpdate, minNbNewChunksPerUpdate, maxNbNewChunksPerUpdate, avgNbNewChunksPerUpdate);
    updateStats(nbNewGridsThisUpdate, minNbNewGridsPerUpdate, maxNbNewGridsPerUpdate, avgNbNewGridsPerUpdate);

    if (duration > 0) {
        const uint64_t updatesPerSecond = static_cast<uint64_t>(1000.0 / static_cast<double>(duration));
        updateStats(updatesPerSecond, minNbUpdatesPerSecond, maxNbUpdatesPerSecond, avgNbUpdatesPerSecond);
    }

}
#include "gpuVersion.h"

#include "loader.h"
#include "outOfCore.h"

void GpuVersion::initHostSide(CuRast* editor, CUcontext* context) {
    // Host side data
    exchangedPointsPointers = malloc(OocSimLodSettings::MAX_NB_NODES_TO_EXCHANGE * sizeof(CUdeviceptr));
    exchangedVoxelsPointers = malloc(OocSimLodSettings::MAX_NB_NODES_TO_EXCHANGE * sizeof(CUdeviceptr));
    batchesToAddPointsPointers = malloc(OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE * sizeof(CUdeviceptr));
    CURuntime::assertCudaSuccess(cuMemAllocHost(&nbExchangedNodes, sizeof(uint32_t)));
    *(uint32_t*)nbExchangedNodes = 0;
}

void GpuVersion::initBuffers(CuRast* editor, CUcontext* context) {

    // Unbounded data
    hostStaging.maxNbConcurrentNodes = OocSimLodSettings::MAX_NB_NODES;
    hostStaging.relationshipMap = alloc<CGlobalVariables::Relationship>(hostStaging.maxNbConcurrentNodes);
    hostStaging.packedNodes = alloc<COctreeNode*>(hostStaging.maxNbConcurrentNodes);
    hostStaging.nodesFlags = alloc<uint32_t>(hostStaging.maxNbConcurrentNodes);

    hostStaging.renderingPackedNodes = alloc<COctreeNode*>(hostStaging.maxNbConcurrentNodes);



    // Exchangeable data
    hostStaging.maxNbNodesExchanged = OocSimLodSettings::MAX_NB_NODES_TO_EXCHANGE;
    hostStaging.exchangedAABBIndices = alloc<CIdAABB>(hostStaging.maxNbNodesExchanged);
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
    
    for(uint32_t i=0; i<hostStaging.maxNbNodesExchanged; i++){
        CUdeviceptr new_ptr = 0;
        CURuntime::assertCudaSuccess(
            cuMemAlloc(
                &new_ptr, 
                OocSimLodSettings::NB_POINTS_PER_CHUNK * hostStaging.maxNbPointsChunksPerExchangedNode * sizeof(CPoint)
            )
        );
        ((CUdeviceptr*)(exchangedPointsPointers))[i] = new_ptr;
        pointers.push_back(new_ptr);
        CURuntime::assertCudaSuccess(
            cuMemAlloc(
                &new_ptr, 
                OocSimLodSettings::NB_POINTS_PER_CHUNK * hostStaging.maxNbVoxelsChunksPerExchangedNode * sizeof(CPoint)
            )
        );
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
    for(uint32_t i=0; i<hostStaging.maxNbBatches; i++){
        CUdeviceptr new_ptr = 0;
        CURuntime::assertCudaSuccess(
            cuMemAlloc(&new_ptr, OocSimLodSettings::MAX_POINTS_PER_BATCHES * sizeof(CPoint))
        );
        ((CUdeviceptr*)(batchesToAddPointsPointers))[i] = new_ptr;
        pointers.push_back(new_ptr);
    }
    CURuntime::assertCudaSuccess(cuMemcpyHtoD(
        (CUdeviceptr)hostStaging.batchesToAddPoints,
        batchesToAddPointsPointers,
        hostStaging.maxNbBatches * sizeof(CUdeviceptr)
    ));
    
    // TODO: put in settings
    hostStaging.residualPoints = alloc<CPoint>(hostStaging.maxNbResidualPoints);
    

    // Lru caches
    hostStaging.updatesCacheSize = OocSimLodSettings::LRU_UPDATES_CACHE_SIZE;
    hostStaging.updatesCache = nullptr;
    hostStaging.visibilityCacheSize = OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE;
    hostStaging.visibilityCache = nullptr;

    
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

    size_t heap_size = 1024 * 1024 * 1024; // 1Gb for now
    CURuntime::assertCudaSuccess(cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, heap_size));

    uint32_t grid_size = OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM 
        * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM
    ;
    OptionalLaunchSettings launch_settings = {
        .gridsize = grid_size,
        .blocksize = 1
    };
    prog->launch("kernel_init_global_allocators", {}, launch_settings);
    prog->launch("kernel_init_global_buffers", {}, launch_settings);
    LoaderGpuVersion::init();
}


void GpuVersion::destroy(CuRast *editor, CUcontext *context){
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

    cudaDeviceSynchronize();
    CURuntime::assertCudaSuccess(cuStreamDestroy(stream));
}













void GpuVersion::octreeUpdateInit(CuRast* editor, CUcontext* context){
    OptionalLaunchSettings launch_settings = {
        .gridsize = OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM,
        .blocksize = 1,
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
        .gridsize = OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM,
        .blocksize = 1,
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
        .gridsize = OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM,
        .blocksize = 1,
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

    uint32_t nb_nodes_to_load = *(uint32_t*)(nbExchangedNodes);
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
    std::vector<CPUFallbackCache::Entry> loaded(nb_nodes_to_load);
    for(uint32_t i=0; i<nb_nodes_to_load; i++){
        CIdAABB aabb_index = ids[i];
        loaded[i] = CPUFallbackCache::Entry::deserializeV2(aabb_index);
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
        ids[i] = loaded[i].serializable_node.aabb_index;
        children_ids[i] = uint32_t(loaded[i].serializable_node.children_ids);
        aabbs[i].maxs = loaded[i].serializable_node.aabb.maxs;
        aabbs[i].mins = loaded[i].serializable_node.aabb.mins;

        std::vector<CPoint> points = {};
        if(loaded[i].serializable_points.has_value()){
            nbs_points[i] = loaded[i].serializable_node.points_counter;

            // Rebuild the points
            points.reserve(nbs_points[i]);
            const ChunkSerializable& loaded_points = loaded[i].serializable_points.value(); 
            for(uint32_t j=0; j<loaded_points.points.size(); j++){
                for(uint32_t k=0; k<loaded_points.sizes[j]; k++){
                    CPoint cur_point = {};
                    cur_point.position = loaded_points.points[j][k].position;
                    cur_point.setColor(
                        loaded_points.points[j][k].color[0],
                        loaded_points.points[j][k].color[1],
                        loaded_points.points[j][k].color[2],
                        loaded_points.points[j][k].color[3]
                    );
                    points.push_back(cur_point);
                }
            }
        }

        std::vector<CPoint> voxels = {};
        if(loaded[i].serializable_voxels.has_value()){
            nbs_voxels[i] = loaded[i].serializable_node.voxels_counter;

            // Rebuild the voxels
            voxels.reserve(nbs_voxels[i]);
            const ChunkSerializable& loaded_voxels = loaded[i].serializable_voxels.value(); 
            for(uint32_t j=0; j<loaded_voxels.points.size(); j++){
                for(uint32_t k=0; k<loaded_voxels.sizes[j]; k++){
                    CPoint cur_voxel = {};
                    cur_voxel.position = loaded_voxels.points[j][k].position;
                    cur_voxel.setColor(
                        loaded_voxels.points[j][k].color[0],
                        loaded_voxels.points[j][k].color[1],
                        loaded_voxels.points[j][k].color[2],
                        loaded_voxels.points[j][k].color[3]
                    );
                    voxels.push_back(cur_voxel);
                }
            }
        }

        if(nbs_points[i]){
            CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync( 
                ((CUdeviceptr*)(exchangedPointsPointers))[i],
                points.data(), nbs_points[i]*sizeof(CPoint), 
                OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
            ));
        }
        if(nbs_voxels[i]){
            CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync( 
                ((CUdeviceptr*)(exchangedVoxelsPointers))[i],
                voxels.data(), nbs_voxels[i]*sizeof(CPoint),
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
        .blocksize = 1,
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

    OptionalLaunchSettings launch_settings = {
        .gridsize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM,
        .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM,
        .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    };
    prog->launch("kernel_simlod_voxel_sampling", {}, launch_settings);
}












void GpuVersion::octreeUpdateSimLODInsertion(CuRast* editor, CUcontext* context){
    uint32_t grid_size = OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM 
        * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM
    ;
    OptionalLaunchSettings launch_settings = {
        .gridsize = grid_size,
        .blocksize = 1,
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
        .gridsize = grid_size,
        .blocksize = 1,
        .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
    };
    prog->launch("kernel_prepare_store_part_1_filling_buffers", {}, launch_settings);
    prog->launch("kernel_prepare_store_part_2_resetting_children", {}, launch_settings);

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
    
    uint32_t nb_nodes_to_store = *(uint32_t*)(nbExchangedNodes);
    if(nb_nodes_to_store == 0){return;}
    // println("\n\nNb nodes to store: {}\n\n\n", nb_nodes_to_store);

    std::vector<CIdAABB> ids(nb_nodes_to_store, CINVALID_ID);
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
    for(uint32_t i=0; i<nb_nodes_to_store; i++){
        uint32_t cur_nb_points = nbs_points[i];
        std::vector<CPoint> points(cur_nb_points);
        if(cur_nb_points > 0){
            CURuntime::assertCudaSuccess(cuMemcpyDtoHAsync(
                points.data(),
                ((CUdeviceptr*)(exchangedPointsPointers))[i],
                cur_nb_points * sizeof(CPoint), OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
            ));
        }

        uint32_t cur_nb_voxels = nbs_voxels[i];
        std::vector<CPoint> voxels(cur_nb_voxels);
        if(cur_nb_voxels > 0){
            CURuntime::assertCudaSuccess(cuMemcpyDtoHAsync(
                voxels.data(),
                ((CUdeviceptr*)(exchangedVoxelsPointers))[i],
                cur_nb_voxels * sizeof(CPoint), OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
            ));
        }

        cudaStreamSynchronize(OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0);

        // Store the node on disk
        COctreeNode node = {};
        node.aabb_index = ids[i];
        node.aabb = aabbs[i];
        node.children_ids = children_ids[i];
        node.points_counter = cur_nb_points;
        node.voxels_counter = cur_nb_voxels;

        OctreeNodeSerializable::serializeV2(&node, points, voxels);
    }
}




void GpuVersion::updateOctree(CuRast* editor, CUcontext* context){
	cuCtxSetCurrent(*context);

    LoaderGpuVersion::run(&stream, editor, context);
    octreeUpdateInit(editor, context);
    octreeUpdateBottomUp(editor, context);
    octreeUpdateSimLOD(editor, context);
    octreeUpdateCacheUpdate(editor, context);
    // Record completion of the update kernels on the UPDATE stream (no host block needed)
    CUevent event_update_completed;
    CURuntime::assertCudaSuccess(cuEventCreate(&event_update_completed, CU_EVENT_DISABLE_TIMING));
    CURuntime::assertCudaSuccess(cuEventRecord(event_update_completed, stream));

    CUevent event_swap_completed;
    CURuntime::assertCudaSuccess(cuEventCreate(&event_swap_completed, CU_EVENT_DISABLE_TIMING));

    {
        // Wait if the scene is being rendered
        std::lock_guard<std::mutex> lock(renderSubmissionMutex);
        // Make stream 0 (GPU-side) wait for the update kernels, without blocking this thread
        CURuntime::assertCudaSuccess(cuStreamWaitEvent(0, event_update_completed, 0));

        OptionalLaunchSettings launch_settings = {
            .gridsize = 0,
            .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            .stream = 0 // Push into stream 0
        };
        prog->launchCooperative("kernel_create_rendereable_octree", {}, launch_settings);

        CURuntime::assertCudaSuccess(cuEventRecord(event_swap_completed, 0));
    }

    // Don't start a new update loop until the swap is actually done on the GPU
    CURuntime::assertCudaSuccess(cuEventSynchronize(event_swap_completed));
    cuEventDestroy(event_swap_completed);
    cuEventDestroy(event_update_completed);


    // TODO: to remove, just to flag the batches and display stuff
    {
        OptionalLaunchSettings launch_settings = {
            .gridsize = 1,
            .blocksize = 1,
            .stream = OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? stream : 0
        };
        prog->launch("kernel_test", {}, launch_settings);
    }

}


void GpuVersion::renderOctree(RenderTarget& target){
    CRenderTarget real_target = {};
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

    // Wait if the update kernel is being added to stream 0
    std::lock_guard<std::mutex> lock(renderSubmissionMutex);

    // Render bounding boxes
    {
        uint32_t grid_size = OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM 
            * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM
        ;

        if(CuRastSettings::showBoundingBoxes){
            OptionalLaunchSettings launch_settings = {
                .gridsize = grid_size,
                .blocksize = 1
            };
            prog->launch("kernel_render_bounding_boxes", {&real_target, &real_settings}, launch_settings);
        }
    }

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
            prog->launch("kernel_drawOctreeLarge", {&real_target, &real_settings}, launch_settings);
            prog->launch("kernel_drawOctreeSmall", {&real_target, &real_settings}, launch_settings);
        }
    }
}
#include "gpuVersion.h"

#include "loader.h"
#include "outOfCore.h"

void GpuVersion::initBuffers(CuRast* editor, CUcontext* context) {
    // Unbounded data
    hostStaging.maxNbAABBs = OocSimLodSettings::MAX_NB_NODES;
    hostStaging.relationshipMap = alloc<CGlobalVariables::Relationship>(hostStaging.maxNbAABBs);
    hostStaging.allAABBs = alloc<CAABB>(hostStaging.maxNbAABBs);
    hostStaging.nodes = alloc<COctreeNode*>(hostStaging.maxNbAABBs);
    hostStaging.packedNodes = alloc<COctreeNode*>(hostStaging.maxNbAABBs);
    hostStaging.nodesFlags = alloc<uint32_t>(hostStaging.maxNbAABBs);
    

    // Exchangeable data
    hostStaging.maxNbNodesReceived = OocSimLodSettings::MAX_NB_NODES_TO_STORE;

    hostStaging.receivedAABBIndices = alloc<CIdAABB>(hostStaging.maxNbNodesReceived);
    hostStaging.receivedChildrenIds = alloc<uint32_t>(hostStaging.maxNbNodesReceived);
    hostStaging.receivedPointsCounters = alloc<uint32_t>(hostStaging.maxNbNodesReceived);
    hostStaging.receivedVoxelsCounters = alloc<uint32_t>(hostStaging.maxNbNodesReceived);
    hostStaging.receivedPoints = alloc<CPoint*>(hostStaging.maxNbNodesReceived);
    hostStaging.receivedVoxels = alloc<CPoint*>(hostStaging.maxNbNodesReceived);
    hostStaging.maxNbPointsChunksPerReceivedNode = 
        (OocSimLodSettings::MAX_POINTS_PER_LEAF + OocSimLodSettings::NB_POINTS_PER_CHUNK - 1) 
        / OocSimLodSettings::NB_POINTS_PER_CHUNK
    ;
    hostStaging.receivedPointsPointers = malloc(hostStaging.maxNbNodesReceived * sizeof(CUdeviceptr));
    hostStaging.receivedVoxelsPointers = malloc(hostStaging.maxNbNodesReceived * sizeof(CUdeviceptr));
    for(uint32_t i=0; i<hostStaging.maxNbNodesReceived; i++){
        CUdeviceptr new_ptr = 0;
        CURuntime::assertCudaSuccess(
            cuMemAlloc(
                &new_ptr, 
                OocSimLodSettings::NB_POINTS_PER_CHUNK * hostStaging.maxNbPointsChunksPerReceivedNode * sizeof(CPoint)
            )
        );
        ((CUdeviceptr*)(hostStaging.receivedPointsPointers))[i] = new_ptr;
        pointers.push_back(new_ptr);
        CURuntime::assertCudaSuccess(
            cuMemAlloc(
                &new_ptr, 
                OocSimLodSettings::NB_POINTS_PER_CHUNK * hostStaging.maxNbVoxelsChunksPerReceivedNode * sizeof(CPoint)
            )
        );
        ((CUdeviceptr*)(hostStaging.receivedVoxelsPointers))[i] = new_ptr;
        pointers.push_back(new_ptr);
        CURuntime::assertCudaSuccess(cuMemcpyHtoD(
            (CUdeviceptr)hostStaging.receivedPoints,
            hostStaging.receivedPointsPointers,
            hostStaging.maxNbNodesReceived * sizeof(CUdeviceptr)
        ));
        CURuntime::assertCudaSuccess(cuMemcpyHtoD(
            (CUdeviceptr)hostStaging.receivedVoxels,
            hostStaging.receivedVoxelsPointers,
            hostStaging.maxNbNodesReceived * sizeof(CUdeviceptr)
        ));
    }

    hostStaging.maxNbNodesToLoad = OocSimLodSettings::MAX_NB_NODES_TO_LOAD;
    hostStaging.nodesToLoadBuffer = alloc<CIdAABB>(hostStaging.maxNbNodesToLoad);

    hostStaging.maxNbBatches = OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE;
    hostStaging.batchesAddedMask = alloc<uint32_t>(hostStaging.maxNbBatches);
    hostStaging.batchesToAddCounts = alloc<uint32_t>(hostStaging.maxNbBatches);
    hostStaging.batchesToAddBottomUpCounts = alloc<uint32_t>(hostStaging.maxNbBatches);
    hostStaging.batchesToAddPoints = alloc<CPoint*>(hostStaging.maxNbBatches);
    hostStaging.batchesToAddPointsPointers = malloc(hostStaging.maxNbBatches * sizeof(CUdeviceptr));
    for(uint32_t i=0; i<hostStaging.maxNbBatches; i++){
        CUdeviceptr new_ptr = 0;
        CURuntime::assertCudaSuccess(
            cuMemAlloc(&new_ptr, OocSimLodSettings::MAX_POINTS_PER_BATCHES * sizeof(CPoint))
        );
        ((CUdeviceptr*)(hostStaging.batchesToAddPointsPointers))[i] = new_ptr;
        pointers.push_back(new_ptr);
    }
    CURuntime::assertCudaSuccess(cuMemcpyHtoD(
        (CUdeviceptr)hostStaging.batchesToAddPoints,
        hostStaging.batchesToAddPointsPointers,
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
    initBuffers(editor, context);
    initAllocators(editor, context, &stream);
    cudaDeviceSynchronize();

    size_t heap_size = 1024 * 1024 * 1024; // 1Gb for now
    CURuntime::assertCudaSuccess(cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, heap_size));

    uint32_t grid_size = max(OocSimLodSettings::NB_ALLOCABLE_CHUNKS,
        max(
            OocSimLodSettings::NB_ALLOCABLE_GRIDS,
            OocSimLodSettings::NB_ALLOCABLE_NODES
        )
    );
    OptionalLaunchSettings launch_settings = {
        .gridsize = grid_size,
        .blocksize = 1
    };
    prog->launch("kernel_init_global_allocators", {}, launch_settings);

    grid_size = max(hostStaging.maxNbAABBs,
        max(hostStaging.maxNbNodesToLoad,
            max(hostStaging.maxNbBatches,
                max(hostStaging.updatesCacheSize, hostStaging.visibilityCacheSize)
            )
        )
    );
    launch_settings = {
        .gridsize = grid_size,
        .blocksize = 1
    };
    prog->launch("kernel_init_global_buffers", {}, launch_settings);
    LoaderGpuVersion::init();

    cudaDeviceSynchronize();
}

void GpuVersion::destroy(CuRast *editor, CUcontext *context){
    cudaDeviceSynchronize();
    for(CUdeviceptr& ptr : pointers){
        if(ptr){
            CURuntime::assertCudaSuccess(cuMemFree(ptr));
        }
    }
    free(hostStaging.receivedPointsPointers);
    free(hostStaging.receivedVoxelsPointers);
    free(hostStaging.batchesToAddPointsPointers);

    cudaDeviceSynchronize();
    CURuntime::assertCudaSuccess(cuStreamDestroy(stream));
}













void GpuVersion::octreeUpdateInit(CuRast* editor, CUcontext* context){
    OptionalLaunchSettings launch_settings = {
        .gridsize = 1024,
        .blocksize = 1
    };
    prog->launch("kernel_init_octree_part_1", {}, launch_settings);

    launch_settings = {
        .gridsize = 1,
        .blocksize = 1
    };
    prog->launch("kernel_init_octree_part_2", {}, launch_settings);
}

void GpuVersion::octreeUpdateBottomUp(CuRast* editor, CUcontext* context){
    // OptionalLaunchSettings launch_settings = {
    //     .gridsize = OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE,
    //     .blocksize = 1024
    // };
    OptionalLaunchSettings launch_settings = {
        .gridsize = OocSimLodSettings::MAX_POINTS_PER_BATCHES,
        .blocksize = 1
    };
    prog->launch("kernel_bottom_up_update_part_1", {}, launch_settings);

    launch_settings = {
        .gridsize = 1,
        .blocksize = 1
    };
    prog->launch("kernel_bottom_up_update_part_2", {}, launch_settings);
}

void GpuVersion::octreeUpdateSimLODLoad(CuRast* editor, CUcontext* context){
    OptionalLaunchSettings launch_settings = {
        .gridsize = OocSimLodSettings::MAX_POINTS_PER_BATCHES,
        .blocksize = 1
    };
    prog->launch("kernel_simlod_load_part_1", {}, launch_settings);
    launch_settings = {
        .gridsize = OocSimLodSettings::MAX_NB_NODES,
        .blocksize = 1
    };
    prog->launch("kernel_simlod_load_part_2", {}, launch_settings);

    // Pointers arithmetic shenanigans to get the correct device address
    CGlobalVariables tmp = {};
    uint64_t pad = uint64_t(&(tmp.nbNodesToLoad)) - uint64_t(&tmp);
    CUdeviceptr dst_device = GpuVersion::deviceStaging + pad;

    // Get the number of nodes to load
    uint32_t nb_nodes_to_load = 0;
    CURuntime::assertCudaSuccess(cuMemcpyDtoH(
		&nb_nodes_to_load, 
		dst_device,
		sizeof(uint32_t)
	));
    if(nb_nodes_to_load == 0){return;}
    println("\nNb nodes to load: {}\n\n", nb_nodes_to_load);

    // Get the ids of the nodes to load
    std::vector<CIdAABB> ids(nb_nodes_to_load, CINVALID_ID);
    CURuntime::assertCudaSuccess(cuMemcpyDtoH(
		ids.data(), 
		(CUdeviceptr)hostStaging.nodesToLoadBuffer,
		nb_nodes_to_load * sizeof(CIdAABB)
	));

    // Load the nodes from disk
    std::vector<CPUFallbackCache::Entry> loaded(nb_nodes_to_load);
    for(uint32_t i=0; i<nb_nodes_to_load; i++){
        CIdAABB aabb_index = ids[i];
        loaded[i] = CPUFallbackCache::Entry::deserializeV2(aabb_index);
    }

    // Send the nodes back to the device
    if(nb_nodes_to_load > hostStaging.maxNbNodesReceived){
        println("ERROR: can't send more than {} nodes back to the device... tried to send {}", 
            hostStaging.maxNbNodesReceived, nb_nodes_to_load
        );
        throw(EXIT_FAILURE);
    }

    std::vector<uint32_t> children_ids(nb_nodes_to_load, 0);
    std::vector<uint32_t> nbs_points(nb_nodes_to_load, 0);
    std::vector<uint32_t> nbs_voxels(nb_nodes_to_load, 0);

    for(uint32_t i = 0; i<nb_nodes_to_load; i++){
        ids[i] = loaded[i].serializable_node.aabb_index;
        children_ids[i] = uint32_t(loaded[i].serializable_node.children_ids);

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
            CURuntime::assertCudaSuccess(cuMemcpyHtoD( 
                ((CUdeviceptr*)(GpuVersion::hostStaging.receivedPointsPointers))[i],
                points.data(), nbs_points[i]*sizeof(uint32_t)
            ));
        }
        if(nbs_voxels[i]){
            CURuntime::assertCudaSuccess(cuMemcpyHtoD( 
                ((CUdeviceptr*)(GpuVersion::hostStaging.receivedVoxelsPointers))[i],
                voxels.data(), nbs_voxels[i]*sizeof(uint32_t)
            ));
        }

        // vec3 first_point = nbs_points[i] ? points[0].position : vec3();
        // vec3 first_voxel = nbs_voxels[i] ? voxels[0].position : vec3();
        // println("HOST side {} / {}:", i+1, nb_nodes_to_load);
        // println("    id: {}, children_ids: {}, points_counter: {}, voxels_counter: {}",
        //     ids[i], children_ids[i], nbs_points[i], nbs_voxels[i]
        // );
        // println("    first point = ({}, {}, {}), first voxel = ({}, {}, {})",
        //     first_point.x, first_point.y, first_point.z,
        //     first_voxel.x, first_voxel.y, first_voxel.z
        // );
    }
    // println("\n\n\n");

    pad = uint64_t(&(tmp.nbNodesReceived)) - uint64_t(&tmp);
    dst_device = GpuVersion::deviceStaging + pad;
    CURuntime::assertCudaSuccess(cuMemcpyHtoD(
		dst_device,	&nb_nodes_to_load, sizeof(uint32_t)
	));
    CURuntime::assertCudaSuccess(cuMemcpyHtoD(
		(CUdeviceptr)hostStaging.receivedAABBIndices,
		ids.data(),
		nb_nodes_to_load * sizeof(CIdAABB)
	));
    CURuntime::assertCudaSuccess(cuMemcpyHtoD(
		(CUdeviceptr)hostStaging.receivedChildrenIds,
		children_ids.data(),
		nb_nodes_to_load * sizeof(uint32_t)
	));
    CURuntime::assertCudaSuccess(cuMemcpyHtoD(
		(CUdeviceptr)hostStaging.receivedPointsCounters,
		nbs_points.data(),
		nb_nodes_to_load * sizeof(uint32_t)
	));
    CURuntime::assertCudaSuccess(cuMemcpyHtoD(
		(CUdeviceptr)hostStaging.receivedVoxelsCounters,
		nbs_voxels.data(),
		nb_nodes_to_load * sizeof(uint32_t)
	));

    cudaDeviceSynchronize();
    launch_settings = {
        .gridsize = nb_nodes_to_load,
        .blocksize = 1
    };
    prog->launch("kernel_simlod_load_part_3", {}, launch_settings);
    launch_settings = {
        .gridsize = hostStaging.maxNbAABBs,
        .blocksize = 1
    };
    prog->launch("kernel_simlod_load_part_4", {}, launch_settings);
}


void GpuVersion::octreeUpdateSimLODCountSplit(CuRast* editor, CUcontext* context){
    OptionalLaunchSettings launch_settings = {
        .gridsize = 0, // Not used with launchCoopertative
        .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    };
    prog->launchCooperative("kernel_simlod_count_split", {}, launch_settings);
    // OptionalLaunchSettings launch_settings = {
    //     .gridsize = 1,
    //     .blocksize = 1
    // };
    // prog->launch("kernel_simlod_count_split", {}, launch_settings);
}

void GpuVersion::octreeUpdateSimLODVoxelSampling(CuRast* editor, CUcontext* context){
    OptionalLaunchSettings launch_settings = {
        .gridsize = OocSimLodSettings::MAX_POINTS_PER_BATCHES,
        .blocksize = 1
    };
    prog->launch("kernel_simlod_voxel_sampling", {}, launch_settings);
}

void GpuVersion::octreeUpdateSimLODInsertion(CuRast* editor, CUcontext* context){
    OptionalLaunchSettings launch_settings = {
        .gridsize = OocSimLodSettings::MAX_NB_NODES,
        .blocksize = 1
    };
    prog->launch("kernel_simlod_insertion_part_1", {}, launch_settings);
    
    launch_settings = {
        .gridsize = OocSimLodSettings::MAX_POINTS_PER_BATCHES,
        .blocksize = 1
    };
    prog->launch("kernel_simlod_insertion_part_2", {}, launch_settings);
}



void GpuVersion::octreeUpdateSimLOD(CuRast* editor, CUcontext* context){
    octreeUpdateSimLODLoad(editor, context);
    // TODO: to remove, just to flag the batches and display stuff
    {
        OptionalLaunchSettings launch_settings = {
            .gridsize = 1,
            .blocksize = 1
        };
        PipelineLevel level = PipelineLevel::LevelSimlodLoad;
        bool display_octree = true;
        GpuVersion::prog->launch("kernel_test_display", {&level, &display_octree}, launch_settings);
    }

    octreeUpdateSimLODCountSplit(editor, context);
    octreeUpdateSimLODVoxelSampling(editor, context);
    octreeUpdateSimLODInsertion(editor, context);
}


void GpuVersion::octreeUpdateCacheUpdate(CuRast* editor, CUcontext* context){
    OptionalLaunchSettings launch_settings = {
        .gridsize = 1,
        .blocksize = 1
    };
    prog->launch("kernel_update_updates_cache", {}, launch_settings);

    launch_settings = {
        .gridsize = OocSimLodSettings::MAX_NB_NODES,
        .blocksize = 1
    };
    prog->launch("kernel_prepare_store_part_1", {}, launch_settings);
    prog->launch("kernel_prepare_store_part_2", {}, launch_settings);
    prog->launch("kernel_prepare_store_part_3", {}, launch_settings);
    launch_settings = {
        .gridsize = 1,
        .blocksize = 1
    };
    prog->launch("kernel_prepare_store_part_4", {}, launch_settings);

    // Readback the nodes and store them
    cudaDeviceSynchronize();

    // Pointers arithmetic shenanigans to get the correct device address
    CGlobalVariables tmp = {};
    uint64_t pad = uint64_t(&(tmp.nbNodesToStore)) - uint64_t(&tmp);
    CUdeviceptr dst_device = GpuVersion::deviceStaging + pad;

    // Get the number of nodes to store
    uint32_t nb_nodes_to_store = 0;
    CURuntime::assertCudaSuccess(cuMemcpyDtoH(
		&nb_nodes_to_store, 
		dst_device,
		sizeof(uint32_t)
	));
    if(nb_nodes_to_store == 0){return;}
    println("\nNb nodes to store: {}\n\n", nb_nodes_to_store);

    std::vector<CIdAABB> ids(nb_nodes_to_store, CINVALID_ID);
    std::vector<uint32_t> children_ids(nb_nodes_to_store, 0);
    std::vector<uint32_t> nbs_points(nb_nodes_to_store, 0);
    std::vector<uint32_t> nbs_voxels(nb_nodes_to_store, 0);

    CURuntime::assertCudaSuccess(cuMemcpyDtoH(
		ids.data(),
		(CUdeviceptr)hostStaging.receivedAABBIndices, 
		nb_nodes_to_store * sizeof(CIdAABB)
	));
    CURuntime::assertCudaSuccess(cuMemcpyDtoH(
		children_ids.data(),
		(CUdeviceptr)hostStaging.receivedChildrenIds, 
		nb_nodes_to_store * sizeof(uint32_t)
	));
    CURuntime::assertCudaSuccess(cuMemcpyDtoH(
		nbs_points.data(),
		(CUdeviceptr)hostStaging.receivedPointsCounters, 
		nb_nodes_to_store * sizeof(uint32_t)
	));
    CURuntime::assertCudaSuccess(cuMemcpyDtoH(
		nbs_voxels.data(),
		(CUdeviceptr)hostStaging.receivedVoxelsCounters, 
		nb_nodes_to_store * sizeof(uint32_t)
	));
    for(uint32_t i=0; i<nb_nodes_to_store; i++){
        uint32_t cur_nb_points = nbs_points[i];
        std::vector<CPoint> points(cur_nb_points);
        if(cur_nb_points > 0){
            CURuntime::assertCudaSuccess(cuMemcpyDtoH(
                points.data(),
                ((CUdeviceptr*)(GpuVersion::hostStaging.receivedPointsPointers))[i],
                cur_nb_points * sizeof(CPoint)
            ));
        }

        uint32_t cur_nb_voxels = nbs_voxels[i];
        std::vector<CPoint> voxels(cur_nb_voxels);
        if(cur_nb_voxels > 0){
            CURuntime::assertCudaSuccess(cuMemcpyDtoH(
                voxels.data(),
                ((CUdeviceptr*)(GpuVersion::hostStaging.receivedVoxelsPointers))[i],
                cur_nb_voxels * sizeof(CPoint)
            ));
        }

        // Store the node on disk
        COctreeNode node = {};
        node.aabb_index = ids[i];
        node.children_ids = children_ids[i];
        node.points_counter = cur_nb_points;
        node.voxels_counter = cur_nb_voxels;

        OctreeNodeSerializable::serializeV2(&node, points, voxels);
    }
}




void GpuVersion::updateOctree(CuRast* editor, CUcontext* context){
    static uint32_t cpt = 0;
    cpt++;
	println("\n\n\n\n\n\n\n\nstep = {}", cpt);

    LoaderGpuVersion::run(editor, context);
    octreeUpdateInit(editor, context);

    octreeUpdateBottomUp(editor, context);
    // TODO: to remove, just to flag the batches and display stuff
    {
        OptionalLaunchSettings launch_settings = {
            .gridsize = 1,
            .blocksize = 1
        };
        PipelineLevel level = PipelineLevel::LevelBottomUp;
        bool display_octree = true;
        GpuVersion::prog->launch("kernel_test_display", {&level, &display_octree}, launch_settings);
    }


    octreeUpdateSimLOD(editor, context);
    // TODO: to remove, just to flag the batches and display stuff
    {
        OptionalLaunchSettings launch_settings = {
            .gridsize = 1,
            .blocksize = 1
        };
        PipelineLevel level = PipelineLevel::LevelSimlod;
        bool display_octree = true;
        GpuVersion::prog->launch("kernel_test_display", {&level, &display_octree}, launch_settings);
    }
    
    octreeUpdateCacheUpdate(editor, context);
    // TODO: to remove, just to flag the batches and display stuff
    {
        OptionalLaunchSettings launch_settings = {
            .gridsize = 1,
            .blocksize = 1
        };
        PipelineLevel level = PipelineLevel::LevelCacheUpdate;
        bool display_octree = true;
        GpuVersion::prog->launch("kernel_test_display", {&level, &display_octree}, launch_settings);
    }

    // TODO: to remove, just to flag the batches and display stuff
    {
        OptionalLaunchSettings launch_settings = {
            .gridsize = 1,
            .blocksize = 1
        };
        GpuVersion::prog->launch("kernel_test", {}, launch_settings);
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
    real_settings.nb_blocks_per_node = OocSimLodSettings::NB_BLOCKS_PER_NODE;
    real_settings.debug_lod_to_render = CuRastSettings::debugLodToRender;
    real_settings.use_voxels_debug_color = CuRastSettings::voxelsDebugColor;
    real_settings.min_pixel_span = CuRastSettings::minPixelSpan;
    real_settings.voxels_nb_points_per_axis = uint32_t(CuRastSettings::voxelsPointsPerAxis);

    // Prepare the octree to be rendered
    {
        OptionalLaunchSettings launch_settings = {
            .gridsize = 1,
            .blocksize = 1
        };
        GpuVersion::prog->launch("kernel_prepare_rendereable_octree", {}, launch_settings);
    }

    // Get the current number of nodes
    uint32_t nb_nodes = 0;
    {
        // Pointers arithmetic shenanigans to get the correct device address
        CGlobalVariables tmp = {};
        uint64_t pad = uint64_t(&(tmp.curNbNodes)) - uint64_t(&tmp);
        CUdeviceptr dst_device = GpuVersion::deviceStaging + pad;
        CURuntime::assertCudaSuccess(cuMemcpyDtoH(
            &nb_nodes, 
            dst_device,
            sizeof(uint32_t)
        ));
    }
    if(nb_nodes > 0){

        // Render Bounding boxes
        if(CuRastSettings::showBoundingBoxes){
            OptionalLaunchSettings launch_settings = {
                .gridsize = nb_nodes,
                .blocksize = 1
            };
            prog->launch("kernel_render_bounding_boxes", {&real_target}, launch_settings);
        }

        // Render nodes
        {
            OptionalLaunchSettings launch_settings = {
                .gridsize = OocSimLodSettings::NB_BLOCKS_PER_NODE * nb_nodes,
                .blocksize = OocSimLodSettings::PER_NODE_KERNEL_BLOCK_SIZE
            };

            prog->launch("kernel_visibilityPass", {&real_target, &real_settings}, launch_settings);
            prog->launch("kernel_drawOctreeLarge", {&real_target, &real_settings}, launch_settings);
            prog->launch("kernel_drawOctreeSmall", {&real_target, &real_settings}, launch_settings);
        }
    }


}
#include "gpuVersion.h"

#include "loader.h"
#include "outOfCore.h"

void GpuVersion::initBuffers(CuRast* editor, CUcontext* context) {
    // Unbounded data
    hostStaging.maxNbAABBs = OocSimLodSettings::INITIAL_MAX_NB_NODES;
    hostStaging.relationshipMap = alloc<CGlobalVariables::Relationship>(hostStaging.maxNbAABBs);
    hostStaging.allAABBs = alloc<CAABB>(hostStaging.maxNbAABBs);
    hostStaging.nodes = alloc<COctreeNode*>(hostStaging.maxNbAABBs);
    hostStaging.nodesFlags = alloc<uint32_t>(hostStaging.maxNbAABBs);
    

    // Exchangeable data
    hostStaging.receivedAABBIndices = alloc<CIdAABB>(hostStaging.maxNbNodesReceived);
    hostStaging.receivedChildrenIds = alloc<uint8_t>(hostStaging.maxNbNodesReceived);
    hostStaging.receivedPointsCounters = alloc<uint32_t>(hostStaging.maxNbNodesReceived);
    hostStaging.receivedVoxelsCounters = alloc<uint32_t>(hostStaging.maxNbNodesReceived);
    hostStaging.receivedPoints = alloc<CPoint*>(hostStaging.maxNbNodesReceived);
    hostStaging.receivedVoxels = alloc<CPoint*>(hostStaging.maxNbNodesReceived);
    hostStaging.maxNbPointsChunkPerReceivedNode = 
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
                OocSimLodSettings::NB_POINTS_PER_CHUNK * hostStaging.maxNbPointsChunkPerReceivedNode * sizeof(CPoint)
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
    hostStaging.nodesToLoadBufferHost = malloc(hostStaging.maxNbNodesToLoad * sizeof(CIdAABB));

    hostStaging.maxNbNodesToStore = OocSimLodSettings::MAX_NB_NODES_TO_STORE;
    hostStaging.nodesToStoreBuffer = alloc<COctreeNode*>(hostStaging.maxNbNodesToStore);

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
    hostStaging.updatesCache = alloc<CIdAABB>(hostStaging.updatesCacheSize);
    hostStaging.visibilityCacheSize = OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE;
    hostStaging.visibilityCache = alloc<CIdAABB>(hostStaging.visibilityCacheSize);

    
    // Temporary buffers
    hostStaging.maxNbSpilledPoints = OocSimLodSettings::MAX_NB_SPILLING_POINTS;
    hostStaging.spilledPoints = alloc<CPoint>(hostStaging.maxNbSpilledPoints);
    hostStaging.spillingNodes = alloc<COctreeNode*>(hostStaging.maxNbSpilledPoints);

    hostStaging.maxNbBacklogVoxels = OocSimLodSettings::MAX_NB_BACKLOG_VOXELS;
    hostStaging.backlogVoxels = alloc<CPoint>(hostStaging.maxNbBacklogVoxels);
    hostStaging.backlogVoxelsNodes = alloc<COctreeNode*>(hostStaging.maxNbBacklogVoxels);
    

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
        "./src/kernels/ooc/render.cu",
        "./src/kernels/ooc/test.cu",
    });

    CURuntime::assertCudaSuccess(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    initBuffers(editor, context);
    println("buffer initialised");
    initAllocators(editor, context, &stream);
    println("allocator initialised");
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
            max(hostStaging.maxNbNodesToStore,
                max(hostStaging.maxNbBatches,
                    max(hostStaging.updatesCacheSize, hostStaging.visibilityCacheSize)
                )    
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
    free(hostStaging.nodesToLoadBufferHost);

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
    OptionalLaunchSettings launch_settings = {
        .gridsize = OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE,
        .blocksize = 1024
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
        .gridsize = 1,
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
    // println("There is {} nodes to load", nb_nodes_to_load);
    if(nb_nodes_to_load == 0){return;}

    // Get the ids of the nodes to load
    CURuntime::assertCudaSuccess(cuMemcpyDtoH(
		&hostStaging.nodesToLoadBufferHost, 
		(CUdeviceptr)hostStaging.nodesToLoadBuffer,
		nb_nodes_to_load * sizeof(CIdAABB)
	));

    // Load the nodes from disk
    std::vector<CPUFallbackCache::Entry> loaded(nb_nodes_to_load);
    for(uint32_t i=0; i<nb_nodes_to_load; i++){
        IdAABB aabb_index = (IdAABB)(((CIdAABB*)(hostStaging.nodesToLoadBufferHost))[i]);
        loaded[i] = CPUFallbackCache::Entry::deserialize(aabb_index);
    }

    // Send the nodes back to the device
    if(nb_nodes_to_load > hostStaging.maxNbNodesReceived){
        println("ERROR: can't send more than {} nodes back to the device... tried to send {}", 
            hostStaging.maxNbNodesReceived, nb_nodes_to_load
        );
        throw(EXIT_FAILURE);
    }
    for(uint32_t i = 0; i<nb_nodes_to_load; i++){
        // TODO: better sending strategy, for now, send 1 node at a time with async
        CIdAABB aabb_index = loaded[i].serializable_node.aabb_index;
        uint8_t children_ids = loaded[i].serializable_node.children_ids;
        std::vector<CPoint> points = {};
        uint32_t points_counter = 0;
        if(loaded[i].serializable_points.has_value()){
            points.reserve(loaded[i].serializable_node.counter);
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
        uint32_t voxels_counter = 0;
        if(loaded[i].serializable_voxels.has_value()){
            voxels.reserve(loaded[i].serializable_node.counter);
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

        CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
            (CUdeviceptr)(hostStaging.receivedAABBIndices + (CUdeviceptr)(i*sizeof(CIdAABB))), 
            &aabb_index, sizeof(CIdAABB), stream
        ));
        CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
            (CUdeviceptr)(hostStaging.receivedChildrenIds + (CUdeviceptr)(i*sizeof(uint8_t))), 
            &children_ids, sizeof(uint8_t), stream
        ));
        CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
            (CUdeviceptr)(hostStaging.receivedPointsCounters + (CUdeviceptr)(i*sizeof(uint32_t))), 
            &points_counter, sizeof(uint32_t), stream
        ));
        CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
            (CUdeviceptr)(hostStaging.receivedVoxelsCounters + (CUdeviceptr)(i*sizeof(uint32_t))), 
            &voxels_counter, sizeof(uint32_t), stream
        ));
        if(points_counter){
            CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync( 
                ((CUdeviceptr*)(GpuVersion::hostStaging.receivedPointsPointers))[i],
                points.data(), points_counter*sizeof(uint32_t), stream
            ));
        }
        if(voxels_counter){
            CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync( 
                ((CUdeviceptr*)(GpuVersion::hostStaging.receivedVoxelsPointers))[i],
                voxels.data(), voxels_counter*sizeof(uint32_t), stream
            ));
        }
    }
    
    pad = uint64_t(&(tmp.nbNodesReceived)) - uint64_t(&tmp);
    dst_device = GpuVersion::deviceStaging + pad;
    CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(
		dst_device,	&nb_nodes_to_load, sizeof(uint32_t), stream
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
        .gridsize = hostStaging.maxNbAABBs,
        .blocksize = 1
    };
    prog->launch("kernel_simlod_count_split", {}, launch_settings);
}



void GpuVersion::octreeUpdateSimLOD(CuRast* editor, CUcontext* context){
    octreeUpdateSimLODLoad(editor, context);
    octreeUpdateSimLODCountSplit(editor, context);
}



void GpuVersion::updateOctree(CuRast* editor, CUcontext* context){
    octreeUpdateInit(editor, context);
    octreeUpdateBottomUp(editor, context);
    octreeUpdateSimLOD(editor, context);

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
    // Prepare the octree to be rendered
    {
        OptionalLaunchSettings launch_settings = {
            .gridsize = 1,
            .blocksize = 1
        };
        GpuVersion::prog->launch("kernel_prepare_rendereable_octree", {}, launch_settings);
    }

    // Render Bounding boxes
    {
        OptionalLaunchSettings launch_settings = {
            .gridsize = hostStaging.maxNbAABBs,
            .blocksize = 1
        };

        CRenderTarget real_target = {};
        real_target.colorbuffer = target.colorbuffer;
        real_target.width = target.width;
        real_target.height = target.height;
        real_target.view = target.view;
        real_target.proj = target.proj;

        prog->launch("kernel_render_bounding_boxes", {&real_target}, launch_settings);
    }
}
#include "gpuVersion.h"

#include "loader.h"
#include "outOfCore.h"

#ifndef COPY_FROM_GPU
#define COPY_FROM_GPU(member, value, type)                                     \
    {                                                                          \
        const uint64_t pad =                                                   \
            reinterpret_cast<uintptr_t>(&(GpuVersion::hostStaging.member)) -   \
            reinterpret_cast<uintptr_t>(&GpuVersion::hostStaging);             \
        const CUdeviceptr src_device = GpuVersion::deviceStaging + pad;        \
        CURuntime::assertCudaSuccess(                                          \
            cuMemcpyDtoH(value, src_device, sizeof(type))                      \
        );                                                                     \
    }
#endif // COPY_FROM_GPU

#ifndef COPY_FROM_GPU_ASYNC
#define COPY_FROM_GPU_ASYNC(member, value, type)                               \
    {                                                                          \
        const uint64_t pad =                                                   \
            reinterpret_cast<uintptr_t>(&(GpuVersion::hostStaging.member)) -   \
            reinterpret_cast<uintptr_t>(&GpuVersion::hostStaging);             \
        const CUdeviceptr src_device = GpuVersion::deviceStaging + pad;        \
        CURuntime::assertCudaSuccess(                                          \
            cuMemcpyDtoHAsync(value, src_device, sizeof(type), 0)              \
        );                                                                     \
    }
#endif // COPY_FROM_GPU_ASYNC

#ifndef COPY_TO_GPU
#define COPY_TO_GPU(member, value, type)                                       \
    {                                                                          \
        const uint64_t pad =                                                   \
            reinterpret_cast<uintptr_t>(&(GpuVersion::hostStaging.member)) -   \
            reinterpret_cast<uintptr_t>(&GpuVersion::hostStaging);             \
        const CUdeviceptr dst_device = GpuVersion::deviceStaging + pad;        \
        CURuntime::assertCudaSuccess(                                          \
            cuMemcpyHtoD(dst_device, value, sizeof(type))                      \
        );                                                                     \
    }
#endif // COPY_TO_GPU

#ifndef COPY_TO_GPU_ASYNC
#define COPY_TO_GPU_ASYNC(member, value, type)                                 \
    {                                                                          \
        const uint64_t pad =                                                   \
            reinterpret_cast<uintptr_t>(&(GpuVersion::hostStaging.member)) -   \
            reinterpret_cast<uintptr_t>(&GpuVersion::hostStaging);             \
        const CUdeviceptr dst_device = GpuVersion::deviceStaging + pad;        \
        CURuntime::assertCudaSuccess(                                          \
            cuMemcpyHtoDAsync(dst_device, value, sizeof(type), 0)              \
        );                                                                     \
    }
#endif // COPY_TO_GPU_ASYNC



void GpuVersion::initConstraints(CuRast* editor, CUcontext* context) {
    hostStaging.maxNbConcurrentNodes = OocSimLodSettings::MAX_NB_NODES;
    hostStaging.maxAllocatedChunks = OocSimLodSettings::NB_ALLOCABLE_CHUNKS;
    hostStaging.maxCountSplitIterations = OocSimLodSettings::MAX_NB_COUNT_SPLIT_ITERATION;

    hostStaging.maxNbNodesExchanged = OocSimLodSettings::MAX_NB_NODES_TO_EXCHANGE;
    hostStaging.maxNbPointsChunksPerExchangedNode = 
        (OocSimLodSettings::MAX_POINTS_PER_LEAF + OocSimLodSettings::NB_POINTS_PER_CHUNK - 1) 
        / OocSimLodSettings::NB_POINTS_PER_CHUNK
    ;
    hostStaging.maxNbVoxelsChunksPerExchangedNode = OocSimLodSettings::MAX_NB_VOXELS_CHUNKS_TO_EXCHANGE;

    hostStaging.maxNbRenderedPoints = OocSimLodSettings::MAX_NB_RENDERED_POINTS;
    hostStaging.maxNbRenderedVoxels = OocSimLodSettings::MAX_NB_RENDERED_VOXELS;

    hostStaging.maxNbBatches = OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE;
    hostStaging.maxBatchSize = OocSimLodSettings::MAX_POINTS_PER_BATCHES;

    hostStaging.updatesCacheSize = OocSimLodSettings::LRU_UPDATES_CACHE_SIZE;
    hostStaging.visibilityCacheSize = OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE;

    hostStaging.maxNbSpilledPoints = OocSimLodSettings::MAX_NB_SPILLING_POINTS;
    hostStaging.maxNbBacklogVoxels = OocSimLodSettings::MAX_NB_BACKLOG_VOXELS;
    hostStaging.maxPointsPerLeaf = OocSimLodSettings::MAX_POINTS_PER_LEAF;
}


void GpuVersion::initHostSide(CuRast* editor, CUcontext* context) {
    // Host side data
    exchangedPointsPointers = std::vector<CUdeviceptr>(hostStaging.maxNbNodesExchanged);
    exchangedVoxelsPointers = std::vector<CUdeviceptr>(hostStaging.maxNbNodesExchanged);
    exchangedGridsPointers = std::vector<CUdeviceptr>(hostStaging.maxNbNodesExchanged);
    batchesToAddPointsPointers = malloc(OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE * sizeof(CUdeviceptr));
    CURuntime::assertCudaSuccess(cuMemAllocHost(&nbExchangedNodes, sizeof(uint32_t)));
    *(uint32_t*)nbExchangedNodes = 0;

    CURuntime::assertCudaSuccess(cuEventCreate(&eventLoadingComplete, CU_EVENT_DISABLE_TIMING));
    CURuntime::assertCudaSuccess(cuEventCreate(&eventStoringComplete, CU_EVENT_DISABLE_TIMING));
    CURuntime::assertCudaSuccess(cuEventCreate(&eventVisibilityUpdateComplete, CU_EVENT_DISABLE_TIMING));

    hostCache = new CLRUCache(OocSimLodSettings::LRU_CPU_CACHE_SIZE);
    parentsMap = std::vector<CIdAABB>(OocSimLodSettings::MAX_NB_NODES, CINVALID_ID);
    aabbsMap = std::vector<CAABB>(OocSimLodSettings::MAX_NB_NODES, CAABB());

    CURuntime::assertCudaSuccess(cuMemAllocHost(&isDoneLoading, sizeof(bool)));
    CURuntime::assertCudaSuccess(cuMemAllocHost(&isDoneStoring, sizeof(bool)));
    CURuntime::assertCudaSuccess(cuMemAllocHost(&isDoneIterating, sizeof(bool)));
    isInitialised = false;
    *(bool*)isDoneLoading = true;
    *(bool*)isDoneStoring = true;
    *(bool*)isDoneIterating = true;

    exchangedIds = allocHost<CIdAABB>(hostStaging.maxNbNodesExchanged);
    exchangedParentsIds = allocHost<CIdAABB>(hostStaging.maxNbNodesExchanged);
    exchangedAABBs = allocHost<CAABB>(hostStaging.maxNbNodesExchanged);
    exchangedChildrenIds = allocHost<uint32_t>(hostStaging.maxNbNodesExchanged);
    exchangedPointsCounters = allocHost<uint32_t>(hostStaging.maxNbNodesExchanged);
    exchangedVoxelsCounters = allocHost<uint32_t>(hostStaging.maxNbNodesExchanged);

    visibilityCache = allocHost<CIdAABB>(OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE);
    voxelsNodesToSend = allocHost<CIdAABB>(hostStaging.maxNbRenderedVoxels);
    nbRenderedPoints = allocHost<uint32_t>(1);
    nbRenderedVoxels = allocHost<uint32_t>(1);
    nbRenderedNodes = allocHost<uint32_t>(1);

    PointsAllocator::init();
}

void GpuVersion::initBuffers(CuRast* editor, CUcontext* context) {

    // Unbounded data
    hostStaging.relationshipMap = alloc<CGlobalVariables::Relationship>(hostStaging.maxNbConcurrentNodes);
    hostStaging.packedNodes = alloc<COctreeNode*>(hostStaging.maxNbConcurrentNodes);
    hostStaging.nodesFlags = alloc<uint32_t>(hostStaging.maxNbConcurrentNodes);


    // Exchangeable data
    hostStaging.exchangedAABBIndices = alloc<CIdAABB>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedAABBParentsIndices = alloc<CIdAABB>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedAABBs = alloc<CAABB>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedChildrenIds = alloc<uint32_t>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedPointsCounters = alloc<uint32_t>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedVoxelsCounters = alloc<uint32_t>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedPoints = alloc<CPoint*>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedVoxels = alloc<CPoint*>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedGrids = alloc<uint64_t*>(hostStaging.maxNbNodesExchanged);
    for(uint32_t i=0; i<hostStaging.maxNbNodesExchanged; i++){
        uint64_t real_size = 0;
    
        CUdeviceptr new_ptr = 0;
        real_size = OocSimLodSettings::NB_POINTS_PER_CHUNK * hostStaging.maxNbPointsChunksPerExchangedNode * sizeof(CPoint);
        totalAllocatedMemory += real_size;
        CURuntime::assertCudaSuccess(cuMemAlloc(&new_ptr, real_size));
        exchangedPointsPointers[i] = new_ptr;
        pointers.push_back(new_ptr);

        real_size = OocSimLodSettings::NB_POINTS_PER_CHUNK * hostStaging.maxNbVoxelsChunksPerExchangedNode * sizeof(CPoint);
        totalAllocatedMemory += real_size;
        CURuntime::assertCudaSuccess(cuMemAlloc(&new_ptr, real_size));
        exchangedVoxelsPointers[i] = new_ptr;
        pointers.push_back(new_ptr);

        real_size = OocSimLodSettings::NB_POINTS_PER_CHUNK * hostStaging.maxNbVoxelsChunksPerExchangedNode * sizeof(uint64_t);
        totalAllocatedMemory += real_size;
        CURuntime::assertCudaSuccess(cuMemAlloc(&new_ptr, real_size));
        exchangedGridsPointers[i] = new_ptr;
        pointers.push_back(new_ptr);

        CURuntime::assertCudaSuccess(cuMemcpyHtoD(
            (CUdeviceptr)hostStaging.exchangedPoints,
            exchangedPointsPointers.data(),
            hostStaging.maxNbNodesExchanged * sizeof(CUdeviceptr)
        ));
        CURuntime::assertCudaSuccess(cuMemcpyHtoD(
            (CUdeviceptr)hostStaging.exchangedVoxels,
            exchangedVoxelsPointers.data(),
            hostStaging.maxNbNodesExchanged * sizeof(CUdeviceptr)
        ));
        CURuntime::assertCudaSuccess(cuMemcpyHtoD(
            (CUdeviceptr)hostStaging.exchangedGrids,
            exchangedGridsPointers.data(),
            hostStaging.maxNbNodesExchanged * sizeof(CUdeviceptr)
        ));
    }

    hostStaging.gridsToInit = alloc<COctreeNode*>(hostStaging.maxNbConcurrentNodes);
    hostStaging.gridsToInitExchangedIndex = alloc<uint32_t>(hostStaging.maxNbConcurrentNodes);

    hostStaging.batchesAddedMask = alloc<uint32_t>(hostStaging.maxNbBatches);
    hostStaging.batchesToAddCounts = alloc<uint32_t>(hostStaging.maxNbBatches);
    hostStaging.batchesToAddPoints = alloc<CPoint*>(hostStaging.maxNbBatches);
    uint64_t real_size = hostStaging.maxBatchSize * sizeof(CPoint);
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
    

    // Lru caches
    hostStaging.updatesCache = alloc<CIdAABB>(hostStaging.updatesCacheSize * 2); // Times 2 for prefix scan
    hostStaging.visibilityCache = alloc<CIdAABB>(hostStaging.visibilityCacheSize);
    hostStaging.renderedPoints = alloc<CPoint>(hostStaging.maxNbRenderedPoints);
    hostStaging.renderedVoxels = alloc<CPoint>(hostStaging.maxNbRenderedVoxels);
    hostStaging.renderedVoxelsNodes = alloc<CIdAABB>(hostStaging.maxNbRenderedVoxels);

    
    // Temporary buffers
    hostStaging.spilledPoints = alloc<CPoint>(hostStaging.maxNbSpilledPoints);
    hostStaging.spillingNodes = alloc<COctreeNode*>(hostStaging.maxNbSpilledPoints);
    hostStaging.spilledChunksCounter = alloc<uint32_t>(hostStaging.maxNbSpilledPoints);
    hostStaging.spillingChunks = alloc<CChunk*>(hostStaging.maxNbSpilledPoints);
    hostStaging.allocatedChunks = alloc<CChunk*>(hostStaging.maxAllocatedChunks);

    hostStaging.backlogVoxels = alloc<CPoint>(hostStaging.maxNbBacklogVoxels);
    hostStaging.backlogVoxelsNodes = alloc<COctreeNode*>(hostStaging.maxNbBacklogVoxels);

    hostStaging.memoizedBatchPointsNodes = alloc<COctreeNode*>(hostStaging.maxNbBatches * hostStaging.maxBatchSize);
    hostStaging.memoizedSpilledPointsNodes = alloc<COctreeNode*>(hostStaging.maxNbSpilledPoints);

    hostStaging.temporaryBufferSize = hostStaging.maxNbConcurrentNodes;
    hostStaging.temporaryIdBuffer = alloc<CIdAABB>(hostStaging.temporaryBufferSize);
    hostStaging.temporaryIdBuffer2 = alloc<CIdAABB>(hostStaging.temporaryBufferSize);
    hostStaging.temporaryNodeBuffer = alloc<COctreeNode*>(hostStaging.temporaryBufferSize);

    

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
        "./src/kernels/ooc/render.cu"
    });

    CURuntime::assertCudaSuccess(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    initConstraints(editor, context);
    initHostSide(editor, context);
    initBuffers(editor, context);
    initAllocators(editor, context, &stream);
    cudaDeviceSynchronize(); // Needed because of the batch copies in the init functions

    // size_t heap_size = 1024 * 1024 * 1024; // 1Gb for now
    // CURuntime::assertCudaSuccess(cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, heap_size));

    // OptionalLaunchSettings launch_settings = {
    //     .gridsize  = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE,
    //     .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    // };

    uint32_t block_size = 256;
    uint32_t grid_size =
        (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1) 
        / block_size
    ;
    OptionalLaunchSettings launch_settings = {
        .gridsize  = grid_size,
        .blocksize = block_size
    };

    // prog->launch("kernel_init_global_allocators", {}, launch_settings);
    prog->launch("kernel_init_global_buffers", {}, launch_settings);
    LoaderGpuVersion::init();
}


void GpuVersion::destroy(CuRast *editor, CUcontext *context){
    if(hostCache){delete(hostCache);}

    cudaDeviceSynchronize();
    for(CUdeviceptr& ptr : pointers){
        if(ptr){
            CURuntime::assertCudaSuccess(cuMemFree(ptr));
        }
    }
    free(batchesToAddPointsPointers);
    CURuntime::assertCudaSuccess(cuMemFreeHost(nbExchangedNodes));
    CURuntime::assertCudaSuccess(cuMemFreeHost(isDoneLoading));
    CURuntime::assertCudaSuccess(cuMemFreeHost(isDoneStoring));
    CURuntime::assertCudaSuccess(cuMemFreeHost(isDoneIterating));

    CURuntime::assertCudaSuccess(cuMemFreeHost(exchangedIds));
    CURuntime::assertCudaSuccess(cuMemFreeHost(exchangedParentsIds));
    CURuntime::assertCudaSuccess(cuMemFreeHost(exchangedAABBs));
    CURuntime::assertCudaSuccess(cuMemFreeHost(exchangedChildrenIds));
    CURuntime::assertCudaSuccess(cuMemFreeHost(exchangedPointsCounters));
    CURuntime::assertCudaSuccess(cuMemFreeHost(exchangedVoxelsCounters));

    CURuntime::assertCudaSuccess(cuMemFreeHost(visibilityCache));
    CURuntime::assertCudaSuccess(cuMemFreeHost(voxelsNodesToSend));
    CURuntime::assertCudaSuccess(cuMemFreeHost(nbRenderedPoints));
    CURuntime::assertCudaSuccess(cuMemFreeHost(nbRenderedVoxels));
    CURuntime::assertCudaSuccess(cuMemFreeHost(nbRenderedNodes));


    CURuntime::assertCudaSuccess(cuEventDestroy(eventLoadingComplete));
    CURuntime::assertCudaSuccess(cuEventDestroy(eventStoringComplete));
    CURuntime::assertCudaSuccess(cuEventDestroy(eventVisibilityUpdateComplete));

    PointsAllocator::destroy();

    cudaDeviceSynchronize();
    CURuntime::assertCudaSuccess(cuStreamDestroy(stream));
}













void GpuVersion::octreeUpdateInit(CuRast* editor, CUcontext* context){
    // OptionalLaunchSettings launch_settings = {
    //     .gridsize  = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE,
    //     .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    // };

    uint32_t block_size = 256;
    uint32_t grid_size =
        (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1) 
        / block_size
    ;
    OptionalLaunchSettings launch_settings = {
        .gridsize  = grid_size,
        .blocksize = block_size
    };

    prog->launch("kernel_init_octree_part_1_aabb_measuring", {}, launch_settings);

    launch_settings = {
        .gridsize = 1,
        .blocksize = 1
    };
    prog->launch("kernel_init_octree_part_2_refining", {}, launch_settings);
}














void GpuVersion::octreeUpdateBottomUp(CuRast* editor, CUcontext* context){
    // OptionalLaunchSettings launch_settings = {
    //     .gridsize  = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE,
    //     .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    // };

    uint32_t block_size = 256;
    uint32_t grid_size =
        (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1) 
        / block_size
    ;
    OptionalLaunchSettings launch_settings = {
        .gridsize  = grid_size,
        .blocksize = block_size
    };

    prog->launch("kernel_bottom_up_update_part_1_counting", {}, launch_settings);

    launch_settings = {
        .gridsize = 1,
        .blocksize = 1
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
        .blocksize = block_size
    };
    prog->launch("kernel_fill_new_grids", {}, launch_settings);
}







void GpuVersion::octreeUpdateSimLODLoad(CuRast* editor, CUcontext* context){
    if(*(bool*)isDoneLoading){
        COPY_TO_GPU(nbNodesExchangedBeforeLoadComplete, &RESET, uint32_t);
    }
    *(bool*)isDoneLoading = true;
    COPY_TO_GPU(isDoneLoading, isDoneLoading, bool);
    COPY_TO_GPU(nbNodesExchanged, &RESET, uint32_t);

    // launch_settings = {
    //     .gridsize  = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE,
    //     .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    // };
    uint32_t block_size = 256;
    uint32_t grid_size =
        (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1) 
        / block_size
    ;
    OptionalLaunchSettings launch_settings = {
        .gridsize  = grid_size,
        .blocksize = block_size
    };
    prog->launch("kernel_simlod_load_part_1_flagging", {}, launch_settings);
    COPY_FROM_GPU(isDoneLoading, isDoneLoading, bool);

    // Get the number of nodes to load
    COPY_FROM_GPU(nbNodesExchanged, nbExchangedNodes, uint32_t);
    uint32_t nb_nodes_to_load = min(*(uint32_t*)(nbExchangedNodes), hostStaging.maxNbNodesExchanged);
    if(nb_nodes_to_load == 0){return;}
    // println("\n\nNb nodes to load: {}\n\n\n", nb_nodes_to_load);

    // Get the ids of the nodes to load
    CURuntime::assertCudaSuccess(cuMemcpyDtoH(
		exchangedIds, 
		(CUdeviceptr)hostStaging.exchangedAABBIndices,
		nb_nodes_to_load * sizeof(CIdAABB)
	));

    // Load the nodes from disk
    std::vector<std::shared_ptr<HostStorageNode>> loaded(nb_nodes_to_load, nullptr);
    
    // Load from CPU cache
    {
        CIdAABB* ids = static_cast<CIdAABB*>(exchangedIds);
        struct TmpToLoadStruct {
            CIdAABB id;
            std::shared_ptr<HostStorageNode> node;
            uint32_t loaded_id;
        };
        std::vector<TmpToLoadStruct> to_load = {};
        for(uint32_t i = 0; i < nb_nodes_to_load; i++){
            if(!persistentStoredNodes.contains(ids[i])){
                to_load.push_back({
                    ids[i], std::make_shared<HostStorageNode>(), i
                });
            } else {
                loaded[i] = persistentStoredNodes[ids[i]];
            }
        }

        auto load_node = [&](TmpToLoadStruct& to_load){
            OctreeNodeSerializable::deserializeV2(to_load.node.get(), to_load.id, "From simlod load");
            loaded[to_load.loaded_id] = to_load.node;
        };
        if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
            std::for_each(std::execution::par, to_load.begin(), to_load.end(), load_node);
        } else {
            std::for_each(to_load.begin(), to_load.end(), load_node);
        }

        for(uint32_t i=0; i<nb_nodes_to_load; i++){
            CIdAABB id = ids[i];
            hostCache->add(id);
            persistentStoredNodes[id] = loaded[i];
        }
    }

    // Send the nodes back to the device
    std::vector<CUdeviceptr> srcs_host = {
        (CUdeviceptr)exchangedChildrenIds,
        (CUdeviceptr)exchangedPointsCounters,
        (CUdeviceptr)exchangedVoxelsCounters
    };
    std::vector<CUdeviceptr> dsts_device = {
        (CUdeviceptr)hostStaging.exchangedChildrenIds,
        (CUdeviceptr)hostStaging.exchangedPointsCounters,
        (CUdeviceptr)hostStaging.exchangedVoxelsCounters
    };
    std::vector<uint64_t> sizes = {
        nb_nodes_to_load * sizeof(uint32_t),
        nb_nodes_to_load * sizeof(uint32_t),
        nb_nodes_to_load * sizeof(uint32_t)
    };

    const uint32_t MAX_NB_VOXELS = hostStaging.maxNbVoxelsChunksPerExchangedNode * OocSimLodSettings::NB_POINTS_PER_CHUNK;
    for(uint32_t i = 0; i<nb_nodes_to_load; i++){
        currentlyInUpdatesCache.insert(loaded[i]->node.aabb_index);
        static_cast<uint32_t*>(exchangedChildrenIds)[i] = loaded[i]->node.children_ids;
        static_cast<uint32_t*>(exchangedPointsCounters)[i] = loaded[i]->node.points_counter;
        static_cast<uint32_t*>(exchangedVoxelsCounters)[i] = loaded[i]->node.voxels_counter;

        // Send grids info
        if(loaded[i]->node.voxels_counter > 0){
            srcs_host.push_back((CUdeviceptr)loaded[i]->occupancy_indices.data());
            dsts_device.push_back(exchangedGridsPointers[i]);
            // TODO: warn
            sizes.push_back(min(loaded[i]->node.voxels_counter, MAX_NB_VOXELS) * sizeof(uint64_t));
        }
    }

    uint64_t nb_copies = sizes.size();
    CURuntime::assertCudaSuccess(cuMemcpyBatchAsync(
        dsts_device.data(), srcs_host.data(), sizes.data(), nb_copies, 
        batchLoadingAttributes.data(), 
        batchLoadingAttributesIndices.data(), 
        batchLoadingAttributes.size(), 
        stream
    ));
    CURuntime::assertCudaSuccess(cuEventRecord(eventLoadingComplete, stream));
    cudaStreamWaitEvent(0, eventLoadingComplete);

    launch_settings = {
        .gridsize = OocSimLodSettings::MAX_NB_NODES_TO_EXCHANGE,
        // .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        .blocksize = 256
    };
    prog->launch("kernel_simlod_load_part_2_rebuilding_nodes", {}, launch_settings);
    octreeUpdateFillNewGrids(editor, context);
    prog->launch("kernel_simlod_load_part_3_rebuilding_children", {}, launch_settings);
}














void GpuVersion::octreeUpdateSimLODCountSplit(CuRast* editor, CUcontext* context){
    bool is_first_iteration = (*(bool*)isDoneIterating == true);
    COPY_TO_GPU(isFirstCountSplitIteration, isDoneIterating, bool);
    *(bool*)isDoneIterating = false;
    COPY_TO_GPU(isDoneIterating, isDoneIterating, bool);

    // OptionalLaunchSettings launch_settings = {
    //     .gridsize = 0, // Not used with launchCoopertative
    //     // .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    //     .blocksize = 256
    // };
    // prog->launchCooperative("kernel_simlod_count_split", {}, launch_settings);

    uint32_t block_size = 256;
    uint32_t grid_size =
        (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1) 
        / block_size
    ;
    OptionalLaunchSettings launch_settings = {
        .gridsize  = grid_size,
        .blocksize = block_size
    };
    OptionalLaunchSettings single_launch = {
        .gridsize  = 1,
        .blocksize = 1
    };

    for(uint32_t i=0; i<OocSimLodSettings::MAX_NB_COUNT_SPLIT_ITERATION; i++){
        prog->launch("kernel_simlod_count_split_part_1_count", {&i, &is_first_iteration}, launch_settings);
        prog->launch("kernel_simlod_count_split_part_1_5_prefix_sum", {}, single_launch);
        prog->launch("kernel_simlod_count_split_part_2_split", {}, launch_settings);
        prog->launch("kernel_simlod_count_split_part_3_chunks_delete", {}, launch_settings);
        prog->launch("kernel_simlod_count_split_part_4_reset", {}, single_launch);
        COPY_FROM_GPU(isDoneIterating, isDoneIterating, bool);
        if(*(bool*)isDoneIterating){break;}
    }

}














void GpuVersion::octreeUpdateSimLODVoxelSampling(CuRast* editor, CUcontext* context){
    octreeUpdateFillNewGrids(editor, context);
    COPY_TO_GPU(nbGridsToInit, &RESET, uint32_t);

    if(!(bool*)isDoneIterating){return;}

    uint32_t block_size = 32;
    uint32_t num_sms = OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM;
    uint32_t max_threads_per_sm = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM;
    uint32_t grid_size = (num_sms * max_threads_per_sm + block_size - 1) / block_size;

    OptionalLaunchSettings launch_settings = {
        .gridsize = grid_size,
        .blocksize = block_size
    };
    prog->launch("kernel_simlod_voxel_sampling", {}, launch_settings);
}












void GpuVersion::octreeUpdateSimLODInsertion(CuRast* editor, CUcontext* context){
    // OptionalLaunchSettings launch_settings = {
    //     .gridsize  = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE,
    //     .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    // };

    uint32_t block_size = 256;
    uint32_t grid_size =
        (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1) 
        / block_size
    ;
    OptionalLaunchSettings launch_settings = {
        .gridsize  = grid_size,
        .blocksize = block_size
    };

    prog->launch("kernel_simlod_insertion_part_1_chunks_allocations", {}, launch_settings);
    prog->launch("kernel_simlod_insertion_part_2_filling", {}, launch_settings);
}















void GpuVersion::octreeUpdateSimLOD(CuRast* editor, CUcontext* context){
    if(*(bool*)isDoneStoring && *(bool*)isDoneIterating){
        octreeUpdateSimLODLoad(editor, context);
    }

    if(*(bool*)isDoneLoading && *(bool*)isDoneStoring){
        octreeUpdateSimLODCountSplit(editor, context);
    }

    if(*(bool*)isDoneLoading && *(bool*)isDoneStoring){
        octreeUpdateSimLODVoxelSampling(editor, context);
    }

    if(*(bool*)isDoneLoading && *(bool*)isDoneStoring && *(bool*)isDoneIterating){
        octreeUpdateSimLODInsertion(editor, context);
    }
}















void GpuVersion::octreeUpdateCacheUpdate(CuRast* editor, CUcontext* context){
    COPY_TO_GPU(nbNodesExchanged, &RESET, uint32_t);
    uint32_t block_size = 256;
    uint32_t grid_size =
        (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1) 
        / block_size
    ;
    OptionalLaunchSettings launch_settings = {
        .gridsize  = grid_size,
        .blocksize = block_size
    };
    OptionalLaunchSettings single_launch = {
        .gridsize  = 1,
        .blocksize = 1
    };
    if(*(bool*)isDoneStoring){
        prog->launch("kernel_update_updates_cache_part_1_counting", {}, launch_settings);
        prog->launch("kernel_update_updates_cache_part_2_sorting", {}, launch_settings);
        prog->launch("kernel_update_updates_cache_part_3_prefix_sum", {}, single_launch);
    }
    *(bool*)isDoneStoring = true;
    COPY_TO_GPU(isDoneStoring, isDoneStoring, bool);

    // launch_settings = {
    //     .gridsize  = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE,
    //     .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    // };
    prog->launch("kernel_prepare_store_part_1_filling_buffers", {}, launch_settings);
    prog->launch("kernel_prepare_store_part_2_resetting_children", {}, launch_settings);

    launch_settings = {
        .gridsize = 0,
        // .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        .blocksize = 256,
    };
    prog->launchCooperative("kernel_prepare_store_part_3_updating_levels", {}, launch_settings);
    COPY_FROM_GPU(isDoneStoring, isDoneStoring, bool);

    COPY_FROM_GPU(nbNodesExchanged, nbExchangedNodes, uint32_t);
    uint32_t nb_nodes_to_store = min(*(uint32_t*)(nbExchangedNodes), hostStaging.maxNbNodesExchanged);
    if(nb_nodes_to_store == 0){return;}
    // println("\n\nNb nodes to store: {}\n\n\n", nb_nodes_to_store);

    storeNodes(nb_nodes_to_store);
}


void GpuVersion::storeNodes(uint32_t nb_nodes_to_store){
    std::vector<CUdeviceptr> srcs_device = {
        (CUdeviceptr)hostStaging.exchangedAABBIndices,
        (CUdeviceptr)hostStaging.exchangedPointsCounters,
        (CUdeviceptr)hostStaging.exchangedVoxelsCounters,
        (CUdeviceptr)hostStaging.exchangedAABBParentsIndices,
        (CUdeviceptr)hostStaging.exchangedChildrenIds,
        (CUdeviceptr)hostStaging.exchangedAABBs
    };
    std::vector<CUdeviceptr> dsts_host = {
        (CUdeviceptr)exchangedIds,
        (CUdeviceptr)exchangedPointsCounters,
        (CUdeviceptr)exchangedVoxelsCounters,
        (CUdeviceptr)exchangedParentsIds,
        (CUdeviceptr)exchangedChildrenIds,
        (CUdeviceptr)exchangedAABBs
    };
    std::vector<uint64_t> sizes = {
        nb_nodes_to_store * sizeof(CIdAABB),
        nb_nodes_to_store * sizeof(uint32_t),
        nb_nodes_to_store * sizeof(uint32_t),
        nb_nodes_to_store * sizeof(CIdAABB),
        nb_nodes_to_store * sizeof(uint32_t),
        nb_nodes_to_store * sizeof(CAABB)
    };

    // Wait for the node properties
    uint64_t nb_copies = sizes.size();
    CURuntime::assertCudaSuccess(cuMemcpyBatchAsync(
        dsts_host.data(), srcs_device.data(), sizes.data(), nb_copies, 
        batchStoringAttributes.data(), 
        batchStoringAttributesIndices.data(), 
        batchStoringAttributes.size(), 
        stream
    ));
    CURuntime::assertCudaSuccess(cuEventRecord(eventStoringComplete, stream));
    CURuntime::assertCudaSuccess(cuEventSynchronize(eventStoringComplete));

    CIdAABB* ids = static_cast<CIdAABB*>(exchangedIds);
    uint32_t* nbs_points = static_cast<uint32_t*>(exchangedPointsCounters);
    uint32_t* nbs_voxels = static_cast<uint32_t*>(exchangedVoxelsCounters);
    uint32_t* children_ids = static_cast<uint32_t*>(exchangedChildrenIds);
    CIdAABB* parents_ids = static_cast<CIdAABB*>(exchangedParentsIds);
    CAABB* aabbs = static_cast<CAABB*>(exchangedAABBs);

    srcs_device.clear();
    dsts_host.clear();
    sizes.clear();

    for(uint32_t i=0; i < nb_nodes_to_store; i++){
        CIdAABB id = ids[i];
        // Load or create the nodes
        if(!persistentStoredNodes.contains(id)){
            std::shared_ptr<HostStorageNode> new_node = std::make_shared<HostStorageNode>();
            new_node->node.aabb_index = id;
            if(storedNodes.contains(id)){
                OctreeNodeSerializable::deserializeV2(new_node.get(), id, "From update cache");
            }
            persistentStoredNodes[id] = new_node;
        }

        // Update node properties
        HostStorageNode* cur_node = persistentStoredNodes[id].get();
        cur_node->node.children_ids = children_ids[i];

        // Update the CPU version of the relationship map
        parentsMap[id] = parents_ids[i];
        aabbsMap[id] = aabbs[i];
        storedNodes.insert(id);
        currentlyInUpdatesCache.erase(id);
        hostCache->add(id);

        // Prepare to load points
        if(nbs_points[i] > 0){
            uint32_t old_counter = cur_node->node.points_counter;
            cur_node->node.points_counter += nbs_points[i];
            // cur_node->points.resize(cur_node->node.points_counter);

            srcs_device.push_back(exchangedPointsPointers[i]);
            // dsts_host.push_back((CUdeviceptr)(cur_node->points.data() + old_counter));
            dsts_host.push_back((CUdeviceptr)(cur_node->points + old_counter));
            sizes.push_back(nbs_points[i] * sizeof(CPoint));
        }

        // Prepare to load voxels and grids
        if(nbs_voxels[i] > 0){
            uint32_t old_counter = cur_node->node.voxels_counter;
            cur_node->node.voxels_counter += nbs_voxels[i];
            cur_node->voxels.resize(cur_node->node.voxels_counter);
            cur_node->occupancy_indices.resize(cur_node->node.voxels_counter);
            
            srcs_device.push_back(exchangedVoxelsPointers[i]);
            dsts_host.push_back((CUdeviceptr)(cur_node->voxels.data() + old_counter));
            sizes.push_back(nbs_voxels[i] * sizeof(CPoint));

            srcs_device.push_back(exchangedGridsPointers[i]);
            dsts_host.push_back((CUdeviceptr)(cur_node->occupancy_indices.data() + old_counter));
            sizes.push_back(nbs_voxels[i] * sizeof(uint64_t));
        }
    }

    // Load the remaining data
    nb_copies = sizes.size();
    if(nb_copies > 0){
        CURuntime::assertCudaSuccess(cuMemcpyBatchAsync(
            dsts_host.data(), srcs_device.data(), sizes.data(), nb_copies, 
            batchStoringAttributes.data(), 
            batchStoringAttributesIndices.data(), 
            batchStoringAttributes.size(), 
            stream
        ));
        CURuntime::assertCudaSuccess(cuEventRecord(eventStoringComplete, stream));
    }
}










#include "visibility.h"

void GpuVersion::visibilityUpdate(CuRast* editor, CUcontext* context){
    // Get the frustum
    const mat4&  view = VKRenderer::view.view;
    const mat4&  proj = VKRenderer::view.proj;
    Frustum frustum = Frustum(proj * view);
    vec3 camera_pos = vec3(glm::inverse(view) * vec4(0.0f, 0.0f, 0.0f, 1.0f));

    // Get all visible nodes and initialise their distances to the camera
    std::vector<std::pair<CIdAABB, float>> visible_nodes = {};
    for(const CIdAABB& id : storedNodes){
        const CAABB& aabb = aabbsMap[id];
        if(frustum.doesIntersect(aabb, camera_pos)){
            float dist = glm::length(aabb.getCentroid() - camera_pos);
            visible_nodes.push_back({id, dist});
        }
    }

    // Order the nodes with respect to the camera
    std::sort(visible_nodes.begin(), visible_nodes.end(), 
        [](const std::pair<CIdAABB, float>& lhs, const std::pair<CIdAABB, float>& rhs){
            return lhs.second < rhs.second; // From closest to furthest 
        }
    );

    // Order to put parent before children
    // From claude
    {
        std::unordered_map<CIdAABB, size_t> indexOf = {};
        indexOf.reserve(visible_nodes.size());
        for(size_t i = 0; i < visible_nodes.size(); i++){
            indexOf[visible_nodes[i].first] = i;
        }

        std::vector<std::pair<CIdAABB, float>> ordered = {};
        ordered.reserve(visible_nodes.size());
        std::vector<bool> placed(visible_nodes.size(), false);

        std::vector<CIdAABB> ancestorChain; // scratch, reused per node
        for(size_t i = 0; i < visible_nodes.size(); i++){
            if(placed[i]){continue;}

            // Climb from this node's parent upward, collecting ancestors
            // that are themselves in visible_nodes and not yet placed.
            ancestorChain.clear();
            CIdAABB parent = parentsMap[visible_nodes[i].first];
            while(parent != CINVALID_ID){
                auto it = indexOf.find(parent);
                if(it == indexOf.end()){break;} // parent isn't in the visible set, stop
                size_t parentIndex = it->second;
                if(placed[parentIndex]){break;} // parent (and its own ancestors) already placed
                ancestorChain.push_back(parent);
                parent = parentsMap[parent];
            }

            // ancestorChain was built immediate-parent-first, reverse so we
            // emit the outermost ancestor first, then down to the immediate parent.
            for(auto it = ancestorChain.rbegin(); it != ancestorChain.rend(); ++it){
                size_t idx = indexOf[*it];
                ordered.push_back(visible_nodes[idx]);
                placed[idx] = true;
            }

            ordered.push_back(visible_nodes[i]);
            placed[i] = true;
        }

        visible_nodes = std::move(ordered);
    }


    // Gather the correct number of nodes to send to the device
    CIdAABB* visibility_cache_to_send = static_cast<CIdAABB*>(visibilityCache);
    CIdAABB* voxels_nodes_to_send = static_cast<CIdAABB*>(voxelsNodesToSend);
    uint32_t* cpt = (uint32_t*)nbRenderedNodes;
    uint32_t* point_cpt = (uint32_t*)nbRenderedPoints;
    uint32_t* voxel_cpt = (uint32_t*)nbRenderedVoxels;
    *cpt = 0;
    *point_cpt = 0;
    *voxel_cpt = 0;

    std::vector<CUdeviceptr> srcs_host = {
        (CUdeviceptr)nbRenderedNodes,
        (CUdeviceptr)nbRenderedPoints,
        (CUdeviceptr)nbRenderedVoxels
    };
    CUdeviceptr dst_nb_nodes = deviceStaging + (
        reinterpret_cast<uintptr_t>(&(hostStaging.visibilityCacheCurrentSize))
        - reinterpret_cast<uintptr_t>(&hostStaging)
    ); 
    CUdeviceptr dst_nb_points = deviceStaging + (
        reinterpret_cast<uintptr_t>(&(hostStaging.nbRenderedPoints))
        - reinterpret_cast<uintptr_t>(&hostStaging)
    ); 
    CUdeviceptr dst_nb_voxels = deviceStaging + (
        reinterpret_cast<uintptr_t>(&(hostStaging.nbRenderedVoxels))
        - reinterpret_cast<uintptr_t>(&hostStaging)
    ); 
    std::vector<CUdeviceptr> dsts_device = {
        dst_nb_nodes,
        dst_nb_points,
        dst_nb_voxels
    };
    std::vector<uint64_t> sizes = {
        sizeof(uint32_t), sizeof(uint32_t), sizeof(uint32_t)
    };

    // Get the LRU_VISIBILTY_CACHE closest nodes
    uint32_t loop_end = min(uint32_t(visible_nodes.size()), OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE);
    for(uint32_t i = 0; i < loop_end; i++){
        const std::pair<CIdAABB, float>& visible_node = visible_nodes[i];
        CIdAABB cur_node = visible_node.first;
        // if(currentlyInUpdatesCache.contains(cur_node)){continue;}

        hostCache->add(cur_node);
        std::shared_ptr<HostStorageNode> node = nullptr;
        // Load wanted node
        if(persistentStoredNodes.contains(cur_node)){
            node = persistentStoredNodes[cur_node];
        } else {
            node = std::make_shared<HostStorageNode>();
            OctreeNodeSerializable::deserializeV2(node.get(), cur_node, "From vis update");
            persistentStoredNodes[cur_node] = node;
        }

        bool has_points = (node->node.points_counter > 0);
        bool has_voxels = (node->node.voxels_counter > 0);
        bool points_can_be_added = 
            // Only send if the maximum of voxels to send is not reached
            (*point_cpt < OocSimLodSettings::MAX_NB_RENDERED_POINTS)
            // Only send if has points
            && (node->node.points_counter > 0)
            // Only send if all points are loaded
            && (node->node.points_counter + *point_cpt <= OocSimLodSettings::MAX_NB_RENDERED_POINTS)
        ;
        bool voxels_can_be_added =
            // Only send if the maximum of voxels to send is not reached
            (*voxel_cpt < OocSimLodSettings::MAX_NB_RENDERED_VOXELS)
            // Only send if has voxels
            && (node->node.voxels_counter > 0)
            // Only send if all voxels are loaded
            && (node->node.voxels_counter + *voxel_cpt <= OocSimLodSettings::MAX_NB_RENDERED_VOXELS)
        ;

        bool points_ok = !has_points || points_can_be_added;
        bool voxels_ok = !has_voxels || voxels_can_be_added;
        if(points_ok && voxels_ok){
            // Add points
            if(has_points){
                uint32_t new_total_points = *point_cpt + node->node.points_counter;
                uint32_t nb_new_points = node->node.points_counter;
                // srcs_host.push_back((CUdeviceptr)node->points.data());
                srcs_host.push_back((CUdeviceptr)node->points);
                dsts_device.push_back((CUdeviceptr)hostStaging.renderedPoints + (CUdeviceptr)(*point_cpt * sizeof(CPoint)));
                sizes.push_back(nb_new_points * sizeof(CPoint));
                *point_cpt = new_total_points;
            }

            // Add voxels
            if(has_voxels){
                uint32_t new_total_voxels = *voxel_cpt + node->node.voxels_counter;
                uint32_t nb_new_voxels = node->node.voxels_counter;
                srcs_host.push_back((CUdeviceptr)node->voxels.data());
                dsts_device.push_back((CUdeviceptr)hostStaging.renderedVoxels + (CUdeviceptr)(*voxel_cpt * sizeof(CPoint)));
                sizes.push_back(nb_new_voxels * sizeof(CPoint));
            
                // Add other voxels properties
                for(uint32_t voxel_id = 0; voxel_id < nb_new_voxels; voxel_id++){
                    voxels_nodes_to_send[*voxel_cpt + voxel_id] = cur_node;
                }
                *voxel_cpt = new_total_voxels;
            }

            // Add the node
            visibility_cache_to_send[*cpt] = cur_node;
            (*cpt)++;
        }

        if(*point_cpt >= OocSimLodSettings::MAX_NB_RENDERED_POINTS && *voxel_cpt >= OocSimLodSettings::MAX_NB_RENDERED_VOXELS){
            break;
        }
        if(*cpt >= OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE){
            break;
        }
    }

    // Add other voxels properties
    if(*voxel_cpt > 0){
        srcs_host.push_back((CUdeviceptr)voxels_nodes_to_send);
        dsts_device.push_back((CUdeviceptr)hostStaging.renderedVoxelsNodes);
        sizes.push_back(*voxel_cpt * sizeof(CIdAABB));
    }

    // Add visibility cache
    if(*cpt > 0){
        srcs_host.push_back((CUdeviceptr)visibility_cache_to_send);
        dsts_device.push_back((CUdeviceptr)hostStaging.visibilityCache);
        sizes.push_back(*cpt * sizeof(CIdAABB));
    }

    // Send the data to the device
    uint64_t nb_copies = sizes.size();
    CURuntime::assertCudaSuccess(cuMemcpyBatchAsync(
        dsts_device.data(), srcs_host.data(), sizes.data(), nb_copies, 
        batchLoadingAttributes.data(), 
        batchLoadingAttributesIndices.data(), 
        batchLoadingAttributes.size(), 
        stream
    ));
    CURuntime::assertCudaSuccess(cuEventRecord(eventVisibilityUpdateComplete, stream));
    // println("Nb nodes: {}, nb voxels: {}, nb points: {}\n", *cpt, *voxel_cpt, *point_cpt);
}



void GpuVersion::updateHostCache(){
    std::vector<std::shared_ptr<HostStorageNode>> nodes_to_store = {};
    for(auto it = persistentStoredNodes.begin(); it != persistentStoredNodes.end();){
        const CIdAABB& id = it->first;
        std::shared_ptr<HostStorageNode> node = it->second;
        if(!hostCache->contains(id)){
            nodes_to_store.push_back(node);
            it = persistentStoredNodes.erase(it);
        } else {
            it++;
        }
    }

    CURuntime::assertCudaSuccess(cuEventSynchronize(eventStoringComplete));
    CURuntime::assertCudaSuccess(cuEventSynchronize(eventVisibilityUpdateComplete));

    // Store all nodes in parallel
    if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
        std::for_each(std::execution::par, nodes_to_store.begin(), nodes_to_store.end(), [](std::shared_ptr<HostStorageNode>& node){
            OctreeNodeSerializable::serializeV2(node);
        });
    } else {
        std::for_each(nodes_to_store.begin(), nodes_to_store.end(), [](std::shared_ptr<HostStorageNode>& node){
            OctreeNodeSerializable::serializeV2(node);
        });
    }

    // Deallocate old points
    std::for_each(nodes_to_store.begin(), nodes_to_store.end(), [](std::shared_ptr<HostStorageNode>& node){
        PointsAllocator::deallocate(node->points);
        node->points = nullptr;
    });
}





void GpuVersion::updateOctree(CuRast* editor, CUcontext* context){
    if(isTakingScreenshots){return;}
	cuCtxSetCurrent(*context);

    GpuVersionUI::lastUpdateStart = high_resolution_clock::now();
    if(GpuVersionUI::nbTotalUpdates == 0){
        GpuVersionUI::firstUpdateStart = GpuVersionUI::lastUpdateStart;
    }

    bool skip_update = false;
    // Only load new points if previous points have been handled
    if(*(bool*)isDoneLoading && *(bool*)isDoneStoring && *(bool*)isDoneIterating){
        // Only run the update if the first batch has been loaded
        skip_update = !LoaderGpuVersion::run(editor, context);
    }

    if(!skip_update){
        // Only run the initialisation kernel once
        if(!isInitialised){
            octreeUpdateInit(editor, context);
            isInitialised = true;
        }

        // Only run the bottom up kernels if nothing else is stalling
        if(*(bool*)isDoneLoading && *(bool*)isDoneStoring && *(bool*)isDoneIterating){
            octreeUpdateBottomUp(editor, context);
        }

        octreeUpdateSimLOD(editor, context);

        if(*(bool*)isDoneLoading && *(bool*)isDoneIterating){
            octreeUpdateCacheUpdate(editor, context);
        }

        if(*(bool*)isDoneLoading && *(bool*)isDoneStoring && *(bool*)isDoneIterating){
            OptionalLaunchSettings launch_settings = {
                .gridsize = 1,
                .blocksize = 1
            };
            prog->launch("kernel_reset_batches", {}, launch_settings);
        }

        GpuVersionUI::update();
    }

    if(isInitialised){
        GpuVersion::visibilityUpdate(editor, context);
        updateHostCache();
    }
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

    // Render nodes
    {
        // uint32_t block_size = min(
        //     OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM,
        //     OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
        // );
        // OptionalLaunchSettings launch_settings = {
        //     .gridsize = OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM,
        //     .blocksize = block_size
        // };
        uint32_t block_size = 256;
        uint32_t grid_size =
            (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1) 
            / block_size
        ;
        OptionalLaunchSettings launch_settings = {
            .gridsize  = grid_size,
            .blocksize = block_size
        };

        if(isTakingScreenshots){
            prog->launch("kernel_test_multi_resolution", {&real_target, &real_settings, &randomOffset}, launch_settings);
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





void GpuVersionUI::update() {
    std::chrono::time_point<std::chrono::high_resolution_clock> now = high_resolution_clock::now();
    uint64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - lastUpdateStart
    ).count();

    COPY_FROM_GPU(nbTotalUpdates, &nbTotalUpdates, uint32_t);
    if(lastNbTotalUpdates == nbTotalUpdates){return;}
    lastNbTotalUpdates = nbTotalUpdates;

    COPY_FROM_GPU(curNbNodes, &currentNbNodes, uint32_t);
    COPY_FROM_GPU(currentNbChunks, &currentNbChunks, uint32_t);
    COPY_FROM_GPU(currentNbGrids, &currentNbGrids, uint32_t);
    COPY_FROM_GPU(currentNbPoints, &currentNbPoints, uint32_t);
    COPY_FROM_GPU(currentNbVoxels, &currentNbVoxels, uint32_t);

    COPY_FROM_GPU(nbTotalPoints, &nbTotalPoints, uint32_t);
    COPY_FROM_GPU(nbTotalVoxels, &nbTotalVoxels, uint32_t);
    COPY_FROM_GPU(nbTotalNewNodes, &nbTotalNewNodes, uint32_t);
    COPY_FROM_GPU(nbTotalNewGrids, &nbTotalNewGrids, uint32_t);
    COPY_FROM_GPU(nbTotalNewChunks, &nbTotalNewChunks, uint32_t);
    COPY_FROM_GPU(nbTotalDeletedNodes, &nbTotalDeletedNodes, uint32_t);
    COPY_FROM_GPU(nbTotalDeletedGrids, &nbTotalDeletedGrids, uint32_t);
    COPY_FROM_GPU(nbTotalDeletedChunks, &nbTotalDeletedChunks, uint32_t);
    COPY_FROM_GPU(nbTotalLoadedNodes, &nbTotalLoadedNodes, uint32_t);
    COPY_FROM_GPU(nbTotalSplitNodes, &nbTotalSplitNodes, uint32_t);
    COPY_FROM_GPU(nbTotalStoredNodes, &nbTotalStoredNodes, uint32_t);
    
    COPY_FROM_GPU(nbNewPointsThisUpdate, &nbNewPointsThisUpdate, uint32_t);
    COPY_FROM_GPU(nbNewVoxelsThisUpdate, &nbNewVoxelsThisUpdate, uint32_t);
    COPY_FROM_GPU(nbNewNodesThisUpdate, &nbNewNodesThisUpdate, uint32_t);
    COPY_FROM_GPU(nbLoadedNodesThisUpdate, &nbLoadedNodesThisUpdate, uint32_t);
    COPY_FROM_GPU(nbStoredNodesThisUpdate, &nbStoredNodesThisUpdate, uint32_t);
    COPY_FROM_GPU(nbSplitNodesThisUpdate, &nbSplitNodesThisUpdate, uint32_t);
    COPY_FROM_GPU(nbDeletedNodesThisUpdate, &nbDeletedNodesThisUpdate, uint32_t);
    COPY_FROM_GPU(nbDeletedChunksThisUpdate, &nbDeletedChunksThisUpdate, uint32_t);
    COPY_FROM_GPU(nbDeletedGridsThisUpdate, &nbDeletedGridsThisUpdate, uint32_t);
    COPY_FROM_GPU(nbNewChunksThisUpdate, &nbNewChunksThisUpdate, uint32_t);
    COPY_FROM_GPU(nbNewGridsThisUpdate, &nbNewGridsThisUpdate, uint32_t);

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

    uint64_t total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - firstUpdateStart
    ).count();
    const uint64_t cur_updates_per_seconds = static_cast<uint64_t>(1000.0 * static_cast<uint64_t>(nbTotalUpdates) / total_duration);
    updateStats(cur_updates_per_seconds, minNbUpdatesPerSecond, maxNbUpdatesPerSecond, avgNbUpdatesPerSecond);

}























/// For another project
void GpuVersion::takeRandomScreenShots(){
    static const uint32_t NB_RESOLUTIONS = 4;
    static const uint32_t NB_SCREENSHOTS = 256;
    static const uint32_t NB_TOTAL_SCREENSHOTS = NB_SCREENSHOTS * NB_RESOLUTIONS * 2;

    // How far the camera sphere's radius can range, relative to the AABB diagonal.
    const double MIN_RADIUS_FACTOR = 0.;
    const double MAX_RADIUS_FACTOR = 1.;
    const double TWO_PI = 6.283185307179586;

    static bool oldSettingsIsLeftDown = false;
    static double oldSettingsScroll = 0;
    static double oldSettingsPosX = 0;
    static double oldSettingsPosY = 0;
    static uint32_t oldSettingsNbPointsPerAxis = 0;
    static vec4 oldSettingsBackgroundColor = {};
    static double oldSettingsYaw = 0;
    static double oldSettingsPitch = 0;
    static double oldSettingsRadius = 0;
    static glm::dvec3 oldSettingsTarget = {};

    static uint32_t screenshotCounter = NB_TOTAL_SCREENSHOTS;
    static bool buttonWasPressed = false;

    // Camera + background shared within one (perturbed, target) pair.
    static double pairYaw = 0;
    static double pairPitch = 0;
    static double pairRadius = 0;
    static vec4 pairBg = {};

    static CAABB mainAABB = {};

    static uint32_t old_width = VKRenderer::width;
    static uint32_t old_height = VKRenderer::height;

    static uint32_t INITIAL_ID = 768;

    // Begin screenshots
    if(CuRastSettings::bruteForceRendering && !buttonWasPressed){
        buttonWasPressed = true;
        screenshotCounter = 0;

        // Save old settings
        oldSettingsBackgroundColor = CuRastSettings::background;
        oldSettingsIsLeftDown      = Runtime::controls->isLeftDown;
        oldSettingsScroll          = Runtime::mouseEvents.wheel_y;
        oldSettingsPosX            = Runtime::mouseEvents.pos_x;
        oldSettingsPosY            = Runtime::mouseEvents.pos_y;
        oldSettingsYaw             = Runtime::controls->yaw;
        oldSettingsPitch           = Runtime::controls->pitch;
        oldSettingsRadius          = Runtime::controls->radius;
        oldSettingsTarget          = Runtime::controls->target;

        // // Get the bounding box of the scene
        // CUdeviceptr main_octree = 0;
        // CIdAABB main_id = CINVALID_ID;
        // CGlobalVariables::Relationship main_relationship = {};
        // COPY_FROM_GPU(mainOctree, &main_octree, CUdeviceptr);
        // COctreeNode tmp = {};
        // const uint64_t pad =
        //     reinterpret_cast<uintptr_t>(&(tmp.aabb_index)) -
        //     reinterpret_cast<uintptr_t>(&tmp);
        // CUdeviceptr src_device = main_octree + pad;
        // CURuntime::assertCudaSuccess(
        //     cuMemcpyDtoH(&main_id, src_device, sizeof(CIdAABB))
        // );
        // src_device = reinterpret_cast<CUdeviceptr>(hostStaging.relationshipMap) 
        //     + static_cast<CUdeviceptr>(main_id * sizeof(CGlobalVariables::Relationship))
        // ;
        // CURuntime::assertCudaSuccess(
        //     cuMemcpyDtoH(&main_relationship, src_device, sizeof(CGlobalVariables::Relationship))
        // );
        // mainAABB = main_relationship.aabb;

        for(auto& [id, node] : GpuVersion::persistentStoredNodes){
            for(uint32_t i = 0; i < node->node.points_counter; i++){
                const CPoint& point = node->points[i];
                mainAABB.mins.x = min(mainAABB.mins.x, point.position.x);
                mainAABB.mins.y = min(mainAABB.mins.y, point.position.y);
                mainAABB.mins.z = min(mainAABB.mins.z, point.position.z);
                mainAABB.maxs.x = max(mainAABB.maxs.x, point.position.x);
                mainAABB.maxs.y = max(mainAABB.maxs.y, point.position.y);
                mainAABB.maxs.z = max(mainAABB.maxs.z, point.position.z);
            }
        }

        // println("HOST: .mins = ({}, {}, {}), .maxs = ({}, {}, {})", 
        //     mainAABB.mins.x, mainAABB.mins.y, mainAABB.mins.z,
        //     mainAABB.maxs.x, mainAABB.maxs.y, mainAABB.maxs.z
        // );
        // throw(EXIT_FAILURE);

        for(int i = INITIAL_ID; i <= 10'000'000; i++){
			fs::create_directories("./screenshots");
			std::string path = format("./screenshots/id_{}_res_{}x{}_perturbed.png",
                i, VKRenderer::width, VKRenderer::height
            );
            INITIAL_ID = i;
			if(!fs::exists(path)) break;
		}


        GpuVersion::isTakingScreenshots = true;
    }

    // Reset after screenshots are done
    if(buttonWasPressed && screenshotCounter == NB_TOTAL_SCREENSHOTS){
        Runtime::controls->isLeftDown = oldSettingsIsLeftDown;
        Runtime::mouseEvents.wheel_y  = oldSettingsScroll;
        Runtime::mouseEvents.pos_x    = oldSettingsPosX;
        Runtime::mouseEvents.pos_y    = oldSettingsPosY;
        CuRastSettings::background    = oldSettingsBackgroundColor;

        Runtime::controls->yaw    = oldSettingsYaw;
        Runtime::controls->pitch  = oldSettingsPitch;
        Runtime::controls->radius = oldSettingsRadius;
        Runtime::controls->target = oldSettingsTarget;
        Runtime::controls->update();

        buttonWasPressed = false;
        CuRastSettings::bruteForceRendering = false;

        VKRenderer::width = old_width;
        VKRenderer::height = old_height;

        GpuVersion::isTakingScreenshots = false;
    }

    // If screenshots
    if(screenshotCounter < NB_TOTAL_SCREENSHOTS){
        fs::create_directories("./screenshots");

        // We drive yaw/pitch/radius directly now, so no fake dragging needed,
        // and zero out any leftover real scroll so it can't sneak a zoom in
        // on a frame while we wait for the screenshot request to be handled.
        Runtime::controls->isLeftDown = false;
        Runtime::mouseEvents.wheel_x = 0;
        Runtime::mouseEvents.wheel_y = 0;

        vec3 aabbMin = mainAABB.mins;
        vec3 aabbMax = mainAABB.maxs;
        glm::dvec3 center = glm::dvec3((aabbMin + aabbMax) * 0.5f);
        double diagonal = double(glm::length(aabbMax - aabbMin));

        uint32_t pairId = screenshotCounter / (2*NB_RESOLUTIONS) + INITIAL_ID;

        if(screenshotCounter % (2*NB_RESOLUTIONS) == 0){
            // First half of the pair: roll a fresh random sphere radius and a
            // random position on that sphere, plus a random background. This
            // camera gets reused by the matching "target" shot right after it.
            double u1 = double(rand()) / double(RAND_MAX);
            double u2 = double(rand()) / double(RAND_MAX);
            double u3 = double(rand()) / double(RAND_MAX);

            double radiusFactor = MIN_RADIUS_FACTOR + u1 * (MAX_RADIUS_FACTOR - MIN_RADIUS_FACTOR);
            pairRadius = radiusFactor * diagonal;

            pairYaw = u2 * TWO_PI;
            // uniform sampling on the sphere surface (avoids bunching near the poles)
            pairPitch = std::asin(2.0 * u3 - 1.0);

            // pairBg.r = float(double(rand()) / double(RAND_MAX));
            // pairBg.g = float(double(rand()) / double(RAND_MAX));
            // pairBg.b = float(double(rand()) / double(RAND_MAX));
            pairBg.r = 1.0f;
            pairBg.g = 1.0f;
            pairBg.b = 1.0f;
            pairBg.a = 1.0f;

            Runtime::controls->target = center;
            Runtime::controls->radius = pairRadius;
            Runtime::controls->yaw    = pairYaw;
            Runtime::controls->pitch  = pairPitch;
            Runtime::controls->update();

            randomOffset = 1 + rand();

            VKRenderer::width = old_width;
            VKRenderer::height = old_height;
            CuRastSettings::background = pairBg;

            CuRastSettings::requestScreenshot = std::make_shared<string>(
                format("./screenshots/id_{}_res_{}x{}_perturbed.png",
                    pairId,
                    VKRenderer::width, VKRenderer::height
                )
            );

        } else if(screenshotCounter % (2*NB_RESOLUTIONS) == NB_RESOLUTIONS) {
            // Second half of the pair: ground-truth shot, same camera and
            // background as the perturbed shot above — only voxelsPointsPerAxis differs.
            Runtime::controls->target = center;
            Runtime::controls->radius = pairRadius;
            Runtime::controls->yaw    = pairYaw;
            Runtime::controls->pitch  = pairPitch;
            Runtime::controls->update();

            randomOffset = 0;

            VKRenderer::width = old_width;
            VKRenderer::height = old_height;
            CuRastSettings::background = pairBg;

            CuRastSettings::requestScreenshot = std::make_shared<string>(
                format("./screenshots/id_{}_res_{}x{}_target.png",
                    pairId,
                    VKRenderer::width, VKRenderer::height
                )
            );
        } else if(screenshotCounter % (2*NB_RESOLUTIONS) < NB_RESOLUTIONS) {
            VKRenderer::width = (old_width >> (screenshotCounter % (2*NB_RESOLUTIONS)));
            VKRenderer::height = (old_height >> (screenshotCounter % (2*NB_RESOLUTIONS)));
            CuRastSettings::background = pairBg;

            CuRastSettings::requestScreenshot = std::make_shared<string>(
                format("./screenshots/id_{}_res_{}x{}_perturbed.png",
                    pairId,
                    VKRenderer::width, VKRenderer::height
                )
            );
        } else if(screenshotCounter % (2*NB_RESOLUTIONS) > NB_RESOLUTIONS) {
            VKRenderer::width = (old_width >> (screenshotCounter % NB_RESOLUTIONS));
            VKRenderer::height = (old_height >> (screenshotCounter % NB_RESOLUTIONS));
            CuRastSettings::background = pairBg;

            CuRastSettings::requestScreenshot = std::make_shared<string>(
                format("./screenshots/id_{}_res_{}x{}_target.png",
                    pairId,
                    VKRenderer::width, VKRenderer::height
                )
            );
        }

        screenshotCounter++;
    }
}
#include "gpuVersion.h"

#include "loader.h"

void GpuVersion::initBuffers(CuRast* editor, CUcontext* context) {
    // Unbounded data
    hostStaging.maxNbAABBs = OocSimLodSettings::INITIAL_MAX_NB_NODES;
    hostStaging.relationshipMap = alloc<CGlobalVariables::Relationship>(hostStaging.maxNbAABBs);
    hostStaging.allAABBs = alloc<CAABB>(hostStaging.maxNbAABBs);
    hostStaging.nodes = alloc<COctreeNode*>(hostStaging.maxNbAABBs);
    

    // Exchangeable data
    hostStaging.maxNbNodesToLoad = OocSimLodSettings::MAX_NB_NODES_TO_LOAD;
    hostStaging.nodesToLoadBuffer = alloc<CIdAABB>(hostStaging.maxNbNodesToLoad);

    hostStaging.maxNbNodesToStore = OocSimLodSettings::MAX_NB_NODES_TO_STORE;
    hostStaging.nodesToStoreBuffer = alloc<COctreeNode*>(hostStaging.maxNbNodesToStore);

    hostStaging.maxNbBatches = OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE;
    hostStaging.batchesAddedMask = alloc<uint32_t>(hostStaging.maxNbBatches);
    hostStaging.batchesToAddCounts = alloc<uint32_t>(hostStaging.maxNbBatches);
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
    hostStaging.maxNbResidualPoints = 1'000'000;
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
    CUdeviceptr global_variables_ptr = prog->getGlobalsPointer("globalVariables");
    if (global_variables_ptr == 0) {
        throw std::runtime_error("globalVariables symbol not found");
    }
    CURuntime::assertCudaSuccess(cuMemcpyHtoD(global_variables_ptr, &hostStaging, sizeof(CGlobalVariables)));
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
        "./src/kernels/ooc/render.cu",
        "./src/kernels/ooc/test.cu",
    });

    CUstream stream;
    CURuntime::assertCudaSuccess(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    initBuffers(editor, context);
    println("buffer initialised");
    initAllocators(editor, context, &stream);
    println("allocator initialised");
    cudaDeviceSynchronize();
    CURuntime::assertCudaSuccess(cuStreamQuery(stream));
    CURuntime::assertCudaSuccess(cuStreamDestroy(stream));

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
    prog->launch("kernel_init", {&grid_size}, launch_settings);
    // prog->launch("kernel_test", {}, launch_settings);

    cudaDeviceSynchronize();

    LoaderGpuVersion::init();
}

void GpuVersion::destroy(CuRast *editor, CUcontext *context){
    for(CUdeviceptr& ptr : pointers){
        if(ptr){
            CURuntime::assertCudaSuccess(cuMemFree(ptr));
        }
    }
    free(hostStaging.batchesToAddPointsPointers);
}